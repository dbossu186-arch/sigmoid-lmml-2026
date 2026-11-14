"""Twins Separated at Birth - find pairs of images that come from the same source photo.

Pipeline
1. Decode every image once (parallel): pHash and SIFT keypoints.
2. Candidates: 8 nearest neighbours by ResNet-50 embedding (cosine) + pairs with
   close pHash. Keeps ~98% of true pairs on train while checking only ~6 per image.
3. Verify each candidate:
   - SIFT matches + RANSAC homography; share of keypoints that agree (frac),
   - warp one image onto the other and measure pixel agreement (lo, hi).
4. Each image keeps at most its 2 best partners (duplicate groups have 2-3 images).

Reads test-input.zip, writes output.csv (image_a,image_b) in the working directory.
Needs: numpy, pillow, opencv-python; torch + torchvision for step 2. If ResNet cannot
run (missing library, no weights, not enough memory) it falls back to hash candidates.

Resource limits: SIFT features are stored in small memory-mapped shard files that all
workers share (sized from the file-size limit), ResNet runs in the main process only
after the decode workers have exited, reading images straight from the zip in small
batches, and the number of workers is chosen from the memory limit.
"""
import csv
import gc
import os
import shutil
import tempfile
import time
import zipfile
from collections import Counter
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps

ZIP = "test-input.zip"
OUT = "output.csv"
EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
WEIGHTS_FILE = "resnet50-11ad3fa6.pth"  # optional local copy next to main.py

SIFT_SIDE = 640        # images are resized to this longest side for SIFT / alignment
SIFT_N = 800           # keypoints per image
TOPK = 8               # embedding neighbours per image
PHASH_MAX = 14         # pHash Hamming distance that also makes a candidate
MIN_FRAC = 0.16        # RANSAC inliers / keypoints
MIN_LO = 0.80          # coarse pixel agreement after alignment
MIN_HI = 0.20          # fine-detail agreement after alignment
MAX_PARTNERS = 2
MAX_WORKERS = 4
EMBED_BATCH = 8        # small batches keep ResNet's activations around 100 MB
MAX_SHARD = 400        # images per feature shard (about 40 MB of descriptors)
MB = 1024 * 1024

_ZF = None
_DESC = _PTS = None    # lists of per-shard memmaps
_CNT = None
_SHARD = MAX_SHARD


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------- resources ----------------
def memory_limit():
    """Container memory limit in bytes (cgroup v2/v1), else physical RAM, else None."""
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(p).read().strip()
            if v != "max" and int(v) < (1 << 50):
                return int(v)
        except Exception:
            pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:
        return None


def file_size_limit():
    """Largest file this process may write (RLIMIT_FSIZE), None if unlimited/unknown."""
    try:
        import resource
        soft, _ = resource.getrlimit(resource.RLIMIT_FSIZE)
        return None if soft in (resource.RLIM_INFINITY, -1) else soft
    except Exception:
        return None


def cpu_count():
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1


def pick_workers(limit):
    w = min(MAX_WORKERS, cpu_count())
    if limit:  # ~600 MB for the main process, ~250 MB per worker
        w = min(w, max(1, int((limit / MB - 600) // 250)))
    return max(1, w)


def pick_shard(fsize):
    per_image = SIFT_N * 128  # descriptor bytes per image (the largest array)
    if not fsize:
        return MAX_SHARD
    return int(max(8, min(MAX_SHARD, (fsize // 2) // per_image)))


# ---------------- workers ----------------
def _init_zip(path):
    global _ZF
    cv2.setNumThreads(1)
    _ZF = zipfile.ZipFile(path)


def _init_match(path, feat_dir, shard, nshards):
    global _DESC, _PTS, _CNT, _SHARD
    _init_zip(path)
    _SHARD = shard
    _DESC = [np.load(os.path.join(feat_dir, f"desc_{s}.npy"), mmap_mode="r") for s in range(nshards)]
    _PTS = [np.load(os.path.join(feat_dir, f"pts_{s}.npy"), mmap_mode="r") for s in range(nshards)]
    _CNT = np.load(os.path.join(feat_dir, "cnt.npy"))


def _open(member, side=SIFT_SIDE, zf=None):
    im = Image.open(BytesIO((zf or _ZF).read(member)))
    im.draft("RGB", (side, side))
    return ImageOps.exif_transpose(im).convert("RGB")


def _gray640(im):
    w, h = im.size
    s = SIFT_SIDE / max(w, h)
    g = im.convert("L").resize((max(8, round(w * s)), max(8, round(h * s))), Image.BILINEAR)
    return np.asarray(g, dtype=np.uint8)


def _phash(im):
    g = np.asarray(im.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float32)
    low = cv2.dct(g)[:8, :8]
    return (low > np.median(low)).flatten()


def decode(member):
    try:
        im = _open(member)
    except Exception:
        return None
    gray = _gray640(im)
    kp, d = cv2.SIFT_create(nfeatures=SIFT_N).detectAndCompute(gray, None)
    if d is None:
        pts, desc = np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8)
    else:
        pts, desc = np.float32([k.pt for k in kp]), np.clip(d, 0, 255).astype(np.uint8)
    return _phash(im), pts[:SIFT_N], desc[:SIFT_N]


def _feat(k):
    s, r = divmod(k, _SHARD)
    c = int(_CNT[k])
    return np.asarray(_PTS[s][r, :c]), np.asarray(_DESC[s][r, :c], dtype=np.float32)


def _ncc(a, b, m):
    a = a[m].astype(np.float64)
    b = b[m].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def verify(job):
    """Return (i, j, frac, lo, hi); zeros mean 'not a duplicate'."""
    i, j, mi, mj = job
    if _CNT[i] < 8 or _CNT[j] < 8:
        return (i, j, 0.0, 0.0, 0.0)
    p1, d1 = _feat(i)
    p2, d2 = _feat(j)
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [a for a, b in (x for x in knn if len(x) == 2) if a.distance < 0.75 * b.distance]
    if len(good) < 8:
        return (i, j, 0.0, 0.0, 0.0)
    src = p1[[g.queryIdx for g in good]]
    dst = p2[[g.trainIdx for g in good]]
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None or mask is None:
        return (i, j, 0.0, 0.0, 0.0)
    frac = float(mask.sum()) / max(1, min(len(d1), len(d2)))
    if frac < MIN_FRAC:
        return (i, j, frac, 0.0, 0.0)
    try:
        A = _gray640(_open(mi)).astype(np.float32)
        B = _gray640(_open(mj)).astype(np.float32)
    except Exception:
        return (i, j, frac, 0.0, 0.0)
    size = (B.shape[1], B.shape[0])
    W = cv2.warpPerspective(A, H, size, flags=cv2.INTER_LINEAR)
    ov = cv2.warpPerspective(np.ones_like(A), H, size, flags=cv2.INTER_NEAREST) > 0.5
    ov = cv2.erode(ov.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    if ov.sum() < 2000:
        return (i, j, frac, 0.0, 0.0)
    lo = _ncc(cv2.GaussianBlur(W, (0, 0), 2.0), cv2.GaussianBlur(B, (0, 0), 2.0), ov)
    e1 = cv2.Laplacian(cv2.GaussianBlur(W, (0, 0), 1.5), cv2.CV_32F)
    e2 = cv2.Laplacian(cv2.GaussianBlur(B, (0, 0), 1.5), cv2.CV_32F)
    hi = _ncc(e1, e2, ov)
    return (i, j, frac, lo, hi)


# ---------------- main process ----------------
def write_pairs(pairs):
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_a", "image_b"])
        w.writerows(pairs)


def _small(zf, member):
    try:
        im = _open(member, side=448, zf=zf)
        return np.asarray(im.resize((224, 224), Image.BILINEAR), dtype=np.uint8)
    except Exception:
        return None


def embed_all(zpath, members, valid):
    """ResNet-50 features for every valid image, or None if ResNet cannot run here.
    Images are read straight from the zip in small batches; nothing is written to disk."""
    try:
        import torch
        from torchvision.models import resnet50, ResNet50_Weights

        here = os.path.dirname(os.path.abspath(__file__))
        local = [p for p in (os.path.join(here, WEIGHTS_FILE), os.path.join(os.getcwd(), WEIGHTS_FILE))
                 if os.path.exists(p)]
        if local:
            m = resnet50()
            m.load_state_dict(torch.load(local[0], map_location="cpu"))
        else:
            m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)  # cached or downloaded
        m.fc = torch.nn.Identity()
        dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        m = m.eval().to(dev)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        E = np.zeros((len(members), 2048), np.float32)
        idx = list(np.flatnonzero(valid))
        with zipfile.ZipFile(zpath) as zf, torch.inference_mode():
            for s in range(0, len(idx), EMBED_BATCH):
                pairs = [(k, _small(zf, members[k])) for k in idx[s:s + EMBED_BATCH]]
                pairs = [(k, a) for k, a in pairs if a is not None]
                if not pairs:
                    continue
                ks = [k for k, _ in pairs]
                x = torch.from_numpy(np.stack([a for _, a in pairs])).permute(0, 3, 1, 2).float() / 255
                f = m(((x - mean) / std).to(dev)).float().cpu()
                E[ks] = torch.nn.functional.normalize(f, dim=1).numpy()
        log(f"ResNet-50 embeddings done on {dev}")
        return E
    except Exception as ex:
        log(f"ResNet unavailable ({type(ex).__name__}: {str(ex)[:200]}); using pHash neighbours")
        return None
    finally:
        gc.collect()


def hamming(bits):
    X = bits.astype(np.float32)
    D = X @ (1 - X).T
    return D + D.T


def nearest(M, k, largest):
    """Indices of the k best entries per row (diagonal already masked)."""
    k = min(k, M.shape[1] - 1)
    return np.argpartition(-M if largest else M, k, axis=1)[:, :k]


def main():
    t0 = time.time()
    write_pairs([])  # a valid file exists even if something fails later
    zpath = os.path.abspath(ZIP)
    with zipfile.ZipFile(zpath) as z:
        members = sorted(
            m for m in z.namelist()
            if m.lower().endswith(EXTS) and not os.path.basename(m).startswith("._") and "__MACOSX" not in m
        )
    n = len(members)
    limit, fsize = memory_limit(), file_size_limit()
    workers, shard = pick_workers(limit), pick_shard(fsize)
    nshards = (n + shard - 1) // shard
    log(f"{n} images | memory limit {limit // MB if limit else '?'} MB | "
        f"file size limit {fsize // MB if fsize else 'none'} MB | cpus {cpu_count()} | "
        f"workers {workers} | {nshards} feature shards of {shard} images")
    if n < 2:
        return

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    feat_dir = tempfile.mkdtemp(prefix="twins_")
    try:
        descs, ptss = [], []
        for s in range(nshards):
            rows = min(shard, n - s * shard)
            descs.append(np.lib.format.open_memmap(os.path.join(feat_dir, f"desc_{s}.npy"), "w+", np.uint8, (rows, SIFT_N, 128)))
            ptss.append(np.lib.format.open_memmap(os.path.join(feat_dir, f"pts_{s}.npy"), "w+", np.float32, (rows, SIFT_N, 2)))
        cnt = np.zeros(n, np.int32)
        ph = np.zeros((n, 64), np.uint8)
        valid = np.zeros(n, bool)

        # ---- 1. decode
        with ctx.Pool(workers, initializer=_init_zip, initargs=(zpath,)) as pool:
            for k, r in enumerate(pool.imap(decode, members, chunksize=8)):
                if r is None:
                    continue
                p, pt, d = r
                s, row = divmod(k, shard)
                valid[k] = True
                ph[k] = p
                cnt[k] = len(d)
                descs[s][row, :len(d)] = d
                ptss[s][row, :len(pt)] = pt
        for a in descs + ptss:
            a.flush()
        del descs, ptss
        np.save(os.path.join(feat_dir, "cnt.npy"), cnt)
        log(f"decoded {valid.sum()} images ({time.time() - t0:.0f}s)")

        # ---- 2. candidate pairs (workers have exited; ResNet runs alone)
        E = embed_all(zpath, members, valid)
        cand = set()
        Dp = hamming(ph)
        np.fill_diagonal(Dp, 1e9)
        Dp[~valid] = 1e9
        Dp[:, ~valid] = 1e9
        ii, jj = np.where(np.triu(Dp <= PHASH_MAX, 1))
        cand |= set(zip(ii.tolist(), jj.tolist()))
        if E is not None:
            S = E @ E.T
            del E
            np.fill_diagonal(S, -2)
            S[~valid] = -2
            S[:, ~valid] = -2
            top = nearest(S, TOPK, largest=True)
            del S
        else:
            top = nearest(Dp, TOPK * 2, largest=False)
        del Dp
        gc.collect()
        for a in np.flatnonzero(valid):
            for b in top[a]:
                b = int(b)
                if a != b and valid[b]:
                    cand.add((min(a, b), max(a, b)))
        cand = sorted(cand)
        log(f"{len(cand)} candidate pairs ({time.time() - t0:.0f}s)")

        # ---- 3. verification
        jobs = [(a, b, members[a], members[b]) for a, b in cand]
        res = []
        with ctx.Pool(workers, initializer=_init_match, initargs=(zpath, feat_dir, shard, nshards)) as pool:
            for r in pool.imap_unordered(verify, jobs, chunksize=32):
                if r[2] >= MIN_FRAC and r[3] >= MIN_LO and r[4] >= MIN_HI:
                    res.append(r)
        log(f"verified ({time.time() - t0:.0f}s)")
    finally:
        shutil.rmtree(feat_dir, ignore_errors=True)

    # ---- 4. at most MAX_PARTNERS per image, best pairs first
    ok = sorted(((frac + hi, a, b) for a, b, frac, lo, hi in res), reverse=True)
    deg = Counter()
    pairs = set()
    for _, a, b in ok:
        if deg[a] < MAX_PARTNERS and deg[b] < MAX_PARTNERS:
            deg[a] += 1
            deg[b] += 1
            x, y = sorted((os.path.basename(members[a]), os.path.basename(members[b])))
            if x != y:
                pairs.add((x, y))
    write_pairs(sorted(pairs))
    log(f"wrote {len(pairs)} pairs to {OUT} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
