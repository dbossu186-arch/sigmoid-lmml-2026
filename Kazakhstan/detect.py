"""
Stamp of Disapproval - automatic localisation of the altered region on every album page.

Detector: non-aligned double-JPEG analysis ("which pixels lost the stamp's own JPEG history").

Every stamp image was JPEG-compressed once on its own (quality ~80-92) before it was placed on
the page at an arbitrary position, and the page was then saved again as JPEG (quality ~95). So
each stamp carries the 8x8 quantisation lattice of its first compression, on a grid that is
shifted against the page grid. The forger edited one small region *after* that first
compression, so in the edited region the luminance DCT coefficients, computed on the stamp's
own grid, no longer sit on multiples of the first quantisation steps.

Per page:
  1. stamp mask = everything that is not the dark album background;
  2. 8x8 block DCT of the luminance for all 64 grid offsets;
  3. for each block, choose the (offset, quality) hypothesis that best explains a 21x21-block
     neighbourhood (sum of per-block log-likelihood ratios "quantised vs not");
  4. per region using the same grid: re-estimate the quantiser step Q and residual sigma of
     each of the 63 AC coefficients from the data (IJG table only as a starting point);
  5. per-block log-likelihood ratio  LLR = sum_k log N(d_k; 0, s_k) - log U(-Q_k/2, Q_k/2),
     d_k = c_k - Q_k round(c_k/Q_k), over coefficients that are informative (|c_k| > 0.35 Q_k);
  6. anomaly = -(3x3 sum of LLR) on blocks fully inside a stamp; the strongest local maximum is
     the altered region. Its box is the connected set of blocks above 35 % of the peak.

Nothing here uses the "letter" layout: each page is scored independently. The letter is read
afterwards from the 26 page marks by fitting every rendered capital letter A-Z to them
(chance-corrected two-way stroke-distance score, see read_letter).

    python detect.py --data data --out .                 # full run (~45-60 min, 4 CPU threads, <1 GB RAM)
    python detect.py --data data --out . --validate      # also score the 30 exhibits against their masks
    python detect.py --data data --out . --cache candidates.csv   # reuse detections, redraw sheets only
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

BG = np.array([38, 36, 34])
LUM = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], np.float32).reshape(8, 8)
C8 = np.array([[np.sqrt((1 if k == 0 else 2) / 8) * np.cos((2 * n + 1) * k * np.pi / 16) for n in range(8)]
               for k in range(8)], np.float32)
QS = list(range(60, 97, 3))
WIN = 21
THR = 0.35
PAGE_Q = 97  # page quantisation table [1,1,1,1,1,2,3,4,...] = IJG quality 97
MODE = 'sim'
K0 = 1  # first DCT coefficient used (0 = include DC)


# ----------------------------------------------------------------------------- JPEG model
def qtable(q):
    s = 5000 / q if q < 50 else 200 - 2 * q
    return np.clip(np.floor((LUM * s + 50) / 100), 1, 255)


def luminance(im):
    f = im.astype(np.float32)
    return 0.299 * f[:, :, 2] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 0]


def stamp_mask(im):
    d = np.abs(im.astype(np.int16) - BG).max(2)
    m = cv2.morphologyEx((d > 30).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros(m.shape, np.uint8)
    for i in range(1, n):
        if st[i, 2] > 120 and st[i, 3] > 120:
            keep[lab == i] = 1
    cnts, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(keep)
    cv2.drawContours(out, cnts, -1, 1, -1)
    return out


def dct_all_offsets(Y):
    """(ox, oy) -> block DCT (nby, nbx, 64); block (i, j) covers rows 8i+oy.., cols 8j+ox.."""
    H, W = Y.shape
    nby, nbx = (H - 8) // 8, (W - 8) // 8
    out = {}
    for oy in range(8):
        for ox in range(8):
            b = Y[oy:oy + nby * 8, ox:ox + nbx * 8].reshape(nby, 8, nbx, 8).transpose(0, 2, 1, 3) - 128
            out[(ox, oy)] = np.einsum('ij,abjk,lk->abil', C8, b, C8).reshape(nby, nbx, 64)
    return out


def coef_llr(c, Q, s):
    d = c - Q * np.round(c / Q)
    return np.clip(-0.5 * (d / s) ** 2 - np.log(s * np.sqrt(2 * np.pi)) + np.log(Q), -5, 5)


def ijg_llr(D, Q, s=1.0):
    Q = Q.ravel()[None, None, :]
    inf = np.abs(D) > THR * Q
    inf[:, :, 0] = False
    return (coef_llr(D, Q, s) * inf).sum(2)


def refine(Dv, q0):
    """data-driven quantiser step Q and residual sigma s per AC coefficient (Q=0: unused)"""
    Qs = np.zeros(64)
    Ss = np.ones(64)
    Q0 = qtable(q0).ravel()
    for k in range(K0, 64):
        c = Dv[:, k]
        best = (0, 0, 1)
        for Q in range(2, int(max(6, Q0[k] * 3)) + 1):
            inf = np.abs(c) > THR * Q
            if inf.sum() < 15:
                continue
            d = c[inf] - Q * np.round(c[inf] / Q)
            s = max(0.5, 1.4826 * np.median(np.abs(d)))
            if s > Q / 3.5:
                continue
            sc = coef_llr(c[inf], Q, s).sum()
            if sc > best[0]:
                best = (sc, Q, s)
        Qs[k], Ss[k] = best[1], best[2]
    return Qs, Ss


def refined_llr(D, Qs, Ss):
    L = np.zeros(D.shape[:2], np.float32)
    for k in range(K0, 64):
        if Qs[k] < 2:
            continue
        c = D[:, :, k]
        inf = np.abs(c) > THR * Qs[k]
        L += np.where(inf, coef_llr(c, Qs[k], Ss[k]), 0)
    return L


def analyse(im):
    """per-block LLR on the locally chosen stamp grid, and the block-level stamp mask"""
    Ds = dct_all_offsets(luminance(im))
    nby, nbx = next(iter(Ds.values())).shape[:2]
    sm = stamp_mask(im)
    bm = sm[:nby * 8, :nbx * 8].reshape(nby, 8, nbx, 8).min((1, 3)).astype(np.float32)
    bm = cv2.erode(bm, np.ones((3, 3), np.uint8))  # drop blocks touching the stamp edge
    keys = [(k, q) for k in Ds for q in QS]
    best = np.full((nby, nbx), -np.inf, np.float32)
    arg = np.zeros((nby, nbx), np.int32)
    for i, ((ox, oy), q) in enumerate(keys):
        if ox == 0 and oy == 0 and q >= 93:  # that is the page's own (last) compression
            continue
        L = ijg_llr(Ds[(ox, oy)], qtable(q)).astype(np.float32)
        sup = cv2.boxFilter(L * bm, -1, (WIN, WIN), normalize=False, borderType=cv2.BORDER_CONSTANT)
        upd = sup > best
        best[upd] = sup[upd]
        arg[upd] = i
    groups = {}
    for i, (off, q) in enumerate(keys):
        sel = (arg == i) & (bm > 0)
        if sel.any():
            groups.setdefault(off, []).append((sel, q))
    out = np.zeros_like(bm)
    sim = np.zeros_like(bm)
    delta = np.zeros(im.shape[:2], np.float32)
    used = []
    for off, lst in groups.items():
        sel = np.zeros(bm.shape, bool)
        qn = []
        for s_, q in lst:
            sel |= s_
            qn += [q] * int(s_.sum())
        if sel.sum() < 30:
            continue
        Qs, Ss = refine(Ds[off][sel], int(np.median(qn)))
        out[sel] = refined_llr(Ds[off], Qs, Ss)[sel]
        used.append((off, sel, Qs, Ss))
        # null simulation: snap the blocks of this region onto the estimated first-compression lattice
        D = Ds[off][sel]
        Qv = np.where(Qs >= 2, Qs, 0)
        Dq = np.where(Qv > 0, Qv * np.round(D / np.maximum(Qv, 1)), D).reshape(-1, 8, 8)
        pix = np.einsum('ji,ajk,kl->ail', C8, Dq - D.reshape(-1, 8, 8), C8)
        ox, oy = off
        for (i, j), blk in zip(np.argwhere(sel), pix):
            delta[8 * i + oy:8 * i + oy + 8, 8 * j + ox:8 * j + ox + 8] = blk
    if used:
        simim = np.clip(np.round(im.astype(np.float32) + delta[:, :, None]), 0, 255).astype(np.uint8)
        simim = cv2.imdecode(cv2.imencode('.jpg', simim, [cv2.IMWRITE_JPEG_QUALITY, PAGE_Q])[1], 1)
        Ysim = luminance(simim)
        for off, sel, Qs, Ss in used:
            sim[sel] = refined_llr(block_dct(Ysim, off, (nby, nbx)), Qs, Ss)[sel]
    return out, bm, sim


def block_dct(Y, off, shape):
    ox, oy = off
    nby, nbx = shape
    b = Y[oy:oy + nby * 8, ox:ox + nbx * 8].reshape(nby, 8, nbx, 8).transpose(0, 2, 1, 3) - 128
    return np.einsum('ij,abjk,lk->abil', C8, b, C8).reshape(nby, nbx, 64)


def anomaly_map(L, bm, sim=None):
    x = -L if (sim is None or MODE == 'plain') else np.maximum(sim - L, 0)
    a = cv2.boxFilter(x * bm, -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
    a[bm < 1] = -1e9
    return a


def peaks(a, k=3):
    out = []
    a2 = a.copy()
    for _ in range(k):
        i = np.unravel_index(np.argmax(a2), a2.shape)
        if a2[i] < -1e8:
            break
        out.append((float(a2[i]), int(i[0]), int(i[1])))
        a2[max(0, i[0] - 12):i[0] + 13, max(0, i[1] - 12):i[1] + 13] = -1e9  # runner-ups >= ~100 px away
    return out


def region_box(a, by, bx):
    """pixel box of the connected blocks around a peak with anomaly above 35 % of the peak"""
    m = (a > 0.35 * a[by, bx]).astype(np.uint8)
    _, lab = cv2.connectedComponents(m)
    sel = lab == lab[by, bx]
    ys, xs = np.nonzero(sel)
    ys = np.clip(ys, by - 8, by + 8)
    xs = np.clip(xs, bx - 8, bx + 8)  # never more than ~130 px across
    x0, y0 = int(xs.min() * 8 - 4), int(ys.min() * 8 - 4)
    return x0, y0, int((xs.max() - xs.min() + 1) * 8 + 8), int((ys.max() - ys.min() + 1) * 8 + 8)


def detect(im, k=3):
    L, bm, sim = analyse(im)
    a = anomaly_map(L, bm, sim)
    pk = peaks(a, k)
    res = []
    for s, by, bx in pk:
        res.append(dict(score=s, cx=bx * 8 + 4, cy=by * 8 + 4, box=region_box(a, by, bx)))
    return res


def confidence(c):
    if not c:
        return 0.0
    s1 = max(c[0]['score'], 0)
    s2 = max(c[1]['score'], 0) if len(c) > 1 else 0
    return s1 / (s1 + s2 + 1e-6)


# ----------------------------------------------------------------------------- letters
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def letter_templates():
    """bold capital letters (Hershey simplex, thick stroke), cropped to their bounding box"""
    t = {}
    for ch in LETTERS:
        img = np.zeros((500, 500), np.uint8)
        cv2.putText(img, ch, (60, 400), cv2.FONT_HERSHEY_SIMPLEX, 12, 255, 40, cv2.LINE_AA)
        ys, xs = np.nonzero(img > 127)
        t[ch] = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 127
    return t


def read_letter(pts, weights, templates, sig=0.05):
    """Fit every capital letter to the marks and keep the best one.

    For each letter a box (height 500..800 px, aspect 0.55..1.0, shifted around the marks' median)
    is searched. Score = chance-corrected fraction of marks lying on the letter body
    (Gaussian of the distance to the stroke, in box-height units, minus what uniformly spread
    marks would get) + fraction of the letter body that has a mark nearby. Marks are weighted by
    their page confidence (1.0 if >= 0.7, else 0.3)."""
    pts = np.asarray(pts, np.float32)
    w = np.asarray(weights, np.float32)
    cx, cy = np.median(pts, 0)
    rng = np.random.default_rng(0)
    res = {}
    for ch, T in templates.items():
        best = (-1e9, None)
        for H in (500, 575, 650, 725, 800):
            for ar in (0.55, 0.7, 0.85, 1.0):
                Wd = int(H * ar)
                R = cv2.resize(T.astype(np.uint8), (Wd, H), interpolation=cv2.INTER_NEAREST)
                dist = cv2.distanceTransform((1 - R).astype(np.uint8), cv2.DIST_L2, 3) / H
                base = np.exp(-0.5 * (dist / sig) ** 2).mean()
                ys, xs = np.nonzero(R)
                idx = rng.choice(len(xs), min(300, len(xs)), replace=False)
                sk = np.stack([xs[idx], ys[idx]], 1).astype(np.float32)
                for dx in range(-150, 151, 25):
                    for dy in range(-150, 151, 25):
                        o = np.array([cx - Wd / 2 + dx, cy - H / 2 + dy], np.float32)
                        p = np.round(pts - o).astype(int)
                        inb = (p[:, 0] >= 0) & (p[:, 1] >= 0) & (p[:, 0] < Wd) & (p[:, 1] < H)
                        d = np.ones(len(p), np.float32)
                        d[inb] = dist[p[inb, 1], p[inb, 0]]
                        s1 = ((w * np.exp(-0.5 * (d / sig) ** 2)).sum() / w.sum() - base) / (1 - base)
                        dd = np.min(np.linalg.norm(sk[:, None] - (pts - o)[None], axis=2), 1) / H
                        s2 = np.exp(-0.5 * (dd / (2 * sig)) ** 2).mean()
                        if s1 + s2 > best[0]:
                            best = (float(s1 + s2), (int(o[0]), int(o[1]), Wd, H))
        res[ch] = best
    order = sorted(res, key=lambda c: -res[c][0])
    return order[0], [(c, round(res[c][0], 3)) for c in order[:3]], res[order[0]][1]


# ----------------------------------------------------------------------------- sheets
def draw_sheet(rows, title, letter=None, alts=None, fitted=None):
    p = np.full((1200, 1600, 3), 255, np.uint8)
    if fitted is not None:  # faint outline of the best-fitting letter, drawn under the marks
        T, (x0, y0, Wd, H) = fitted
        R = cv2.resize(T.astype(np.uint8), (Wd, H), interpolation=cv2.INTER_NEAREST)
        full = np.zeros((1200, 1600), np.uint8)
        xa, ya = max(x0, 0), max(y0, 0)
        xb, yb = min(x0 + Wd, 1600), min(y0 + H, 1200)
        full[ya:yb, xa:xb] = R[ya - y0:yb - y0, xa - x0:xb - x0]
        p[full > 0] = (235, 235, 235)
    for r in rows:
        x, y, w, h = r['x'], r['y'], r['w'], r['h']
        col = (0, 0, 200) if r['conf'] >= 0.7 else (0, 150, 255)
        cv2.rectangle(p, (x, y), (x + w, y + h), col, -1)
        if r.get('alt'):
            ax, ay = r['alt']
            cv2.circle(p, (ax, ay), 10, (170, 170, 170), 2)
        cv2.putText(p, r['page'].replace('page_', ''), (x + w + 4, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (110, 110, 110), 1, cv2.LINE_AA)
    cv2.putText(p, title, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 2, cv2.LINE_AA)
    if letter:
        cv2.putText(p, 'reads: %s   (fit: %s)' % (letter, ', '.join('%s %.2f' % a for a in alts)), (20, 1170),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(p, 'red: confident (>=0.7), orange: weaker; grey ring: runner-up; grey: fitted letter', (900, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 1, cv2.LINE_AA)
    return p


def overlay_page(im, cands):
    o = im.copy()
    for j, c in enumerate(cands):
        x, y, w, h = c['box']
        cv2.rectangle(o, (x, y), (x + w, y + h), (0, 0, 255) if j == 0 else (0, 200, 255), 3 if j == 0 else 1)
    return o


# ----------------------------------------------------------------------------- main
def validate(data, out):
    rows = []
    for f in sorted((data / 'exhibits').glob('exhibit_??.jpg')):
        im = cv2.imread(str(f))
        m = cv2.imread(str(f).replace('.jpg', '_mask.png'), 0) > 127
        c = detect(im)
        near = cv2.dilate(m.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
        hit = [bool(near[d['cy'], d['cx']]) for d in c]
        x, y, w, h = c[0]['box']
        pb = np.zeros_like(m)
        pb[max(y, 0):y + h, max(x, 0):x + w] = True
        iou = (pb & m).sum() / (pb | m).sum()
        rows.append(dict(exhibit=f.stem, top1=hit[0], top3=any(hit), iou=round(float(iou), 3),
                         score=round(c[0]['score'], 1), conf=round(confidence(c), 3)))
        print('%s top1=%s top3=%s iou=%.2f conf=%.2f' % (f.stem, hit[0], any(hit), iou, confidence(c)), flush=True)
    t1 = np.mean([r['top1'] for r in rows])
    t3 = np.mean([r['top3'] for r in rows])
    print('exhibits: top-1 hit %.1f%%, top-3 hit %.1f%%, mean IoU of box %.2f' % (
        100 * t1, 100 * t3, np.mean([r['iou'] for r in rows])))
    with open(out / 'exhibit_validation.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--out', default='.')
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--cache', default=None,
                    help='candidates.csv of an earlier run: skip detection, only redraw sheets and read letters')
    args = ap.parse_args()
    cv2.setNumThreads(args.threads)
    data, out = Path(args.data), Path(args.out)
    (out / 'figures' / 'pages').mkdir(parents=True, exist_ok=True)
    if args.validate:
        validate(data, out)

    templates = letter_templates()
    cache = {}
    if args.cache:
        for r in csv.DictReader(open(args.cache)):
            cache.setdefault((r['album'], r['page']), []).append(dict(
                cx=int(r['cx']), cy=int(r['cy']), score=float(r['score']),
                box=(int(r['x']), int(r['y']), int(r['w']), int(r['h']))))
    albums = sorted(p for p in (data / 'albums').iterdir() if p.is_dir())
    boxes, cand_rows, sheets, word, report = [], [], [], '', {}
    for album in albums:
        rows = []
        for f in sorted(album.glob('*.jpg')):
            t = time.time()
            im = cv2.imread(str(f))
            c = cache[(album.name, f.stem)] if cache else detect(im)
            conf = confidence(c)
            x, y, w, h = c[0]['box']
            rows.append(dict(page=f.stem, x=x, y=y, w=w, h=h, conf=conf, cx=c[0]['cx'], cy=c[0]['cy'], score=c[0]['score'],
                             alt=(c[1]['cx'], c[1]['cy']) if len(c) > 1 else None))
            boxes.append([album.name, f.stem, x, y, w, h])
            for j, d in enumerate(c):
                cand_rows.append([album.name, f.stem, j + 1, d['cx'], d['cy'], *d['box'], round(d['score'], 1),
                                  round(conf, 3) if j == 0 else ''])
            cv2.imwrite(str(out / 'figures' / 'pages' / ('%s_%s.jpg' % (album.name, f.stem))),
                        cv2.resize(overlay_page(im, c), None, fx=0.5, fy=0.5), [cv2.IMWRITE_JPEG_QUALITY, 85])
            print('%s/%s  (%d,%d)  score %.0f  conf %.2f  %.0fs' % (album.name, f.stem, c[0]['cx'], c[0]['cy'],
                                                                   c[0]['score'], conf, time.time() - t), flush=True)
        ch, alts, lbox = read_letter([(r['cx'], r['cy']) for r in rows],
                                     [1.0 if r['conf'] >= 0.7 else 0.3 for r in rows], templates)
        word += ch
        x0, y0, Wd, H = lbox
        R = cv2.resize(templates[ch].astype(np.uint8), (Wd, H), interpolation=cv2.INTER_NEAREST)
        R = cv2.dilate(R, np.ones((int(0.08 * H) * 2 + 1,) * 2, np.uint8))  # letter body + 8 % of its height
        on = []
        for r in rows:
            px, py = r['cx'] - x0, r['cy'] - y0
            on.append(bool(0 <= px < Wd and 0 <= py < H and R[py, px]))
            r['on_letter'] = on[-1]
        report[album.name] = dict(letter=ch, alternatives=alts, letter_box=list(lbox), pages=len(rows),
                                  n_confident=int(sum(r['conf'] >= 0.7 for r in rows)),
                                  marks_on_letter=int(sum(on)),
                                  confident_marks_on_letter=int(sum(o and r['conf'] >= 0.7 for o, r in zip(on, rows))),
                                  off_letter_pages=[r['page'] for r in rows if not r['on_letter']])
        sheet = draw_sheet(rows, '%s - marks of all %d pages' % (album.name, len(rows)), ch, alts,
                           (templates[ch], lbox))
        cv2.imwrite(str(out / 'figures' / ('sheet_%s.png' % album.name)), sheet)
        sheets.append(sheet)
        print('%s reads %s  %s' % (album.name, ch, alts), flush=True)

    with open(out / 'boxes.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['album', 'page', 'x', 'y', 'w', 'h'])
        w.writerows(boxes)
    with open(out / 'candidates.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['album', 'page', 'rank', 'cx', 'cy', 'x', 'y', 'w', 'h', 'score', 'confidence'])
        w.writerows(cand_rows)
    small = [cv2.resize(s, (800, 600), interpolation=cv2.INTER_AREA) for s in sheets]
    while len(small) % 3:
        small.append(np.full_like(small[0], 255))
    grid = np.vstack([np.hstack(small[i:i + 3]) for i in range(0, len(small), 3)])
    cv2.imwrite(str(out / 'figures' / 'sheet_all.png'), grid)
    json.dump(report, open(out / 'letters.json', 'w'), indent=1)
    print('WORD:', word)
    print('FLAG: SIGMOID_' + word)


if __name__ == '__main__':
    main()
