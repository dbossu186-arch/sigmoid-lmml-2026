"""Reassemble a vertically shredded image.

Reads every PNG strip from ./input, finds the left-to-right order and writes
reconstructed_order.json and reconstructed_image.png next to this script.

Method
1. Edge compatibility D[i, j] = cost of putting strip j directly right of strip i.
   It uses the "prediction" (gradient-extrapolation) dissimilarity: the column
   just right of i is predicted as 2*last - second_last, likewise the column
   just left of j, and both predictions are compared with the real boundary
   columns (L1 in Lab-ish float colour, robust square-root norm per row).
2. Costs are normalised per row and per column (best-buddy style) so a strip
   whose edges are uniformly smooth does not attract everything.
3. Ordering = shortest Hamiltonian path: greedy chain merging (Kruskal on the
   sorted edge list, keeping every chain a path), then 2-opt / Or-opt
   refinement on the open path. All O(n^2)-O(n^3) with n ~ 200: well below a
   second.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
IN_DIR = Path("input") if Path("input").is_dir() else HERE / "input"


def load_strips(folder: Path):
    names = sorted(p.name for p in folder.iterdir() if p.suffix.lower() == ".png")
    imgs = []
    for n in names:
        im = cv2.imread(str(folder / n), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise RuntimeError(f"cannot read {n}")
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        elif im.shape[2] == 4:
            im = im[:, :, :3]
        imgs.append(im)
    h = min(i.shape[0] for i in imgs)
    imgs = [i[:h] for i in imgs]
    return names, imgs


def edge_cost(imgs):
    n = len(imgs)
    f = [cv2.cvtColor(i, cv2.COLOR_BGR2LAB).astype(np.float32) for i in imgs]
    R1 = np.stack([x[:, -1] for x in f])  # n,h,3
    L1 = np.stack([x[:, 0] for x in f])
    if imgs[0].shape[1] >= 2:
        R2 = np.stack([x[:, -2] for x in f])
        L2 = np.stack([x[:, 1] for x in f])
    else:
        R2, L2 = R1, L1
    predR = 2 * R1 - R2  # predicted column right of i
    predL = 2 * L1 - L2  # predicted column left of j
    h = R1.shape[1]
    a = predR.reshape(n, -1)
    b = L1.reshape(n, -1)
    c = R1.reshape(n, -1)
    d = predL.reshape(n, -1)
    D = np.zeros((n, n), np.float64)
    # prediction errors, robust power norm, computed in chunks
    for i in range(n):
        e1 = np.sqrt(np.abs(a[i][None, :] - b)).sum(1, dtype=np.float64)
        e2 = np.sqrt(np.abs(c[i][None, :] - d)).sum(1, dtype=np.float64)
        e3 = np.sqrt(np.abs(c[i][None, :] - b)).sum(1, dtype=np.float64)
        D[i] = (e1 + e2) * 0.5 + 0.5 * e3
    D /= h
    np.fill_diagonal(D, np.inf)
    return D


def normalise(D):
    n = D.shape[0]
    Dm = D.copy()
    np.fill_diagonal(Dm, np.nan)
    # second-best in row / column
    srow = np.sort(np.where(np.isnan(Dm), np.inf, Dm), axis=1)[:, 1][:, None]
    scol = np.sort(np.where(np.isnan(Dm), np.inf, Dm), axis=0)[1][None, :]
    N = D / (np.minimum(srow, scol) + 1e-6)
    np.fill_diagonal(N, np.inf)
    return N


def greedy_path(C):
    n = C.shape[0]
    nxt = -np.ones(n, int)
    prv = -np.ones(n, int)
    head = np.arange(n)  # chain id via union-find
    def find(x):
        while head[x] != x:
            head[x] = head[head[x]]
            x = head[x]
        return x
    order = np.argsort(C, axis=None)
    links = 0
    for k in order:
        i, j = divmod(int(k), n)
        if i == j or nxt[i] != -1 or prv[j] != -1:
            continue
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        nxt[i] = j
        prv[j] = i
        head[rj] = ri
        links += 1
        if links == n - 1:
            break
    start = int(np.where(prv == -1)[0][0])
    path = [start]
    while nxt[path[-1]] != -1:
        path.append(int(nxt[path[-1]]))
    return path


def path_cost(p, C):
    return sum(C[p[k], p[k + 1]] for k in range(len(p) - 1))


def refine(path, C, max_rounds=30, deadline=None):
    """Or-opt: move blocks (len 1..L) to the best other position, both orientations kept fixed."""
    p = list(path)
    n = len(p)
    best = path_cost(p, C)
    for _ in range(max_rounds):
        improved = False
        for L in range(1, min(40, n - 1)):
            if deadline is not None and time.time() > deadline:
                return p
            for s in range(0, n - L + 1):
                blk = p[s:s + L]
                rest = p[:s] + p[s + L:]
                # cost of rest + insert block at best position
                base = best
                # removal delta
                rem = 0.0
                if s > 0:
                    rem -= C[p[s - 1], p[s]]
                if s + L < n:
                    rem -= C[p[s + L - 1], p[s + L]]
                if s > 0 and s + L < n:
                    rem += C[p[s - 1], p[s + L]]
                # insertion positions in rest
                m = len(rest)
                cand = np.empty(m + 1)
                first, last = blk[0], blk[-1]
                cand[0] = C[last, rest[0]]
                cand[m] = C[rest[-1], first]
                if m > 1:
                    ra = np.array(rest[:-1]); rb = np.array(rest[1:])
                    cand[1:m] = C[ra, first] + C[last, rb] - C[ra, rb]
                pos = int(np.argmin(cand))
                if pos == s:
                    continue
                newc = base + rem + cand[pos]
                if newc < best - 1e-9:
                    p = rest[:pos] + blk + rest[pos:]
                    best = newc
                    improved = True
        if not improved:
            break
    return p


def main() -> int:
    t0 = time.time()
    in_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else IN_DIR
    names, imgs = load_strips(in_dir)
    D = edge_cost(imgs)
    C = normalise(D)
    path = greedy_path(C)
    path = refine(path, C, deadline=t0 + 12.0)
    order = [names[k] for k in path]
    with open(HERE / "reconstructed_order.json", "w") as fh:
        json.dump(order, fh, indent=4)
    out = np.concatenate([imgs[k] for k in path], axis=1)
    cv2.imwrite(str(HERE / "reconstructed_image.png"), out)
    print(f"{len(order)} strips -> {out.shape[1]}x{out.shape[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
