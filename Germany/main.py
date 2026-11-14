"""
GERMANY "Black Box" solver.

Model family assumed (and checked against the data):
    f(x) = g(x) + J * [w . x >= t]
where g is smooth = intercept + per-variable main effects (polynomial up to
cubic) + pairwise products, and the step is a single linear half-space.

Strategy (150-query budget):
  1. Design: all 32 cube corners (a non-empty half-space always contains a
     vertex, so the step can never hide from them) + space-filling interior
     points.  Mixed corners/interior separates "step on a sub-cube"
     from "bilinear interaction", which corners alone cannot.
  2. Robust regression: L1 (IRLS) fit of g, then alternate
     (fit g + J*label) <-> (relabel each point by nearest of g, g+J) until
     the noise-free data are fitted exactly.  Gives g, J, and a label for
     every queried point.
  3. Boundary: bisection along segments joining differently-labelled points
     (each query = 1 bit, read from the residual f - g), then a second round
     of short segments perpendicular to the rough plane.  Weighted total
     least squares on the crossing points -> hyperplane; tiny components
     are zeroed -> gate variables.
  4. Final joint refit of g and J on every query, structure readout,
     predictions.
"""
import itertools
import math
import os
import time

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")      # tiny matrices: threads only hurt

import numpy as np  # noqa: E402

T0 = time.time()

VARS = ["x1", "x2", "x3", "x4", "x5"]
PAIRS = list(itertools.combinations(range(5), 2))
BUDGET = 150
RESERVE = 1            # never touch the last query(s)


# ---------------------------------------------------------------- basis
def basis(X, level):
    """level 1: 1, x_i, x_i x_j ; level 2: + x_i^2 ; level 3: + x_i^3."""
    X = np.atleast_2d(np.asarray(X, float))
    cols = [np.ones(len(X))]
    cols += [X[:, i] for i in range(5)]
    cols += [X[:, i] * X[:, j] for i, j in PAIRS]
    if level >= 2:
        cols += [X[:, i] ** 2 for i in range(5)]
    if level >= 3:
        cols += [X[:, i] ** 3 for i in range(5)]
    return np.column_stack(cols)


def lstsq(A, y):
    return np.linalg.lstsq(A, y, rcond=None)[0]


def lad(A, y, iters=150):
    beta = lstsq(A, y)
    for _ in range(iters):
        r = np.abs(y - A @ beta)
        w = 1.0 / np.sqrt(np.maximum(r, 1e-8))
        nb = lstsq(A * w[:, None], y * w)
        if np.max(np.abs(nb - beta)) < 1e-11:
            beta = nb
            break
        beta = nb
    return beta


def alternate(A, y, labels, iters=40):
    """Fit y = A beta + J*labels, relabel, repeat."""
    beta, J = None, 0.0
    for _ in range(iters):
        if labels.any() and (~labels).any():
            sol = lstsq(np.column_stack([A, labels.astype(float)]), y)
            beta, J = sol[:-1], sol[-1]
        else:
            beta, J = lstsq(A, y), 0.0
        r = y - A @ beta
        if abs(J) < 1e-9:
            return beta, 0.0, np.zeros(len(y), bool), r
        new = np.abs(r - J) < np.abs(r)
        if (new == labels).all():
            break
        labels = new
    r = y - A @ beta - J * labels
    return beta, J, labels, r


def make_directions(rng):
    """Candidate boundary normals: axis, 2-, 3-, 4-variable and dense 5-D."""
    D = [np.eye(5)[i] for i in range(5)]
    for i, j in PAIRS:
        for a in np.linspace(0, np.pi, 120, endpoint=False)[1:]:
            v = np.zeros(5)
            v[i], v[j] = np.cos(a), np.sin(a)
            if abs(v[i]) > 1e-9 and abs(v[j]) > 1e-9:
                D.append(v)
    for sub in itertools.combinations(range(5), 3):
        R = rng.normal(size=(250, 3))
        for r in R:
            v = np.zeros(5)
            v[list(sub)] = r
            D.append(v)
    for sub in itertools.combinations(range(5), 4):
        R = rng.normal(size=(300, 4))
        for r in R:
            v = np.zeros(5)
            v[list(sub)] = r
            D.append(v)
    D += list(rng.normal(size=(3000, 5)))
    D = np.array(D)
    return D / np.linalg.norm(D, axis=1, keepdims=True)


def scan_halfspaces(M, ry, X, D, batch=400):
    """For each direction d and each split 'top-k of X@d', the residual sum of
    squares of y ~ A beta + J*z (z = top-k indicator), computed exactly via
    the annihilator M = I - A A^+.  Returns (best_ss, best_k) per direction."""
    n = len(ry)
    yy = float(ry @ ry)
    best_ss = np.empty(len(D))
    best_k = np.empty(len(D), int)
    rows = np.arange(n)
    for b0 in range(0, len(D), batch):
        Db = D[b0:b0 + batch]
        S = X @ Db.T                                   # n x K
        order = np.argsort(-S, axis=0, kind="stable").T  # K x n
        num = np.cumsum(ry[order], axis=1)             # K x n
        Mp = M[order[:, :, None], order[:, None, :]]   # K x n x n
        L = np.cumsum(Mp, axis=2)
        diag = Mp[:, rows, rows]
        prev = np.zeros_like(diag)
        prev[:, 1:] = L[:, rows[1:], rows[1:] - 1]
        den = np.cumsum(2 * prev + diag, axis=1)
        gain = np.where(den > 1e-10, num ** 2 / np.maximum(den, 1e-10), 0.0)
        gain[:, -1] = 0.0                              # all-active = no step
        k = np.argmax(gain, axis=1)
        best_k[b0:b0 + batch] = k + 1
        best_ss[b0:b0 + batch] = yy - gain[np.arange(len(Db)), k]
    return best_ss, best_k


def robust_fit(A, y, X, D, rng):
    """Return (beta, J, labels, max_abs_resid) of the best two-level fit."""
    n = len(y)
    Q, _ = np.linalg.qr(A)
    M = np.eye(n) - Q @ Q.T
    ry = M @ y
    cands = []
    if float(np.max(np.abs(ry))) < 5e-6:               # already smooth
        beta = lstsq(A, y)
        return beta, 0.0, np.zeros(n, bool), float(np.max(np.abs(y - A @ beta)))
    ss, kk = scan_halfspaces(M, ry, X, D)
    top = np.argsort(ss)[:6]
    for t in top:
        d, sbest = D[t].copy(), ss[t]
        # local hill-climb on the direction
        sig = 0.15
        for _ in range(6):
            P = d + sig * rng.normal(size=(60, 5)) * (rng.random((60, 5)) < 0.7)
            P = P / np.linalg.norm(P, axis=1, keepdims=True)
            s2, k2 = scan_halfspaces(M, ry, X, P)
            j = int(np.argmin(s2))
            if s2[j] < sbest:
                d, sbest = P[j], s2[j]
            else:
                sig *= 0.5
        _, kb = scan_halfspaces(M, ry, X, d[None])
        lab = np.zeros(n, bool)
        lab[np.argsort(-(X @ d), kind="stable")[:kb[0]]] = True
        cands.append(alternate(A, y, lab))
    for b0 in (lad(A, y), lstsq(A, y)):
        r = y - A @ b0
        tol = max(1e-3, 20 * float(np.median(np.abs(r))))
        out = np.abs(r) > tol
        if out.any():
            J0 = float(np.median(r[out]))
            cands.append(alternate(A, y, np.abs(r - J0) < np.abs(r)))
    best = None
    for beta, J, lab, r in cands:
        score = float(np.sum(r ** 2))
        if best is None or score < best[0]:
            best = (score, beta, J, lab, float(np.max(np.abs(r))))
    return best[1], best[2], best[3], best[4]


# ---------------------------------------------------------------- plane
def fit_plane(B, w, dims):
    """Weighted TLS plane through points B restricted to dims.
    Returns unit normal (len 5) and offset t (n.x = t)."""
    Bd = B[:, dims]
    m = (w[:, None] * Bd).sum(0) / w.sum()
    M = np.sqrt(w)[:, None] * (Bd - m)
    _, _, vt = np.linalg.svd(M, full_matrices=True)
    nd = vt[-1]
    n = np.zeros(5)
    n[dims] = nd
    return n, float(nd @ m)


def plane_from_points(B, w):
    dims = list(range(5))
    n, t = fit_plane(B, w, dims)
    for _ in range(4):
        big = np.abs(n) / np.max(np.abs(n))
        keep = [i for i in range(5) if big[i] >= 0.06]
        if keep == dims:
            break
        dims = keep
        if len(dims) == 1:
            n = np.zeros(5)
            n[dims[0]] = 1.0
            t = float(np.average(B[:, dims[0]], weights=w))
            break
        n, t = fit_plane(B, w, dims)
    return n, t


# ---------------------------------------------------------------- solve
def solve(oracle, points):
    with np.errstate(all="ignore"):     # spurious BLAS fp warnings on macOS
        return _solve(oracle, points)


def _solve(oracle, points):
    global T0
    T0 = time.time()
    rng = np.random.default_rng(20261114)
    budget_left = lambda: oracle.remaining - RESERVE  # noqa: E731

    X, Y = [], []

    def ask(pts, phase):
        pts = [list(map(float, np.clip(p, 0.0, 1.0))) for p in pts]
        ys = oracle.ask_many(pts, phase)
        for p, v in zip(pts, ys):
            X.append(p)
            Y.append(float(v))
        return ys

    # ---- 1. design: corners + scrambled LHS interior points
    corners = [list(c) for c in itertools.product([0.0, 1.0], repeat=5)]
    n_int = 48
    lhs = np.empty((n_int, 5))
    for k in range(5):
        lhs[:, k] = (rng.permutation(n_int) + rng.uniform(0.15, 0.85, n_int)) / n_int
    lhs = 0.03 + 0.94 * lhs
    ask(corners, "corners")
    ask(lhs.tolist(), "interior-lhs")

    D = make_directions(rng)
    warm_mx = [0.0]

    def fit_all(level_max=3, warm=None):
        Xa, Ya = np.array(X), np.array(Y)
        res = None
        if warm is not None:              # (level, beta, J): relabel + polish
            lv, bt, jj = warm
            A = basis(Xa, lv)
            r0 = Ya - A @ bt
            b2, J2, lab2, r2 = alternate(A, Ya, np.abs(r0 - jj) < np.abs(r0))
            mx2 = float(np.max(np.abs(r2)))
            res = (lv, b2, J2, lab2, mx2)
            if mx2 < max(5e-5, 3 * warm_mx[0]) or time.time() - T0 > 25:
                return res
        for level in (1, 2, 3)[:level_max]:
            A = basis(Xa, level)
            beta, J, lab, mx = robust_fit(A, Ya, Xa, D, rng)
            if res is None or mx < res[4] * 0.5:
                res = (level, beta, J, lab, mx)
            if mx < 5e-5 or time.time() - T0 > 25:
                break
        warm_mx[0] = res[4]
        return res

    level, beta, J, lab, mx = fit_all()
    print(f"[fit] level={level} J={J:+.4f} active={int(lab.sum())}/{len(lab)} "
          f"maxres={mx:.2e}", flush=True)

    def g(x):
        return float((basis(np.array([x]), level) @ beta).reshape(-1)[0])

    def label_of(x, y):
        r = y - g(x)
        return abs(r - J) < abs(r)

    step_found = abs(J) > 1e-3 and lab.any() and (~lab).any()
    bpts, bw = [], []
    n_hat, t_hat = None, None

    def bisect(a, b, la, steps, phase, min_width=0.0):
        a, b = np.array(a, float), np.array(b, float)
        L = float(np.linalg.norm(b - a))
        lo, hi = 0.0, 1.0
        for _ in range(steps):
            if budget_left() < 1 or (hi - lo) * L <= min_width:
                break
            mid = 0.5 * (lo + hi)
            x = a + mid * (b - a)
            y = ask([x], phase)[0]
            if label_of(x, y) == la:
                lo = mid
            else:
                hi = mid
        p = a + 0.5 * (lo + hi) * (b - a)
        width = max((hi - lo) * L, 1e-4)
        bpts.append(p)
        bw.append(1.0 / width ** 2)

    def orient(n, t):
        # make the label-True side the n.x >= t side
        Xa_ = np.array(X)
        r_ = np.array(Y) - basis(Xa_, level) @ beta
        lab_ = np.abs(r_ - J) < np.abs(r_)
        if np.mean(((Xa_ @ n - t) >= 0) == lab_) < 0.5:
            return -n, -t
        return n, t

    if step_found:
        Xa = np.array(X)
        A_idx = np.where(lab)[0]
        I_idx = np.where(~lab)[0]
        # ---- 2. phase 1: bisection between nearby opposite-label points
        pairs = []
        for i in A_idx:
            for j in I_idx:
                pairs.append((float(np.linalg.norm(Xa[i] - Xa[j])), i, j))
        pairs.sort()
        chosen, mids = [], []
        n_p1 = 7
        for d, i, j in pairs:
            m = 0.5 * (Xa[i] + Xa[j])
            if all(np.linalg.norm(m - q) > 0.35 for q in mids):
                chosen.append((i, j))
                mids.append(m)
            if len(chosen) >= n_p1:
                break
        if len(chosen) < n_p1:   # relax diversity
            for d, i, j in pairs:
                if (i, j) not in chosen:
                    chosen.append((i, j))
                if len(chosen) >= n_p1:
                    break
        for i, j in chosen:
            if budget_left() < 6:
                break
            bisect(Xa[i], Xa[j], True, 6, "bisect-1", min_width=0.008)

        # refit g with the new labelled points (they are informative too)
        level, beta, J, lab, mx = fit_all(warm=(level, beta, J))
        B, W = np.array(bpts), np.array(bw)
        if len(B) >= 2:
            n_hat, t_hat = plane_from_points(B, W) if len(B) >= 3 else (None, None)
        if n_hat is not None:
            n_hat, t_hat = orient(n_hat, t_hat)

        # ---- 3. phase 2: short perpendicular segments across the rough plane
        if n_hat is not None:
            cand = rng.uniform(0, 1, (4000, 5))
            proj = cand - np.outer(cand @ n_hat - t_hat, n_hat)
            ok = np.all((proj >= 0.01) & (proj <= 0.99), axis=1)
            proj = proj[ok]
            sel = []
            existing = list(B)
            while len(proj) and budget_left() >= 7 and len(sel) < 12:
                if existing:
                    dmin = np.min(np.linalg.norm(proj[:, None, :] -
                                                 np.array(existing)[None], axis=2), axis=1)
                    k = int(np.argmax(dmin))
                else:
                    k = 0
                p0 = proj[k]
                sel.append(p0)
                existing.append(p0)
                y0 = ask([p0], "plane-probe")[0]
                l0 = label_of(p0, y0)
                direc = -n_hat if l0 else n_hat   # head to the other side
                found = None
                prev = p0
                for delta in (0.02, 0.05, 0.12):
                    q = p0 + delta * direc
                    if np.any(q < 0) or np.any(q > 1):
                        q = np.clip(q, 0, 1)
                    if budget_left() < 1:
                        break
                    yq = ask([q], "plane-probe")[0]
                    if label_of(q, yq) != l0:
                        found = (prev, q)
                        break
                    prev = q
                if found is None:
                    continue
                bisect(found[0], found[1], l0, 6, "bisect-2", min_width=0.0015)
            B, W = np.array(bpts), np.array(bw)
            n_hat, t_hat = plane_from_points(B, W)
        print(f"[plane] {len(bpts)} crossings, widths "
              f"{np.round(1 / np.sqrt(np.array(bw)), 4).tolist()} "
              f"n={np.round(n_hat, 4).tolist()} t={t_hat:.4f} "
              f"spent={oracle.spent}", flush=True)

        # ---- 4. final refit with all data
        level, beta, J, lab, mx = fit_all(warm=(level, beta, J))
        if n_hat is not None:
            Xa = np.array(X)
            s = Xa @ n_hat - t_hat
            # orient: active (label True) side should be n.x >= t
            agree = np.mean((s >= 0) == lab)
            if agree < 0.5:
                n_hat, t_hat = -n_hat, -t_hat
            # majority-side check: report the minority side as the step
            mc = rng.uniform(0, 1, (20000, 5))
            frac = float(np.mean(mc @ n_hat >= t_hat))
            if frac > 0.5:
                # f = (g + J) - J*[n.x < t]  ->  boundary -n.x >= -t
                n_hat, t_hat = -n_hat, -t_hat
                beta = beta.copy()
                beta[0] += J
                J = -J
                lab = ~lab
            # joint refit of g and J with plane labels, away from the plane
            Xa, Ya = np.array(X), np.array(Y)
            s = Xa @ n_hat - t_hat
            ind = (s >= 0).astype(float)
            far = np.abs(s) > 0.004
            A = basis(Xa, level)
            sol = lstsq(np.column_stack([A, ind])[far], Ya[far])
            beta2, J2 = sol[:-1], sol[-1]
            r2 = Ya[far] - np.column_stack([A, ind])[far] @ sol
            if np.max(np.abs(r2)) <= max(mx * 2, 1e-4):
                beta, J = beta2, J2

    # ---------------------------------------------------- structure readout
    nb = 1 + 5 + 10
    lin = beta[1:6]
    inter = beta[6:16]
    sq = beta[16:21] if level >= 2 else np.zeros(5)
    cu = beta[21:26] if level >= 3 else np.zeros(5)
    grid = np.linspace(0, 1, 21)
    main_range = []
    for i in range(5):
        h = lin[i] * grid + sq[i] * grid ** 2 + cu[i] * grid ** 3
        main_range.append(float(h.max() - h.min()))
    k_top = int(np.argmax(np.abs(inter)))
    ci, cj = PAIRS[k_top]
    c_top = float(inter[k_top])
    inter_scale = max(abs(c_top), 1e-12)

    gate = []
    if n_hat is not None:
        gate = [i for i in range(5) if abs(n_hat[i]) > 1e-12]
    relevant = set(gate)
    for i in range(5):
        if main_range[i] > 0.01:
            relevant.add(i)
    if abs(c_top) > 0.02:
        relevant.update([ci, cj])
    for k, (i, j) in enumerate(PAIRS):
        if abs(inter[k]) > max(0.05, 0.25 * inter_scale):
            relevant.update([i, j])

    interactions = [{
        "variables": [VARS[ci], VARS[cj]],
        "type": "product",
        "coefficient": round(c_top, 6),
        "equivalent_centered_form":
            f"{c_top:.6f}*({VARS[ci]}-0.5)*({VARS[cj]}-0.5)",
    }]

    discs = []
    cond = None
    if n_hat is not None:
        scale = float(np.max(np.abs(n_hat)))
        coefs = {v: round(float(n_hat[i] / scale), 6) for i, v in enumerate(VARS)}
        thr = round(float(t_hat / scale), 6)
        cond = " + ".join(f"{coefs[v]}*{v}" for v in VARS if coefs[v] != 0.0)
        cond = f"{cond} >= {thr}"
        discs.append({
            "gate_variables": [VARS[i] for i in gate],
            "boundary": {"form": "linear", "coefficients": coefs,
                         "threshold": thr, "active_when": "greater_equal"},
            "jump": round(float(J), 6),
            "condition": cond,
        })

    # ---------------------------------------------------- expression
    names = (["1"] + VARS + [f"{VARS[i]}*{VARS[j]}" for i, j in PAIRS]
             + ([f"{v}**2" for v in VARS] if level >= 2 else [])
             + ([f"{v}**3" for v in VARS] if level >= 3 else []))
    terms = [f"{beta[0]:.6f}"]
    for c, nm in zip(beta[1:], names[1:]):
        if abs(c) >= 5e-6:
            terms.append(f"{'+' if c >= 0 else '-'} {abs(c):.6f}*{nm}")
    expr = " ".join(terms)
    if cond:
        expr += f" + ({float(J):.6f} if {cond} else 0.0)"

    P = np.array(points, float)
    pred = basis(P, level) @ beta
    if n_hat is not None:
        pred = pred + J * ((P @ n_hat - t_hat) >= 0)
    preds = [round(float(v), 6) for v in pred]

    notes = (f"{oracle.spent} queries: 32 cube corners + 48 LHS interior points "
             f"for a robust (L1 + alternating relabel) fit of smooth part "
             f"(basis level {level}: linear+pairwise products"
             f"{', squares' if level >= 2 else ''}{', cubes' if level >= 3 else ''}) "
             f"plus a step indicator; {len(bpts)} boundary crossings located by "
             f"bisection (each query read as one bit via the residual f-g). "
             f"Max residual of the final fit {mx:.2e}. Main-effect ranges: "
             + ", ".join(f"{v}:{r:.3f}" for v, r in zip(VARS, main_range)))
    return {
        "method": "corners+LHS robust regression, residual-labelled bisection "
                  "of the switching surface, weighted TLS hyperplane",
        "relevant_variables": [VARS[i] for i in sorted(relevant)],
        "irrelevant_variables": [VARS[i] for i in range(5) if i not in relevant],
        "interactions": interactions,
        "discontinuities": discs,
        "model_expression": expr,
        "predictions": preds,
        "notes": notes,
    }
