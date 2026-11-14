"""Check a set of predictions for the sealed records.

    python evaluation_script.py predictions.csv

predictions.csv must have the columns  record_id, vrancea  with one row per sealed record
(dataset/sealed.csv) and vrancea in {0, 1}.

The flag is printed when the predictions are good enough. Nothing else is printed: no score,
no count.

Standard library + numpy + pandas.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------- arithmetic mod p
def _trim(a):
    nz = np.flatnonzero(a)
    return a[: nz[-1] + 1] if len(nz) else a[:0]


def _divmod(a, b, p):
    """Polynomial division mod p; coefficients little-endian (index = power)."""
    a = a.copy()
    db = len(b) - 1
    inv = pow(int(b[-1]), p - 2, p)
    q = np.zeros(max(len(a) - db, 1), dtype=np.int64)
    for i in range(len(a) - 1 - db, -1, -1):
        c = int(a[i + db]) * inv % p
        if c:
            q[i] = c
            a[i:i + db + 1] = (a[i:i + db + 1] - c * b) % p
    return _trim(q), _trim(a[:db]) if db else a[:0]


def _mul(a, b, p):
    if len(a) == 0 or len(b) == 0:
        return a[:0]
    if len(a) < len(b):
        a, b = b, a
    out = np.zeros(len(a) + len(b) - 1, dtype=np.int64)
    for i, c in enumerate(b):
        if c:
            out[i:i + len(a)] = (out[i:i + len(a)] + int(c) * a) % p
    return out


def _sub(a, b, p):
    out = np.zeros(max(len(a), len(b)), dtype=np.int64)
    out[:len(a)] = a
    out[:len(b)] = (out[:len(b)] - b) % p
    return _trim(out)


def _interpolate(x, y, p, inv):
    """Coefficients of the polynomial of degree < len(x) through the points (x, y), and of prod(X - x_i)."""
    m = len(x)
    c = y.astype(np.int64).copy()
    for j in range(1, m):                      # Newton divided differences
        d = x[j:] - x[:-j]
        dinv = np.where(d > 0, inv[np.abs(d)], (p - inv[np.abs(d)]) % p)
        c[j:] = (c[j:] - c[j - 1:-1]) % p * dinv % p
    poly = np.zeros(m, dtype=np.int64)         # Newton form -> ordinary coefficients (Horner)
    poly[0] = c[m - 1]
    for i in range(m - 2, -1, -1):
        deg = m - 1 - i
        nxt = np.zeros(m, dtype=np.int64)
        nxt[1:deg + 1] = poly[:deg]
        nxt[:deg] = (nxt[:deg] - int(x[i]) * poly[:deg]) % p
        nxt[0] = (nxt[0] + c[i]) % p
        poly = nxt
    g0 = np.zeros(m + 1, dtype=np.int64)       # prod (X - x_i)
    g0[0] = 1
    for i in range(m):
        g0[1:i + 2], g0[0] = (g0[:i + 1] - int(x[i]) * g0[1:i + 2]) % p, (-int(x[i]) * g0[0]) % p
    return _trim(poly), g0


def _decode(x, y, k, p, inv):
    """The polynomial of degree < k that agrees with at least (len(x) + k) / 2 of the points, or None (Gao)."""
    m = len(x)
    if m < k:
        return None
    g1, g0 = _interpolate(x, y, p, inv)
    if len(g1) <= k:                           # no wrong point at all
        return g1
    r0, r1 = g0, g1
    v0, v1 = np.zeros(0, dtype=np.int64), np.ones(1, dtype=np.int64)
    limit = (m + k) // 2 if (m + k) % 2 == 0 else (m + k + 1) // 2
    while len(r1) - 1 >= limit:
        q, r = _divmod(r0, r1, p)
        r0, r1 = r1, r
        v0, v1 = v1, _sub(v0, _mul(q, v1, p), p)
        if len(r1) == 0:
            return None
    f, rem = _divmod(r1, v1, p)
    if len(rem) or len(f) > k:
        return None
    return f


def _evaluate(f, x, p):
    out = np.zeros(len(x), dtype=np.int64)
    for c in f[::-1]:
        out = (out * x + int(c)) % p
    return out


def _word(f, k, z):
    coeffs = np.zeros(k, dtype="<u2")
    coeffs[:len(f)] = f
    key = hashlib.sha256(coeffs.tobytes()).digest()
    if hashlib.sha256(key + b"door").digest()[:8] != bytes(z["tag"]):
        return None
    stream = hashlib.sha256(key + b"word").digest()
    return bytes(a ^ b for a, b in zip(bytes(z["sealed_word"]), stream)).decode("ascii")


# ----------------------------------------------------------------------------- the check
def flag_for(pred, vault_path=None):
    """pred: int array of 0/1 in the order of dataset/sealed.csv. Returns the flag word or None."""
    z = np.load(vault_path or HERE / "dataset" / "vault.npz")
    p, n, k = int(z["p"]), int(z["n"]), int(z["k"])
    pred = np.asarray(pred, dtype=np.int64)
    if pred.shape != (n,) or not np.isin(pred, [0, 1]).all():
        raise SystemExit(f"expected {n} predictions in {{0, 1}}")
    inv = np.zeros(n + 2, dtype=np.int64)
    for d in range(1, n + 2):
        inv[d] = pow(d, p - 2, p)
    xs = np.arange(1, n + 1, dtype=np.int64)
    ys = z["table"][np.arange(n), pred].astype(np.int64)
    f = _decode(xs, ys, k, p, inv)
    if f is None:
        return None
    agree = int((_evaluate(f, xs, p) == ys).sum())
    if 2 * agree < n + k:
        return None
    return _word(f, k, z)


def load_predictions(path, sealed_path=None):
    sealed = pd.read_csv(sealed_path or HERE / "dataset" / "sealed.csv", usecols=["record_id"])
    pred = pd.read_csv(path)
    for col in ("record_id", "vrancea"):
        if col not in pred.columns:
            raise SystemExit(f"{path}: missing column {col}")
    if pred.record_id.duplicated().any():
        raise SystemExit(f"{path}: duplicated record_id")
    merged = sealed.merge(pred[["record_id", "vrancea"]], on="record_id", how="left")
    if merged.vrancea.isna().any():
        missing = int(merged.vrancea.isna().sum())
        raise SystemExit(f"{path}: {missing} sealed records have no prediction")
    return merged.vrancea.astype(int).to_numpy()


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    word = flag_for(load_predictions(sys.argv[1]))
    if word:
        print(f"SIGMOID_{word}")
    else:
        print("No flag for these predictions.")


if __name__ == "__main__":
    main()
