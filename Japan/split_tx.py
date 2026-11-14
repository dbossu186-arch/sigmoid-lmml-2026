"""Separate camera T (Toorisugari's phone) from hidden camera X using anonymous-only fingerprints,
then score every credited Toorisugari photo (and all others) against K_T and K_X.
usage: split_tx.py <cache> <seedX comma list> <seedT comma list> [rounds]"""
import sys, os
import numpy as np, torch
args = sys.argv[1:]
sys.argv = [sys.argv[0], args[0], "x"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openset as O
seedX = args[1].split(","); seedT = args[2].split(","); rounds = int(args[3]) if len(args) > 3 else 3
idx_of = {(O.who[i], str(O.names[i])): i for i in range(O.N)}
X = [idx_of[("anonymous", n)] for n in seedX]; T = [idx_of[("anonymous", n)] for n in seedT]


def build(idx):
    num = torch.zeros(O.H, O.W, device=O.dev); den = torch.zeros_like(num); parts = {}
    for i in idx:
        a, b = O.contrib(i); num += a; den += b; parts[i] = (a, b)
    return num, den, parts


def score(i, nd):
    num, den, parts = nd
    if i in parts:
        a, b = parts[i]; return O.pce(i, O.zm((num - a) / (den - b + 1.0)))
    return O.pce(i, O.zm(num / (den + 1.0)))


anon = [i for i in range(O.N) if O.who[i] == "anonymous"]
for rnd in range(rounds):
    kx, kt = build(X), build(T)
    sx = {i: score(i, kx) for i in range(O.N)}; st = {i: score(i, kt) for i in range(O.N)}
    newX = [i for i in anon if sx[i] > 15 and sx[i] > 2 * max(st[i], 1)]
    newT = [i for i in anon if st[i] > 15 and st[i] > 2 * max(sx[i], 1)]
    print(f"round {rnd}: |X| {len(X)} -> {len(newX)}   |T| {len(T)} -> {len(newT)}", flush=True)
    X, T = newX, newT
print("\nCREDITED toorisugari vs anon-only K_X / K_T:")
for i in np.where(O.who == "toorisugari")[0]:
    print(f"  {O.names[i]}  X={sx[i]:7.1f}  T={st[i]:7.1f}  -> {'X' if sx[i] > st[i] else 'T'}")
print("\nother credited with X > 10:", [(O.who[i], str(O.names[i]), round(sx[i], 1)) for i in range(O.N) if O.who[i] not in ("anonymous", "toorisugari") and sx[i] > 10])
print("\nANON X members:", sorted(str(O.names[i]) for i in X))
print("ANON T members:", sorted(str(O.names[i]) for i in T))
np.savez("work/split_tx.npz", names=O.names, who=O.who, sx=np.array([sx[i] for i in range(O.N)]), st=np.array([st[i] for i in range(O.N)]),
         X=np.array([str(O.names[i]) for i in X]), T=np.array([str(O.names[i]) for i in T]))
