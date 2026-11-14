"""Attribution in gram space (fast, exact for sum-fingerprints).

x_i = cleaned residual * luminance, unit norm. G_id[i,j]=<x_i,x_j>, G_rot[i,j]=<x_i,rot180 x_j>.
Camera c with members j and orientations s_j (0 id, 1 rot): v_c = sum_j T_{s_j} x_j.
<T_o x_i, v_c> = sum_j G_{o xor s_j}[i,j];  ||v_c||^2 = sum_{j,k} G_{s_j xor s_k}[j,k].
score(i,c,o) = <T_o x_i, v_c without i> / ||v_c without i||   (a correlation; null std ~ 1/sqrt(N)).

    python v2/gattrib.py work/gram_clean.npz work/gattrib.json
"""
import json, sys, collections
import numpy as np

z = np.load(sys.argv[1]); names = list(z["names"]); Gi = z["G_id"].astype(np.float64); Gr = z["G_rot"].astype(np.float64)
n = len(names); who = np.array([s.split("__")[0] for s in names])
people = ["amayadori", "jihanki", "rojiura", "shuuden", "toorisugari", "yakin"]
G = [Gi, Gr]
SCALE = 1000.0  # ~sqrt(N_eff): turns correlations into rough z-scores

def orient_spectral(idx):
    D = Gi[np.ix_(idx, idx)] - Gr[np.ix_(idx, idx)]
    np.fill_diagonal(D, 0)
    w, V = np.linalg.eigh(D)
    v = V[:, -1]
    s = (v < 0).astype(int)
    return dict(zip(idx, s))

def cam_score(i, mem, o):
    """mem: dict j->s_j. returns correlation of T_o x_i with v_c (excluding i)."""
    js = [j for j in mem if j != i]
    if not js: return 0.0
    s = np.array([mem[j] for j in js]); js = np.array(js)
    dot = np.where((o ^ s) == 0, Gi[i, js], Gr[i, js]).sum()
    S = s[:, None] ^ s[None, :]
    nrm2 = np.where(S == 0, Gi[np.ix_(js, js)], Gr[np.ix_(js, js)]).sum()
    return SCALE * dot / np.sqrt(max(nrm2, 1e-12))

def scores(i, cams):
    return {(p, o): cam_score(i, cams[p], o) for p in people for o in (0, 1)}

cams = {p: orient_spectral(list(np.where(who == p)[0])) for p in people}
for p in people:
    print(p, "rot", sum(cams[p].values()), "/", len(cams[p]))

def loo(cams):
    ok = 0; per = collections.Counter()
    for i in np.where(who != "anonymous")[0]:
        s = scores(i, cams); pred = max(s, key=s.get)[0]
        ok += pred == who[i]; per[who[i]] += pred == who[i]
    return ok / (who != "anonymous").sum(), per

acc, per = loo(cams); print("credited LOO acc %.3f" % acc, dict(per))

anon = list(np.where(who == "anonymous")[0])
THR = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
assign = {}
for rnd in range(8):
    cur = {p: {j: s for j, s in cams[p].items() if who[j] != "anonymous"} for p in people}
    for i, (p, o) in assign.items():
        cur[p][i] = o
    # re-orient each camera's set jointly
    cur = {p: orient_spectral(list(cur[p])) if len(cur[p]) > 1 else cur[p] for p in people}
    res = []
    new = {}
    for i in anon:
        s = scores(i, cur)
        per = {p: max(s[(p, 0)], s[(p, 1)]) for p in people}
        rk = sorted(per, key=per.get, reverse=True)
        b = rk[0]; o = 0 if s[(b, 0)] >= s[(b, 1)] else 1
        # orientation relative to the camera frame chosen by spectral sign
        res.append({"name": names[i], "pred": b, "ori": o, "z": per[b], "margin": per[b] - per[rk[1]], "second": rk[1],
                    **{f"{p}|{oo}": s[(p, oo)] for p in people for oo in (0, 1)}})
        if per[b] >= THR and per[b] - per[rk[1]] >= THR / 2:
            new[i] = (b, o)
    changed = set(new.items()) ^ set(assign.items())
    assign = new
    print(f"round {rnd}: confident {len(assign)}", dict(collections.Counter(p for p, _ in assign.values())), "changed", len(changed))
    cams_final = cur
    if not changed: break
acc, per = loo({p: {j: s for j, s in cams_final[p].items()} for p in people})
print("credited LOO with anon in fingerprints %.3f" % acc, dict(per))
print("final preds", collections.Counter(r["pred"] for r in res))
zs = np.array([r["z"] for r in res]); ms = np.array([r["margin"] for r in res])
print("z quantiles", np.round(np.quantile(zs, [0, .1, .25, .5, .75, .9, 1]), 2))
print("margin quantiles", np.round(np.quantile(ms, [0, .1, .25, .5, .75, .9, 1]), 2))
json.dump({"anonymous": res, "cams": {p: {names[j]: int(s) for j, s in cams_final[p].items()} for p in people}},
          open(sys.argv[2], "w"), indent=1)
