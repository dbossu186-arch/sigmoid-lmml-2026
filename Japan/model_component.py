"""Same-model hypothesis: Toorisugari and Yakin share a model-level component. Score every photo against
pooled(T+Y), T only, Y only, and pooled(other four). Photos high on pooled(T+Y) but on neither unit
fingerprint would come from a third camera of the same model.
usage: model_component.py <cache> <attrib.json>"""
import sys, os, json
import numpy as np, torch
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2]]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openset as O

att = {r["name"]: r for r in json.load(open(sys.argv[2]))["anon"]}
groups = {"T": np.where(O.who == "toorisugari")[0], "Y": np.where(O.who == "yakin")[0]}
groups["TY"] = np.concatenate([groups["T"], groups["Y"]])
groups["OTH"] = np.where(np.isin(O.who, ["amayadori", "jihanki", "rojiura", "shuuden"]))[0]
sums = {}
for g, idx in groups.items():
    num = torch.zeros(O.H, O.W, device=O.dev); den = torch.zeros_like(num)
    for i in idx:
        a, b = O.contrib(i); num += a; den += b
    sums[g] = (num, den)


def score(i, g):
    num, den = sums[g]
    if i in groups[g]:
        a, b = O.contrib(i); num = num - a; den = den - b
    return O.pce(i, O.zm(num / (den + 1.0)))


rows = []
for i in range(O.N):
    s = {g: score(i, g) for g in ["T", "Y", "TY", "OTH"]}
    rows.append((O.who[i], O.names[i], s))
    lab = att.get(str(O.names[i]), {})
    print(f"{O.who[i][:5]:5s} {O.names[i]} T={s['T']:7.1f} Y={s['Y']:7.1f} TY={s['TY']:7.1f} OTH={s['OTH']:7.1f}  att={lab.get('pred','-')}:{lab.get('best',0):.0f}", flush=True)
