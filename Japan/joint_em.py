"""Joint orientation-aware EM over credited + anonymous photos.
Credited photos keep their camera label but get the best orientation; anonymous photos get the best
(camera, orientation) when confident. Fingerprints are re-estimated from all members each round.
usage: joint_em.py <cache> <orient_attrib.json> <out_dir> [--thr 15] [--rounds 4]"""
import sys, os, json, argparse
import numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument("cache"); ap.add_argument("init"); ap.add_argument("out")
ap.add_argument("--thr", type=float, default=15.0); ap.add_argument("--rounds", type=int, default=4)
args = ap.parse_args()
sys.argv = [sys.argv[0], args.cache, "x"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openset as O
os.makedirs(args.out, exist_ok=True)
people = ["amayadori", "jihanki", "rojiura", "shuuden", "toorisugari", "yakin"]
ORI = ("id", "rot180")


def T(t, o):
    return t if o == "id" else torch.flip(t, (0, 1))


def pce(i, K, o):
    r = T(O.R[i].to(O.dev).float(), o); x = T(O.I[i].to(O.dev).float(), o); w = T(O.Wt[i].to(O.dev).float(), o).sqrt()
    a = r * w; b = x * K * w; a = a - a.mean(); b = b - b.mean()
    xc = torch.fft.irfft2(torch.fft.rfft2(a.to(O.FFTDEV)) * torch.fft.rfft2(b.to(O.FFTDEV)).conj(), s=(O.H, O.W))
    peak = xc[0, 0].item(); m = torch.ones_like(xc, dtype=torch.bool)
    for dy in range(-5, 6):
        m[dy % O.H, :6] = False; m[dy % O.H, O.W - 5:] = False
    return float(np.sign(peak) * peak * peak / (xc[m] ** 2).mean().item())


init = json.load(open(args.init))
key = lambda i: f"{O.who[i]}/{O.names[i]}"
idx = {key(i): i for i in range(O.N)}
# membership: i -> (camera, orientation)
mem = {}
for k, o in init["cred_ori"].items():
    i = idx[k]; mem[i] = (str(O.who[i]), o)
for r in init["anon"]:
    if r["best"] >= args.thr and r["best"] >= 3 * max(r["second"], 1):
        mem[idx["anonymous/" + r["name"]]] = (r["pred"], r["ori"])

for rnd in range(args.rounds):
    num = {p: torch.zeros(O.H, O.W, device=O.dev) for p in people}; den = {p: torch.zeros(O.H, O.W, device=O.dev) for p in people}
    parts = {}
    for i, (p, o) in mem.items():
        a, b = O.contrib(i); a, b = T(a, o), T(b, o); num[p] += a; den[p] += b; parts[i] = (a.half().cpu(), b.half().cpu())
    Kfull = {p: O.zm(num[p] / (den[p] + 1.0)) for p in people}
    scores = {}
    for i in range(O.N):
        sc = {}
        for p in people:
            if i in mem and mem[i][0] == p:
                a, b = parts[i]; K = O.zm((num[p] - a.to(O.dev).float()) / (den[p] - b.to(O.dev).float() + 1.0))
            else:
                K = Kfull[p]
            for o in ORI:
                sc[(p, o)] = pce(i, K, o)
        scores[i] = sc
    new = {}
    for i in range(O.N):
        sc = scores[i]
        if O.who[i] != "anonymous":
            p = str(O.who[i]); new[i] = (p, max(ORI, key=lambda o: sc[(p, o)]))
        else:
            per = {p: max(sc[(p, o)] for o in ORI) for p in people}
            srt = sorted(per.values(), reverse=True); bp = max(per, key=per.get)
            if srt[0] >= args.thr and srt[0] >= 3 * max(srt[1], 1):
                new[i] = (bp, max(ORI, key=lambda o: sc[(bp, o)]))
    changed = sum(1 for i in set(new) | set(mem) if new.get(i) != mem.get(i))
    cnt = {p: sum(1 for i, v in new.items() if v[0] == p and O.who[i] == "anonymous") for p in people}
    # credited LOO accuracy (best camera over both orientations)
    cred = [i for i in range(O.N) if O.who[i] != "anonymous"]
    acc = np.mean([max(people, key=lambda p: max(scores[i][(p, o)] for o in ORI)) == O.who[i] for i in cred])
    print(f"round {rnd}: changed {changed}  anon assigned {sum(cnt.values())} {cnt}  credited LOO acc {acc:.3f}", flush=True)
    mem = new
    if changed == 0:
        break

out = []
for i in range(O.N):
    sc = scores[i]
    per = {p: max(sc[(p, o)] for o in ORI) for p in people}
    srt = sorted(per.values(), reverse=True); bp = max(per, key=per.get)
    out.append(dict(who=str(O.who[i]), name=str(O.names[i]), pred=bp, ori=max(ORI, key=lambda o: sc[(bp, o)]),
                    best=srt[0], second=srt[1], member=list(mem[i]) if i in mem else None,
                    **{f"{p}|{o}": sc[(p, o)] for p in people for o in ORI}))
json.dump(out, open(f"{args.out}/joint.json", "w"), indent=0)
for p in people:
    np.save(f"{args.out}/K_{p}.npy", Kfull[p].cpu().numpy().astype(np.float32))
for r in out:
    if r["who"] != "anonymous":
        if r["pred"] != r["who"]:
            print("CRED-MISMATCH", r["who"], r["name"], "->", r["pred"], round(r["best"], 1), "own", round(max(r[f"{r['who']}|id"], r[f"{r['who']}|rot180"]), 1))
low = [r for r in out if r["who"] == "anonymous" and r["member"] is None]
print("unassigned anonymous:", len(low))
for r in sorted(low, key=lambda r: -r["best"]):
    print(f"  {r['name']} best={r['pred']}|{r['ori']} {r['best']:.1f} second {r['second']:.1f}")
