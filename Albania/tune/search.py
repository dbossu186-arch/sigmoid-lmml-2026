"""Random search over Detector params on train (halves) + synthetic streams.
usage: python search.py N_CONFIGS SEED OUT"""
import sys, random, json, os
from multiprocessing import Pool
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D + "/.."); sys.path.insert(0, D)
import harness as h, synth

mod = h.load_detector(os.path.join(D, "..", "detector.py"))
rows, y, t = h.load_data(*[os.path.join(D, "..", "data", f) for f in ("train-input.csv", "train-output.csv")])
rows = [{k: (v if k in ("timestamp", "bunker_zone") else float(v)) for k, v in r.items()} for r in rows]
H = 1800
SETS = dict(h1=(rows[:H], y[:H], t[:H]), h2=(rows[H:], y[H:], t[H:]))
for s in range(4): SETS[f"sn{s}"] = synth.make(100 + s, wide=False)
for s in range(3): SETS[f"sw{s}"] = synth.make(200 + s)
for s in range(2): SETS[f"zo{s}"] = synth.make(300 + s, zone_offsets=True, wide=False)
for s in range(2): SETS[f"zi{s}"] = synth.make(400 + s, zone_offsets=True, interleave=True, wide=False)

SPACE = dict(
    a_lev=[0.015, 0.02, 0.03, 0.04, 0.06],
    b_tr=[0.0, 0.005, 0.01, 0.02, 0.04],
    tr_max=[0.01, 0.03, 0.1],
    gate=[0.8, 1.0, 1.0, 1.5, 3.0],
    a_drift=[0.03, 0.05, 0.08, 0.12],
    a_fast=[0.25, 0.35, 0.5, 0.7],
    th_drift=[5.0, 6.0, 7.0, 8.0, 10.0, 12.0],
    th_fast=[2.0, 2.5, 3.0, 3.5, 4.0],
    a_coll=[0.25, 0.35, 0.5, 0.7],
    th_coll=[14.0, 18.0, 25.0, 32.0, 40.0],
    th_point=[4.5, 5.0, 6.0, 7.0],
    hold_max=[70, 90, 120],
    clip=[2.0, 2.5, 3.0],
    a_var=[0.005, 0.01, 0.02],
)

def evalp(p):
    out = {}
    for name, (r, yy, tt) in SETS.items():
        sc = h.run(mod, r, p)
        out[name] = h.score([s >= 0.5 for s in sc], yy, tt)["score"]
    tr = (out["h1"] + out["h2"]) / 2
    sn = sum(out[f"sn{s}"] for s in range(4)) / 4
    sw = sum(out[f"sw{s}"] for s in range(3)) / 3
    zz = sum(out[k] for k in out if k[:2] in ("zo", "zi")) / 4
    out["obj"] = 0.55 * tr + 0.25 * sn + 0.1 * sw + 0.1 * zz
    return p, out

if __name__ == "__main__":
    n, seed, outf = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    rnd = random.Random(seed)
    cfgs = [{k: rnd.choice(v) for k, v in SPACE.items()} for _ in range(n)]
    if len(sys.argv) > 4:
        cfgs = [json.loads(l) for l in open(sys.argv[4])] + cfgs
    with Pool(16) as pool:
        res = pool.map(evalp, cfgs, chunksize=4)
    with open(outf, "w") as f:
        for p, o in res:
            f.write(json.dumps(dict(p=p, o=o)) + "\n")
    res.sort(key=lambda r: -r[1]["obj"])
    for p, o in res[:15]:
        print({k: round(v, 1) for k, v in o.items()}, p)
