"""Fine tau scan from a run's val_probs.npz: held-out known acc and Zappos rejection."""
import sys, numpy as np
for f in sys.argv[1:]:
    d = np.load(f); pv, yv, pu, pk, zky = d["va"], d["va_y"], d["zu"], d["zk"], d["zk_y"]
    print(f)
    for t in np.arange(0.55, 0.80, 0.02):
        ka = np.mean((pv.argmax(1) == yv) & (pv.max(1) >= t)); rj = np.mean(pu.max(1) < t)
        kz = np.mean((pk.argmax(1) == zky) & (pk.max(1) >= t))
        print(f"  tau={t:.2f} heldout known={ka:.3f} zk known={kz:.3f} rej={rj:.3f}" + ("  <- both gates" if ka >= 0.85 and rj >= 0.70 else ""))
