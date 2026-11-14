"""Synthetic streams resembling the train generator, with wider anomaly variety
(any feature may drift/spike, random collective subsets, zone offsets)."""
import math, random

F = ("temperature_c", "humidity_pct", "vibration_g", "acoustic_db", "air_pressure_kpa", "co2_ppm")
MEAN = [13.0, 72.0, 0.12, 32.1, 101.5, 510.0]
SD = [0.2, 0.57, 0.010, 0.35, 0.045, 10.5]


def make(seed, n=3600, zone_offsets=False, interleave=False, wide=True):
    rnd = random.Random(seed)
    ph_t, ph_p = rnd.uniform(0, 6.3), rnd.uniform(0, 6.3)
    per_t, per_p = rnd.uniform(600, 900), rnd.uniform(1100, 1600)
    amp_t, amp_p = rnd.uniform(0.3, 0.6), rnd.uniform(0.08, 0.16)
    zones = ["north_gallery", "south_gallery"]
    off = {z: [0.0] * 6 for z in zones}
    if zone_offsets:
        off["south_gallery"] = [rnd.uniform(-3, 3) * s for s in SD]
    X = []
    for t in range(n):
        z = zones[t % 2] if interleave else zones[(t // 900) % 2]
        base = list(MEAN)
        base[0] += amp_t * math.sin(2 * math.pi * t / per_t + ph_t)
        base[4] += amp_p * math.sin(2 * math.pi * t / per_p + ph_p)
        X.append([z, [base[i] + off[z][i] + rnd.gauss(0, SD[i]) for i in range(6)]])
    lab = ["normal"] * n
    # schedule events, non-overlapping, after warmup
    t = rnd.randint(150, 300)
    while t < n - 120:
        kind = rnd.choice(["point", "point", "drift", "collective"])
        if kind == "point":
            f = rnd.randrange(6) if wide else rnd.choice([2, 3, 5])
            X[t][1][f] += rnd.choice([-1, 1] if wide else [1]) * rnd.uniform(10 if wide else 14, 25) * SD[f]
            lab[t] = "point"; t += rnd.randint(150, 400)
        elif kind == "drift":
            f = rnd.randrange(6) if wide else rnd.choice([1, 4, 5])
            L = rnd.randint(45, 75) if wide else 60
            fin = rnd.uniform(9, 14) * rnd.choice([-1, 1]) * SD[f]
            pre = 25  # ramp starts ~25 rows before the labelled part (as in train)
            for k in range(-pre, L):
                X[t + k][1][f] += fin * (k + pre + 1) / (L + pre)
                if k >= 0:
                    lab[t + k] = "drift"
            t += L + rnd.randint(150, 400)
        else:
            fs = rnd.sample(range(6), rnd.randint(2, 4)) if wide else [2, 3, 5]
            L = rnd.randint(25, 40) if wide else rnd.randint(30, 35)
            sh = {f: rnd.uniform(3.0, 5.5) * rnd.choice([-1, 1] if wide else [1]) * SD[f] for f in fs}
            for k in range(L):
                for f in fs:
                    X[t + k][1][f] += sh[f] + rnd.gauss(0, 0.8 * SD[f])
                lab[t + k] = "collective"
            t += L + rnd.randint(150, 400)
    rows = []
    for k, (z, v) in enumerate(X):
        r = {"timestamp": str(k), "bunker_zone": z}
        r.update({F[i]: v[i] for i in range(6)})
        rows.append(r)
    y = [0 if l == "normal" else 1 for l in lab]
    return rows, y, lab
