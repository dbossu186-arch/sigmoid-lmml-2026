"""Streaming anomaly detector for the Albania bunker sensor stream.

Online, no future data, pure-python (stdlib only), O(1) per reading.

Model of "normal": every sensor = slowly varying baseline + white noise.  The
baseline is tracked per feature with a Holt filter (level + bounded trend) so
slow daily cycles (temperature / pressure) are followed.  While an anomaly is
flagged the filter is frozen and only extrapolated, so a drift is not "learned"
as normal.  Noise scale per feature = EWMA of squared standardized residuals.

Detectors on standardized residuals z:
  * point      : max |z| of the current reading
  * drift      : signed EWMA of z per feature (normalised to unit variance),
                 confirmed by a faster EWMA so the alarm ends promptly
  * collective : EWMA of multivariate energy sum(z^2) - n (several moderate
                 simultaneous deviations add up)
Safety net: an alarm lasting far longer than any plausible anomaly resets the
baseline to the current values (level change => new normal).

Zones: one shared baseline (in train both galleries follow the same process)
plus a slowly learned per-zone additive offset.  A persistent alarm that began
right at a zone switch is attributed to that zone's offset.
"""
import math

FEATURES = ("temperature_c", "humidity_pct", "vibration_g",
            "acoustic_db", "air_pressure_kpa", "co2_ppm")

DEFAULT_PARAMS = dict(
    warmup=60,          # readings used to initialise the baseline
    a_lev=0.02,         # Holt level rate
    b_tr=0.02,         # Holt trend rate
    tr_max=0.01,        # max |trend| per step, in sigma units
    a_var=0.005,         # noise-variance EWMA rate
    clip=3.0,           # residual clip (sigma) for baseline / variance updates
    gate=1.0,           # stat/threshold ratio above which the filter freezes
    a_drift=0.08,       # slow drift EWMA rate
    a_fast=0.5,        # fast confirmation EWMA rate
    th_drift=7.0,       # threshold on normalised slow drift EWMA
    th_fast=3.5,        # threshold on normalised fast EWMA (confirmation)
    a_coll=0.7,        # collective energy EWMA rate
    th_coll=32.0,       # threshold on EWMA of (sum z^2 - n)
    th_point=4.5,       # |z| threshold for a point anomaly
    hold_max=120,        # max consecutive frozen steps before forced re-baseline
    sharp=6.0,          # sigmoid sharpness for score mapping
    a_off=0.003,        # per-zone offset learning rate
    sw_win=5,           # alarm starting <= this many steps after a zone switch
                        # and persisting => attributed to a zone offset
)


def _to_float(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (math.inf, -math.inf):
        return None
    return f


class _State:
    __slots__ = ("buf", "lev", "tr", "var", "var0", "d", "f", "e", "hold",
                 "ready")

    def __init__(self, n):
        self.buf = [[] for _ in range(n)]
        self.lev = [0.0] * n
        self.tr = [0.0] * n
        self.var = [1.0] * n
        self.var0 = [1.0] * n
        self.d = [0.0] * n
        self.f = [0.0] * n
        self.e = 0.0
        self.hold = 0
        self.ready = False

    def init_baseline(self):
        for i in range(len(self.lev)):
            b = self.buf[i]
            m = len(b)
            tm = (m - 1) / 2.0
            ym = sum(b) / m
            sxx = sum((k - tm) ** 2 for k in range(m))
            slope = sum((k - tm) * (b[k] - ym) for k in range(m)) / sxx
            res = [b[k] - (ym + slope * (k - tm)) for k in range(m)]
            s = math.sqrt(sum(r * r for r in res) / max(1, m - 2))
            if not s > 0:
                s = abs(ym) * 1e-3 + 1e-6
            self.lev[i] = ym + slope * (m - 1 - tm)
            self.tr[i] = 0.0
            self.var[i] = s * s
            self.var0[i] = s * s
        self.buf = [[] for _ in self.lev]
        self.ready = True


class Detector:
    def __init__(self, params=None):
        p = dict(DEFAULT_PARAMS)
        if params:
            p.update(params)
        self.p = p
        self.n = len(FEATURES)
        self.st = _State(self.n)
        self.off = {}           # per-zone additive offset (raw units)
        self.step = 0
        self.last_zone = None
        self.switch_step = -10**9
        self.alarm_start = 0
        a = p["a_drift"]
        self.dnorm = math.sqrt((2.0 - a) / a)
        a = p["a_fast"]
        self.fnorm = math.sqrt((2.0 - a) / a)

    def update(self, reading):
        try:
            s = self._update(reading)
        except Exception:
            return 0.0
        if not (s == s) or s < 0.0:
            return 0.0
        return float(s) if s <= 1.0 else 1.0

    # ------------------------------------------------------------------
    def _update(self, reading):
        p = self.p
        n = self.n
        st = self.st
        self.step += 1
        get = reading.get if isinstance(reading, dict) else (lambda k: None)
        zone = get("bunker_zone")
        zone = "_" if zone is None else str(zone).strip()
        if zone != self.last_zone:
            if self.last_zone is not None:
                self.switch_step = self.step
            self.last_zone = zone
        off = self.off.get(zone)
        if off is None:
            off = [0.0] * n
            self.off[zone] = off

        x = [None] * n
        for i, f in enumerate(FEATURES):
            v = _to_float(get(f))
            if v is None and st.ready:
                v = st.lev[i] + st.tr[i] + off[i]  # impute with forecast
            x[i] = v

        if not st.ready:
            if all(v is not None for v in x):
                for i in range(n):
                    st.buf[i].append(x[i])
            if len(st.buf[0]) >= p["warmup"]:
                st.init_baseline()
            return 0.0

        # forecast and standardized residual
        z = [0.0] * n
        sds = [0.0] * n
        for i in range(n):
            sds[i] = math.sqrt(st.var[i])
            z[i] = (x[i] - st.lev[i] - st.tr[i] - off[i]) / sds[i]

        zmax = 0.0
        for v in z:
            if abs(v) > zmax:
                zmax = abs(v)
        r_point = zmax / p["th_point"]

        # drift: slow + fast EWMAs (spike-limited input)
        cap = p["th_point"]
        a, af = p["a_drift"], p["a_fast"]
        thd, thf = p["th_drift"], p["th_fast"]
        dn, fn = self.dnorm, self.fnorm
        r_drift = 0.0
        for i in range(n):
            zi = z[i]
            if zi > cap:
                zi = cap
            elif zi < -cap:
                zi = -cap
            st.d[i] += a * (zi - st.d[i])
            st.f[i] += af * (zi - st.f[i])
            ds = st.d[i] * dn / thd
            fs = st.f[i] * fn / thf
            if ds * fs > 0:  # same sign
                rr = abs(ds) if abs(ds) < abs(fs) else abs(fs)
                if rr > r_drift:
                    r_drift = rr

        # collective: EWMA of energy, spike-limited
        en = 0.0
        for v in z:
            vv = v * v
            en += vv if vv < 49.0 else 49.0
        st.e += p["a_coll"] * ((en - n) - st.e)
        r_coll = st.e / p["th_coll"]

        r = max(r_point, r_drift, r_coll)
        score = 1.0 / (1.0 + math.exp(-p["sharp"] * (r - 1.0)))
        alarm = score >= 0.5

        # --- baseline update ------------------------------------------------
        quiet = (not alarm) and r_drift < p["gate"] and r_coll < p["gate"]
        if quiet:
            st.hold = 0
            al, bt, av, c = p["a_lev"], p["b_tr"], p["a_var"], p["clip"]
            ao = p["a_off"]
            tmf = p["tr_max"]
            for i in range(n):
                zi = z[i]
                if zi > c:
                    zi = c
                elif zi < -c:
                    zi = -c
                sd = sds[i]
                new = st.lev[i] + st.tr[i] + al * zi * sd
                tr = st.tr[i] + bt * (new - st.lev[i] - st.tr[i])
                tmax = tmf * sd
                st.tr[i] = tmax if tr > tmax else (-tmax if tr < -tmax else tr)
                st.lev[i] = new
                off[i] += ao * zi * sd
                st.var[i] += av * (zi * zi - 1.0) * st.var[i]
                fl = st.var0[i] * 0.01
                if st.var[i] < fl:
                    st.var[i] = fl
        else:
            if st.hold == 0:
                self.alarm_start = self.step
            st.hold += 1
            for i in range(n):
                st.lev[i] += st.tr[i]
            if st.hold > p["hold_max"]:
                # persistent change: new normal.  If it began right at a zone
                # switch, attribute it to a zone offset, else to the process.
                to_zone = self.alarm_start - self.switch_step <= p["sw_win"]
                for i in range(n):
                    dev = x[i] - st.lev[i] - st.tr[i] - off[i]
                    if to_zone:
                        off[i] += dev
                    else:
                        st.lev[i] += dev
                        st.tr[i] = 0.0
                    st.d[i] = 0.0
                    st.f[i] = 0.0
                st.e = 0.0
                st.hold = 0
        return score
