import sys; sys.path.insert(0, "C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix/lbcal")
import numpy as np, time, json
from scipy.optimize import least_squares
from model import *
th_full = np.load(f"{D}/th_full.npy")
def fit(name, drop=(), sig0=None, th0=None, start=None):
    t = time.time()
    res = least_squares(resid_fast, (th_full if start is None else start).copy(), diff_step=1e-3, max_nfev=12, ftol=1e-5, xtol=1e-5,
                        kwargs=dict(drop=drop, sig0=sig0, th0=th0))
    o = predict_fast(res.x)
    print(f"[{name}] {time.time()-t:.0f}s cost {res.cost:.3f} th {dict(zip(PARAMS, np.round(res.x, 3)))} NT {np.exp(res.x[8]):.0f}")
    print("    pred", {k: round(v, 5) for k, v in o.items()}, flush=True)
    np.save(f"{D}/th_{name}.npy", res.x)
    return res.x
for k in ["m", "i", "j", "gn"]:
    fit("loo_" + k, drop=(k,))
s = SIG0.copy(); s[3:8] = 0.5; fit("tightname", sig0=s)
s = SIG0.copy(); s[1:3] = 3.0; fit("wideslope", sig0=s)
t0 = TH0.copy(); t0[1], t0[2] = 1.0, 0.0; fit("prior_p3", th0=t0)
t0 = TH0.copy(); t0[1], t0[2] = 0.0, 1.0; fit("prior_g", th0=t0)
s = SIG0.copy(); s[8] = 0.003; t0 = TH0.copy(); t0[8] = np.log(860e3); fit("nt860", sig0=s, th0=t0)
s = SIG0.copy(); s[8] = 0.003; t0 = TH0.copy(); t0[8] = np.log(930e3); fit("nt930", sig0=s, th0=t0)
