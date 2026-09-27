import sys; sys.path.insert(0, "C:/ber_scratch/frfix/lbcal")
import numpy as np, time, json
from scipy.optimize import least_squares
from model import *
t = time.time()
res = least_squares(resid, TH0.copy(), diff_step=1e-3, max_nfev=200)
th = res.x
print("fit", time.time() - t, "cost", res.cost)
print(dict(zip(PARAMS, np.round(th, 3))), "NT", round(np.exp(th[8])))
o = predict(th); print({k: round(v, 5) for k, v in o.items()})
np.save(f"{D}/th_full.npy", th)
