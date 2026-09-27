import sys; sys.path.insert(0, "C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix/lbcal")
import numpy as np, time
from model import *
th = np.load(f"{D}/th_full.npy")
t = time.time(); o1 = predict(th); t1 = time.time(); o2 = predict_fast(th); t2 = time.time()
print(o1); print(o2); print(t1 - t, t2 - t1)
