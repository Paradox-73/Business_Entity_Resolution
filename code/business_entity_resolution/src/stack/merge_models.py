"""Put the model pairs of several fit.py runs into one model file; apply_test.py averages every model in the file.
  python merge_models.py <tag> [<tag> ...] <out_tag>
  v10d: python merge_models.py _rob _robtr _avg   -> OUT/lgb_models_avg.pkl, models 0-1 from lgb_models_rob.pkl
        (fitted on the two parts of the evaluation half) and 2-3 from lgb_models_robtr.pkl (fitted on the training half)
"""
import os
import sys
import pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feats import OUT  # noqa: E402

*tags, dst = sys.argv[1:]
models = {}
for t in tags:
    for m in pickle.load(open(os.path.join(OUT, f"lgb_models{t}.pkl"), "rb")).values():
        models[len(models)] = m
pickle.dump(models, open(os.path.join(OUT, f"lgb_models{dst}.pkl"), "wb"))
print(f"{len(models)} models from {tags} -> {os.path.join(OUT, f'lgb_models{dst}.pkl')}")
