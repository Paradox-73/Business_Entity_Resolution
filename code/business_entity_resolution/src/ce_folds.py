"""One command for a GPU machine: train and score the 3 fold transformers of the reranker (rerank.py fold sides).

  python ce_folds.py --model intfloat/multilingual-e5-base --name base            # full run
  python ce_folds.py --model intfloat/multilingual-e5-base --name base --smoke    # 3-minute check first

For each fold k = 0, 1, 2: train on the close calls whose record is in another fold (2/3 of all labelled close calls,
incl. stage-1 ranks 3-5), then score fold k's close calls and ALL test close calls. Each step runs in its own process
and is retried up to 3 times after a crash (out of memory etc.): training resumes from its last checkpoint (every
1,000 steps), scoring from its last saved chunk (200k pairs). Finished steps are skipped when the command is re-run.

Inputs (WORK = <repo>/work): WORK/{train,test}_s{1,2,3}.parquet (names/addresses), WORK/ce_x/{train,test}_rows.parquet.
Outputs to send back: WORK/ce_x/train_ce_<name>f{0,1,2}.parquet and WORK/ce_x/test_ce_<name>f{0,1,2}.parquet.
Log: WORK/ce_folds_<name>.log.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import WORK  # noqa: E402  (also opts this process out of Windows power throttling)


def main():
    """Parse the options, check the inputs and the GPU, then for each fold train (rerank.py train) and score
    (rerank.py score), each step with up to 3 tries; steps already finished are skipped."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--name", default="base")
    ap.add_argument("--dir", default=os.path.join(WORK, "ce_x"))
    ap.add_argument("--folds", default="0,1,2")
    ap.add_argument("--batch", default="")
    ap.add_argument("--train-rows", default="0", help="cap on training rows per fold (0 = all)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gc", action="store_true", help="gradient checkpointing (less GPU memory, ~30%% slower)")
    ap.add_argument("--half-emb", action="store_true", help="frozen word table in 16 bit (saves 0.4-0.5 GB)")
    ap.add_argument("--no-freeze", action="store_true", help="train the word table too (big GPUs, e.g. 48 GB)")
    ap.add_argument("--lr", default="", help="peak learning rate (default 3e-5; 1e-5 to 2e-5 for large models)")
    ap.add_argument("--score-batch", default="", help="pairs per scoring batch (default 256; 1024 on a 48 GB GPU)")
    a = ap.parse_args()
    logf = os.path.join(WORK, f"ce_folds_{a.name}{'_smoke' if a.smoke else ''}.log")

    def log(msg):
        """Print msg with the time of day and append it to WORK/ce_folds_<name>.log."""
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(logf, "a", encoding="utf8") as f:
            f.write(line + "\n")

    for need in [os.path.join(WORK, f"{sp}_s{k}.parquet") for sp in ("train", "test") for k in (1, 2, 3)] + \
                [os.path.join(a.dir, f"{sp}_rows.parquet") for sp in ("train", "test")]:
        if not os.path.exists(need):
            sys.exit(f"missing input {need} (unzip the data bundle into the repo root first)")
    import torch
    if not torch.cuda.is_available():
        sys.exit("no CUDA GPU visible to torch: install the CUDA build of torch (README.md section 2)")
    log(f"GPU {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB; "
        f"model {a.model}; name {a.name}; smoke {a.smoke}")

    def run(what, env_extra, args):
        """Run 'rerank.py <args>' with the fold's environment, output to WORK/ce_folds_<name>_<what>.log; up to 3
        tries, 60 s apart. Returns True on success."""
        env = dict(os.environ, PYTHONIOENCODING="utf8", BER_CE_DIR=a.dir, BER_CE_NAME=a.name, **env_extra)
        if a.batch:
            env["BER_CE_BS"] = a.batch
        if a.gc:
            env["BER_CE_GC"] = "1"
        if a.half_emb:
            env["BER_CE_HALF_EMB"] = "1"
        if a.no_freeze:
            env["BER_CE_FREEZE"] = "0"
        if a.lr:
            env["BER_CE_LR"] = a.lr
        if a.score_batch:
            env["BER_CE_SCORE_BS"] = a.score_batch
        out = os.path.join(WORK, f"ce_folds_{a.name}_{what.replace(' ', '_')}.log")
        for k in range(1, 4):
            log(f"START {what} (try {k})")
            with open(out, "a", encoding="utf8") as f:
                rc = subprocess.call([sys.executable, os.path.join(HERE, "rerank.py")] + args, env=env, cwd=HERE,
                                     stdout=f, stderr=subprocess.STDOUT)
            if rc == 0:
                log(f"OK {what}")
                return True
            with open(out, encoding="utf8", errors="replace") as f:
                tail = [x.strip() for x in f.readlines()[-3:]]
            log(f"FAILED {what} (try {k}, exit {rc}): {' | '.join(tail)[:300]}")
            time.sleep(60)
        return False

    for fold in [int(x) for x in a.folds.split(",")]:
        side = f"f{fold}"
        model_dir = os.path.join(WORK, "ce", f"model_{a.name}{side}{'_smoke' if a.smoke else ''}")
        env = {"BER_CE_SIDE": side, "BER_CE_MODEL": model_dir}
        if a.smoke:
            env["BER_CE_LIMIT"] = "640"
        elif a.train_rows != "0":
            env["BER_CE_LIMIT"] = a.train_rows
        if os.path.exists(os.path.join(model_dir, "config.json")):
            log(f"fold {fold}: model already trained ({model_dir})")
        elif not run(f"train fold {fold}", env, ["train", a.model]):
            sys.exit(f"fold {fold}: training failed 3 times, see the logs in {WORK}")
        env_score = {k: v for k, v in env.items() if k != "BER_CE_LIMIT" or a.smoke}
        if a.smoke:
            env_score["BER_CE_LIMIT"] = "2000"
        if not run(f"score fold {fold}", env_score, ["score"]):
            sys.exit(f"fold {fold}: scoring failed 3 times, see the logs in {WORK}")
    sfx = "_smoke" if a.smoke else ""
    outs = [os.path.join(a.dir, f"{sp}_ce_{a.name}f{f}{sfx}.parquet") for f in a.folds.split(",") for sp in ("train", "test")]
    log("DONE. Send back: " + ", ".join(os.path.relpath(o, os.path.dirname(WORK)) for o in outs if os.path.exists(o)))


if __name__ == "__main__":
    main()
