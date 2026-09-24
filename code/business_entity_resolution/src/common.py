"""Shared paths, file reading and the competition metric."""
import os
import time
import polars as pl

ROOT = os.environ.get("BER_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "student_resource", "dataset"))
WORK = os.environ.get("BER_WORK", os.path.join(ROOT, "work"))
OUT = os.environ.get("BER_OUT", os.path.join(ROOT, "output"))
os.makedirs(WORK, exist_ok=True)

_T0 = time.time()


def log(*a):
    """Print with elapsed seconds since start."""
    print(f"[{time.time() - _T0:7.1f}s]", *a, flush=True)


def read_tsv(path):
    """Read a challenge TSV with every column as string; no quote handling (names contain quotes)."""
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema_length=0)


def raw_path(split, k):
    return os.path.join(DATA, split, f"{split}_source{k}.tsv")


def read_truth():
    """Ground truth as a (s1_id, q_id) pair table; singletons are absent."""
    gt = read_tsv(os.path.join(DATA, "train", "train_ground_truth.tsv"))
    return (gt.with_columns(q_id=pl.col("matched_entity_ids").fill_null("").str.split(","))
              .explode("q_id").filter(pl.col("q_id") != "")
              .select(pl.col("source1_entity_id").alias("s1_id"), "q_id"))


def f05(pred, true):
    """F0.5 for one Source 1 entity. Empty/empty scores 1.0 as in the problem statement."""
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred_map, true_map, s1_ids):
    """Average F0.5 over all s1_ids. pred_map/true_map: s1_id -> set of matched ids."""
    return sum(f05(pred_map.get(s, set()), true_map.get(s, set())) for s in s1_ids) / len(s1_ids)


if __name__ == "__main__":
    # the worked example from the problem statement: expected 0.714
    print(round(f05({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"}), 3))
