"""Shared paths, file reading, id encoding and the competition metric."""
import os
import time
import polars as pl

ROOT = os.environ.get("BER_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "student_resource", "dataset"))
WORK = os.environ.get("BER_WORK", os.path.join(ROOT, "work"))
OUT = os.environ.get("BER_OUT", os.path.join(ROOT, "output"))
os.makedirs(WORK, exist_ok=True)

_T0 = time.time()
SRC_MULT = 10_000_000_000   # integer id = source * SRC_MULT + numeric part of the id


def _no_power_throttling():
    """Windows 11 puts windowless background processes on the slow efficiency cores of hybrid CPUs (EcoQoS);
    26 Sep: transformer training ran at 2.3 steps/s throttled vs 11.6 unthrottled. Opt this process out."""
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        class State(ctypes.Structure):
            """PROCESS_POWER_THROTTLING_STATE of the Windows API."""
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]
        k = ctypes.WinDLL("kernel32")
        st = State(1, 0x1 | 0x4, 0)             # execution speed + timer resolution: control on, throttling off
        k.SetProcessInformation(wintypes.HANDLE(k.GetCurrentProcess()), 4, ctypes.byref(st), ctypes.sizeof(st))
    except Exception:
        pass


_no_power_throttling()


def log(*a):
    """Print with elapsed seconds since start."""
    print(f"[{time.time() - _T0:7.1f}s]", *a, flush=True)


def read_tsv(path):
    """Read a challenge TSV with every column as string; no quote handling (names contain quotes)."""
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema_length=0)


def raw_path(split, k):
    """Path of a raw challenge file: DATA/<split>/<split>_source<k>.tsv."""
    return os.path.join(DATA, split, f"{split}_source{k}.tsv")


def id_to_int(col):
    """'S2-000123' -> 2*SRC_MULT + 123 (works for S1/S2/S3)."""
    c = pl.col(col)
    return c.str.slice(1, 1).cast(pl.Int64) * SRC_MULT + c.str.slice(3).cast(pl.Int64)


def int_to_id(col):
    """Inverse of id_to_int: integer id -> 'S<source>-<number>'."""
    c = pl.col(col)
    return pl.format("S{}-{}", (c // SRC_MULT).cast(pl.Utf8), (c % SRC_MULT).cast(pl.Utf8))


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


def macro_f05_df(pred, truth, s1):
    """Vectorised macro F0.5. pred/truth: DataFrames with int columns s, q. s1: DataFrame with column s."""
    tp = pred.join(truth, on=["s", "q"]).group_by("s").len("tp")
    d = (s1.join(pred.group_by("s").len("np"), on="s", how="left")
           .join(truth.group_by("s").len("nt"), on="s", how="left")
           .join(tp, on="s", how="left").fill_null(0))
    p, r = pl.col("tp") / pl.col("np"), pl.col("tp") / pl.col("nt")
    f = (pl.when(pl.col("nt") == 0).then((pl.col("np") == 0).cast(pl.Float64))
           .when(pl.col("tp") == 0).then(0.0)
           .otherwise(1.25 * p * r / (0.25 * p + r)))
    return d.select(f.mean()).item()


if __name__ == "__main__":
    # the worked example from the problem statement: expected 0.714
    print(round(f05({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"}), 3))
    pr = pl.DataFrame({"s": [1, 1, 1], "q": [47, 193, 812]})
    tr = pl.DataFrame({"s": [1, 1], "q": [47, 812]})
    print(round(macro_f05_df(pr, tr, pl.DataFrame({"s": [1]})), 3))
