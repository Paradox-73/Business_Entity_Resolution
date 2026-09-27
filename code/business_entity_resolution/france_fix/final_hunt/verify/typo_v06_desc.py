"""For pairs whose S1 word a is a descriptor: is the garbled b better explained by another descriptor B (same first letter)?
Calibrated on analog true garbles (any a) with the same alternative vocabulary."""
import sys
from collections import Counter
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/france_fix/artifacts/census")
import polars as pl
from ops import DESC

pl.Config.set_tbl_rows(110); pl.Config.set_fmt_str_lengths(40); pl.Config.set_tbl_width_chars(250)
print(len(DESC), sorted(DESC)[:200])
d = pl.read_parquet(r"C:/ber_scratch/final/verify/typo_mix.parquet")
ALT = sorted(w for w in DESC if len(w) > 2)


def dice(a, b):
    ca, cb = Counter(a), Counter(b)
    return 2 * sum(min(v, cb[k]) for k, v in ca.items()) / (len(a) + len(b))


def lr(a, b):
    own = dice(a, b)
    alts = [(dice(w, b), w) for w in ALT if w != a and w[:1] == b[:1]]
    bo, bw = max(alts) if alts else (0.0, None)
    return own, bo, bw


rows = []
for r in d.iter_rows(named=True):
    own, bo, bw = lr(r["a"], r["b"])
    rows.append(dict(own=own, alt=bo, alt_w=bw))
d = pl.concat([d, pl.DataFrame(rows)], how="horizontal").with_columns(alt_wins=pl.col("alt") >= pl.col("own"))
print(d.group_by("grp").agg(n=pl.len(), alt_wins=pl.col("alt_wins").mean(), own=pl.col("own").mean(), alt=pl.col("alt").mean()).sort("grp"))
print(d.filter((pl.col("grp") == "fr_set_aDESC")).sort("own").select("a", "b", "own", "alt", "alt_w", "rec").head(100))
d.write_parquet(r"C:/ber_scratch/final/verify/typo_desc.parquet")
