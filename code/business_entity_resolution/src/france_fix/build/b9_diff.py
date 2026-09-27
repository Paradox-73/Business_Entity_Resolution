"""France pair diff of each built file vs v7ens; non-France rows identical to the main file; V2 additions vs add set."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))  # src/
from common import ROOT, WORK  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
from common import WORK, id_to_int
W = f"{WORK}/frfix"; S = f"{ROOT}/submissions"
fr = set(pl.read_parquet(f"{WORK}/test_s1.parquet", columns=["entity_id", "country"]).filter(pl.col("country") == "France")["entity_id"].cast(pl.Utf8).to_list())
def rd(p):
    return pl.read_csv(p, separator="\t", schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8}, quote_char=None).with_columns(pl.col("matched_entity_ids").fill_null(""))
def pairs(df):
    d = df.filter(pl.col("source1_entity_id").is_in(fr) & (pl.col("matched_entity_ids") != ""))
    return d.with_columns(q=pl.col("matched_entity_ids").str.split(",")).explode("q").select(s="source1_entity_id", q="q")
E = rd(f"{S}/v7ens/matching_results.tsv"); P = rd(f"{S}/v7p/matching_results.tsv")
Ep = pairs(E); print("v7ens France pairs", Ep.height)
add2 = pl.read_parquet("add_set_v2_final.parquet").select("q", "s")
for v in ("descveto", "descveto_gnadd"):
    for mname, M in (("v7ens", E), ("v7p", P)):
        X = rd(f"{W}/out_{v}_on_{mname}/matching_results.tsv")
        j = X.join(M, on="source1_entity_id", suffix="_m")
        nonfr = j.filter(~pl.col("source1_entity_id").is_in(fr) & (pl.col("matched_entity_ids") != pl.col("matched_entity_ids_m"))).height
        print(f"{v} on {mname}: rows {X.height}, non-France rows differing from main: {nonfr}")
        if mname != "v7ens":
            continue
        Xp = pairs(X)
        a = Xp.join(Ep, on=["s", "q"], how="anti"); r = Ep.join(Xp, on=["s", "q"], how="anti")
        jf = j.filter(pl.col("source1_entity_id").is_in(fr))
        ch = jf.filter(pl.col("matched_entity_ids") != pl.col("matched_entity_ids_m"))
        print(f"   France pairs {Xp.height} (+{a.height} / -{r.height}); S1 rows changed {ch.height}; "
              f"made empty {ch.filter(pl.col('matched_entity_ids') == '').height}; made non-empty {ch.filter(pl.col('matched_entity_ids_m') == '').height}")
        if v == "descveto_gnadd":
            ai = a.with_columns(q=id_to_int("q"), s=id_to_int("s"))
            print("   added pairs in the V2 add set:", ai.join(add2, on=["q", "s"]).height, "of", add2.height)
old = rd(f"{SCRATCH}/frfix/namechg/v7ens_descveto/matching_results.tsv")
X = rd(f"{W}/out_descveto_on_v7ens/matching_results.tsv")
print("V1 build identical to the analyst's spliced file:", X.equals(old))
