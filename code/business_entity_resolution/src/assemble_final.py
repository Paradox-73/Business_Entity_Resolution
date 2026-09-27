"""Final submission from parts: US/India rows of one file, plus (optionally) gathik v8's US/India pairs our candidate
lists never had (for records the US/India file leaves unmatched; v9y's recipe), plus France rows of another file.

  python assemble_final.py <us_india.tsv> <france.tsv> <out_dir> [gathik.tsv [our_candidate_scores.parquet]]
e.g. v9y = assemble_final.py submissions/v9e/... submissions/v9e/... work/out_x work/gathik/v8_matching_results.tsv
         work/test_scores_blend_v7p.parquet
"""
import os
import sys
import polars as pl
from common import WORK, id_to_int, int_to_id, log


def pairs(p, s1):
    """(s, q, country) integer pairs of a matching_results.tsv."""
    d = pl.read_csv(p, separator="\t", schema_overrides={"matched_entity_ids": pl.Utf8}).filter(
        pl.col("matched_entity_ids").fill_null("") != "")
    return (d.select(s=id_to_int("source1_entity_id").cast(pl.Int64), q=pl.col("matched_entity_ids").str.split(","))
             .explode("q").with_columns(q=id_to_int("q").cast(pl.Int64)).join(s1.select("s", "country"), on="s"))


def main(ui_tsv, fr_tsv, out, g_tsv=None, lists=os.path.join(WORK, "test_scores_blend_v7p.parquet")):
    """Write <out>/matching_results.tsv: the US/India pairs of ui_tsv, the France pairs of fr_tsv and, with
    g_tsv, the US/India pairs of g_tsv that are absent from the score table `lists`, for records ui_tsv leaves unmatched."""
    s1 = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=["entity_id", "country"]).select(
        s1_id="entity_id", s=id_to_int("entity_id").cast(pl.Int64), country="country")
    ui = pairs(ui_tsv, s1).filter(pl.col("country") != "France")
    fr = pairs(fr_tsv, s1).filter(pl.col("country") == "France")
    parts = [ui, fr]
    if g_tsv:
        g = pairs(g_tsv, s1).filter(pl.col("country") != "France")
        known = pl.read_parquet(lists, columns=["q", "s"]).with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
        add = g.join(known, on=["q", "s"], how="anti").join(ui.select("q"), on="q", how="anti")
        log(f"gathik pairs our lists never had, records unmatched: {add.height} "
            f"({add.group_by('country').len().sort('country').rows()})")
        parts.append(add)
    P = pl.concat(parts)
    df = P.group_by("s").agg(pl.col("q").sort()).with_columns(
        matched_entity_ids=pl.col("q").list.eval(int_to_id("")).list.join(","))
    o = (s1.join(df.select("s", "matched_entity_ids"), on="s", how="left").with_columns(pl.col("matched_entity_ids").fill_null(""))
           .select(source1_entity_id="s1_id", matched_entity_ids="matched_entity_ids"))
    os.makedirs(out, exist_ok=True)
    o.write_csv(os.path.join(out, "matching_results.tsv"), separator="\t", quote_style="never")
    log(f"wrote {out}: pairs US {P.filter(pl.col('country') == 'US').height}, India {P.filter(pl.col('country') == 'India').height}, "
        f"France {P.filter(pl.col('country') == 'France').height}")


if __name__ == "__main__":
    main(*sys.argv[1:])
