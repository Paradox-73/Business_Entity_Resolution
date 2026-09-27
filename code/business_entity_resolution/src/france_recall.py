"""Recall for the countries without training labels (common.unlabelled_countries; France in this test set) from a
second candidate generator: add its matches in those countries that our candidate lists never contained, for records the
base file leaves unmatched, when the record differs from the S1 row only by generator noise.

Kept: no word-level name change (a swapped/added/dropped word is how France look-alikes are made), no legal-form change
or addition, house number not moved up (the look-alike direction), and the generator's probability >= 0.8.

Why (27 Sep 18:40): France rows come from our old candidate lists. Gathik v9's wider search (name max_df 20000, dense e5
top-10) finds 7,266 France matches for records v10a leaves unmatched: acronyms ('PC' = 'Paranormal Club'), web domains
('nantesclubsas.com'), typos, word order. US/India analog on labels (eval half, `france_fix/recall/train_analog.py`): Gathik's
accepted pairs outside our lists for records we leave unmatched are 98.4% true (28,113 pairs), 98.8% under this filter,
and each adds +0.098 / (S1 rows) macro F0.5. France checks without labels: none of the kept non-domain record names is
all-lowercase (look-alikes 2-4%, true records 0.2-0.4%); France rows are under-matched (6.0% empty vs 5.7% US/India,
3.29 vs 3.39 matches per S1).

  python france_recall.py <base matching_results.tsv> <gathik matching_results.tsv> <gathik test scores.parquet> <out_dir>
"""
import os
import sys
import polars as pl
from common import WORK, id_to_int, int_to_id, is_unlabelled, log, read_tsv

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "france_fix", "artifacts", "census"))
from ops import ops  # noqa: E402  (called without its country argument, so its output does not depend on the country)

REJECT = r"n_swap:|n_add:|n_drop:|n_legal_change|n_legal_add|a_num_up"
OUR_LISTS = ["test_scores_full_cons.parquet", os.path.join("ce_x", "test_rows.parquet")]


def pairs(p):
    """(s, q) integer pairs of a matching_results.tsv."""
    return (read_tsv(p).with_columns(q_id=pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("q_id")
            .filter(pl.col("q_id") != "").select(s=id_to_int("source1_entity_id").cast(pl.Int64), q=id_to_int("q_id").cast(pl.Int64)))


def main(base_tsv, g_tsv, g_scores, out):
    """Add to base_tsv the pairs of the second pipeline's matching file in countries without training labels that are
    outside our candidate lists, for records base_tsv leaves unmatched, when the change detector (ops.py) finds no word-level or legal-form
    change and no house number moved up, and the second pipeline's p2 >= 0.8. Writes <out>/matching_results.tsv and
    <out>/france_recall_added.parquet."""
    s1 = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=["entity_id", "country", "business_name", "business_address"]).select(
        s1_id="entity_id", s=id_to_int("entity_id").cast(pl.Int64), country="country", sn="business_name", sa="business_address")
    rec = pl.concat([pl.read_parquet(os.path.join(WORK, f"test_s{k}.parquet"), columns=["entity_id", "business_name", "business_address"])
                     for k in (2, 3)]).select(q=id_to_int("entity_id").cast(pl.Int64), qn="business_name", qa="business_address")
    lists = pl.concat([pl.read_parquet(os.path.join(WORK, f), columns=["q", "s"]).with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
                       for f in OUR_LISTS]).unique()
    base = pairs(base_tsv)
    fr = s1.filter(is_unlabelled()).select("s")
    g = pairs(g_tsv).join(fr, on="s")
    gp = pl.read_parquet(g_scores, columns=["q", "s", "p2"]).with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
    cand = g.join(lists, on=["q", "s"], how="anti").join(base.select("q"), on="q", how="anti").join(gp, on=["q", "s"], how="left")
    cand = cand.join(s1.select("s", "sn", "sa"), on="s").join(rec, on="q")
    cand = cand.with_columns(ops=pl.Series([";".join(sorted(ops(r["qn"] or "", r["qa"] or "", r["sn"] or "", r["sa"] or "")))
                                            for r in cand.iter_rows(named=True)], dtype=pl.Utf8))
    add = cand.filter(~pl.col("ops").str.contains(REJECT) & (pl.col("p2").fill_null(0) >= 0.8))
    log(f"pairs of the second generator in countries without training labels, outside our lists, record unmatched: "
        f"{cand.height}; kept {add.height} "
        f"(S1 rows {add['s'].n_unique()}, of them empty in the base file {add.join(base.select('s').unique(), on='s', how='anti')['s'].n_unique()})")
    P = pl.concat([base, add.select("s", "q")])
    df = P.group_by("s").agg(pl.col("q").sort()).with_columns(matched_entity_ids=pl.col("q").list.eval(int_to_id("")).list.join(","))
    o = (s1.join(df.select("s", "matched_entity_ids"), on="s", how="left").with_columns(pl.col("matched_entity_ids").fill_null(""))
           .select(source1_entity_id="s1_id", matched_entity_ids="matched_entity_ids"))
    os.makedirs(out, exist_ok=True)
    o.write_csv(os.path.join(out, "matching_results.tsv"), separator="\t", quote_style="never")
    add.select("s", "q", "p2", "ops").write_parquet(os.path.join(out, "france_recall_added.parquet"))
    log(f"wrote {out}/matching_results.tsv: {P.height} pairs (base {base.height} + {add.height})")


if __name__ == "__main__":
    main(*sys.argv[1:])
