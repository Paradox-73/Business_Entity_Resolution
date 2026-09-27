"""Apply pair sets to a submission: remove the pairs of 'veto' sets, then add the pairs of 'add' sets.

Each set is a parquet file with integer columns s (S1 id) and q (record id). An added record that the file already
matches elsewhere is moved (its old pair is removed), so every record keeps at most one S1 row. Checks: added pairs join
S1 rows and records of the same country; ids exist in the test files.

  python apply_pair_sets.py <base matching_results.tsv> <out_dir> [+add.parquet ...] [-veto.parquet ...]
e.g. python apply_pair_sets.py submissions/v10b/matching_results.tsv work/out_final +adds_fr.parquet -veto_fr.parquet
"""
import os
import sys
import polars as pl
from common import WORK, id_to_int, int_to_id, log, read_tsv


def pairs(p):
    return (read_tsv(p).with_columns(q_id=pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("q_id")
            .filter(pl.col("q_id") != "").select(s=id_to_int("source1_entity_id").cast(pl.Int64), q=id_to_int("q_id").cast(pl.Int64)))


def main(base_tsv, out, *sets):
    s1 = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=["entity_id", "country"]).select(
        s1_id="entity_id", s=id_to_int("entity_id").cast(pl.Int64), country="country")
    rec = pl.concat([pl.read_parquet(os.path.join(WORK, f"test_s{k}.parquet"), columns=["entity_id", "country"]) for k in (2, 3)]).select(
        q=id_to_int("entity_id").cast(pl.Int64), qc="country")
    P = pairs(base_tsv)
    log(f"base {base_tsv}: {P.height} pairs")
    for spec in sets:
        kind, path = spec[0], spec[1:]
        d = pl.read_parquet(path, columns=["s", "q"]).with_columns(pl.col("s").cast(pl.Int64), pl.col("q").cast(pl.Int64)).unique()
        if kind == "-":
            n0 = P.height
            P = P.join(d, on=["s", "q"], how="anti")
            log(f"veto {path}: {d.height} pairs, {n0 - P.height} removed")
        elif kind == "+":
            chk = d.join(s1.select("s", "country"), on="s", how="left").join(rec, on="q", how="left")
            bad = chk.filter(pl.col("country").is_null() | pl.col("qc").is_null() | (pl.col("country") != pl.col("qc")))
            assert bad.height == 0, f"{path}: {bad.height} pairs with unknown ids or a country mismatch"
            assert d["q"].n_unique() == d.height, f"{path}: a record is added to two S1 rows"
            new = d.join(P, on=["s", "q"], how="anti")
            moved = P.join(new.select("q"), on="q").height
            P = pl.concat([P.join(new.select("q"), on="q", how="anti"), new])
            log(f"add {path}: {d.height} pairs, {new.height} new ({moved} records moved from another S1 row)")
        else:
            raise SystemExit(f"prefix each set with + (add) or - (veto): {spec}")
    assert P["q"].n_unique() == P.height, "a record is matched to two S1 rows"
    df = P.group_by("s").agg(pl.col("q").sort()).with_columns(matched_entity_ids=pl.col("q").list.eval(int_to_id("")).list.join(","))
    o = (s1.join(df.select("s", "matched_entity_ids"), on="s", how="left").with_columns(pl.col("matched_entity_ids").fill_null(""))
           .select(source1_entity_id="s1_id", matched_entity_ids="matched_entity_ids"))
    os.makedirs(out, exist_ok=True)
    o.write_csv(os.path.join(out, "matching_results.tsv"), separator="\t", quote_style="never")
    log(f"wrote {out}/matching_results.tsv: {P.height} pairs, {(o['matched_entity_ids'] != '').sum()} S1 rows non-empty")


if __name__ == "__main__":
    main(*sys.argv[1:])
