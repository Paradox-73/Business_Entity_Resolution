"""Candidate slices: noise-only word change; rename (no shared core word); by pg band. France committee vs US/India analog vs France refs."""
import os, re
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
T = rf"{SCRATCH}/final/fr-committee"
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(220); pl.Config.set_fmt_str_lengths(40)
STOP = set("sarl sas sasu sa eurl sci snc ei eirl selarl scp gie earl llc inc ltd pvt private limited co corp company the de la le les du des et and & of".split())


def core(x):
    import unicodedata
    x = unicodedata.normalize("NFKD", x or "").encode("ascii", "ignore").decode().lower()
    return {w for w in re.split(r"[^a-z0-9]+", x) if w and w not in STOP}


def add_cols(d):
    return d.with_columns(
        slice=pl.when(pl.col("ops").str.contains(r"n_(swap|add|drop):(other|desc|title|loc|legal)")).then(pl.lit("word_nonnoise"))
        .when(pl.col("ops").str.contains(r"n_(swap|add|drop):noise")).then(pl.lit("noise_only"))
        .otherwise(pl.lit("other")),
        share=pl.Series([len(core(a) & core(b)) > 0 for a, b in zip(d["qn"].to_list(), d["sn"].to_list())]),
        up=pl.col("ops").str.contains("a_num_up"), down=pl.col("ops").str.contains("a_num_down"),
        lowok=~pl.col("ops").str.contains("n_domain|n_squash|n_acronym"))


fr = add_cols(pl.read_parquet(os.path.join(T, "cand2.parquet")).filter(pl.col("cls").is_in(["word_other", "noword"])))
tr = add_cols(pl.read_parquet(os.path.join(T, "train_committee.parquet")).filter((pl.col("kind") == "add") & pl.col("cls").is_in(["word_other", "noword"])))
cal = add_cols(pl.read_parquet(os.path.join(T, "calib.parquet")).filter(pl.col("cls").is_in(["word_other", "noword"])))
st = lambda d, k: d.group_by(k).agg(n=pl.len(), low=pl.col("low").filter(pl.col("lowok")).mean().round(4), up=pl.col("up").mean().round(3), dn=pl.col("down").mean().round(3)).sort(k)
print("FRANCE committee"); print(st(fr, ["cls", "slice", "share"]))
print("FRANCE refs"); print(st(cal, ["src", "cls", "slice", "share"]))
print("TRAIN analog"); print(tr.group_by("cls", "slice", "share").agg(n=pl.len(), prec=pl.col("y").mean().round(3), low=pl.col("low").filter(pl.col("lowok")).mean().round(4)).sort("cls", "slice", "share"))
x = tr.filter(pl.col("cls") == "word_other").with_columns(pgb=pl.col("pg").cut([0.9, 0.95, 0.99]))
print(x.group_by("slice", "share", "pgb").agg(n=pl.len(), prec=pl.col("y").mean().round(3)).sort("slice", "share", "pgb"))
print(fr.filter(pl.col("slice") == "noise_only").sample(25, seed=2).select("qn", "sn", "qa", "sa", "po", "pg"))
fr.write_parquet(os.path.join(T, "cand3.parquet"))
