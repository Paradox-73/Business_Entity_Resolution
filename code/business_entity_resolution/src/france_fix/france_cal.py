"""France recalibration by change pattern, with rates learned from labelled US train (26 Sep 16:30).

Finding: for each kind of change between a record and its best S1 candidate (name: same / word swapped / word added /
dropped / other; house number: same / different / missing), the model's acceptance rate on US and India TEST equals the
true rate on labelled US/India TRAIN, but in France it is far lower (word swapped + same number: accepted 64.6% vs
91.8% true in US; same name + different number: 15.8% vs 42.6%). Descriptor swaps and number changes are normal noise
of true records in the labelled data (98-99% and 43% true). France's LB (~0.952 vs ~0.989 US/India) fits ~10% of true
France records being rejected. The model's France probabilities are too low (French address forms, department vs
region names, no French abbreviation map), but their ORDER inside a pattern is still used.

Method: per pattern, France accepts the best-candidate records whose probability is in the top X% of that pattern,
X = the pattern's true rate on labelled US train (best candidate of every record). The legal-form veto is kept
(validated on labels: US conflicts 5.2% true, India 0.0%).

  python france_cal.py check                    # same procedure on labelled US train vs the model's own decision
  python france_cal.py build <scores.parquet> <base.tsv> <out_dir> [strength=1.0]
      scores: test probabilities (q, s, p2) incl. France; base.tsv: submission whose US/India rows are kept;
      strength 1.0 = full US rates, 0.5 = halfway between France's current acceptance and the US rate.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/: shared modules
import json
import os
import re
import sys
import unicodedata
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from common import WORK, log, id_to_int, int_to_id, read_truth, macro_f05_df

STOP = set("sarl sas sasu sa eurl sci snc ei eirl selarl scp gie earl inc llc ltd corp co company pvt private limited "
           "llp plc the and of de des du la le les et d l a s e r u www com net org in fr".split())
FR_FORMS = ["sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei", "eirl", "selarl", "scp", "gie", "earl"]


def toks(x):
    x = unicodedata.normalize("NFKD", x or "").encode("ascii", "ignore").decode().lower()
    return [w for w in re.sub(r"[^a-z0-9]+", " ", x).split() if w not in STOP]


def name_kind(s, q):
    ts, tq = toks(s), toks(q)
    if not ts or not tq:
        return "empty"
    if len(tq) == 1 and len(tq[0]) > 8 and any(w in tq[0] for w in ts):
        return "squashed"
    if len(tq) == 1 and tq[0] == "".join(w[0] for w in ts):
        return "acronym"
    ms = [any(fuzz.ratio(a, b) >= 80 for b in tq) for a in ts]
    mq = [any(fuzz.ratio(b, a) >= 80 for a in ts) for b in tq]
    if all(ms) and all(mq):
        return "same"
    if not any(mq):
        return "other"
    if all(ms):
        return "added"
    if all(mq):
        return "dropped"
    return "swap"


def num(a):
    m = re.search(r"\d+", a or "")
    return (m.group(0).lstrip("0") or "0") if m else None


def pattern(sn, qn, sa, qa):
    x, y = num(sa), num(qa)
    nr = "nmiss" if x is None or y is None else ("nsame" if x == y else "ndiff")
    return name_kind(sn, qn) + "|" + nr


def attach_text(b, split):
    """b: q, s -> + sn, sa, qn, qa (only the needed rows are read)."""
    q = pl.concat([pl.scan_parquet(os.path.join(WORK, f"{split}_s{k}.parquet")).select(
        q=id_to_int("entity_id"), qn="business_name", qa="business_address").join(b.select("q").unique().lazy(), on="q").collect()
        for k in (2, 3)])
    s = pl.scan_parquet(os.path.join(WORK, f"{split}_s1.parquet")).select(
        s=id_to_int("entity_id"), sn="business_name", sa="business_address", country="country").join(
        b.select("s").unique().lazy(), on="s").collect()
    b = b.join(q, on="q").join(s, on="s")
    return b.with_columns(pat=pl.Series([pattern(*r) for r in zip(b["sn"].to_list(), b["qn"].to_list(),
                                                                   b["sa"].to_list(), b["qa"].to_list())]))


def us_rates(n=250000):
    """True rate of the best candidate per pattern, labelled US train (all records, incl. look-alikes)."""
    oof = pl.read_parquet(os.path.join(WORK, "models", "full_cons_ce", "oof.parquet"), columns=["q", "s", "p2", "label"])
    top = oof.sort("p2", descending=True).unique("q", keep="first")
    s1 = pl.read_parquet(os.path.join(WORK, "train_s1.parquet"), columns=["entity_id", "country"]).select(
        s=id_to_int("entity_id"), c="country").filter(pl.col("c") == "US")
    top = top.join(s1, on="s").sample(n=n, seed=7)
    d = attach_text(top, "train")
    r = d.group_by("pat").agg(n=pl.len(), rate=pl.col("label").mean(), acc=(pl.col("p2") >= 0.5).mean())
    return {p: (float(x), int(k)) for p, x, k in r.select("pat", "rate", "n").iter_rows()}, d


def calibrated_accept(d, prob, rates, cur=None, strength=1.0, min_n=200):
    """Per pattern keep the top X% by prob; X = strength * US rate + (1 - strength) * current acceptance."""
    keep = []
    for pat, g in d.group_by("pat"):
        pat = pat[0]
        rate = rates.get(pat, (None, 0))
        if rate[0] is None or rate[1] < min_n:            # pattern too rare in US train: keep the model's own decision
            keep.append(g.filter(pl.col(prob) >= 0.5).select("q", "s"))
            continue
        c = (g[prob] >= 0.5).mean() if cur is None else cur
        x = strength * rate[0] + (1 - strength) * c
        k = int(round(x * g.height))
        keep.append(g.sort(prob, descending=True).head(k).select("q", "s"))
    return pl.concat(keep)


def check():
    rates, d = us_rates()
    log("US train true rate per pattern (best candidate, 250k records): " +
        ", ".join(f"{p} {r:.3f} (n {k})" for p, (r, k) in sorted(rates.items(), key=lambda x: -x[1][1])[:14]))
    # procedure on a DIFFERENT labelled US sample, scored against the truth of its S1 rows
    oof = pl.read_parquet(os.path.join(WORK, "models", "full_cons_ce", "oof.parquet"), columns=["q", "s", "p2", "label"])
    s1 = pl.read_parquet(os.path.join(WORK, "train_s1.parquet"), columns=["entity_id", "country"]).select(
        s=id_to_int("entity_id"), c="country").filter(pl.col("c") == "US")
    ss = s1.filter((pl.col("s").hash(seed=9) % 20) == 0).select("s")                  # 5% of US S1 rows
    truth = read_truth().select(s=id_to_int("s1_id"), q=id_to_int("q_id")).join(ss, on="s")
    top = oof.sort("p2", descending=True).unique("q", keep="first").join(ss, on="s")
    d2 = attach_text(top, "train")
    model = d2.filter(pl.col("p2") >= 0.5).select("q", "s")
    cal = calibrated_accept(d2, "p2", rates)
    log(f"US check on {ss.height} S1 rows: model threshold 0.5 -> macro F0.5 {macro_f05_df(model, truth, ss):.5f}; "
        f"per-pattern calibration -> {macro_f05_df(cal, truth, ss):.5f} (should be close if the method is sound)")
    json.dump({p: r for p, r in rates.items()}, open(os.path.join(WORK, "us_pattern_rates.json"), "w"), indent=1)


def veto_fr(b):
    """France legal-form veto (same rule as finalize.py BER_FR_LEGAL_VETO)."""
    def forms(col):
        t = (pl.col(col).fill_null("").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()
             .str.replace_all(r"\b([a-z])\.", "$1").str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars())
        return t.str.split(" ").list.eval(pl.element().replace({"5arl": "sarl", "5as": "sas"})).list.eval(
            pl.element().filter(pl.element().is_in(FR_FORMS))).list.unique()
    b = b.with_columns(fs=forms("sn"), fq=forms("qn"))
    v = (pl.col("fs").list.len() > 0) & (pl.col("fq").list.len() > 0) & (pl.col("fs").list.set_intersection("fq").list.len() == 0)
    return b.with_columns(p2=pl.when(v).then(0.0).otherwise(pl.col("p2"))).drop("fs", "fq")


def build(scores, base_tsv, out_dir, strength=1.0):
    rates = {p: tuple(v) for p, v in json.load(open(os.path.join(WORK, "us_pattern_rates.json"))).items()}
    s1 = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=["entity_id", "country"]).select(
        s=id_to_int("entity_id"), c="country")
    b = pl.read_parquet(scores, columns=["q", "s", "p2"]).join(s1.filter(pl.col("c") == "France"), on="s")
    b = attach_text(b.drop("c"), "test")
    b = veto_fr(b)
    top = b.sort("p2", descending=True).unique("q", keep="first")
    cur = top.filter(pl.col("p2") >= 0.5)
    keep = calibrated_accept(top, "p2", rates, strength=strength)
    log(f"France best candidates {top.height}: accepted at 0.5 {cur.height}; per-pattern calibration (strength {strength}) {keep.height}")
    per = (top.join(keep.with_columns(k=pl.lit(True)), on=["q", "s"], how="left").group_by("pat")
              .agg(n=pl.len(), before=(pl.col("p2") >= 0.5).mean(), after=pl.col("k").is_not_null().mean())
              .sort("n", descending=True).head(12))
    for p, n, a, c in per.iter_rows():
        log(f"   {p:14s} n {n:7d}  accepted {a:.3f} -> {c:.3f}  (US true rate {rates.get(p, (float('nan'),))[0]:.3f})")
    base = pl.read_csv(base_tsv, separator="\t", quote_char=None, infer_schema_length=0).with_columns(
        pl.col("matched_entity_ids").fill_null(""))
    fr = keep.group_by("s").agg(pl.col("q").sort()).with_columns(
        fr_ids=pl.col("q").list.eval(int_to_id("")).list.join(","), source1_entity_id=int_to_id("s")).select("source1_entity_id", "fr_ids")
    frs = s1.filter(pl.col("c") == "France").select(source1_entity_id=int_to_id("s"), is_fr=pl.lit(True))
    out = (base.join(frs, on="source1_entity_id", how="left", maintain_order="left").join(fr, on="source1_entity_id", how="left", maintain_order="left")
               .with_columns(matched_entity_ids=pl.when(pl.col("is_fr")).then(pl.col("fr_ids").fill_null("")).otherwise(pl.col("matched_entity_ids")))
               .select("source1_entity_id", "matched_entity_ids"))
    os.makedirs(out_dir, exist_ok=True)
    out.write_csv(os.path.join(out_dir, "matching_results.tsv"), separator="\t", quote_style="never")
    json.dump({"base": base_tsv, "scores": scores, "strength": strength, "france_accepted": keep.height},
              open(os.path.join(out_dir, "result.json"), "w"), indent=1)
    log(f"wrote {out_dir}/matching_results.tsv (US/India rows from {base_tsv})")


def build_ndiff(scores, base_tsv, out_dir, strength=1.0):
    """France only, ADD-only: for best-candidate patterns with a different house number ("|ndiff"), raise acceptance
    to strength x (US train true rate) + (1 - strength) x (current rate), adding the not-yet-accepted records with the
    highest probability first. Pairs set to 0 by the legal-form veto are never added; every other France decision of
    base_tsv is kept, US/India rows are copied unchanged."""
    rates = {p: tuple(v) for p, v in json.load(open(os.path.join(WORK, "us_pattern_rates.json"))).items()}
    s1 = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=["entity_id", "country"]).select(
        s=id_to_int("entity_id"), c="country")
    frs = s1.filter(pl.col("c") == "France").select("s")
    base = pl.read_csv(base_tsv, separator="\t", quote_char=None, infer_schema_length=0).with_columns(
        pl.col("matched_entity_ids").fill_null(""))
    bpairs = (base.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
                  .filter(pl.col("matched_entity_ids") != "").select(s=id_to_int("source1_entity_id"), q=id_to_int("matched_entity_ids"))
                  .join(frs, on="s"))
    b = pl.read_parquet(scores, columns=["q", "s", "p2"]).with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64)).join(frs, on="s")
    b = veto_fr(attach_text(b, "test"))
    top = b.sort("p2", descending=True).unique("q", keep="first").filter(pl.col("p2") > 0)
    taken = bpairs.select("q").unique().with_columns(tk=pl.lit(True))
    top = top.join(taken, on="q", how="left").with_columns(acc=pl.col("tk").is_not_null()).drop("tk")
    adds = []
    for (pat,), g in top.filter(pl.col("pat").str.ends_with("|ndiff")).group_by("pat"):
        r = rates.get(pat, (None, 0))
        if r[0] is None or r[1] < 200:
            continue
        cur = g["acc"].mean()
        target = strength * r[0] + (1 - strength) * cur
        k = int(round(target * g.height)) - int(g["acc"].sum())
        if k > 0:
            add = g.filter(~pl.col("acc")).sort("p2", descending=True).head(k)
            adds.append(add.select("q", "s"))
            log(f"   {pat:14s} n {g.height:7d} accepted {cur:.3f} -> {(g['acc'].sum() + add.height) / g.height:.3f} "
                f"(US true {r[0]:.3f}); added {add.height}, their p2 from {add['p2'].min():.3f} to {add['p2'].max():.3f}")
    adds = pl.concat(adds) if adds else pl.DataFrame({"q": [], "s": []}, schema={"q": pl.Int64, "s": pl.Int64})
    newp = pl.concat([bpairs.select("q", "s"), adds.select("q", "s")]).unique()
    log(f"France: base {bpairs.height} matched records + {adds.height} number-changed records = {newp.height}")
    fr = newp.group_by("s").agg(pl.col("q").sort()).with_columns(
        fr_ids=pl.col("q").list.eval(int_to_id("")).list.join(","), source1_entity_id=int_to_id("s")).select("source1_entity_id", "fr_ids")
    isfr = frs.select(source1_entity_id=int_to_id("s"), is_fr=pl.lit(True))
    out = (base.join(isfr, on="source1_entity_id", how="left", maintain_order="left")
               .join(fr, on="source1_entity_id", how="left", maintain_order="left")
               .with_columns(matched_entity_ids=pl.when(pl.col("is_fr")).then(pl.col("fr_ids").fill_null("")).otherwise(pl.col("matched_entity_ids")))
               .select("source1_entity_id", "matched_entity_ids"))
    os.makedirs(out_dir, exist_ok=True)
    out.write_csv(os.path.join(out_dir, "matching_results.tsv"), separator="\t", quote_style="never")
    json.dump({"base": base_tsv, "scores": scores, "strength": strength, "france_added": adds.height, "mode": "ndiff add-only"},
              open(os.path.join(out_dir, "result.json"), "w"), indent=1)
    log(f"wrote {out_dir}/matching_results.tsv")


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check()
    elif sys.argv[1] == "ndiff":
        build_ndiff(sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]) if len(sys.argv) > 5 else 1.0)
    else:
        build(sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]) if len(sys.argv) > 5 else 1.0)
