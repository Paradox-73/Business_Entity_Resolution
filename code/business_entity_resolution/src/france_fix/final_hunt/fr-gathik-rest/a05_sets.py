"""Build the two France add-sets from the 2,441 rest pairs; lowercase bound; expected LB change per set."""
import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
from common import log

T = rf"{SCRATCH}/final/fr-gathik-rest"
d = pl.read_parquet(os.path.join(T, "rest_cls2.parquet"))
N_FR = 259452
base = (pl.col("p2") >= 0.9) & ~pl.col("legalbad") & ~pl.col("numup") & (pl.col("n_twin") == 0) & (pl.col("n_rec_twin") == 0)
sets = {
    "fr_noise_in": d.filter(base & (pl.col("cls2") == "N_noise_in")),
    "fr_typo_in": d.filter(base & pl.col("cls2").is_in(["G_typo_own", "G_garble_other", "D_drop_stopword_only"])),
}
# reference classes (not proposed) for the lowercase contrast
refs = {"X_desc_in": d.filter(pl.col("cls2") == "X_desc_in"), "R_realword_in": d.filter(pl.col("cls2") == "R_realword_in")}
DECOY_LOW = 0.0338   # France records with a pure decoy word (docs/METHODOLOGY.md)


def row_gain(m, k, t):
    """expected per-row F0.5 change when k pairs (each true w.p. t) are added to a row with m matches (assumed true)."""
    def F(tp, np_, nt):
        if nt == 0:
            return 1.0 if np_ == 0 else 0.0
        if tp == 0:
            return 0.0
        p, r = tp / np_, tp / nt
        return 1.25 * p * r / (0.25 * p + r)
    exp = 0.0
    for j in range(k + 1):
        pr = math.comb(k, j) * t ** j * (1 - t) ** (k - j)
        exp += pr * (F(m + j, m + k, m + j) - F(m, m, m + j))
    return exp


pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(40); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(12)
for name, x in {**sets, **refs}.items():
    ok = x.filter(~pl.col("handle") & pl.col("lowtest_ok"))
    nlow, nok = int(ok["low"].sum()), ok.height
    up95 = (3.0 if nlow == 0 else nlow + 2.0 * math.sqrt(nlow) + 1) / max(nok, 1)
    t_pt = max(0.0, min(1.0, 1 - (nlow / max(nok, 1)) / DECOY_LOW))
    t_lo = max(0.0, 1 - up95 / DECOY_LOW)
    log(f"{name}: pairs {x.height}, S1 rows {x['s'].n_unique()} (empty in v10b {x.filter(pl.col('n_v10b') == 0)['s'].n_unique()}); "
        f"lowercase {nlow}/{nok} -> true share point {t_pt:.3f}, 95% low {t_lo:.3f}; num up {int(x['numup'].sum())}, "
        f"down {int(x['opsF'].str.contains('a_num_down').sum())}; gathik p mean {x['p2'].mean():.3f}")
    if name in sets:
        g = x.group_by("s").agg(k=pl.len(), m=pl.col("n_v10b").first())
        for t in (0.85, 0.90, 0.95, 0.99):
            tot = sum(row_gain(m, k, t) for m, k in g.select("m", "k").iter_rows())
            log(f"   t={t:.2f}: sum dF {tot:+.1f} -> France {tot / N_FR:+.5f} -> LB {0.15 * tot / N_FR:+.6f}")
        be = [t / 100 for t in range(50, 100) if sum(row_gain(m, k, t / 100) for m, k in g.select("m", "k").iter_rows()) > 0]
        log(f"   break-even true share ~{be[0] if be else None}")
        log("   top S1-side -> record-side word changes:\n" + str(x.group_by("w_s", "w_q").len().sort("len", descending=True).head(15)))
        log("   by S1-side word class:\n" + str(x.with_columns(sw=pl.col("opsF").str.extract_all(r"n_(?:swap|drop):[a-z>]+").list.join("+"))
                                               .group_by("sw").len().sort("len", descending=True).head(8)))
        x.select("s", "q", "p2", "pgx", "cls2", "opsF", "w_s", "w_q", "n_v10b", "low", "sn", "qn", "sa", "qa").write_parquet(
            os.path.join(T, f"{name}.parquet"))
