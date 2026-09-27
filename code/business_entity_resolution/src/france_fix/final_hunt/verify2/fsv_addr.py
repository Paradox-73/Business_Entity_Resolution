import sys, re, unicodedata, polars as pl
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
from common import WORK  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
from rapidfuzz import fuzz
from common import id_to_int
W = f"{WORK}/"
P = f'{SCRATCH}/final2/fr-strong-veto/'
STOP = set("rue r av ave avenue bd blvd boulevard ch chemin imp impasse all allee pl place cours crs quai route rte de du des la le les l d et no n bis ter apt residence".split())
def norm(a):
    a = unicodedata.normalize('NFKD', a or ''); a = ''.join(ch for ch in a if not unicodedata.combining(ch)).lower(); return a
def street(a):
    a = norm(a)
    for p in a.split(','):
        m = re.search(r'(\d+)', p)
        if m:
            toks = [t for t in re.findall(r"[a-z]{2,}", p) if t not in STOP]
            return int(m.group(1)), ' '.join(toks)
    return None, ''
v = pl.read_parquet(P+'fr_strong_veto_street.parquet')
k = pl.read_parquet(P+'fr_prof.parquet').filter(pl.col('grp')=='keep_cc')
exec(open(P+'s17_street.py', encoding='utf8').read().split("r = pl.read_parquet")[0].split("def words")[0])
src = open(P+'s17_street.py', encoding='utf8').read()
exec('def words'+src.split('def words')[1].split('r = pl.read_parquet')[0])
k = k.with_columns(rep=pl.Series([repl(a,b) for a,b in zip(k['sa'].to_list(), k['qa'].to_list())], dtype=pl.Boolean)).filter(pl.col('rep').fill_null(False))
s1 = pl.scan_parquet(W+"test_s1.parquet").filter(pl.col("country")=="France").select(s=id_to_int("entity_id"), n="business_name", a="business_address").collect()
rows = []
for s, n, a in s1.iter_rows():
    num, st = street(a)
    if num is not None and st: rows.append((s, num, st, n))
idx = {}
for s, num, st, n in rows: idx.setdefault(num, []).append((s, st, n))
def probe(d, name):
    hit = []; hitn = []
    for s, qa in zip(d['s'].to_list(), d['qa'].to_list()):
        num, st = street(qa)
        best = (0, None, None)
        if num is not None and st:
            for s2, st2, n2 in idx.get(num, []):
                if s2 == s: continue
                r = fuzz.token_set_ratio(st, st2)
                if r > best[0]: best = (r, s2, n2)
        hit.append(best[0]); hitn.append(best[2])
    d = d.with_columns(other_s1_addr_sim=pl.Series(hit), other_s1_name=pl.Series(hitn, dtype=pl.Utf8))
    for t in (80, 90):
        print(name, d.height, f'record address = another France S1 address (same number, street sim>={t}):', (d['other_s1_addr_sim']>=t).sum(), round((d['other_s1_addr_sim']>=t).mean(),3))
    return d
vv = probe(v, 'VETO')
kk = probe(k.sample(min(837, k.height), seed=1), 'KEEP_cc street-replaced (control)')
vv.write_parquet('fsv_veto_addr.parquet'); kk.write_parquet('fsv_keep_addr.parquet')
pl.Config.set_tbl_rows(30); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(40)
print(vv.filter(pl.col('other_s1_addr_sim')>=90).select('sn','sa','qn','qa','other_s1_name','other_s1_addr_sim').head(25))
