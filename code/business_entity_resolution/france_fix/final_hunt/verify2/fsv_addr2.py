import sys, re, unicodedata, polars as pl
from rapidfuzz import fuzz
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
from common import id_to_int
W = "E:/Projects/Amazon ML Challenge/work/"
STOP = set("rue r av ave avenue bd blvd boulevard ch chemin imp impasse all allee pl place cours crs quai route rte de du des la le les l d et no n bis ter apt residence".split())
REG = {'hauts de france','nouvelle aquitaine','pays de la loire','nord','gironde','loire atlantique','pas de calais','nord pas de calais','france'}
def norm(a):
    a = unicodedata.normalize('NFKD', a or ''); a = ''.join(ch for ch in a if not unicodedata.combining(ch)).lower(); return a
def parse(a):
    a = norm(a); num=None; st=''; city=set()
    for p in a.split(','):
        m = re.search(r'(\d+)', p)
        if m and num is None:
            num=int(m.group(1)); st=' '.join(t for t in re.findall(r"[a-z]{2,}", p) if t not in STOP)
        else:
            c = ' '.join(re.findall(r"[a-z]+", p.replace('saint','st')))
            if c and c not in REG: city.add(c)
    return num, st, city
nm = lambda x: ' '.join(re.findall(r'[a-z0-9]+', norm(x)))
LEG = {'sarl','sas','sasu','sa','eurl','ei','sci','sarl','sàrl'}
core = lambda x: ' '.join(t for t in nm(x).split() if t not in LEG and t!='france')
s1 = pl.scan_parquet(W+"test_s1.parquet").filter(pl.col("country")=="France").select(s=id_to_int("entity_id"), n="business_name", a="business_address").collect()
idx = {}
for s, n, a in s1.iter_rows():
    num, st, city = parse(a)
    if num is not None and st: idx.setdefault(num, []).append((s, st, city, n))
def probe(d, name):
    out=[]
    for s, sa, qn, qa in zip(d['s'].to_list(), d['sa'].to_list(), d['qn'].to_list(), d['qa'].to_list()):
        num, st, city = parse(qa); _, _, scity = parse(sa)
        cty = city | scity
        best=(0,None)
        if num is not None and st:
            for s2, st2, c2, n2 in idx.get(num, []):
                if s2==s or not (c2 & cty): continue
                r = fuzz.ratio(st, st2)
                if r>best[0]: best=(r,n2)
        rel = None
        if best[0]>=80:
            a, b = core(qn), core(best[1])
            rel = 'same_name' if a==b else ('share_first_tok' if a.split()[:1]==b.split()[:1] else 'other')
        out.append((best[0], best[1], rel))
    d = d.with_columns(o_sim=pl.Series([o[0] for o in out]), o_name=pl.Series([o[1] for o in out], dtype=pl.Utf8), o_rel=pl.Series([o[2] for o in out], dtype=pl.Utf8))
    h = d.filter(pl.col('o_sim')>=80)
    print(name, d.height, 'record addr = another same-city S1 addr (same no., street ratio>=80):', h.height, round(h.height/d.height,3), h['o_rel'].value_counts().sort('o_rel').rows())
    return d
v = pl.read_parquet('C:/ber_scratch/final2/fr-strong-veto/fr_strong_veto_street.parquet')
kk = pl.read_parquet('fsv_keep_addr.parquet')
vv = probe(v, 'VETO'); kc = probe(kk, 'KEEP control')
vv.write_parquet('fsv_veto_addr2.parquet')
pl.Config.set_tbl_rows(30); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(40)
print(vv.filter(pl.col('o_sim')>=80).select('sn','qn','qa','o_name','o_sim','o_rel').head(20))
print(kc.filter(pl.col('o_sim')>=80).select('sn','qn','qa','o_name','o_sim','o_rel').head(10))
