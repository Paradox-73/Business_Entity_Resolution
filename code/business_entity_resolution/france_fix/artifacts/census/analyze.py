import sys, itertools, collections
import polars as pl
pl.Config.set_tbl_rows(300); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(120)
d = pl.read_parquet('ops_all.parquet')
what = sys.argv[1]
if what == 'nops':
    # ops per record, excluding pure formatting ops listed as FMT
    d = d.with_columns(k=pl.col('ops').list.len())
    print(d.group_by('grp','country').agg(mean=pl.col('k').mean(), k0=(pl.col('k')==0).mean(), k1=(pl.col('k')==1).mean(),
          k2=(pl.col('k')==2).mean(), k3=(pl.col('k')==3).mean(), k4p=(pl.col('k')>=4).mean()).sort('country','grp'))
if what == 'pairs':
    c = sys.argv[2]
    x = d.filter(pl.col('country')==c)
    res = []
    for g in ['T','D']:
        y = x.filter(pl.col('grp')==g)['ops'].to_list()
        N = len(y)
        single = collections.Counter(o for s in y for o in s)
        pair = collections.Counter(p for s in y for p in itertools.combinations(sorted(s), 2))
        res.append((N, single, pair))
    (NT, sT, pT), (ND, sD, pD) = res
    rows = []
    keys = set(pT) | set(pD)
    for a, b in keys:
        eT = sT[a]*sT[b]/NT; eD = sD[a]*sD[b]/ND
        rows.append(dict(a=a, b=b, T=pT[(a,b)], eT=round(eT,1), D=pD[(a,b)], eD=round(eD,1),
                         rT=round((pT[(a,b)]+1)/(eT+1),3), rD=round((pD[(a,b)]+1)/(eD+1),3)))
    r = pl.DataFrame(rows)
    print('--- pairs that co-occur far LESS than chance on TRUE records (expected >= 50)')
    print(r.filter(pl.col('eT')>=50).sort('rT').head(60))
    print('--- pairs that co-occur far MORE than chance on TRUE records')
    print(r.filter(pl.col('T')>=100).sort('rT', descending=True).head(30))
