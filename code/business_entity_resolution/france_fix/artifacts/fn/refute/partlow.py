# Mechanism check: are true word-changed records lowercased BEFORE the word change (retained words lowercase, new word not)?
import polars as pl, re, unicodedata
pl.Config.set_tbl_rows(120); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(60); pl.Config.set_float_precision(4)
def sa(x): return unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode()
def case_kind(qn, sn):
    """full: record all lowercase; part: all record tokens also in S1 are lowercase (and S1 had capitals there) but another token has capitals;
       upfull / uppart likewise for uppercase; none otherwise"""
    if not qn or not sn: return 'na'
    if not re.search(r'[A-Za-z]', qn): return 'na'
    if qn == qn.lower() and sn != sn.lower(): return 'full'
    if qn == qn.upper() and sn != sn.upper(): return 'upfull'
    st = {sa(w).lower() for w in re.findall(r'[^\W\d_]+', sn)}
    sraw = {w for w in re.findall(r'[^\W\d_]+', sn)}
    toks = re.findall(r'[^\W\d_]+', qn)
    ret = [w for w in toks if sa(w).lower() in st and len(w) > 1]
    new = [w for w in toks if sa(w).lower() not in st and len(w) > 1]
    if ret and new and all(w == w.lower() for w in ret) and any(w != w.lower() for w in new) and not all(w in sraw for w in ret):
        return 'part'
    if ret and new and all(w == w.upper() for w in ret) and any(w != w.upper() for w in new) and not all(w in sraw for w in ret):
        return 'uppart'
    return 'none'
if __name__ == '__main__':
    u = pl.read_parquet('usi_keys.parquet', columns=['key', 'country', 'T', 'qn', 'sn', 'p'])
    keys = ['SWAP|n_swap:noise|NSAME', 'ADD|n_add:noise|NSAME', 'TYPO||NSAME', 'SWAP|n_swap:other|NSAME', 'SWAP|n_swap:desc|NSAME', 'SWAP|n_swap:noise|NMISS', 'ADD|n_add:noise|NMISS']
    x = u.filter(pl.col('key').is_in(keys))
    x = x.with_columns(ck=pl.Series([case_kind(a, b) for a, b in zip(x['qn'].to_list(), x['sn'].to_list())]))
    g = x.group_by('key', 'country', 'T').agg(n=pl.len(), full=(pl.col('ck') == 'full').mean(), part=(pl.col('ck') == 'part').mean(),
                                            upfull=(pl.col('ck') == 'upfull').mean(), uppart=(pl.col('ck') == 'uppart').mean())
    print(g.sort('key', 'country', 'T'))
    print(x.filter((pl.col('ck') == 'part') & pl.col('T')).head(12).select('key', 'qn', 'sn'))
    print(x.filter((pl.col('ck') == 'full') & ~pl.col('T')).head(8).select('key', 'qn', 'sn'))
