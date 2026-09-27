import polars as pl, hashlib
A = "E:/Projects/Amazon ML Challenge/work/frfix/test_scores_frmin_descveto.parquet"
B = "C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix/namechg/test_scores_frmin_descveto.parquet"
h = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
print("same file:", h(A) == h(B))
print(pl.read_parquet_schema(B))
FR = "C:/Users/kanav/.claude/jobs/ef6f152d/tmp/france"
t = pl.read_parquet(f"{FR}/fr_top.parquet"); print(t.columns, t.height)
u = pl.read_parquet(f"{FR}/usi_top.parquet"); print(u.columns, u.height)
