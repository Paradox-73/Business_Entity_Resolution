# Local cleanup, 28 Sep 2026

After the final package was built, validated and pushed, 91.0 GB of intermediate files under `work/` were deleted to free disk space. Before deleting, they were moved aside and `src/build_final.py` was run from an empty build folder: all five md5 values still matched the uploaded files ("ALL IDENTICAL"), so nothing the exact rebuild needs was removed.

Kept under `work/` (11.5 GB): the cleaned test files, the score files of the final chain (GBDT stage 2, the three transformer families of the final file, France step-10 scores), the second pipeline's files, the final GBDT models (`models/full`, `models/full_cons`), the locally trained transformer weights (`ce/model`, `model_b`, `model_a2`, `model_smallf0-2`, `e5_ft_addr`), all logs and json notes. The bge-reranker and e5-large fold weights stay on the teammates' GPU servers.

| Deleted | Items | GB | How to rebuild |
|---|---:|---:|---|
| pairs/ (pair feature files: train full, test, wide test, tl2) | 1 | 66.5 | `pipeline.py build train full`, `pipeline.py build test test`, `blocking_b2.py build` (README steps 3, 5, 13; 2-4 h each) |
| out_*/ (build folders of earlier versions; the uploaded and saved files are in submissions/) | 52 | 6.2 | rerun the version's build (submissions/*/finalize.json); final files: `build_final.py` |
| emb/ (e5 embeddings of S1 rows and records) | 1 | 4.0 | `embed.py encode` (README step 2) |
| frfix*/ France analysis intermediates (the final sets ship in sets/) | 36 | 3.8 | scripts in `src/france_fix/` (sets/README.md names each) |
| ce/ files of versions and transformer families not in the final file, training-resume snapshots, partial score chunks | 20 | 3.6 | `rerank.py score` / `blend.py` (README steps 8-15); resume snapshots are not needed |
| test_scores_* / test_base_* of earlier versions | 16 | 2.7 | `blend.py` / `pipeline.py predict` from the kept score files (README steps 5-15) |
| ce_x/ files of versions and transformer families not in the final file, training-resume snapshots, partial score chunks | 26 | 1.4 | `ce_folds.py` runs of the unused families (EXPERIMENTS.md) |
| train_s{1,2,3}.parquet (cleaned train files) | 3 | 1.1 | `learn_maps.py` + `prep.py` (README step 1, minutes) |
| models/ of experiments not in the final file (full_sib, full_sib_all, full_cons_ce, s10, testlike ...) | 3 | 0.6 | `pipeline.py train ...` (EXPERIMENTS.md names each run) |
| other (friend-machine bundles f2/, testfr_*, tl2 data, older runs) | 43 | 0.6 | not needed for the final files |
| e5_ft/ (older name-only embedding model, superseded by e5_ft_addr) | 1 | 0.5 | not needed (superseded) |
| ce_b2/ files of versions and transformer families not in the final file, training-resume snapshots, partial score chunks | 3 | 0.0 | `rerank.py score` with `BER_CE_REUSE` (README step 14) |

`submissions/`: the matching and candidate TSVs of versions other than v7p, v7q, v9y and v10a-v10d were compressed with gzip (`<file>.tsv.gz`; read with `pl.read_csv(path, separator='	')` after `gzip -d`, or pandas `compression='gzip'`). v7p, v7q and v9y stay plain because `src/movers/build_movers.py` reads them; v10a-v10d because they are the final chain.

## Other space savings

- `submissions/`: 47 TSV files of versions other than v7p, v7q, v9y and v10a-v10d gzip-compressed (4.57 GB -> 1.97 GB); every file's decompressed md5 was checked against the original before the original was removed.
- `2026_official/6ab10eb3b23ba_student_resource.zip` (1.1 GB, the original dataset download) deleted after checking by CRC that all 10 files in it are identical to the unpacked `student_resource/` (the only entries not unpacked were two macOS `.DS_Store` files).
- `output/matching_results.tsv` and `output/candidate_pairs.tsv` are hard links to `submissions/v10d/` (same md5), so the bytes are stored once.

Result: the project folder went from about 100 GB to about 17 GB (`work/` 11 GB, `student_resource/dataset` 2.4 GB, `submissions/` 3.0 GB, the rest under 0.5 GB).
