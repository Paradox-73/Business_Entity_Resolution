# Pair sets of the final submission

This folder holds every fixed (record, S1 row) pair set and the stacker model file that `src/build_final.py` applies on
top of the saved score files to rebuild the final submission (v10d, public leaderboard 0.990565). A pair set is a parquet
table with integer columns `s` (Source 1 id) and `q` (Source 2/3 record id); the other columns are the analysis values
the defining script saved with each pair.

- The parquet and pkl files are not in git (the repository ignores `*.parquet` and `*.pkl`). They ship in the
  submission zip next to this file.
- Row counts, columns and md5 below were measured on the files in this folder on 27 Sep 2026. Each file is a byte copy
  of the file the build used then (source path in the last table).
- Script paths are relative to `src/`. "One-off command" means the set was written by a command typed during the run,
  not by a script in this repository; the rule it applied is stated.

## Files

| File | Rows | md5 | What it is | Derived by | Added in |
|---|---|---|---|---|---|
| `desc_veto_set.parquet` | 22,436 | `0e65b2e78ba95252695cd81f99aeb492` | France pairs the model accepted whose record adds or swaps in a descriptor word (amicale, comite, ecole, club, centre, ...); build sets p2 = 0 | `france_fix/namechg/a1_build.py` ... `a11_veto.py` | v9a (kept through v9y) |
| `noise_add_set.parquet` | 9,800 | `e8aad4b0cd80d661bfcd8c2dd00e263b` | Rejected France best-candidate pairs with the same house number and street whose added or swapped word is a noise word (et fils, & associes, cie, services, ...) or groupe/developpement/france; build sets p2 = max(p2, 0.95) | `france_fix/build/b4_addset.py`, `france_fix/build/b8_mkscores_v2.py` | v9b |
| `fp_veto_set.parquet` | 1,145 | `b5d41271195aa926b164995ccbae0a28` | Accepted France pairs with a look-alike signature (column `tier`: A legal form added and house number up by 1-20; B word changed and number up by 1-20; C all-lowercase record with a name change); build sets p2 = 0 | `france_fix/artifacts/fp/fp1_dec.py` ... `fp10_veto.py` | v9e |
| `recall_add_set.parquet` | 5,883 | `4290d065039dcded471e3202f4578913` | Missed France matches: single-change copies, acronym / web-domain / dotted-legal-form names at the same address, exact-name twins at the same address (column `grp`, estimated true share `est`); build sets p2 = 0.95 (2 pairs absent from the score table are appended) | `france_fix/v9z/recall/r1_base.py` ... `r5_build.py` | v9z |
| `moreveto_stem2_set.parquet` | 73 | `36892522bd956551c00896991eb861ae` | Accepted France pairs whose record swaps a descriptor word for a different word with the same 5-letter stem (sport -> sportive); build sets p2 = 0 | `france_fix/v9z/moreveto/m1_acc.py` ... `m10_veto_stem.py` (71 pairs) + 2 pairs listed in `france_fix/v9z/build/b1_scores.py` | v9z |
| `tier12_removed.parquet` | 839 | `44e0c2ffd59c4e1d506c7e4e16cdcab8` | US/India pairs v9y matches that only the wide-candidate build accepted (the old build rejected them), where the list-independent transformer probability is below 0.5 (tier 1: wide build did not rescore the pair) or 0.2 (tier 2: both builds rescored it); build sets p2 = 0 | `movers/build_movers.py v7p v7q v9y <dir>` | v9zm |
| `tier12_restored.parquet` | 832 | `dd2f090642a60861adcfeaf6108a8edd` | US/India pairs only the old-candidate build accepted, for records v9y leaves unmatched, where that probability is at least 0.8 (tier 1) or 0.95 (tier 2); build sets p2 = max(p2, 0.95) (0.95 for the 29 pairs neither pipeline scored) | `movers/build_movers.py v7p v7q v9y <dir>` | v9zm |
| `fr_typo_safe.parquet` | 416 | `1c82f7d76b9f3ff85ad217cced772a97` | France matches of the second pipeline whose record garbles the S1 word (typo sharing its letters), for records v10b leaves unmatched; added | `france_fix/final_hunt/fr-gathik-rest/a01_rest.py` ... `a05_sets.py`, then `france_fix/final_hunt/verify/typo_v01_impl.py` ... `typo_v07_safe.py` (`typo_v07_safe.py` writes it) | v10c |
| `fr_same_address_safe.parquet` | 298 | `d4cdb4d5f9ff31de933c124dc4f5cddc` | France pairs from the same-address rule (same house number and street words, name differs by acronym or noise, one S1 at the address), verified subset; added. Also an extra candidate source for `candidate_pairs.tsv` | `france_fix/final_hunt/same-address/test_apply.py`, `make_sets.py`, then `france_fix/final_hunt/verify/v1.py` ... `v8.py` (`v8.py` writes it). The pipeline module `same_address.py` regenerates exactly these 298 pairs from the cleaned test files and v10b; `build_final.py` step 6 checks it | v10c |
| `fr_amp_safe.parquet` | 88 | `e434878696947eaf8bc228eb01affec9` | France matches of the second pipeline where the record writes '&' as 'et' or '+'; added | `france_fix/final_hunt/fr-gathik-rest/a05_sets.py`, then `france_fix/final_hunt/verify/nz_*.py` (`nz_gain.py` writes it) | v10c |
| `fr_street_veto.parquet` | 256 | `97ecb200d6a23f1c370fa7a68fa936cc` | Matched France pairs with the same generic name and house number but a completely different street, both bge families below 0.3; removed | `france_fix/final_hunt/fr-strong-veto/s1_cov.py` ... `s19_final.py` (258 pairs), checked by `france_fix/final_hunt/verify2/fsv_addr.py`, `fsv_addr2.py`; a one-off command then dropped the 2 pairs whose S1 row would have lost every match and kept columns `s`, `q` | v10d |
| `usi_stack_add.parquet` | 7,021 | `6267908133292fd838da9acc44c3a2e0` | US/India pairs the stacker matches that v10c does not; not applied by the build, which recomputes the difference and checks it equals this file | one-off command: stacker pairs (`stack/apply_test.py _avg`) minus v10c US/India pairs; same pairs as `added_vs_v10c_avg.parquet` of `stack/apply_test.py` | v10d |
| `usi_stack_remove.parquet` | 1,275 | `3ae1106081c1701c394b5df12e3a5c2a` | US/India pairs of v10c the stacker drops; used as a check as above | one-off command: v10c US/India pairs minus stacker pairs; same pairs as `removed_vs_v10c_avg.parquet` | v10d |
| `lgb_models_avg.pkl` | 4 models | `b1ef5f249461286b6e9cc46102d6eaeb` | Python dict of 4 LightGBM boosters (300 trees each, 24 features); the stacker's probability is their mean | `stack/fit.py` run twice (tags `_rob`, `_robtr`; commands in `stack/README.md`) + `stack/merge_models.py _rob _robtr _avg` | v10d |

All row counts equal the number of distinct (s, q) pairs and the number of distinct records `q`: no record appears twice
in a set.

## How the build uses them

| Step of `src/build_final.py` | Sets |
|---|---|
| 1. France scores of v9y (README step 16) | `desc_veto_set`, `noise_add_set`, `fp_veto_set` |
| 2. France rows of v9z | `recall_add_set`, `moreveto_stem2_set` |
| 3 and 7. US/India blend (v10a) and stacker (v10d) | `tier12_removed`, `tier12_restored` (applied after the model), `lgb_models_avg.pkl` (step 7) |
| 6. v10c | `fr_typo_safe`, `fr_same_address_safe` (first regenerated by `same_address.py` with the veto sets `desc_veto_set`, `fp_veto_set`, `moreveto_stem2_set` and compared), `fr_amp_safe` |
| 7. check | `usi_stack_add`, `usi_stack_remove` |
| 8. v10d France | `fr_street_veto` |
| 10. candidate file | `fr_same_address_safe` (its matched pairs join the candidates) |

## Source of each copy

| File | Copied from (27 Sep 2026) |
|---|---|
| `desc_veto_set.parquet` | `$BER_SCRATCH/frfix/namechg/veto_set.parquet` |
| `noise_add_set.parquet` | `$BER_SCRATCH/frfix/build/add_set_v2_final.parquet` |
| `fp_veto_set.parquet` | `$BER_WORK/frfix2/fp_veto_set.parquet` |
| `recall_add_set.parquet` | `$BER_WORK/frfix3/recall_add_set.parquet` |
| `moreveto_stem2_set.parquet` | `$BER_WORK/frfix3/moreveto_stem2_set.parquet` |
| `tier12_removed.parquet`, `tier12_restored.parquet` | `$BER_SCRATCH/maxplan/s5/` (rebuilt identically by `movers/build_movers.py`) |
| `fr_*`, `usi_stack_*`, `lgb_models_avg.pkl` | `$BER_WORK/final_sets/` |

`$BER_SCRATCH` is the scratch folder of the analysis runs (default `C:/ber_scratch`); `$BER_WORK` is the work folder of
the pipeline (README section 3).
