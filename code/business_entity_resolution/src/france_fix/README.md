# France analyses

France is 15% of the test S1 rows and has no training labels. The scripts in this folder are the analyses that chose
each France rule and each France pair set of the final file.

- They ran once, interactively, on the laptop between 26 and 27 Sep 2026, and were not re-run for the package.
- Each script has its input and output folders as constants at the top of the file: the scratch folder `$BER_SCRATCH`
  (default `C:/ber_scratch`) or `$BER_WORK`.
- The final build does not run them. It reads the pair sets they wrote, copied into `../../sets/`
  (`sets/README.md` names the scripts behind each set; `src/build_final.py` applies them).
- Script paths below are relative to this folder.

| Folder or file | What the scripts do | Pair set or decision it led to |
|---|---|---|
| `artifacts/census/` | `ops.py`: detector of the change between a record and an S1 row (legal form dropped, word swapped, house number moved up, all-lowercase, ...); `groups.py` maps detector flags to generator operations; `run_ops.py`, `run_fr.py`, `analyze.py`: counts of each change on labelled US/India pairs and on France | the label-free tests (lowercase share, house-number direction). `ops.py` is also imported by `../france_recall.py` (README step 25) |
| `tables/` | `fr1.py`: best-candidate table of France test records (`fr_top.parquet`); `fr4.py`: the same for labelled US/India train records (`usi_top.parquet`); `fr8.py`: France pairs of earlier submissions (`pairs_<version>.parquet`) | inputs of `namechg/`, `build/` and `lbcal/` |
| `france_cal.py` | France recalibration by change pattern towards US true rates; `ndiff` adds number-changed records | probe versions v7j and v7j_full; v7j lost France -0.0163 on the leaderboard, rule refuted |
| `fr_restore.py` | the transformer may not lower a France pair with the same house number and street | probe version v7m; France -0.0091, rule refuted; led to the descriptor veto |
| `namechg/` | `a1_build.py` ... `a11_veto.py`: France pairs whose record adds or swaps a word, grouped by word class, with sibling and lowercase tests | `desc_veto_set.parquet` (22,436 pairs, p2 = 0; v9a, in v9y) |
| `refute_desc/` | `r1_feat.py` ... `r5_lb.py`: checks that tried to refute the descriptor veto (surface-noise, casing and address fingerprints on France and on US/India labels; lowercase share of each uploaded France change) | no set; the veto was kept |
| `build/` | `b1_groups.py` ... `b11_free.py`: lowercase-share estimates of rejected same-address France pairs by class; `b4_addset.py`, `b8_mkscores_v2.py` write the set; `b10_classmodel.py`, `b11_free.py` fit class true rates with the `lbcal/` model | `noise_add_set.parquet` (9,800 pairs, p2 = max(p2, 0.95); v9b, in v9y) |
| `lbcal/` | a model of the France probabilities fitted to the leaderboard results of the France-only uploads | no set of its own; `build/b10_classmodel.py` and `build/b11_free.py` reuse its logistic model |
| `refute_lbcal/` | checks that tried to refute a France change proposed from the `lbcal/` fit (adversarial fits, chi-square tests, US analogs) | no set |
| `twins/`, `twins_check/` | France S1 "twins" (the same generic name on the same street): how often a record's best candidate has a twin, and a label-free false-share estimate from the twin excess; `twins_check/` checks the candidate coverage behind it | no set |
| `artifacts/fp/` | `fp1_dec.py` ... `fp10_veto.py`: accepted France pairs with a look-alike signature, tiers A, B, C, true-share estimate per tier; `refute/`: checks | `fp_veto_set.parquet` (1,145 pairs, p2 = 0; v9e, in v9y) |
| `artifacts/fn/`, `artifacts/build/` | missed France matches: single-change copies of the S1 row at the same address that the transformer rejected; `artifacts/build/b2_scores.py` writes the set of tiers A1-A4; `fn/refute/`: checks | `$BER_WORK/frfix2/fn_add_set_final.parquet` (4,150 pairs; probe version v9f), later part of `recall_add_set.parquet` |
| `recall/` | `decomp.py`: where the held-out loss of the v10a blend sits; `fr_recall.py`, `train_analog.py`, `train_gain.py`: the second pipeline's France matches outside our lists and their US/India analog on labels | the rule of `../france_recall.py` (v10b, +4,825 pairs) |
| `v9z/` | recoveries and the same-stem descriptor veto (`v9z/README.md`) | `recall_add_set.parquet` (5,883 pairs), `moreveto_stem2_set.parquet` (73 pairs) |
| `final_hunt/fr-committee/` | France pairs both pipelines scored that only the second pipeline accepted | no set: US/India analog 69.6% true |
| `final_hunt/fr-gathik-rest/` | the 2,441 France matches of the second pipeline left out of v10b, by class | proposals for `fr_typo_safe` and `fr_amp_safe` |
| `final_hunt/same-address/` | the same-address rule: same house number and street words, one S1 row at the address, name differs only by acronym or noise; checked on train labels | proposal for `fr_same_address_safe` |
| `final_hunt/verify/` | `typo_v01_impl.py` ... `typo_v07_safe.py`, `v1.py` ... `v8.py`, `nz_*.py`: checks of the three proposals, each by a separate attempt to refute it; they write the kept subsets | `fr_typo_safe.parquet` (416), `fr_same_address_safe.parquet` (298), `fr_amp_safe.parquet` (88); v10c |
| `final_hunt/fr-strong-veto/`, `final_hunt/verify2/` | France vetoes where both bge families score low; `s19_final.py` keeps only same name and house number on a completely different street; `verify2/` checks the addresses | `fr_street_veto.parquet` (256 pairs; v10d) |

The numbers in the last column come from `sets/README.md`, `submissions/LOG.md` and `EXPERIMENTS.md`.
