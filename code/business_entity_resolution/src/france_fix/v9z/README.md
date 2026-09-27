# France changes of v9z (27 Sep 2026)

v9z = v9y with 5,883 missed France matches added and 73 France matches removed (`submissions/LOG.md`). These scripts
chose the two pair sets; they ran once, interactively, and read and write the scratch folder `$BER_SCRATCH` (default
`C:/ber_scratch`) and `$BER_WORK/frfix3/`. `src/build_final.py` (step 2) rebuilds the v9z France rows from the saved
sets without them.

| Folder | What the scripts do | Output used by the build |
|---|---|---|
| `recall/` | `r1_base.py`-`r4_c.py`: three kinds of candidate recovery: (a) vetoed records with an exact-name S1 at the same address, (b) the earlier missed-match set `$BER_WORK/frfix2/fn_add_set_final.parquet` (`france_fix/artifacts/fn/`, written by `france_fix/artifacts/build/b2_scores.py`; tiers A1-A4), (c) rejected single-change copies at the same address whose change is an acronym, a web domain or a dotted legal form; `r5_build.py`: the final set (column `grp`) and a score file with p2 = 0.95 on it | `recall_add_set.parquet` (5,883 pairs) |
| `moreveto/` | `m1_acc.py`-`m8_words.py`: accepted v9y France pairs grouped by name-change signature, with the all-lowercase-record share as a label-free test; `m9_veto.py`: a wider descriptor-swap veto (not used); `m10_veto_stem.py`: the same-stem descriptor swaps | `moreveto_stem_set.parquet` (71 pairs) |
| `refute_moreveto/` | `r1.py`-`r7.py`: checks that tried to refute the veto | none |
| `verify_recall/` | `v1_scores.py`-`v18_central.py`: checks of the recoveries (US/India analog rates, address-format tests, expected France F0.5 change) | none |
| `build/` | `b1_scores.py`: adds 2 pairs to the 71 (`moreveto_stem2_set.parquet`, 73 pairs) and writes `test_scores_v9y_combo.parquet` (recoveries at p2 = 0.95, then the 73 pairs at p2 = 0); `b2_diff.py`: pair-level difference of the spliced file from v9y | `moreveto_stem2_set.parquet` |

The v9z file was then made by two commands (run from `src/`, `W=$BER_WORK/frfix3`):

```bash
BER_CALIB=$BER_WORK/ce/rule_blend_ab_a2.json BER_FR_LEGAL_VETO=1 python finalize.py full_cons $W/out_combo_all $W/test_scores_v9y_combo.parquet
python experiments/splice_country.py <v9y matching_results.tsv> $W/out_combo_all/matching_results.tsv $W/out_combo_on_v9y
```

followed by a one-off filter that keeps only matched pairs present in `test_scores_blend_v7p.parquet` or in the second
pipeline's matches; it removed 0 pairs (measured by comparing `out_combo_all` France rows with `submissions/v9z`).
