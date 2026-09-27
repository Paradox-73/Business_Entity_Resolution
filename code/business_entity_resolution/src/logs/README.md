# Run logs cited by the documentation

| File | What it is |
|---|---|
| `stack_fit_rob.log` | output of `stack/fit.py gbdt` with `DROP=nq,rank_s,margin_s TAG=_rob` (27 Sep 2026 20:29-20:32): 2 models fitted on the two parts of the held-out half, 42 s and 20 s; 180 s in all |
| `stack_fit_robtr.log` | the same with `TAG=_robtr FITON=1`: 2 models fitted on the training half, 25 s and 22 s; 156 s in all |
| `stack_res_gbdt_rob.json`, `stack_res_gbdt_robtr.json` | held-out macro F0.5 of the two runs (`base` = the v10a blend, `stack_0.5_1.0` = the stacker), on all held-out S1 rows, on the two parts (h0, h1) and per country (keys `US` and `IN`; `fit.py` and `score_avg.py` now name India by its full label, `India`) |
| `stack_res_avg.json` | output of `stack/score_avg.py`: held-out macro F0.5 of the v10a blend (`p`), of each run (`pst`, `pst2`) and of the 4-model mean (`pa`, 0.99226) |
| `build_final.log` | a full run of `build_final.py` from an empty build folder (27 Sep 2026, laptop, 304 s): every step, the official validator's PASS and the md5 table (all five files identical to the uploaded ones) |
| `build_final_memory.txt` | peak memory of each process of that run (largest: `stack/build_test.py`, 5.46 GB) |
| `build_final_28sep.log` | a second full run of `build_final.py` from an empty build folder (28 Sep 2026, laptop, 370 s wall-clock while other checks ran at the same time), after the country names were removed from the code and the same-address check was added: the stage counts of `same_address.py` (step 26), "equal to fr_same_address_safe: True", both checkers' PASS and the same five md5 values |

- The stacker fit logs were copied unchanged from the stacker's output folder of that evening. The fit commands
  themselves were not logged; `stack/README.md` gives them as reconstructed from the saved models.
- In `build_final.log` and `build_final_28sep.log` the folder names were shortened: `$BER_BUILD` and `$BER_OUT`
  stand for the build and output folders of that run (both in a scratch folder), `<root>` for the repository folder.
  Nothing else was changed.
- `build_final.log` was written by the code of 27 Sep, so its lines name countries ("France legal-form veto"); the
  28 Sep log shows the current wording ("countries without training labels"). Its last line, the wall-clock, was
  added by the shell that ran it; the 28 Sep log has no such line.
