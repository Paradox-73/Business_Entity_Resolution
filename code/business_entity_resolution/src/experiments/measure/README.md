# Measurement scripts of the documentation

The methodology document (`Documentation_template.md` in the zip, `docs/METHODOLOGY.md` in the repository) marks
numbers measured for it with [M]. These scripts compute them. They were run on 27 Sep 2026 on the laptop, after the
final upload, from the saved files of the final build.

- They read `$BER_WORK` (the intermediate files of README steps 1-21), `../../../sets/` (or `$BER_SETS`) and, where
  stated, the uploaded files in `<root>/submissions/v10d/`.
- Scripts that write files write them to `$BER_SCRATCH/measure/` (`BER_SCRATCH` defaults to `C:/ber_scratch`).
- Each runs from any folder: `python <script>`.
- None of them changes a pipeline file.

| Script | Numbers it gives | Section of the methodology |
|---|---|---|
| `data_facts.py` | row counts per split and country; label structure (7,638,365 true pairs, 26.0% decoys, 5.58% S1 rows without a match); chains (38.3% exact name shared, 50.0% core name, "Primary Care Group" 253); all-lowercase share of records with and without a changed word; house-number difference of true pairs and decoys (Artefact 1) | 2.1 |
| `change_sample.py` | samples of 80,000 true pairs and 80,000 decoys per country run through the change detector `france_fix/artifacts/census/ops.py`; input of the next two scripts | 2.1 |
| `change_table.py` | share of true pairs with each name and address change ("Noise in true pairs") | 2.1 |
| `lowercase_test.py` | all-lowercase share of records whose name adds or swaps in a noise or descriptor word, and of other changed words (Artefact 2) | 2.1 |
| `test_candidates.py` | test pairs at each stage, per S1 row and per record, reduction ratios, final candidate file by country, composition by source | 3 |
| `candidate_sources.py` | final candidate pairs outside our score table by source; S1 rows with the most candidates (more than 50 and 100) | 3 |
| `heldout_recall.py` | recall ceilings of each candidate definition on the held-out half, by country and with / without address | 3 |
| `compute_cost.py` | search and feature rates, laptop wall-clock per step, transformer throughput, disk use of `work/`, parsed from `work/*.log` | 3, B.6 |
| `stack_importance.py` | split gain of each stacker feature, mean over the 4 models of `sets/lgb_models_avg.pkl` | 4 |
| `set_summary.py` | sizes and saved columns of the pair sets: recovery groups and their mean estimated true share (0.93), false-accept veto tiers A / B / C with their estimated true share and lowercase share, descriptor veto by kind | 4 |
| `desc_veto_examples.py` | "Club" <-> "Ecole" swaps inside the descriptor veto (230) | 4, 5 |
| `france_examples.py` | raw texts of the France example pairs of section 5 and of the street veto | 5 |
| `ce_auc.py` | held-out AUC of the three cross-encoder families per fold | B.1 |
| `peak_memory.py` | peak memory per process while another command runs (`src/logs/build_final_memory.txt`) | A |

The recall ceilings and the candidate counts in the root `README.md` and in `README.md` step 28 of this folder come from
`heldout_recall.py` and `test_candidates.py`.
