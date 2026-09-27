# TEMPLATE — Methodology document (1–2 pages)

How to use this file:
- Copy it at hour 0 into your shared drive. Fill each section **as you go**, not in the last hour.
- Hard limit: 2 pages at 11 pt. 2025 rules said "1 page" in the problem repo and "1–2 pages" on Unstop, so aim for 1.5 pages and put the rest in an appendix only if allowed.
- The document is used to pick the top 10 together with the leaderboard (host, `videos/A/notes.md` `[00:13:40]`). Judges at the 2025 finale rewarded: ablation tables, data insights backed by examples, small efficient models, production thinking.
- Every claim gets a number. Replace every `<...>`.

---

## Team `<name>` — `<college>` — `<problem name>`

**Members:** `<name (role)>`, `<name (role)>`, `<name (role)>`, `<name (role)>`
**Final score:** `<metric>` = `<CV score ± std over k folds>` (cross-validation), `<public LB score>` (public leaderboard)

### 1. Problem understanding (3–4 lines)

- Task in one sentence: `<predict X from Y>`.
- Metric and what it rewards: `<e.g. SMAPE punishes relative error, so cheap items matter as much as expensive ones; over-prediction and under-prediction are penalised differently>`.
- Key constraint(s): `<licence MIT/Apache-2.0, ≤8B params, no external target lookups, inference time>`.

### 2. Data insights that drove decisions (the most important section)

Write each as **observation → evidence → decision**. 3–5 bullets.

- `<Target is right-skewed (skew 13.6, max 2,796)>` → `<histogram>` → `<trained on log1p(target)>`.
- `<~80 unit spellings; weight/volume/count cover 98.6%>` → `<counts>` → `<normalised to g / ml / count; features log(size), log(pack), log(size/pack)>`.
- `<Same image, different prices (21 copies of one card priced $10–$20)>` → `<example>` → `<need text + image>`.
- `<Error analysis: "gluten-free" variant priced 3× higher but predicted the same>` → `<example row ids>` → `<fine-tuned text encoder / added keyword features>`.
- `<Duplicates / missing images / label noise>` → `<counts>` → `<how handled>`.

### 3. Final approach (with a small diagram if space allows)

- Inputs and preprocessing: `<cleaning, image size, missing-image handling (zero tensor + image-present flag)>`.
- Model(s): `<backbone name, licence, param count, frozen or fine-tuned (which layers)>`.
- Fusion / head: `<concat + MLP / cross-attention / transformer over modality tokens>`.
- Loss and target: `<Huber on log1p; hybrid 10×MSE + SMAPE; why>`.
- Training: `<optimizer, LRs (backbone vs head), schedule, epochs, batch, precision, EMA>`.
- Ensemble: `<k fold models averaged; blend weights fitted on out-of-fold predictions: w1=…, w2=…>`.
- Post-processing: `<clip to [min,max], snap to valid values, TTA>`.

### 4. Validation strategy

- `<Stratified 5-fold on target quantile bins; same folds for all models; seed 42>`.
- CV–leaderboard agreement: `<CV 40.9 vs public LB 40.6>`. Say how you guarded against leakage (`<grouped duplicates into the same fold>`).

### 5. Ablation table (judges asked for this in almost every Q&A)

| # | Change | CV score | Δ vs previous |
|---|---|---|---|
| 0 | Baseline `<TF-IDF + Ridge>` | `<..>` | — |
| 1 | `<+ engineered features, LightGBM>` | `<..>` | `<..>` |
| 2 | `<+ text encoder fine-tuned>` | `<..>` | `<..>` |
| 3 | `<+ image embeddings>` | `<..>` | `<..>` |
| 4 | `<+ k-fold ensemble + EMA>` | `<..>` | `<..>` |
| 5 | `<+ TTA / blend>` | `<..>` | `<..>` |

- Things that did **not** work (1 line each, with numbers): `<LightGBM on learned embeddings: +1.2 worse>`, `<raw SMAPE loss: unstable>`.

### 6. Efficiency and scalability

- Total parameters: `<..>`; model size on disk: `<.. GB>`; peak GPU memory: `<.. GB>`.
- Training time and hardware: `<.. h on ..>`. Inference: `<.. ms/item>`, full test set in `<.. min>`.
- Production sketch (2 lines): `<embeddings cached, encoders as separate services, handles missing image / rotated image>`.

### 7. Compliance

- All models: `<name — licence — params>`. No external data for targets. Code and exact commands in the attached zip (`README.md`).

---

Checklist before submitting:
- [ ] Every number in the doc matches the logs / experiment sheet.
- [ ] Ablation table present.
- [ ] At least one concrete example (row / product) in "Data insights".
- [ ] Model licences and parameter counts listed.
- [ ] Fits in 2 pages; exported to PDF; file name as required.
