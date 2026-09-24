# Amazon ML Challenge 2026 — Team Playbook

Built 22 Sep 2026 from:
- `videos/A/notes.md` and `videos/B/notes.md`: the 2025 finale, where 10 teams presented to 3 Amazon scientists. Cited as `A [hh:mm:ss]` / `B [hh:mm:ss]` (time in the Twitch recording).
- `research/past_solutions.md`: public write-ups from 2021–2025 and the 2026 rules pages.
- `videos/C/notes.md`: the 21 Sep 2026 AWS session (Builder Center, Free Tier, SageMaker). Cited as `C [hh:mm:ss]`.

Words used below:
- **CV**: cross-validation. Split the training data into k parts ("folds"), train k times, and score each part with the model that did not see it.
- **OOF**: out-of-fold predictions, meaning the prediction each training row got from the fold model that did not train on it. Used to fit ensemble weights honestly.
- **LB**: leaderboard. **Public LB** is scored on part of the test set during the contest. The **private LB** uses the full test set and is revealed afterwards.
- **SMAPE**: symmetric mean absolute percentage error, the 2025 metric. Lower is better, range 0–200.
- **Ablation**: removing or adding one component and measuring the score change.
- **Embedding**: a fixed-length vector a pretrained model outputs for a text or image.
- **Frozen / fine-tuned**: frozen means the pretrained model's weights are not trained; fine-tuned means they are updated on our data.

---

## Dates that matter (from the Unstop page; recheck on the day)

- **Registration closes 22 Sep 2026, 23:59 IST (today).** Every member needs a valid AWS Builder Center alias (Builder Center is Amazon's developer community site). An invalid alias leaves the registration incomplete.
- **The ML round is 60 hours, not 72**: 25 Sep 09:00 IST → **27 Sep 21:00 IST** (Unstop round page: "2 days 12:00:00").
- The top 500 teams **at the 48-hour mark** (≈ 27 Sep 09:00 IST) get an extra $100 AWS credit, on top of $200 per participant.
- Results 2 Oct; finale 7 Oct 2026, 10:00–15:00 IST, virtual.

---

## (a) What a winning solution looks like

- The finale ranking **did not follow the leaderboard** in 2025 `B [02:52:32]`:
  - Winner MESSI reported 40.03. 1st runner-up Zenith reported 41.21. 2nd runner-up Rocket reported 39.83.
  - Antrix reported 39.2 ("top of the leaderboard") and Test Data ~39.2. Neither placed.
  - The host said the top-10 scores were "very minute" apart `B [02:52:12]`, so the presentation and the document decide.
- What the three winners had in common:
  - **An ablation table**: every step with its score gain. Named as the reason for Zenith (runner-up) `B [02:57:33]` and MESSI (winner) `B [03:00:29]`.
  - **Data insight shown with a concrete example**: MESSI's "gluten-free" sweetener, actual price 109, predicted 37 `A [02:16:57]`. The judge named it: "example of gluten-free stood out for me" `B [03:00:12]`.
  - **Small, efficient, justified models**: MESSI ~630M params; Rocket 441M, "18× smaller than 8B"; Zenith 2.1B with a head trained in 10 minutes.
  - **Real-world thinking**: missing images (zero tensor + image-present flag), rotation, keeping expensive items instead of deleting them as "outliers" `A [02:09:59]`.
  - **Knowing why each choice was made**, including dropping a whole modality with evidence (Rocket dropped images) `B [02:53:33]`.
- The judges' own words:
  - "You justified your architecture with data-driven insights. That is going to matter" `B [02:48:50]`.
  - "Dive deep… understand why the results are good, why they are bad, and try out different approaches" `B [02:50:25]`.
  - Leadership principles they cited: **Frugality** (Rocket), **Working Backwards** from the customer (Zenith), **Dive Deep** (the advice given to everyone).
- Past years:
  - 2024 winner: fine-tuned the strongest allowed model (Qwen2-VL-7B), then **fixed 1,600 labels by hand**. That gave +0.186 F1, while doubling the data gave +0.001 (`research/past_solutions.md`).
  - Fine-tuning a pretrained model end to end appeared in 9 of 12 published top-100 solutions.

## (b) Judging criteria as best understood

| Stage | Criterion | Evidence | Weight |
|---|---|---|---|
| Top-10 selection | Leaderboard score (full test set) | 2025 rules; host `A [00:13:40]` | not published |
| Top-10 selection | Approach document (1–2 pages) | host `A [00:13:40]`: "solution submission, the approach document… innovative solutions"; Unstop 2026 | not published |
| Finale | Novelty and approach | host `A [00:15:01]` | not stated |
| Finale | Intelligent use of data | host `A [00:15:01]` | not stated |
| Finale | Efficiency and scalability | host `A [00:15:01]`; judges asked GB / latency / hardware in 4 of 10 Q&As | not stated |
| Finale | Presentation | host `A [00:15:01]`; 10 min, warning at 8 min | not stated |

- Unstated but clearly rewarded: ablations, error analysis, frugality, answering questions with numbers.

## (c) 2025 finalist comparison

| Team (college) | Result | Reported score | Text model | Image model | Frozen or fine-tuned | Fusion / head | Loss, target | Validation | Special idea |
|---|---|---|---|---|---|---|---|---|---|
| MESSI (IIT Madras) | **Winner** | 40.03 test | DeBERTa | Swin Transformer | last 3 blocks fine-tuned | concat + MLP; pack-qty embedding; text "hints" | Huber, log1p | stratified 5-fold, fold average | error analysis ("gluten-free"), EMA, horizontal-flip TTA, image-present flag |
| Zenith (IIEST Shibpur) | **1st runner-up** | 41.21 val | OpenCLIP ViT-bigG text | OpenCLIP ViT-bigG | staged: head → last 4 blocks → full | 4 tokens (CLS, image, text, numeric) → 6-layer transformer | Huber, log1p | hold-out (not stated) | unit standardisation to g/ml/count; log(size), log(pack), log(size/pack) |
| Rocket (IIIT Delhi) | **2nd runner-up** | 39.83 val | DeBERTa-v3-large | none (dropped with evidence) | fully fine-tuned | cross-attention: text CLS queries 128-d feature embeddings | Smooth L1, log1p | 5% hold-out | "Occam's razor", 441M params; keyword flags (organic, gourmet, bulk) |
| Antrix (Thapar) | — | 39.2 | ModernBERT (fine-tuned) | PaliGemma + Qwen-VL (frozen) | mixed | MLP + GCN over 4 nearest neighbours (FAISS) | SMAPE-variant, log1p | not stated | graph of similar products; missing pack qty filled by a zero-shot NER model |
| Test Data (IIT Patna) | — | ~39.2 (ens.), 40.55 best single | BERT / CLIP text | EfficientNet-B4, CLIP ViT-L/14, ViT-H/14 | mostly frozen | concat + deep MLP; weighted ensemble 0.1/0.35/0.55 | log target | 15% hold-out | de-duplication rules; 99.6th-pct clipping |
| Trailblazers (BITS Pilani) | — | 40.8 | Qwen2.5-VL-3B (image+text) | same | frozen | 4096-d VLM embedding + 18–30 features → MLP | Huber, log | 5-fold | EMA, residual MLP; features: brand regex, pack, text stats |
| Spam LLMs (IIT ISM Dhanbad) | — | not stated | Qwen3-4B embeddings | SigLIP2 + DINOv3 | frozen | per-model tower MLPs → concat → head | MSE, log1p | 5-fold | each extra encoder ≈ −2 SMAPE |
| Abhimanyu (IIT Jodhpur) | — | 41.08 | SigLIP / CoCa | SigLIP / CoCa | frozen | 2 MLPs, weighted average | L1 mix, log1p | 5-fold | 73 keyword flags (keto, gluten-free, dairy-free…) |
| Royal Recruits (IIIT Delhi) | — | 40.2 | CLIP ViT-L/14 text | CLIP ViT-L/14 + CNN from scratch | top 5 layers fine-tuned | concat + deep MLP; value/unit embeddings | **10×MSE + SMAPE**, log1p | hold-out | two different learning rates (1e-5 backbone, 1e-4 new layers) |
| CTRL+ALT+DEV (Amrita, Coimbatore) | — (public LB rank 8) | 40.78 public | DistilBERT + CLIP-L text | CLIP ViT-L/14 | fine-tuned | concat → MLP | Huber + contrastive (InfoNCE), log2 | hold-out | text contrastive loss with 6% masking; white placeholder for missing image |

## (d) Top 15 techniques, ranked by how many of the 10 finalists used them

| Rank | Technique | Finalists using it (of 10) | Notes |
|---|---|---|---|
| 1 | Predict **log(1+price)**, invert at the end | 10 | Every finalist. Zenith showed linear scale converged far slower `A [00:47:09]`. |
| 2 | **Parse quantity / unit / pack size** from text; normalise units | 8 | Zenith: ~80 unit spellings, 3 types cover 98.6% `A [00:35:17]`. |
| 3 | **Text + image** together | 9 | Rocket placed 3rd without images, but proved why. Images gave >10 points for Test Data `A [00:53:54]`. |
| 4 | **MLP head on concatenated embeddings** | 8 | The default fusion; attention fusion in 3. |
| 5 | **CLIP-family encoder** (CLIP, OpenCLIP, SigLIP, CoCa) | 6 | CLIP ViT-L/14 was the most common single choice. |
| 6 | **Fine-tune the backbone** (at least the top layers) | 6 (all top 3) | Frozen embeddings only: 4 teams, none placed. |
| 7 | **Text cleaning** (HTML, emoji, unicode, whitespace) | 6 | Cheap; do in hour 1. |
| 8 | **Huber / Smooth L1** loss on log target | 5 (+3 SMAPE-aware) | Plain SMAPE loss was worse / unstable (MESSI `A [02:15:36]`; 2025 rank 11). |
| 9 | **Warm-up + cosine** (or linear) LR schedule, AdamW | 5 | Standard. |
| 10 | **k-fold CV, average fold models** at test time | 4 | Winner used stratified 5-fold on price bins. |
| 11 | **Staged unfreezing** (train head first, then top layers, then all) | 4 (all top 3 had a staged recipe) | Zenith: −8.6 SMAPE from unfreezing the last 4 blocks `A [00:40:22]`. |
| 12 | **Missing-image handling** (zero/black/white image; image-present flag) | 4 | Winner added the flag; judges praised it `A [02:20:57]`. |
| 13 | **Keyword / text-statistic features** (word count, "organic", "gluten-free", "pack of") | 4 | Rocket found sentence and word counts correlated most with log price. |
| 14 | **Brand extraction** (first words, regex, small NER model) | 4 | One team found it useless; test it. |
| 15 | **EMA of weights, TTA, weighted ensembles** | 2 / 1 / 3 | EMA + k-fold took MESSI from 43.15 to 40.18 `A [02:19:29]`. EMA keeps a running average of the weights over training; TTA averages predictions over flipped copies of each test image. |

- Tried by several teams and **did not help**: LightGBM on learned embeddings (MESSI, Spam LLMs); plain KNN neighbour averaging (Spam LLMs); larger text models (CTRL+ALT+DEV, Test Data); raw SMAPE as the only loss.

## (e) Top 10 mistakes to avoid

1. **Chasing the public leaderboard.** In 2025 the public LB used 25k of 75k test rows, and the finale order ignored small LB gaps. Trust CV.
2. **No ablation log.** Keep a shared sheet from hour 1: experiment, change, CV score, Δ, time. The finale deck and document are built from it.
3. **Not knowing your own numbers.** A finalist couldn't state their ensemble weights `A [00:59:57]`. Another said "fine-tuned" on slides and "no fine-tuning" in Q&A `A [01:01:16]`.
4. **Deleting expensive items as "outliers".** A judge objected: "highly priced items are quite natural in e-commerce" `A [01:18:38]`. The winner kept them on purpose.
5. **Leaky validation.** A 2025 retrieval model showed 10.8 CV vs ~47 on test. Put near-duplicates in the same fold.
6. **Underestimating image download and inference time.** The winner lost time just downloading images `B [03:01:53]`. In 2024 one team predicted only 84k of 131k rows. Start downloads in hour 0; time one batch and multiply by the row count.
7. **Loss that fights the metric.** Log+MSE mismatched SMAPE, and raw SMAPE exploded on cheap items (2025 rank 11). Use Huber on log, or a hybrid like 10×MSE+SMAPE; check CV.
8. **Using a model with the wrong licence or size.** 2025 required MIT/Apache-2.0 and ≤8B params. Llama and CC-BY-NC models are out. Check each model card before using it.
9. **Running over time and AV failures at the finale.** Two teams were cut off at 10 minutes; one lost minutes to camera/mic problems.
10. **Writing the document in the last hour / missing the deadline.** One 2025 participant with SMAPE 48 never submitted. Submit a valid file in the first 3 hours and refresh it every few hours.

## (f) Questions judges asked in 2025, and strong answers

| Question (paraphrased, with time) | Strong answer pattern |
|---|---|
| Why this big model, not a smaller fine-tuned one? What does the complexity buy? `A [00:31:23]` | "We compared X (N params) vs Y (M params) on the same folds: Y was 0.8 better but 5× slower; we kept X because…". Have the table. |
| What if seller images are rotated in production? `A [00:27:14]` | "We tested: rotating test images by 90° raised SMAPE from 40.1 to 41.3; flip-TTA / rotation augmentation cut that to 40.5." Measure it before the finale. |
| Pack of 5 vs 5 × pack of 1 — how do you prevent arbitrage? `A [00:28:53]` | Predict unit price (price ÷ pack quantity × size) and multiply back, or post-check that a pack of n is predicted at most n × the single-unit price. Show a plot. |
| Two 500 ml milks: ₹25 vs ₹100 predicted — any per-category sanity check? `A [01:01:40]` | Per-category / per-brand residual table; list top-10 worst predictions and what caused them. |
| Ablation: log vs linear? with vs without images? why two encoders? `A [00:46:38]`, `A [01:21:06]`, `A [01:49:00]` | Single table with the numbers. |
| Model size in GB, latency, hardware for production? `A [00:48:37]` | "2.1B params, 4.2 GB fp16, 12 ms/item on a T4, full test in 15 min, embeddings cacheable." |
| Missing images — how does the model degrade? `A [01:20:11]` | Show score on the test rows without images; describe the flag / zero tensor. |
| What exactly were your outliers? `A [01:17:24]` | Only remove proven errors (price 0, unit mismatch). Show counts. |
| Which layer did you take embeddings from? `A [01:19:39]` | Exact layer and pooling ("last hidden state, mean-pooled over tokens"). |
| Did you test the real-world robustness claim? `A [02:23:01]` | A stress test (rotated, missing text, out-of-range price bucket) with numbers. |
| How were ensemble weights learned? `A [02:38:42]` | "Non-negative weights summing to 1, fitted on OOF predictions with scipy to minimise the metric: 0.55/0.30/0.15." |
| Which handcrafted features mattered most? `A [01:22:57]` | Feature importance or drop-one ablation, top 5 with Δ. |
| Do you have data-driven evidence for dropping a modality? `A [02:04:22]` | CV with and without, and the reason (e.g. embedding PCA shows no price structure). |

## (g) Plan for the 60-hour round (team of 4)

The plan asked for 72 hours; the 2026 round is **60 h: Fri 25 Sep 09:00 → Sun 27 Sep 21:00 IST**. Adjust if the problem page says otherwise.

Roles:
- **P1 Data & features lead**: download data and images, EDA, parsing features, error analysis.
- **P2 Text models**: TF-IDF baseline, text encoder fine-tuning.
- **P3 Image / multimodal models**: image embeddings, fusion model.
- **P4 Validation, ensembling, submissions, document**: owns the CV folds, the experiment sheet, the submission validator, the methodology doc and the deck.

| Hours (IST) | P1 Data | P2 Text | P3 Image/multimodal | P4 Validation/docs |
|---|---|---|---|---|
| 0–1 (Fri 09–10) | Read problem, rules, licences, metric. Start image download | Same | Book GPUs: Kaggle, Colab, AWS credits | Read rules; write the metric function; create experiment sheet; fix CV folds and seed |
| 1–3 | EDA: target distribution, duplicates, units, missing fields | TF-IDF + Ridge baseline | Check image download rate; estimate total time | **Submit baseline** after `validate_submission.py` passes |
| 3–8 | Parsing features (quantity, unit, pack, keywords, brand) | Sentence-embedding + LightGBM | CLIP/SigLIP embeddings of all images (cached to disk) | Log every result; test that the CV score matches the LB score |
| 8–14 (Fri evening) | Error analysis on OOF: worst rows by category | Start fine-tuning DeBERTa-v3 / a similar text model, fold 0 only | Fusion MLP on text + image + features | Draft doc sections 1–2 |
| 14–20 (night) | Sleep (shift A) | Fine-tune folds 1–4 overnight | Sleep (shift A) | Monitor runs; sleep (shift B) |
| 20–30 (Sat 05–15) | Fix feature bugs; new features from error analysis | Loss experiments: Huber vs hybrid | Unfreeze top layers of the image encoder; missing-image flag | Blend OOF; submit best |
| 30–40 (Sat 15–01) | Robustness tests: rotated images, missing text | Second text model for diversity | Second multimodal variant | Ablation table filled; doc 70% |
| 40–48 (Sun 01–09) | Sleep shift | Final fold runs | Final fold runs | **48-h mark ≈ Sun 09:00**: make sure a strong submission is on the LB (top-500 extra credit) |
| 48–54 (Sun 09–15) | Freeze features | Inference on test | Inference on test | Final blend on OOF; validate; submit |
| 54–58 (Sun 15–19) | Doc: data insights + examples | Doc: model details | Doc: efficiency numbers | Finish doc, zip code, re-run validator |
| 58–60 (Sun 19–21) | Buffer | Buffer | Buffer | **Final submission by 20:00**; upload doc + zip; screenshot confirmation |

Rules for the round:
- Every run writes OOF + test predictions under one name, so anything can be blended at the end.
- No experiment without a line in the sheet.
- Keep one teammate asleep in every 8-hour block from hour 14.

## (h) AWS Builder Center checklist

Builder Center is Amazon's developer community site. The 2026 registration asks for each member's Builder Center alias.
Sources: `videos/C/notes.md` (the 21 Sep AWS session) and the Unstop page.

Today (22 Sep, before 23:59 IST):
- [ ] Each member: https://builder.aws.com → Sign in → set an **alias**: 3–19 characters, a–z and 0–9, no `@`. "This alias is required for your ML challenge evaluation" `C [00:26:57]`.
- [ ] Team leader enters every alias on Unstop exactly. Invalid alias = incomplete registration.
- [ ] Each member answers "Yes" to "Are you a student?" and verifies student status with a college ID. This unlocks Skill Builder Premium and badge rewards: 7 badges → $10, 14 → $20, 21 → $100 certification voucher `C [00:24:56]`.

Before 25 Sep:
- [ ] At least 2 members create an **AWS Free Tier account**:
  - Needs UPI autopay (up to a ₹15,000 mandate) or a credit card, plus PAN / Aadhaar / driving licence.
  - Gives $100 at sign-up plus $100 for 5 starter tasks ($20 each: launch an EC2 instance, try the Bedrock playground, create a budget, build a Lambda web app, …).
- [ ] Turn on MFA; create a budget alert.
- [ ] Set up **SageMaker Studio** now. Setup failed on a fresh account in the live demo and needed 20–30 minutes to initialise `C [00:57:10]`.
- [ ] **Request a GPU quota increase** (SageMaker ml.g4dn / ml.g5, EC2 G instances). GPUs were never mentioned in the session; the default notebook is CPU only (ml.t3.medium). $200 won't buy many GPU hours, so keep Kaggle/Colab as the main GPUs.
- [ ] Note the $100 bonus for the top 500 at the 48-hour mark.

During the round:
- [ ] The dataset arrives **by email** on 25 Sep `C [00:19:42]`. Upload it to your S3 bucket.
- [ ] Use Batch Transform or notebook inference, not a live endpoint. **Delete endpoints** (≈$0.12/h) and **stop JupyterLab** every night `C [01:39:33]`.

## (i) Open questions to verify yourself

- Round length: the AWS presenter said "exactly 72 hours" `C [00:19:49]`, but Unstop shows 60 h. Plan for 60 h until the 25 Sep email says otherwise.
- How the "free AWS credits for every participant" are delivered: the session only showed the standard Free Tier $200. Ask the organisers.
- Is the round really 60 h (Unstop) or 72 h (older pages)? Is the registration deadline 22 Sep (Unstop) or 20 Sep (Internshala)?
- 2026 model rules: licence list and parameter cap; whether closed APIs (GPT, Claude, Gemini) are banned; whether pretrained weights count as "external data".
- Daily submission limit and public/private LB split.
- Document length (1 vs 2 pages) and format (PDF?).
- Spellings of 2025 team members and some model names come from speech recognition — check the VOD before quoting them.
- Reported 2025 scores are what teams said on stage (mix of validation, public LB and test); they are not official.
