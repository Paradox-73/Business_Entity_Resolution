# Video A — Amazon ML Challenge 2025 Grand Finale, part 1

Source: https://www.twitch.tv/videos/2593891085 (3h22m, streamed 17 Oct 2025). Timestamps are VOD time `[HH:MM:SS]`.
Transcript made by Whisper `medium.en`; names of people, teams and models are often misheard — spellings below are best guesses, checked against slide OCR where possible.

## Event format (stated by host)

- 82,000+ registrations in 2025, more than 2024 `[00:09:40]`.
- Round 1: 11–13 Oct 2025, 3 days, live leaderboard `[00:13:26]`.
- Top 10 chosen using leaderboard score **plus the approach document and "innovative solutions"** `[00:13:40]`.
- Prizes: winner ₹1,00,000; 1st runner-up ₹75,000; 2nd runner-up ₹50,000, plus certificates/goodies `[00:13:59]`.
- 2025 task: predict product price from catalog text + image `[00:14:26]`.
- **Finale judging criteria (stated):** novelty and approach; intelligent use of data; efficiency and scalability; presentation `[00:15:01]`. No weights were given.
- Format: 10 min presentation (buzzer in chat at 8 min, last 2 min to conclude) + 5 min Q&A `[00:15:16]`.
- Judges ("grand jury") `[00:10:13]`:
  - Deepak Gupta — Senior Manager / Principal Scientist, International Machine Learning (catalog quality, recommendations, payments, image/video generation).
  - Ajay Srinivasa Murthy — Applied Science Manager, AGI team Bangalore (speech/audio), adjunct faculty IIT Hyderabad.
  - Arunita (Anindita?) Das — Senior Applied Scientist, International ML (uncertainty estimation, responsible AI, computer vision).

## Team 1 — Team Antrix (Thapar Institute of Engineering and Technology) `[00:15:37]`

- Members: Aditya Garg, Garima Singla, Akshat Bakshi.
- Final result: **did not place** in top 3 despite the best reported score (see `videos/B/notes.md`).
- Reported score: SMAPE **39.7 → 39.2** (said "top of the leaderboard") `[00:24:27]`.
- **Data processing** `[00:17:00]`:
  - Regex cleaning of text (stray HTML tags).
  - Missing Item Pack Quantity (IPQ) / unit fields filled from bullet points using a zero-shot NER model (said "NuNER Zero") `[00:17:36]`; same model extracted item name and brand.
- **Models** `[00:17:54]`:
  - Three embedding sources: BERT-family text encoder (fine-tuned; "ModernBERT performed best" among ModernBERT/RoBERTa/others `[00:32:11]`), plus two frozen vision-language models, PaliGemma and Qwen (VL) — frozen because of compute.
  - BERT fine-tuned with an MLP regression head on 50% of train, then head dropped and encoder used as embedder `[00:18:38]`.
  - Model choice by benchmark: embed 10% of train with each candidate, put a regression head on top, compare `[00:19:11]`.
  - MLP on concatenated embeddings with residual connection, `log1p` target, a SMAPE-variant loss, dropout `[00:19:32]`.
- **Score progression**: PaliGemma+Qwen MLP 44.25 → + BERT embeddings 42.70 → + graph features 39.7 → + skip connection 39.2 `[00:19:55]`–`[00:24:47]`.
- **Graph convolutional network (their novelty)** `[00:21:27]`:
  - Each product = node; PCA-reduced embeddings; edges to 4 nearest neighbours (FAISS).
  - GCN learns neighbour weights (unlike plain KNN, which weights all neighbours equally) and allows multi-hop context.
  - GCN alone scored 46.85; used only as extra context embeddings concatenated into the final MLP.
- Stayed under the 8B-parameter limit; bigger models would have broken it `[00:20:56]`.
- **Scalability pitch** `[00:24:53]`: encoders as separate EC2 services, embeddings cached in S3, <7B params total, FAISS sublinear search; business use: cold-start pricing for new sellers, flag over/under-priced listings.
- **Judge questions** `[00:26:56]`:
  - Q (Ajay/Arunita): VLMs are not rotation-invariant; what happens if seller images are rotated in production? `[00:27:14]`
    - A: pretrained on augmentations so fairly robust; could add an auto-orientation module. Judge: "think a little bit more on this aspect" — weak answer.
  - Q: Pack-size parity — 5 × pack-of-1 should cost at least a pack-of-5, otherwise arbitrage. How does the model guarantee it? `[00:28:53]`
    - A: normalised price by IPQ (price per gram / per unit), trained on unit price, multiplied back by IPQ at the end `[00:29:35]`.
  - Q (Deepak): PaliGemma/Qwen are heavy; did you try fine-tuning smaller text models (BERT, DistilBERT, T5)? Complexity vs gain? `[00:31:23]`
    - A: tried Florence, ModernBERT, RoBERTa; ModernBERT best on 10% subset.

## Team 2 — Team Zenith (IIEST Shibpur) `[00:33:12]`

- Members: Anshuman Roy, Somyajit Das, Suraj Kashyap, Hanzala Sharik.
- **EDA findings** `[00:34:36]`:
  - Price heavily right-skewed → predict log price; Huber loss.
  - ~80 unit spellings, but weight/volume/count cover 98.6% → standardised to grams / millilitres / count.
  - Features: log(size+1), log(pack+1), log(size/pack+1) (e.g. "150 ml pack of 6").
  - Images mostly 1:1; resized to 448 px; missing image → black image.
- **Model** `[00:37:12]`:
  - OpenCLIP ViT-bigG text and image encoders, L2-normalised, projected to 1024-d; numeric features projected by small MLP to 1024-d.
  - 4 tokens (learned CLS, image, text, numeric) → 6-layer transformer encoder → CLS → regression head (Linear-GELU-Dropout-Linear), target log1p(price), Huber loss; SMAPE tracked.
- **Training in phases** `[00:38:55]`:
  - Phase 1 head-only warm-up (CLIP frozen): SMAPE 52.92.
  - Phase 2 unfreeze last 4 CLIP blocks: −8.6 points.
  - Phase 3 joint fine-tune, very low LR, gradient checkpointing + accumulation: 42.89; full-model with bf16 and smaller batch: 41.96.
  - Final: precompute embeddings, train only regressor 30 more epochs with exponential LR decay: **41.21** validation `[00:41:35]`.
  - Warm-up + cosine LR each phase.
- Tried but dropped (compute/time): image adapter + DeBERTa text branch; mixture-of-experts self/cross-attention branches `[00:42:29]`. Future idea: retrieval (KNN) augmentation `[00:44:15]`.
- Ran over time; host cut them off at `[00:44:22]` — keep to 8 minutes.
- **Judge questions** `[00:44:59]`:
  - Q: Your numeric features (size, pack) come from units — in a general problem they may be missing. How would you handle that? `[00:45:07]`
    - A: kept few numeric features because they were present in almost all rows; pack missing → assume 1 `[00:45:50]`.
  - Q: Did you do ablations on log vs linear target scale? `[00:46:38]`
    - A: linear scale converged much slower (even head-only); showed loss curve.
  - Q: Other multimodal models besides CLIP? `[00:47:48]`
    - A: tried DeBERTa text + CLIP image; no other architectures due to time/GPU.
  - Q (praise): "I really like the slide where you summarized all the experiments and the incremental gains" `[00:48:23]`. Then: memory footprint, model size in GB, inference time, what machine for production? `[00:48:37]`
    - A: CLIP ~2B + head ~100M = 2.1B; a 2×T4 would be enough; trained on Kaggle 2×T4 16 GB; regressor 30 epochs in ~10 min on precomputed embeddings `[00:48:50]`.

## Team 3 — Team Test Data / "TestRadar" (IIT Patna) `[00:50:19]`

- Members: Kanhaiya(?), Anubhav Chandak, Animesh Tripathi, Kaustubh Kumar.
- **Data** `[00:51:36]`:
  - Three duplicate patterns: exact duplicates (same image + text) collapsed by averaging price; same image/different text (or reverse) kept as genuine variants.
  - Price skewness ~13.6, max 2796; trained on log price; clipped outliers above the 99.6th percentile (~0.4% of rows).
- **Staged models** `[00:52:52]`:
  - M1: sentence-transformer text embeddings + XGBoost → SMAPE ~53.78.
  - M2: EfficientNet-B4 image + BERT-base text, multi-head attention fusion → 43.17 (images worth >10 points).
  - M3: fine-tuned OpenAI CLIP ViT-L/14 → 41.97.
  - M4: LAION CLIP ViT-H/14, 1024+1024 → 2048-d concat, deep head (Linear+ReLU+BatchNorm) → **40.55** best single.
  - Final: weighted ensemble of M2/M3/M4, weights by grid search on validation (≈0.10 / 0.35 / 0.55) → **39.45** validation; test ~39.45 single M4 → ~39.2 ensemble (as said in Q&A) `[01:00:36]`.
  - AdamW + warm-up; 15% hold-out (no k-fold — "started setting up k-fold but couldn't complete") `[00:57:08]`.
- Efficiency: 11.37 ms/item for a single model; all models <1B params; async pipeline for new SKUs `[00:56:19]`.
- Tried without gain: EfficientNet-B5 + DeBERTa-v3-large; planned Qwen2-VL-2B but no compute.
- **Judge questions** `[00:59:24]`:
  - Q: What were the ensemble weights? `[00:59:46]` — A: roughly 0.1/0.35/0.55 (didn't know offhand; weak).
  - Q: Which model gives the final result? — A: M4 alone ~39.45; ensemble only 39.2 `[01:00:36]`.
  - Q: Were the encoders fine-tuned? — A: no, frozen embeddings + MLP (this contradicts "fine-tuned" on slides).
  - Q: **Category-level price parity** — two 500 ml milk brands; how do you make sure you don't predict ₹25 for one and ₹100 for the other? Any post-hoc anomaly analysis per category? `[01:01:40]`
    - A: pointed to duplicate handling — judge: "I'm not talking about duplicates… no need to answer now… think this through when you build real-time models" `[01:03:22]`. Weak answer; prepare a post-hoc check (per-category/brand residual analysis, unit-price consistency).

## Team 4 — Team Trailblazers (BITS Pilani) `[01:04:44]`

- Members: Vaibhav Sharma, Asmit Jain, Suyash Handa, Aryan Jain.
- **Approach** `[01:06:27]`:
  - Cleaning: HTML tags (`<br>`), duplicates, outliers, zero/invalid prices.
  - Day 1: text-only — ~300 hand-crafted features → kept 18–30 high-signal ones; MLP regressor; SMAPE stuck at 50–58.
  - Then Qwen2.5-VL-3B-Instruct (Apache-2.0, under 8B limit) fed image + text → single 4096-d embedding ("from the fusion layer") `[01:08:03]`.
  - Concatenated with engineered features → MLP, 5-fold CV, early stopping; fold models averaged.
  - Features: regex brand extraction; item pack quantity, weight, volume (packs of 4/6 are discounted); 8 text statistics (word count, bullet count — "luxury products have richer descriptions") `[01:11:03]`.
  - MLP: residual connections, LayerNorm, Huber loss, AdamW + cosine warm-up, EMA of weights, dropout, weight decay, gradient clipping, early stopping `[01:14:36]`.
- **Score progression**: features-only MLP 50–58 → Qwen embeddings 45.4 → + features 43.1 → EMA/ensembling/scheduler **40.8–40.9** `[01:12:24]`.
- Future work: attention fusion instead of concatenation; stacking with XGBoost/LightGBM; error analysis of worst predictions.
- **Judge questions** `[01:17:02]`:
  - Q: What exactly were the outliers and how did you cap them? `[01:17:24]` — Judge pushed back: "highly priced items are quite natural in e-commerce… not exactly outliers". A: removed per category (e.g. tea/spices ranges), not globally.
  - Q: Which Qwen layer did you take embeddings from? `[01:19:39]` — A: "fusion head" (vague).
  - Q: Products without images — how does prediction suffer? Any ablation removing images from test? `[01:20:11]` — A: black placeholder; text-only was ~50 SMAPE.
  - Q: What were you ensembling? `[01:21:46]` — A: average of 5 fold models (judge seemed unimpressed by textbook explanation).
  - Q: Which handcrafted features had most impact? `[01:22:57]` — A: category, brand, description length/density, pack quantity.
- Ran over time (cut at `[01:16:25]`).
- Break `[01:24:30]`–~`[01:29]`.

## Team 5 — Team Spam LLMs (IIT (ISM) Dhanbad) `[01:39:13]`

- Members: Manav Jain (lead), Prashanth Naidu Karaka, Alok Raj, Rudraksh Joshi.
- **Data** `[01:40:27]`:
  - log1p + normalisation of price; below the 99th percentile mean 21.6 / variance 511; above it mean 223 / variance >20,000.
  - Found same image with different prices and same text with different prices (e.g. a Papyrus card: 21 copies, price $10–$20) → need multimodality.
  - Split catalog text into description / bullet points / value+unit; removed emojis; unit spelling normalisation (grams/gm/g → gm), unknown units → "others"; if no description, used top-5 bullet points; label-encoded units. Brand feature "did not prove useful at all".
- **Model** `[01:43:16]`:
  - Frozen backbones: Qwen3 (4B, text embeddings), SigLIP 2 (2B, image+text), DINOv3 (0.8B, image).
  - Each embedding → own tower MLP → 512-d → concatenated → regression head; MSE on log1p price.
  - MLP 8.9M params × 5 folds (5-fold CV, fold models averaged).
  - Latency: embeddings ~84 ms/sample, MLP head ~30 µs/sample (Kaggle P100/T4×2).
  - Ablation: Qwen3 text-only → + DINOv3 ≈ −2 SMAPE → + SigLIP2 ≈ another −2 `[01:49:15]`.
- **Tried without success** `[01:45:04]`: LightGBM on engineered features + embeddings; FAISS KNN neighbour features (naive average — suspected small dataset); cross-attention fusion (too few experiments); min-max-scaled target + log-loss/BCE (close to best but constrained to training range).
- Future: exponentially distance-weighted KNN with a price-band filter around the predicted price (hyperparameters k, band x, λ, α).
- **Judge questions** `[01:48:38]`:
  - Q: Ablation — why both DINOv3 and Qwen, why not one? `[01:49:00]` — A: gave per-model gains (above); each modality adds.
  - Q: Was the KNN idea implemented or only conceptualised? `[01:50:11]` — A: naive version implemented, weighted one not; post-hoc KNN analysis gave ~1 point. Judge: "It would be good to implement this part and visualise some of the items" (judge liked the idea).

## Team 6 — Team Rocket (IIIT Delhi) `[01:52:42]`

- Members: Angadjit Singh, Parth Rastogi, Abhishek Jha, Harsh Kumar.
- **EDA** `[01:53:55]`:
  - Prices log-normal → log transform (cited Gibrat's law: relative changes more stable).
  - Correlation of handcrafted features with log price: sentence count, word count strongest; binary flags gourmet / bulk / organic / gluten-free; capitalisation ratio.
  - PCA of BERT text embeddings showed price-coherent clusters; CLIP image embeddings were noisy (two look-alike teas, 7× price difference) → **dropped images entirely**, text + features only `[01:56:00]`.
- **Model (441M params)** `[01:56:31]`:
  - Stage 1: DeBERTa-v3-large + linear head, trained on 95% of data, log1p target, Smooth L1 loss → 43.4 validation.
  - Stage 2: handcrafted numeric/binary features → 128-d; cross-attention with DeBERTa CLS (1024-d) as query, features as key/value; DeBERTa unfrozen; 60 epochs Smooth L1 → **39.83** on 5% hold-out.
  - Hyperparameters: AdamW, linear decay, 6% warm-up, max length 128, batch 32, LR 3e-5 `[02:00:05]`.
- **Experiments** `[01:58:38]`: BERT+MLP MSE 49.1 → Smooth L1 48.3 → DeBERTa-v3-large 43.4; differentiable SMAPE loss and pseudo-labelling did not beat fusion; XGBoost/CatBoost on same features ~59 ("overfit, weak on text-heavy regression").
- Pitch: "Occam's razor" — 441M vs 8B params, 18× smaller; trained on 3×H100 `[02:00:37]`. Future: confidence-weighted pseudo-labelling (MC dropout), cross-architecture ensembling, Optuna.
- **Judge questions** `[02:03:20]`:
  - Q: Are you using images? — A: no.
  - Q: Tried newer models like Flan-T5? — A: DeBERTa judged best for NLU.
  - Q (Ajay): Do you have data-driven evidence for dropping images? It may be an artifact of the embedding model. What in the image would help predict price? `[02:04:22]` — A: colours, OCR of packaging text; conceded no clear evidence.
  - Judge (Deepak) summary `[02:07:13]`: liked the data-driven, lightweight approach and the justification, but "it may be an artifact of the CLIP model… since you use cross-attention anyway, attend to image tokens from a better model."

## Team 7 — Team Messi / "MSE" (IIT Madras) `[02:08:26]`

- Members: Aditya Sai, Kriti Shailia, Akshaya Midkar(?), Aravindan Mohanraj — MS students, Data Science & AI dept.
- **Framing** `[02:09:59]`: treat outliers as meaningful, not noise; ~73k items <100, ~2k >100 — a model ignoring expensive items scores OK but is fragile; wanted robustness to bad/missing images.
  - Avoided VLMs/LLMs: hallucinate, non-deterministic, slow to fine-tune, low GPU budget `[02:11:06]`.
- **Features** `[02:11:58]` ("how does a human judge price?"): word/char length (scaled 0–1); item pack quantity, base value (12 oz), total value; log transform; units standardised (kg/mg → g, l → ml); one-hot unit type (grams / ml / count); target log1p.
- **Model** `[02:13:59]`:
  - DeBERTa (text) + Swin Transformer (image) + MLP (numeric), pack quantity as a learned embedding.
  - Text input enriched with "hints": numeric features, item name, brand guess (first word of item name / regex).
  - Concatenate → MLP head. Huber loss; zero image tensor + image-present flag for missing images.
  - Baseline 45.5; plain SMAPE loss worse than Huber (not smooth).
- **LightGBM attempts failed** `[02:15:48]`: LGBM on learned embeddings worse; LGBM on DeBERTa embeddings averaged with main model pulled score down (weaker subset of same patterns).
- **Error analysis** `[02:16:46]`: sweetener actual 34, predicted 36.7; "gluten-free" version actual 109, still predicted ~37 → model can't price modifier words; fix: fine-tune more toward price.
- **Training** `[02:17:33]`:
  - Stratified k-fold on price bins (5 folds, consistent scores), fold-average at test.
  - Two-stage: freeze DeBERTa/Swin and train MLP; then unfreeze last 3 attention blocks.
  - LRs: DeBERTa ~1e-5, Swin ~1e-4, MLP ~1e-3; cosine annealing + warm-up; AdamW; Huber.
  - Baseline test 43.15 → k-fold + EMA of weights **40.18** → horizontal-flip test-time augmentation **40.03** `[02:19:04]`.
- Future: two-stage — classify price <$10 or not, then regress.
- More future work: hyperparameter tuning (relied on intuition); nearest-neighbour price averaging with a cross-encoder over latent features `[02:20:18]`.
- **Judge questions** `[02:20:50]`:
  - Judge praise: "attention to details… made a lot of sense in a real-world environment", image-present flag, rotation-invariance observations `[02:20:57]`.
  - Q: Final model size and memory footprint? — A: ~630M params; 5 folds trained in parallel, ~4.5 h each; used ~35 GB of an 80 GB GPU `[02:21:39]`.
  - Q: How were images used? — A: only ImageNet normalisation into Swin; text carried most signal.
  - Q (Ajay): You claim robustness in the real world — did you test that claim outside the challenge's scope? `[02:23:01]` — A: not yet; explained deliberate choices (kept expensive items even though dropping them would improve the score; null-image tensor + flag).
  - Q: Is the classify-then-regress idea motivated only by the price range, or other reasons? `[02:24:56]` — A: more uniform distribution per bucket; bucket-specific features (brand/luxury cues for expensive items).

## Team 8 — Team Abhimanyu (IIT Jodhpur) `[02:28:01]`

- Members: Rahul Maurya, Vishwas Patel, Om Patel, Shubham Rabukhanolkar(?).
- Camera/audio problems at start (host asked them to turn camera to speaker) — test your AV setup.
- **Features** `[02:29:15]`:
  - Emoji removal (~3,000 rows); log1p price.
  - 73 regex binary keyword features grouped by category: health/diet (keto, gluten-free, sugar-free), allergens (dairy-free, nut-free, soy-free), quality (organic, gourmet, artisanal).
  - Two aggregated features: counts of the top-5 positively and top-3 negatively price-correlated keywords.
  - Pack size, text length, quantity; brand via a small NER model + cleanup; price per unit; outliers removed with 1.5×IQR rule `[02:33:53]`.
- **Model** `[02:36:09]`:
  - SigLIP (~500M) and CoCa (~500M?) embeddings + engineered features + "cross-modal interaction" (image–text match score) → two MLPs, each predicting price; weighted average with weights learned by gradient descent / chosen by loss, "dynamically based on predicted price".
  - 5-fold CV (seed 42), log1p target, mix of L1 and other losses, early stopping. Best **41.08** (said "41.0795") `[02:35:31]`.
  - ~870M params total; ~0.1 s inference per item.
- **Judge questions** `[02:38:42]`:
  - Q (Deepak): How are the ensemble weights W1/W2 learned? — A: gradient descent on the loss.
  - Q: Did you try attention layers instead of MLP? — A: tried fine-tuning a transformer, dropped for GPU limits.
- Lunch break `[02:41:12]` until ~1:25 pm IST; "last two teams" and the award ceremony are in **video B**.

## Judging patterns seen in video A

- Judges repeatedly asked for:
  - **Ablations / evidence** for every design decision (log vs linear; why drop images; why two backbones).
  - **Production concerns**: model size in GB, inference latency, hardware needed, missing images, rotated images.
  - **Business consistency**: pack-size price parity (no arbitrage), per-category price sanity, anomaly checks after training.
  - **Why this model and not a smaller/simpler one** (complexity vs gain).
  - Knowing your own numbers (ensemble weights, per-model scores) without hesitation.
- Explicit praise went to: a slide summarising all experiments with incremental gains; attention to real-world details (image-present flag, robustness); data-driven justification for choices; lightweight models.
- Weak moments: vague answers ("fusion head"), not knowing ensemble weights, answering a different question (duplicates vs category parity), calling high-priced items "outliers", running over 8 minutes.
- All 8 teams in video A used a log / log1p price target. Most used Huber / Smooth L1 rather than raw SMAPE loss.

## Final results (announced in video B `[02:52:32]`)

- Winner: Team MESSI (Team 7 here, IIT Madras) — reported 40.03.
- 1st runner-up: Team Zenith (Team 2, IIEST Shibpur) — reported 41.21 validation.
- 2nd runner-up: Team Rocket (Team 6, IIIT Delhi) — reported 39.83, text-only.
- Antrix (39.2, "top of the leaderboard") and Test Data (~39.2) did not place. Leaderboard score alone did not decide the finale.
