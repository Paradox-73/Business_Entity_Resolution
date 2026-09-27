# Video B — Amazon ML Challenge 2025 Grand Finale, part 2

Source: https://www.twitch.tv/videos/2593971640 (3h10m, 17 Oct 2025). Timestamps are VOD time `[HH:MM:SS]`.
- Most of this VOD is break music: `[00:00]–[00:38]`, `[01:02]–[02:45]` ("Your event will return shortly").
- Real content: Team 9 `[00:40:27]`, Team 10 `[00:49:56]`, jury remarks and **results** `[02:45:23]`–`[03:05:51]`.
- Transcript by Whisper `medium.en`; names are best guesses.

## Team 9 — Royal Recruits (IIIT Delhi) `[00:40:27]`

- Members: Karthikeya Chitkara, Saksham Singh, Siddharth Garg, Swarnima Prasad.
- **Pre-processing** `[00:41:47]`: log1p price; catalog split into name / bullets / description / value / unit; unit standardisation (e.g. "oz" variants); HTML tags and special characters removed; images resized/centre-cropped to 224 for CLIP ahead of time.
- **Model** `[00:42:45]`:
  - "Dual-branch vision": CLIP ViT-L/14 image encoder (semantics — "what is this object") + a small CNN trained from scratch (low-level cues: packaging quality, texture, colour, print resolution).
  - CLIP text encoder for text; learned embedding networks for parsed value and unit.
  - All concatenated → deep MLP with LayerNorm + dropout → log price.
- **Training** `[00:44:28]`:
  - Unfroze only the top 5 layers of CLIP vision and text (trainable 450M → 50M params) — faster and regularising.
  - Differential learning rates: 1e-5 for CLIP layers, 1e-4 for CNN and fusion.
  - Mixed precision (~1.5× faster), gradient accumulation ×2 → effective batch 32; ~6 GB VRAM at batch 16.
  - **Hybrid loss: 10 × MSE(log price) + SMAPE** (also tried 5×MSE+SMAPE) — stability of MSE plus direct push on the metric `[00:46:20]`.
- **Ablation table** `[00:45:37]`: CLIP vision only 41.7 → + CNN + fine-tuning lower → 3→4 unfrozen layers −0.7 → 5 layers + CNN **40.2**.
- Future: distillation to a smaller student; categorise first then regress per group; better text encoder.
- **Judge questions**: none — all three judges said "no questions" `[00:48:58]`. (Did not place.)

## Team 10 — CTRL + ALT + DEV (Amrita Vishwa Vidyapeetham, Coimbatore) `[00:49:56]`

- Members: Vishal Kail(?), Amrita Nandini, S P Saran Darshan, Surya K P.
- Public write-up exists: https://github.com/VishalTheHuman/Amazon-ML-Challenge-2025/ (rank 8, SMAPE 40.777 — see `research/past_solutions.md`).
- **EDA** `[00:52:11]`: all text in English → English-specialised models; mostly grocery; log-normalised price; missing images → plain **white** placeholder (catalog images have white backgrounds).
- **Model** `[00:53:07]`:
  - CLIP ViT-L/14 text (first 64 tokens ≈ item name — less noise than full catalog) and image embeddings (768-d).
  - DistilBERT-base-uncased fine-tuned on the full catalog, two views (unmasked and 6% masked) with a contrastive loss as regulariser (SimCSE-style).
  - Projected to 256-d → "fusion regressor" → log price.
  - Loss: symmetric InfoNCE contrastive (image–text) + Huber, ratio 2:1; LR 2e-5, batch 16 (VRAM-limited), temperature 0.07; ~500M trainable params (ViT ~400M, BERT ~100M).
- **Progression** `[00:55:55]` (partly garbled): LLM-embedding + LightGBM baseline ~57 → small model + engineered features (brand, unit, value, word counts) + TF-IDF + LightGBM ~51.5 → fine-tuned DistilBERT with contrastive loss ~45 (text-only peak; a larger BERT was worse, 48) → CLIP-L + DistilBERT fusion → final ~40.8.
- Inference: 16.8 ms per item; 75k items in ~21 min; no neighbour dependency (each item independent) `[00:58:57]`.
- Future: cross-attention fusion; 8-bit quantisation for edge; decoder (GPT-style) models.
- **Judge questions**: none `[01:00:56]`. (Did not place.)

## Jury remarks before results `[02:46:16]`

- Ajay `[02:47:02]`: price prediction is "a problem that we solve in Amazon day in and day out".
- Arunita `[02:48:50]`: **"what stayed with me is you not only threw a major amount of compute or large pre-trained models at it, but you justified your architecture with data-driven insights. That is going to matter… when we build solutions using real, noisy data."**
- Deepak `[02:50:01]`: **"dive deep is an important part of the ML journey… understand why the results are good, why they are bad, and try out different approaches"**; "here there was a benchmark available, sometimes there won't be."
- Host `[02:52:12]`: the top-10 leaderboard differences were "very minute", so it was hard for the judges to decide.

## RESULTS — 2025 winners `[02:52:32]`

| Place | Team | College | Score they reported | Judge's stated reason |
|---|---|---|---|---|
| **Winner** | Team MESSI | IIT Madras | 40.03 (test) | "holistic view of the problem, thorough experimentation, scalable small model, in-depth insight… the **gluten-free example** stood out… technical rigour… ablation studies" — Deepak `[02:59:58]` |
| **1st runner-up** | Team Zenith | IIEST Shibpur | 41.21 (validation) | "deep experimentation… valuable data insights… you exactly knew what things were giving you what gain… scalability… **the detailed ablation studies stood out**" — Ajay `[02:56:24]` |
| **2nd runner-up** | Team Rocket | IIIT Delhi | 39.83 (validation, text-only) | "Occam's razor… being **frugal** (Amazon leadership principle) about data, compute, modalities… minimal compute led to an efficient solution, what we need for production" — Arunita `[02:53:33]` |

- **Most important takeaway:** the finale ranking did **not** follow the leaderboard.
  - Team Antrix reported 39.2 ("top of the leaderboard") and did not place.
  - Test Data reported ~39.2–39.45 and did not place.
  - The winner reported 40.03; the 1st runner-up reported 41.21.
  - The places went to teams with the clearest **ablation tables**, **data insights** (error analysis like "gluten-free"), and **small, efficient, justified** models.
- Judges tied praise to **Amazon Leadership Principles**: Frugality (Rocket), Working Backwards from the customer (Zenith), Dive Deep (Deepak's advice, MESSI).
- Winners' process comments:
  - Zenith `[02:57:53]`: competed the year before and did badly; this year "planned it through from the beginning", "systematic approach… didn't go for something big… experiments from the ground up"; spent until mid-Sunday only experimenting, then trained the final model to finish Monday. Had a college exam on Saturday (lost ~12 h), slept only 1–2 h naps.
  - MESSI `[03:01:44]`: "expected a lot less data than we had… took a lot of time just **downloading the images**" and processing them; setbacks with scores not improving; "really looking into what we had in the data helped us".
- Prizes (from video A): ₹1,00,000 / ₹75,000 / ₹50,000.
