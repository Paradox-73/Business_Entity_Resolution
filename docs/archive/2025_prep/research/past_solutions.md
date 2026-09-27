# Amazon ML Challenge: past problems, top solutions, rules, 2026 format

Compiled 22 Sep 2026 from public GitHub repos, LinkedIn posts, college news pages and Unstop/Internshala listings.

How to read the tags:
- **(unverified)**: the claim appeared only in a search-result snippet, or comes from a single self-reported source that could not be cross-checked.
- **LB**: leaderboard, the live ranking table on the competition site.
- **Public LB**: ranking computed on part of the test set, shown during the contest.
- **Private LB**: ranking on the full test set, revealed after the contest closes.
- **SMAPE**: symmetric mean absolute percentage error. Lower is better.
- **F1**: the harmonic mean of precision and recall. Higher is better.
- **VLM**: vision-language model, a model that takes an image plus a text prompt and outputs text.
- **QLoRA**: a cheap fine-tuning method. The base model is loaded in 4/8-bit and only small added weight matrices are trained.
- **OOF**: out-of-fold predictions, meaning each training row is predicted by a model that did not see that row.

---

## 1. Summary table per year

| Year | Platform | Task | Input data | Metric | Data size | Winning / top score |
|---|---|---|---|---|---|---|
| 2021 | HackerEarth | Classify products into a browse node (Amazon category ID). About 9,919 classes | Text: TITLE, DESCRIPTION, BULLET_POINTS, BRAND | Accuracy (subset accuracy) | Train 2,903,024 rows; test 110,775 | Not found. Rank 16 scored 67.465 on public LB |
| 2022 | — | No 2022 edition found. Repos tagged "2022" contain the 2021 browse-node task | — | — | — | — |
| 2023 | HackerEarth | Predict PRODUCT_LENGTH (a number) from product text | PRODUCT_ID, TITLE, DESCRIPTION, BULLET_POINTS, PRODUCT_TYPE_ID | `score = max(0, 100*(1 - MAPE(actual, predicted)))`. Higher is better | Train 2,249,698 rows; test 734,736 | Not found. Rank 118 scored 35.129 |
| 2024 | Unstop | Read an attribute value plus unit (weight, volume, voltage, wattage, dimensions) from a product image. Output string "x unit" | index, image_link, group_id, entity_name; target entity_value | F1 on exact string match (definitions in section 2) | Train about 263,859 rows (255,906 images); test 131,187 rows (90,666 unique images) | LB #1 NeuralNinjas 0.8655; #2 0.8364; #3 0.8019 (unverified) |
| 2025 | Unstop | Predict product price from catalog text plus one product image | sample_id, catalog_content (title + description + "Item Pack Quantity"), image_link, price (train only) | SMAPE, 0–200%, lower is better | Train 75,000; test 75,000 (public LB used 25k of them) | "Winning score 39.7" (self-reported by the rank-80 team, unverified). Rank 5 had 39.19 on public LB (unverified) |

- Task pattern across years:
  - 2021 was text classification. 2023 was text regression. 2024 was image-to-text extraction. 2025 was multimodal (text plus image) regression.
  - Each year added one modality or changed output type. Every year used Amazon catalog data.
- Metric pattern:
  - Two of the last three years (2023, 2025) used a percentage-error metric.
  - A percentage-error metric punishes errors on small true values much more than errors on large ones.

---

## 2. Per-year sections

### 2025: Smart Product Pricing (Unstop, 11–13 Oct 2025, finale 17 Oct 2025)

**Problem**
- Predict `price` for each product from `catalog_content` text and the image at `image_link`.
- `catalog_content` holds the title, the description and the item pack quantity.
- Source: https://github.com/naveen2200080142/amazon_ml_hackathon_2025

**Metric**
- `SMAPE = (1/n) * Σ |pred − actual| / ((|actual| + |pred|) / 2)`, reported as a percentage from 0 to 200.
- Predictions must be positive floats.

**Data**
- 75,000 train rows and 75,000 test rows.
- Prices ranged from $0.13 to $2,796 (from the rank-11 post).

**Leaderboard split**
- The public LB used 25k test rows.
- The final ranking used the full 75k test rows, plus the quality of the methodology document.
- Sources: https://github.com/naveen2200080142/amazon_ml_hackathon_2025 and https://github.com/sai-123-code/AMAZON_ML_CHALLENGE_2025

**Rules**
- Models had to be MIT or Apache-2.0 licensed and at most 8B parameters.
- Getting prices from the internet, external databases or any other outside source was STRICTLY NOT ALLOWED.

**Deliverables**
- `test_out.csv` with columns `sample_id,price`, matching the format of `sample_test_out.csv`.
- A 1-page methodology document. The Unstop page says 1–2 pages.
- Zipped code.

**Known final standings**
- #3: 00_Team_Rocket (IIIT Delhi; Angadjeet Singh, Abhishek Jha, Harsh Kumar), second runner-up.
  - Search snippet: "40.3% on the public test set" and a "simple, focused, well-engineered approach" (unverified). https://sites.google.com/iiitd.ac.in/iiit-delhi-emailers/2025/sep-oct
- #5: "Test Data" (IIT Patna; Animesh Tripathy, Kanahia, Anubhav Chandak, Kaustubh Kumar).
  - Public SMAPE 39.19 (unverified snippet). Approach not published.
  - https://patnapress.com/iit-patna-amazon-ml-challenge-2025-results/
- #24: "Apex" (IIT Patna). Same source.
- Winner and runner-up (#1, #2): names and approaches not found.

#### Solution entries (2025)

**Rank 8: CTRL + ALT + DEV. Public SMAPE 40.777%.** https://github.com/VishalTheHuman/Amazon-ML-Challenge-2025/
- Final model:
  - Image branch: CLIP ViT-L/14 image encoder, 768-dim output.
  - Text branch: DistilBERT, projected to 256 dims.
  - The two embeddings are normalised, concatenated and passed to an MLP regression head.
  - Both backbones were fine-tuned. Trainable parameters: 501M.
- Loss:
  - Huber regression loss (δ=1.0) with weight 1.0.
  - Plus a CLIP image–text contrastive loss (InfoNCE) with weight 0.20.
  - Plus a SimCSE-style text contrastive loss with 6% word masking, weight 0.10.
- Target: log2(price).
- Training: lr 2e-5, batch 16 with gradient accumulation, 15 epochs, FP16. About 5 hours of training (hardware not stated).
- Missing images: a zero vector was used in place of the image, so no rows were dropped.
- Final submission: the average of the best 2 models.
- Earlier results they reported:
  - Text-only BERT scored 45–49 SMAPE.
  - Frozen (not fine-tuned) embeddings scored 50–57.
  - Classical ML on engineered features scored 51.
- Their claim "all top performers use contrastive learning" is their opinion (unverified).

**Rank 11: Rudr Pratap Singh, Swayam Singh, Antriksh Arya, Prasann.** https://www.linkedin.com/posts/rudr-pratap-singh_rank-11-in-amazon-ml-challenge-2025-just-activity-7386578786881822720-give
- Final model: Qwen3-1.7B (a small LLM) fine-tuned as a text regressor on H100 GPUs.
  - Liger Kernel (fused GPU kernels for training) cut memory use by 60% and made training 20% faster. They called it "the game changer".
- Loss: pseudo-Huber, a smooth version of Huber loss.
  - The loss thresholds were set from the interquartile range of prices.
- Target: raw price, with no log transform.
- Features: pack quantity and pack value extracted from the text.
- Failed experiments:
  - Log target plus MSE loss gave a mismatch with the SMAPE metric.
  - Training directly on SMAPE loss blew up the gradients near zero: a $0.50 item got gradients 100x larger than a $50 item.
- Process: they spent 1.5 of the 3 days on EDA (exploratory data analysis, meaning looking at the data before modelling).

**Rank 17: Team Helios. SMAPE 41.17 (unverified; LinkedIn search snippet only).**
- Baseline: GPU XGBoost.
- Brand extraction: Phi-3-mini, zero-shot.
- Final model: DeBERTa-v3-large fine-tuned for regression with a SMAPE-based objective.

**Top 40: raptor7197.** https://github.com/raptor7197/amazon-ML-school
- Features:
  - Text: TF-IDF, BERT and sentence-transformer embeddings.
  - Image: ResNet and CLIP embeddings.
  - Tabular features.
- Models: LightGBM, neural networks, a cross-modal transformer and a "retrieval regressor". A retrieval regressor predicts from the prices of the most similar training items.
- Validation: K-fold cross-validation.
- Tuning: Optuna. Tracking: MLflow and W&B. Exact SMAPE not given.

**Rank 80 of about 23,000 teams: ML Mavericks (Neel Shah, Sneh Shah, Harsh Maheshwari, Harsh Shah). Final SMAPE 43.28.** https://github.com/NeelDevenShah/Amazon-ML-Challenge-2025
- The README says the winning score was 39.7 (unverified).
- They tried 30+ implementations:
  - LLMs: Granite, Qwen2.5, FLAN-T5.
  - CLIP and Qwen2.5-VL multimodal models.
  - Gradient boosting and FAISS nearest-neighbour search.
  - BERT, T5 and LSTM.
- LLM-based feature extraction produced 15+ fields plus brand. Target was log-transformed.
- Final submission: a text-only hybrid ensemble, notebook `hybrid-ensemble-validation-test-gap-fixes.ipynb`.
- Lessons in the README:
  - "Trust your validation over leaderboard fluctuations."
  - "Biggest gains came from deeply understanding the data before training."

**Rank 476 of 6,500+: Spkap.** https://github.com/Spkap/amazon-ml-challenge-2025-Multimodal-Product-Price-Prediction
- Features (2,450 in total):
  - SigLIP2 image embedding, 1,024 dims.
  - SigLIP2 text embedding, 1,024 dims.
  - Image–text cosine similarity.
  - MiniLM text embedding, 384 dims.
  - 17 hand-made catalog features: quantity, unit, brand, product class, text structure.
- Models: LightGBM, XGBoost, CatBoost and a PyTorch MLP.
- Validation: 5-fold CV with the same folds for every model.
- Ensemble weights: fitted on OOF predictions to minimise SMAPE directly, constrained to be non-negative and sum to 1.
- Predictions clipped to at least 0.01.

**Rank 629 of about 23,000: Ishaan Bhatt.** Quoted in https://www.linkedin.com/posts/khadga-a_amazonmlchallenge-hackathon-activity-7379016384493600768-03PC
- Approach: TF-IDF, quantity parsing and keyword features, fed to LightGBM tuned with Optuna.
- Lesson: "Thoughtful feature engineering > complex deep learning" when compute is limited.

**Rank 985 of 7,100: Optimizers.** https://github.com/sai-123-code/AMAZON_ML_CHALLENGE_2025
- Best result: DistilBERT on text, 49.61 SMAPE.
- A ViT (image-only) model scored 190 SMAPE, which is close to the metric's 200 maximum.
- LightGBM with 111 hand-made features scored 61.
- They also used Phi-3.5-mini on H100.

**About rank 3000: Shubh Mehrotra's team.** Same LinkedIn thread as Ishaan Bhatt.
- They scored 56 during the contest and reached 40 after the contest ended.
- The gain came from moving XGBoost/LightGBM and CuPy to GPU and fixing bugs.
- Lesson: slow iteration and bugs cost them about 16 SMAPE points.

**Not submitted in time: Keshav Khandelwal. SMAPE 48.28.** https://www.linkedin.com/posts/kesav01_amazonmlchallenge-aws-amazon-activity-7384142261049094144-fRzG
- Approach: BERT text embedding plus BLIP-2 image embedding (2,176 dims), combined late in the network with a small head.
- Loss: an "adaptive SMAPE weighting" loss. Target: log price. Validation: 5-fold CV.
- Mistake: missing the submission deadline.
- The same thread mentions Sharish Sasikumar's team: 2-level stacking on MiniLM (384-dim) plus ResNet50 (2,048-dim) features, and a large gap between validation score and LB score.

**Vision-RAG: SunaynaPadhye. No rank given.** https://github.com/SunaynaPadhye/Amazon-ML-Challenge-2025
- Approach:
  - Text: BGE-base embeddings. Image: ViT-B-16 embeddings. Plus 29 numeric features.
  - A learned MLP combines them into one vector, trained so that similarly priced products sit close together.
  - Price is predicted by weighted KNN (k=10) over training items.
- Their CV SMAPE was 10.8%, but their expected test SMAPE was 46–48%.
- That 4x gap is a warning: a KNN method validated this way can find duplicates or leaked neighbours inside the training set.

**Other 2025 repos (ranks not given):**
- https://github.com/Aditya-Ranjan1234/Amazon-ML-Challenge-2025: TF-IDF plus NER, CNN image features, XGBoost/LightGBM plus a neural network.
- https://github.com/vishalkumar-ai25/Amazon-ML-Challenge-2025: repeats the rule that violations mean disqualification.
- https://github.com/uditjain100/Amazon-ML-Challange-2025: word and character TF-IDF, SVD, regex pack sizes, stacked regressors.
- Search snippet (source page not opened; unverified): one team used SigLIP 2 Giant (1B), cross-attention in both directions between text and image, and trained directly on SMAPE.

### 2024: Entity value extraction from product images (Unstop, 13–16 Sep 2024, finale 24 Sep 2024)

**Problem**
- Given a product image and an `entity_name`, output the value with its unit. Examples of `entity_name`: `item_weight`, `width`, `voltage`.
- Output format: `"x unit"`, where x is a float and the unit comes from an allowed list in `src/constants.py`.
- Example outputs: `"2 gram"`, `"12.5 centimetre"`.
- Scientific notation and unit abbreviations were invalid. Output `""` if the value is not present.
- Every submission had to pass `src/sanity.py`.
- Source: https://github.com/hemanthkarthick03/Amazon-ML-Challenge-2024

**Metric**
- GT = ground truth. OUT = prediction.
- TP (true positive): OUT≠"" and GT≠"" and OUT==GT.
- FP (false positive): OUT≠"" and GT≠"" and OUT≠GT, **or** OUT≠"" and GT=="".
- FN (false negative): OUT=="" and GT≠"".
- TN (true negative): OUT=="" and GT=="".
- `F1 = 2PR/(P+R)`, where P = TP/(TP+FP) and R = TP/(TP+FN).
- A wrong non-empty answer counts as an FP, so predicting "" when unsure can raise F1.

**Data**
- Train: 263,859 rows (255,906 images). Test: 131,187 rows (90,666 images). Source: https://github.com/hwaseem04/Amazon-ML-Challenge-2024
- Test images took up more than 52 GB.
- Labels were noisy. The winning team said the labels were likely AI-annotated.

**Leaderboard and finale results**
- LB top 3 (unverified snippet):
  - NeuralNinjas (IIT Jodhpur) 0.86545549.
  - BuffJezos (IIT Patna) 0.83644522.
  - fambruh (NSUT) 0.80186399.
- Finale result (unverified snippet): 1 NeuralNinjas, 2 "ART in Artificial Intelligence" (NSUT), 3 BuffJezos.
  - BuffJezos was #2 on the LB but finished 3rd at the finale. If true, the finale presentation can change the order.

#### Solution entries (2024)

**Rank 1: NeuralNinjas (Alli Khadga Jyoth, Kushal Agrawal, Nachiketa Purohit, Ritu Singh; IIT Jodhpur). F1 0.865.** https://github.com/KhadgaA/Amazon-ML-Challenge (same team: https://github.com/nachiketashunya/Amazon-ML-Challenge-2024). Finale slides PDF read in full: `ML Challenge 2024_NeuralNinjas.pdf` in that repo.
- Base model: Qwen2-VL-7B-Instruct. Chosen because it was pre-trained on VQA and OCR data and handles images at their native resolution.
  - They rejected an OCR+NER pipeline because it cannot read unlabeled values or use where text sits on the image.
- Stage 1: QLoRA 8-bit fine-tuning on 20k samples with LLaMA-Factory.
  - 3 epochs, batch 8, gradient accumulation 8, cosine schedule.
- Stage 2: fine-tuning on **1,600 hand-curated samples**.
  - 30 epochs, batch 4, gradient accumulation 4, ReduceLROnPlateau schedule.
  - Curation meant fixing wrong labels and setting the label to "NA" when the value is not visible in the image.
- Prompt: `What is the {entity_name}?`. Learning rate 5e-5.
- Preprocessing:
  - Labels with invalid units (for example horsepower) were set to NA.
  - Ranges were replaced by their maximum ("[24,30] volt" became "30 volt").
- Post-processing: invalid units were fixed and NA was turned into "".
- F1 by stage:
  - Zero-shot base model: 0.617.
  - Fine-tuned on 10k samples: 0.678.
  - Fine-tuned on 20k samples: 0.679.
  - Plus the curated 1.6k stage: **0.865**.
- Compute: **2× A100 40GB**. Inference about 0.6 s per sample using 8-bit quantisation.
- Main lesson: doubling data from 10k to 20k gained 0.001 F1. Fixing 1,600 labels gained 0.186.

**Rank 6 at finale: DBkaScam (Arnav Goel, Medha Hira, Mihir Aggarwal, AS Poornash). F1 71.8.** https://github.com/arnav10goel/Amazon-ML-Challenge-24
- Final submission: a voting ensemble of three parts.
  - MiniCPM-V-2.6 zero-shot with a structured prompt.
  - Qwen2-VL-7B with dynamic few-shot examples. The examples were chosen by product category from a list of frequent errors.
  - Qwen2-VL-7B fine-tuned with QLoRA 8-bit via LLaMA-Factory on 150k samples for 1 epoch.
  - InternVL2-8B was tested but not kept.
- Post-processing rules:
  - Regex conversion of fractions and decimals.
  - Quote marks turned into foot and inch units.
  - Rule-based choice of one value from a range.
  - Footnote symbols removed.
- F1 progression:
  - Zero-shot: 66.2.
  - Plus post-processing: 69.3 (+3.1 from rules alone).
  - Few-shot: 70.9.
  - Ensemble: 71.8.

**Rank 15: Akhil Gupta's team. F1 0.689 (unverified snippet).**

**Rank 18: IITians_In_De_North.** https://github.com/Stormbreakerr20/amazon-ml-challenge-2024
- Each entity type went to a different small model:
  - MoonDream2, fine-tuned, for width and height.
  - InternVL2-1B for OCR (reading text in the image).
  - Phi-3-Vision for reasoning.
- Regex parsing of the model outputs.

**Top 20: Piyush Aggarwal.** https://www.linkedin.com/posts/aggarwal-piyush_amazonmlchallenge-amazon-machinelearning-activity-7243358458333274112-qO_3
- Seen in search results only. Details not read.

**Rank 45 of 18,000+ teams: Muhammad Waseem et al. F1 0.616.** https://github.com/hwaseem04/Amazon-ML-Challenge-2024
- Model: Idefics-2 8B, fine-tuned with a custom prompt template.
- Augmentation: image rotation. Images larger than 980×960 were resized.
- Data cleaning: noisy rows removed and units standardised.
- F1 progression: zero-shot 0.44, basic fine-tuning 0.56, final 0.616.
- Post-processing: upper bound of ranges, invalid units filtered out.
- Inference was split across several parallel `tmux` sessions (terminal sessions left running), which cut it from 14 h to 4 h.

**Rank 64: ML Mavericks.** https://github.com/Sneh-T-Shah/Amazon-ML-Challenge-2024
- Model: MiniCPM-Llama3-V-2.5, mostly used without training, plus unit-normalising post-processing.
- Compute: inference spread over **15+ Kaggle GPUs**, at 2–3.5 s per image.

**Rank 172: Debopam Chowdhury et al. F1 0.47.** https://www.linkedin.com/posts/debopam-chowdhury-param-600619229_amazonmlchallenge-hackathon-machinelearning-activity-7243330346522447872-WeAR
- Approaches in order:
  - Tesseract OCR + Gemma-2 failed because the OCR text was poor.
  - YOLO + OCR + Llama-3 took about 2 minutes per sample.
  - Qwen2-VL-2B took 1.5 s per sample.
- They split inference across 53 free Colab accounts and still predicted only 84k of the 131k test rows.
- Lesson: they found the VLM approach too late.

**Rank 357 of 74,850: Aman Prakash.** https://medium.com/@aman_prakash/how-we-secured-357th-spot-among-74-850-in-amazon-ml-challenge-2024-aef31f4e31b0
- The page returned 403. Search snippet: top-10 teams planned GPU use better and rented A100/H100 on Runpod and Lightning AI (unverified).

**Rank 503: Spartan-71. F1 0.03.** https://github.com/Spartan-71/Amazon-ML-Challenge-2024
- Approach: OCR plus regex. Inference took more than 20 h.
- Shows how badly plain OCR did on this task.

**Label conflict (unverified)**
- A LinkedIn snippet lists "CTRL + ALT + DEV AIR 8, F1 0.628" for 2024.
- The same team name is rank 8 in 2025. It may be the same team two years running, or a mislabelled snippet.

### 2023: Product length prediction (HackerEarth, registration closed 20 Apr 2023)

**Problem**
- Predict PRODUCT_LENGTH from TITLE, DESCRIPTION, BULLET_POINTS and PRODUCT_TYPE_ID.

**Metric**
- `score = max(0, 100*(1 − MAPE))`, where MAPE is mean absolute percentage error. Higher score is better.

**Data**
- Train 2,249,698 rows (6 columns). Test 734,736 rows.
- Submission: CSV with columns `PRODUCT_ID, PRODUCT_LENGTH`.
- Source: https://github.com/Marinto-Richee/Amazon-ML-Challenge-2023 and https://github.com/VectorNd/Amazon-ML-Challenge-2023

**Winners**
- An Amazon LinkedIn post lists the top 3 colleges as NSUT, IIT (ISM) Dhanbad and DTU (unverified; team names not given). https://www.linkedin.com/posts/amazon_amazon-ml-challenge-2023-activity-7057361786227875841-7r4u
- About 25,000+ students took part.
- Winning score not found. The HackerEarth LB page returned an error.

#### Solution entries (2023)

**Self-reported 2nd of about 7,000 teams: Team Fishes (Tashvik Dhamija, Pranav Balaji, Adarsh Jha).** https://github.com/greenfish8090/AmazonML
- Input text: all text fields joined into one string.
- Target: log of the length, clipped at 12.
- Stage 1: frozen BERT embeddings fed to a small neural network.
- Stage 2: BERT and RoBERTa fine-tuned end to end.
  - A learned embedding for each PRODUCT_TYPE_ID was trained jointly with the model.
- Ensemble:
  - Each prediction was rounded to the nearest value that appears in the training set.
  - Then the **minimum** of the BERT and RoBERTa predictions was taken.
- Why the minimum works: MAPE punishes over-prediction without limit but under-prediction by at most 100%, so erring low is safer.
- Conflict: the "2nd" claim does not match the college list above (unverified).

**Rank 118: VikramxD. Score 35.129.** https://github.com/VikramxD/ML-Challenge
- Approach: scikit-learn gradient boosting on engineered features.

**No rank: Ash469. About 49.33% MAPE.** https://github.com/Ash469/AmazonMLChallenge
- Features: word TF-IDF (1–2 grams) and character TF-IDF (3–5 grams).
- Models: Ridge regression on text, LightGBM/XGBoost/CatBoost on structured features, then combined.
- Outlier trimming: 2% of rows at the low end, 1% at the high end.
- Lesson: character n-grams caught measurement strings such as "12cm" that word tokens missed.

**No rank: VectorNd.** https://github.com/VectorNd/Amazon-ML-Challenge-2023
- Model: multi-input network with a GRU/LSTM on word and character tokens plus a product-type embedding.
- Target: log, then PowerTransformer.

### 2021 (brief): Browse-node classification (HackerEarth, launched 30 Jul 2021)

**Problem and data**
- Classify products into about 9,919 browse nodes using TITLE, DESCRIPTION, BULLET_POINTS and BRAND.
- Train 2,903,024 rows. Test 110,775 rows.
- Participation: 12,000+ registrations, 7,000+ submissions.

**Winners**
- 1st no_bert_train_challenge (IIT ISM Dhanbad), 2nd DeVaSh.Ai (IIT Guwahati), 3rd Patanjali Noodles (IIT ISM).
- The finale was streamed on Twitch.
- Source: https://www.aboutamazon.in/news/job-creation-and-investment/amazon-ml-challenge-a-unique-upskilling-opportunity-to-tackle-a-real-world-problem
- Top-10 finale teams per a search snippet (unverified): N00bs, no_bert_train_challenge, CL_everywhere, Tyche, Hungry For Gold, code_ml, bert_hi_train_kiya, DeVaSh.Ai, Patanjali Noodles, PERO_CODERS.

**Solution entries (2021)**
- **Rank 16 of 3,000+: team Panaroma. Public LB 67.465.** https://github.com/nikhil6041/AmazonMLChallenge2021
  - Multilingual BERT and XLM-RoBERTa fine-tuned end to end for 3 epochs.
- **Rank 193 of 3,300: av1paul. Accuracy 63.36%.** https://github.com/av1paul/Product-Browse-Node-Classification---Amazon-ML-Challenge-2021
  - Model: SVM.
- **kavanpatel18. Accuracy 66.85%.** https://github.com/kavanpatel18/amazon-ml-challenge
  - Sentence-transformer embeddings, GPU KNN and majority vote.

---

## 3. Recurring official rules (seen in 2 or more years)

- **Format**
  - A 2-stage competition: a 3-day (about 72 h) online ML round, then a virtual finale about 1–3 weeks later.
  - At the finale, the top 10 teams present to Amazon scientists. Years seen: 2021, 2023, 2024, 2025.
- **Finale selection**
  - Based on LB rank plus the Round-1 methodology document (2025 and 2026 pages).
  - 2025 also said the final evaluation used the full test set plus documentation quality.
- **Team rules**
  - 3–4 members in 2021–2025. The 2026 Unstop page says 2–4; Internshala says 3–4.
  - One designated leader. Cross-college teams allowed. A student may be on only one team.
- **Eligibility**
  - Full-time engineering students in India (B.E./B.Tech/M.E./M.Tech/M.S./PhD).
  - Graduating in the next two years: 2025/26 for the 2024 edition, 2026/27 for 2025, 2027/28 for 2026.
- **Deliverables**
  - A predictions CSV in the exact sample format, with an ID column and a prediction column.
  - A **1–2 page approach document**. The 2025 problem repo says "one-page".
  - Code, script or notebook as a zip.
- **Model limits** (explicit in 2025)
  - The **MIT or Apache-2.0 licence only**. Llama-licence and CC-BY-NC models are excluded.
  - At most **8B parameters**.
  - 2024 finalists used Qwen2-VL-7B, MiniCPM-V, InternVL2-8B and Idefics2-8B, which suggests an 8B-class limit then too (unverified).
- **External data**
  - Getting target values (prices) from the internet, external databases or any outside source is strictly not allowed. Violation means disqualification (2025).
  - Pretrained open models are allowed.
- **Leaderboard**
  - The live public LB is computed on a subset (2025: 25k of 75k = 33%).
  - The private LB on the full test set is revealed after the contest.
  - 2024: the public/private split was not found.
- **Submission format checks**
  - 2024 shipped `sanity.py` and required it to pass.
  - 2025 required positive floats for every `sample_id`.
- **Daily submission limit**
  - **Not stated in any page read.** Check the Unstop problem page on Day 1.
- **Prizes** (2024, 2025 and 2026 are identical)
  - ₹1,00,000 / ₹75,000 / ₹50,000 for the top three at the finale.
  - PPIs (pre-placement interviews) for the Applied Scientist Intern role for the top 50 teams.
  - Certificates and merchandise for the top 10.

---

## 4. 2026 format and dates

Sources:
- https://unstop.com/hackathons/crp-amazon-ml-challenge-2026-amazon-1743604 (read through the `/amp` version; the main page needs cookies).
- https://internshala.com/competitions/amazon-ml-challenge-2026-win-%E2%82%B9225000/

**Registration**
- Opened 7 Sep 2026.
- Deadline:
  - Unstop says **22 Sep 2026, 11:59 PM IST**, which is today.
  - Internshala says 20 Sep 2026.
  - Treat Unstop as current.
- About 83,353 people were registered at the time of reading.

**AWS Builder Center requirement**
- "Before registering for the ML Challenge 2026, participants are required to create their AWS Builder Center Profile ID."
- Registration asks for the Builder Center alias. "Participants with an invalid alias will have an incomplete registration."
- **Check that every teammate's alias is valid.**
- No separate student-ID verification step is mentioned on either page (unverified whether one exists).

**Team rules**
- 2–4 members per Unstop, 3–4 per Internshala.
- One leader. Cross-college teams allowed. One team per student.

**Eligibility**
- Full-time engineering students in India, graduating in 2027 or 2028.

**ML round timing**
- 25 Sep 2026, 9:00 AM IST to 27 Sep 2026, 9:00 PM IST.
- The Unstop round page gives the same window in UTC: 25 Sep 03:30 to 27 Sep 15:30, with duration "2 days 12:00:00".
- **That is 60 hours, not 72.** Plan for 9 PM IST on 27 Sep as the hard stop.
- The problem statement and dataset are released on Day 1.
- Deliverables (same as past years):
  - Predictions file.
  - 1–2 page approach document.
  - Zipped code.

**AWS credits**
- $200 for every registered participant.
- A further $100 for the **top 500 teams at the 48-hour mark**. That is about 9 AM IST on 27 Sep if counted from the start. Rank on the public LB at that point matters.

**After the round**
- Results: 2 Oct 2026.
- Grand Finale: 7 Oct 2026, 10:00 AM–3:00 PM IST, virtual. The top 10 teams present to Amazon scientists.

**Prizes**
- Winner ₹1,00,000. First runner-up ₹75,000. Second runner-up ₹50,000. All three also get certificates and merchandise.
- Top 50 teams get PPIs for the Applied Scientist Intern role.
- Top 10 teams and the top 10 women-only teams get merchandise and certificates.

**Other**
- Unstop lists a "best practices" session with a live demo ("Train, deploy & test an ML model"), probably on AWS.
- Not stated anywhere: daily submission cap, public/private split, model licence and size limits for 2026. Expect them in the Day-1 problem statement.

---

## 5. Cross-year patterns in top solutions

**Sample used for counting**
- 11 solutions ranked in the top 100 whose method is published:
  - 2025: #8, #11, #17, top-40, #80.
  - 2024: #1, #6, #18, #45, #64.
  - 2023: #2.
- The 2021 rank-16 solution is added as a 12th where it applies.
- The 2025 #3 and #5 teams did not publish methods.
- Counts are how many of these 12 used each technique.

| # | Technique | Count (of 12) | Which solutions |
|---|---|---|---|
| 1 | Fine-tune a pretrained transformer or VLM end to end. Frozen embeddings plus a small head was not enough | **9** | 2025 #8, #11, #17; 2024 #1, #6, #18, #45; 2023 #2; 2021 #16 |
| 2 | Rule or regex post-processing of outputs: unit fixes, range handling, snapping to training values, clipping | **7** | 2024 #1, #6, #18, #45, #64; 2023 #2; 2025 #8 (log inverse; others clip) |
| 3 | Ensemble of 2 or more models or prompts: averaging, voting, min, stacking | **6** | 2025 #8, top-40, #80; 2024 #6, #18; 2023 #2 |
| 4 | Hand-made features parsed from text (pack quantity, unit, brand), often extracted by a small LLM or regex | **5** | 2025 #11, #17, top-40, #80; 2023 #2 (product-type embedding) |
| 5 | Loss or target chosen to fit the metric: Huber or pseudo-Huber, log target, SMAPE-aware objective, min-ensemble for MAPE | **5** | 2025 #8, #11, #17, #80; 2023 #2 |
| 6 | Cleaning or curating training labels | **3** | 2024 #1 (the key step), #45; 2023 #2 (clipping) |
| 7 | QLoRA or other low-cost fine-tuning of a 7B model with LLaMA-Factory | **2** | 2024 #1, #6 |
| 8 | Image and text fused in one model (2025 only) | **2 of 5** from 2025 | #8, top-40. Text-only: #11, #17, #80 |
| 9 | Prompt engineering or few-shot examples for a VLM | **3** | 2024 #1, #6, #45 |
| 10 | Gradient-boosted trees (LightGBM/XGBoost) in the final model | **1** (top-40). Used as a baseline by #17 | Common at ranks 400–3000 (#476, #629, #985) |
| 11 | Paid or multi-GPU compute: A100/H100, many Kaggle or Colab GPUs in parallel | **4+** | 2024 #1 (2×A100), #64 (15 GPUs); 2025 #11 (H100); snippet about top-10 on Runpod/Lightning |

**Observations**
- In every year, the top solution fine-tunes the strongest open model that fits the licence and size limits.
- Top teams had strong models and clean data.
  - 2024 #1: "simple method with a powerful model > complex method with a simple model".
- In 2025, text-only models reached ranks 11, 17 and 80.
  - Images added a small gain: rank 8 reported multimodal beating text-only by roughly 5–8 SMAPE points.
  - Image-only failed badly (190 SMAPE).
- For percentage-error metrics, 3 of the 4 teams that reported loss experiments found that plain MSE on a log target was not the best choice.
  - What worked: Huber or pseudo-Huber loss, a SMAPE-aware objective, or biasing predictions low.
- Classical ML (TF-IDF plus GBDT) reached about rank 600 in 2025.
  - It is a fast baseline and ensemble member, not a winner.

---

## 6. Mistakes and lessons reported by teams

- **Label noise matters more than data volume.**
  - 2024 #1: going from 10k to 20k samples gave +0.001 F1. Fixing 1,600 labels gave +0.186.
  - Khadga (2024 #1) tip for 2025: first check whether labels are human- or AI-annotated.
- **Mismatch between loss and metric** (2025 #11).
  - Log target with MSE did not optimise SMAPE.
  - Raw SMAPE loss had exploding gradients on cheap items. Pseudo-Huber on the raw price worked best.
- **Validation that leaks gives a false score.**
  - SunaynaPadhye's KNN retrieval showed 10.8 CV vs about 47 on test.
  - Sharish Sasikumar's team reported a big gap between validation and LB.
  - Rank 80: "trust your validation over leaderboard fluctuations".
  - Rank 80's final notebook was named for "validation-test gap fixes".
- **Public LB ≠ final result.**
  - 2025 public LB was 25k of 75k rows.
  - Reported public scores (39.19 for #5, 40.3 for #3, 40.777 for #8) do not follow the final order.
  - In 2024, the #2 team on the LB finished 3rd at the finale (unverified).
  - The document and presentation count.
- **Inference time can be the real bottleneck.**
  - 2024: one team could only predict 84k of 131k rows despite using 53 Colab accounts.
  - Another team needed 20 h for OCR.
  - #45 used parallel tmux sessions to go from 14 h to 4 h.
  - Estimate time per sample × test rows in the first hours.
- **Slow iteration costs rank.**
  - About rank 3000 (2025) reached 40 SMAPE after the contest (from 56) just by moving to GPU and fixing bugs.
- **Missed deadline.**
  - One 2025 participant with 48.3 SMAPE did not submit in time.
- **Picking the approach too late.**
  - 2024 #172 said finding VLMs earlier would have put them in the top 50.
- **EDA pays.**
  - 2025 #11 spent 1.5 days on EDA and ranked 11th.
  - Rank 80 said the "biggest gains came from deeply understanding the data".
- **Tooling for memory and speed** (Liger Kernel, 8-bit QLoRA, AWQ quantisation) let top teams fine-tune bigger models on the GPUs they had.
- **Output-format rules cost points.**
  - 2024: exact string match on "x unit". Rule-based post-processing alone added 3.1 F1 points for #6.

---

## Sources opened or seen (by year)

- 2025:
  - naveen2200080142/amazon_ml_hackathon_2025
  - NeelDevenShah/Amazon-ML-Challenge-2025 (plus raw README)
  - VishalTheHuman/Amazon-ML-Challenge-2025 (plus raw README)
  - Spkap/amazon-ml-challenge-2025-Multimodal-Product-Price-Prediction
  - raptor7197/amazon-ML-school
  - sai-123-code/AMAZON_ML_CHALLENGE_2025
  - SunaynaPadhye/Amazon-ML-Challenge-2025
  - Aditya-Ranjan1234/Amazon-ML-Challenge-2025
  - vishalkumar-ai25/Amazon-ML-Challenge-2025
  - uditjain100/Amazon-ML-Challange-2025
  - LinkedIn posts: Rudr Pratap Singh (rank 11), Khadga A (tips thread), kesav01
  - patnapress.com (IIT Patna ranks)
  - IIIT-Delhi emailers page
  - unstop.com 2025 `/amp`
  - github.com/topics/amazon-ml-challenge-2025
- 2024:
  - KhadgaA/Amazon-ML-Challenge (plus finale PDF)
  - nachiketashunya/Amazon-ML-Challenge-2024
  - arnav10goel/Amazon-ML-Challenge-24
  - Stormbreakerr20/amazon-ml-challenge-2024
  - hwaseem04/Amazon-ML-Challenge-2024
  - Sneh-T-Shah/Amazon-ML-Challenge-2024
  - Spartan-71/Amazon-ML-Challenge-2024
  - hemanthkarthick03/Amazon-ML-Challenge-2024
  - LinkedIn: Khadga A (win post), Debopam Chowdhury
  - unstop.com 2024 `/amp`
  - Aman Prakash Medium (403; snippet only)
- 2023:
  - greenfish8090/AmazonML
  - VikramxD/ML-Challenge
  - VectorNd/Amazon-ML-Challenge-2023
  - Ash469/AmazonMLChallenge
  - Marinto-Richee/Amazon-ML-Challenge-2023
  - Amazon LinkedIn posts (2023)
- 2021:
  - aboutamazon.in article
  - nikhil6041/AmazonMLChallenge2021
  - av1paul and kavanpatel18 repos (search results only)
- 2026:
  - unstop.com 2026 `/amp`
  - internshala.com 2026 page
  - Unstop round page (search snippet for the UTC window)
