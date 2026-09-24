# TEAM PLAN — Amazon ML Challenge 2026

One page to run the team. Details and reasons are in `PLAYBOOK.md`.

Round: **Fri 25 Sep 09:00 IST → plan for a hard stop at Sun 27 Sep 21:00 IST (60 h)**. If the email says 72 h, the extra 12 h is a bonus.

---

## 1. Roles

| Person | Hardware | Role | Owns |
|---|---|---|---|
| **A — Kanav (captain)** | RTX 3050 4 GB laptop + Kaggle | Data, features, integration | Data download, `folds.csv`, features, LightGBM/CatBoost, **blending, every submission**, experiment sheet, runs Claude Code |
| **B — Gupta** | RTX 4060 8 GB laptop + Kaggle | Images | Image download, image embeddings (CLIP/SigLIP), image fine-tuning on Kaggle |
| **C — Gathik** | RTX 3050 Ti 4 GB laptop + Kaggle | Text | Text embeddings, DeBERTa/DistilBERT fine-tuning on Kaggle |
| **D — Bhavya** | No GPU; Kaggle only | Fusion + write-up | Fusion model on Kaggle (text + image + features); error analysis; **methodology document**; finale deck draft |

- One owner per task. If two people need the same thing, the owner does it and shares the file.
- **Only A submits.** Nobody else uploads to the leaderboard. This avoids wasted submissions and mixed-up files.
- Kaggle GPU budget: about 30 h per person per week (Kaggle shows your remaining hours). Across 4 people that is ~120 h. Spend at most ~8 h per person per round day. Run long jobs overnight.
- AWS: backup only, unless the 25 Sep email requires it. Whoever has the new-account $200 credit keeps it untouched until Sunday.

---

## 2. Before 25 Sep (checklist per person)

**Everyone (by Wed 23 Sep night):**
- [ ] Builder Center alias set and registration confirmed on Unstop (deadline 22 Sep 23:59 IST).
- [ ] Kaggle account, **phone-verified** (without it there is no GPU). Settings → turn on "Phone verification".
- [ ] GitHub account; accept the team repo invite from A.
- [ ] Join the team WhatsApp/Discord group with 3 channels (or pinned topics): `#results` (scores only), `#blockers`, `#general`.
- [ ] Read `SUMMARY.md` (5 min) and your own section of this file.

**A, B, C (the laptops with GPUs):**
- [ ] Update the NVIDIA driver to the latest (NVIDIA App or GeForce Experience). Check with `nvidia-smi`: "CUDA Version" should say 12.x or higher.
- [ ] Install Python 3.11 or 3.12 and Git.
- [ ] Install PyTorch with GPU support and check it sees the GPU:
  ```
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  ```
  It must print `True` and the GPU name. If not, paste the output to Claude.
- [ ] Post "GPU ready" in `#general`.

**A only:**
- [ ] Create a **private** GitHub repo `amlc2026`, push the contents of this folder, invite B, C and D. (Claude can do the local git part: ask "set up the git repo".)
- [ ] Create the shared experiment sheet (Google Sheets), columns:
  `id | time | who | model_name | change vs previous | CV score | CV std | public LB | GPU & minutes | notes`

**D only:**
- [ ] Open a Kaggle notebook, turn on Accelerator → GPU T4×2, run `!nvidia-smi` to confirm it works.
- [ ] Copy `TEMPLATE_methodology_doc.md` into a shared Google Doc; fill in team names and headings now.

**Thu 24 Sep: practice run (2 hours, everyone):**
- [ ] A gives Claude a public Kaggle dataset with text + images + a numeric target; everyone follows the Hour 0–3 protocol below exactly. Goal: a valid submission file in 2 hours. Fix whatever broke.
- [ ] Sleep early. The round starts at 09:00.

---

## 3. Friday 25 Sep: the first 6 hours

| Time (IST) | A (captain) | B (images) | C (text) | D (fusion / docs) |
|---|---|---|---|---|
| 08:45 | Laptop on, Claude Code open in the project folder | Online | Online | Online |
| **09:00–09:20** | Paste the problem email + rules into Claude (prompt in §5). Get the **rules check** and the plan | Read the problem statement | Read the problem statement | Read the problem statement; start the doc's "Problem" section |
| 09:20 | **Team call (10 min):** A reads Claude's rules check aloud: metric, allowed models, AI-tool rule, submission limit, deadline. Confirm roles | | | |
| 09:30–10:30 | Download data; upload it as a **private Kaggle Dataset** shared with B, C, D; make `folds.csv`; push it to the repo | Start the **image download** at once (script from Claude at hour 0); report the speed in `#general` | Text EDA: lengths, language, fields, missing values | Open the Kaggle Dataset in a notebook; check it loads; plot the target |
| 10:30–12:00 | Features + TF-IDF/Ridge baseline (a simple text model). **First submission by 12:00** | Image embeddings with a small CLIP on the laptop (4060) | Text embeddings (MiniLM) + first DeBERTa fine-tune on Kaggle (fold 0 only) | Write EDA findings (charts + 3 insights) into the doc |
| 12:00 | **Status call (5 min):** everyone posts their CV score in `#results` | | | |
| 12:00–15:00 | LightGBM on features + embeddings; blend; second submission | Full image embeddings; start a CLIP fine-tune on Kaggle | Full 5-fold DeBERTa on Kaggle | Fusion model v1 on Kaggle using B's and C's embeddings |
| 15:00 | **Status call.** Decide the main direction for the night | | | |

After hour 6, follow the 60-hour grid in `PLAYBOOK.md` section (g).

---

## 4. Rules that prevent chaos

1. **Same folds for everyone.** Only use `folds.csv` from A. Different folds = models can't be blended.
2. **Every run saves two files** in `artifacts/<model_name>/`: `oof.csv` (predictions for training rows) and `test.csv` (predictions for test rows). Claude writes this into every training script. Upload them to the shared Drive folder `predictions/`.
3. **Naming:** `<who>_<model>_<version>`, e.g. `C_debertaV3small_v2`. Never reuse a name.
4. **Every run gets one line** in the experiment sheet before you start the next one.
5. **Only A submits**, only after Claude's submission check prints PASS (row count, ids, no blanks).
6. **Status calls** at 12:00, 15:00, 19:00 and 23:00 each day, 5 minutes max: each person says score, what's next, what's blocked.
7. **Blocked for more than 20 minutes?** Post in `#blockers`. A asks Claude.
8. **Sleep shifts** from Friday night: at least 2 people awake and 1 GPU job running at all times. Nobody goes more than 20 h without sleep.
9. **Freeze times:** no new model ideas after **Sun 15:00**; final blend by **Sun 19:00**; final submission by **Sun 20:00**; document and code zip uploaded by **Sun 20:30**.

---

## 5. How to use Claude during the round

A runs Claude Code on the laptop in this folder (`claude`). Everyone sends questions and errors to A, or runs Claude on their own laptop in their copy of the repo.

**Hour-0 prompt (A pastes this at 09:00 with the problem email):**
> The Amazon ML Challenge 2026 problem is below. Read TEAM_PLAN.md, PLAYBOOK.md and research/past_solutions.md. Then:
> 1. Rules check: metric (with formula), allowed model licences and sizes, external-data rule, AI-assistant rule, submission limit, deadline, deliverables.
> 2. Is this like a past year's task? Which winning recipe applies?
> 3. Write the shared code: data loading, image download, `folds.csv`, the metric, a baseline, and a submission checker.
> 4. Give each of A, B, C, D their first 3 hours of tasks with exact commands.
> [paste problem statement + data description + rules]

**During the round, ask Claude things like:**
- "Here is C's error from Kaggle: [paste]. Fix it."
- "Write a Kaggle notebook for B that fine-tunes CLIP ViT-L/14 on 5 folds with our folds.csv."
- "Blend everything in predictions/ and tell me the best weights and CV score."
- "Show the 20 worst predictions and what they have in common." (error analysis, which won the finale in 2025)
- "Update the methodology doc with the latest ablation table."
- "It's 23:00 Friday. What should each person run overnight?"

**Check first:** if the rules ban AI assistants, stop using Claude for the round itself (preparation is still fine).

---

## 6. Deliverables tracker

| Deliverable | Owner | Due |
|---|---|---|
| Baseline submission | A | Fri 12:00 |
| EDA findings in the doc | D | Fri 15:00 |
| Ablation table (kept current) | A → D | continuous |
| Final submission (validated) | A | Sun 20:00 |
| Methodology document (1–2 pages, PDF) | D, reviewed by A | Sun 20:30 |
| Code zip with README | C | Sun 20:30 |
| Finale deck (if top 10) | D + everyone | 6 Oct |
