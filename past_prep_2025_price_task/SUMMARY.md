# SUMMARY — 10 things the team must know (Amazon ML Challenge 2026)

**Who does what, and the plan for 25 Sep: `TEAM_PLAN.md`.**

Sources: `PLAYBOOK.md` (full detail), `videos/A/notes.md` and `videos/B/notes.md` (2025 finale), `research/past_solutions.md`.

1. **Register today.** Registration closes **22 Sep 2026, 23:59 IST** (Unstop). Every member needs a valid AWS Builder Center alias (Builder Center is Amazon's developer site); an invalid alias means an incomplete registration.

2. **The round is 60 hours, not 72.** It runs Fri 25 Sep 09:00 → **Sun 27 Sep 21:00 IST**. Plan the final submission for 20:00 Sunday.

3. **Leaderboard rank gets you to the finale; the finale is judged on rigour.** In 2025 the winner reported SMAPE 40.03 and the 1st runner-up 41.21. The team with 39.2, "top of the leaderboard", won nothing. Judges named the reasons: ablation tables, data insights, small justified models.

4. **Keep an ablation sheet from hour 1.** One line per experiment: change, CV score, difference, time. It becomes the methodology document (`TEMPLATE_methodology_doc.md`) and the finale's key slide (`TEMPLATE_finale_deck.md`). Judges praised this slide by name.

5. **Find one concrete error example and fix it.** The 2025 winner showed a "gluten-free" sweetener priced 3× higher that the model priced the same. A judge said this example "stood out". Do error analysis on the worst predictions every day.

6. **The recipe every 2025 finalist shared:**
   - Log target; parse quantity, unit and pack size from the text.
   - Text + image embeddings from a CLIP-type model, fine-tuned (at least the top layers) rather than frozen.
   - MLP head, Huber loss, k-fold with fold averaging.
   - All 3 winners fine-tuned; none of the 4 frozen-embedding teams placed.

7. **Trust cross-validation, not the public leaderboard.** In 2025 the public board used only 25k of 75k test rows. Keep out-of-fold predictions for every model and fit ensemble weights on them.

8. **Start image downloads and time inference in hour 0.** The 2025 winner lost time just downloading images. In 2024 a team finished only 84k of 131k test predictions. Submit a valid baseline within 3 hours.

9. **Check each model's licence and size before using it.** 2025 allowed only MIT/Apache-2.0 models of at most 8B parameters. External price lookups meant disqualification. Expect the same in 2026; confirm on the problem page.

10. **Prepare for the judges' questions:**
    - Model size in GB, latency, hardware.
    - What if images are rotated or missing.
    - Pack-of-5 vs 5 × pack-of-1 price consistency.
    - Per-category sanity checks.
    - "Why this big model?" and "Do you have an ablation for that?"
    - Answer with a number. 10 minutes max, warning at 8.

From the 21 Sep AWS session (`videos/C/notes.md`):
- Set your alias at https://builder.aws.com: 3–19 characters, a–z and 0–9. It is "required for your ML challenge evaluation". Also verify student status there.
- The dataset arrives by email on 25 Sep.
- The presenter said "72 hours" but Unstop says 60 h. Plan for 60 h.
- If you'll use AWS: create the Free Tier account and SageMaker Studio **before** 25 Sep (setup took 20–30 minutes on a new account), and request a GPU quota increase now. Delete endpoints and stop notebooks nightly.
