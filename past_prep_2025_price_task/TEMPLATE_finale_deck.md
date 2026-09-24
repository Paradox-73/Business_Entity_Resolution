# TEMPLATE — Finale deck (10 minutes + 5 minutes Q&A)

Format in 2025 (`videos/A/notes.md` `[00:15:16]`):
- 10 minutes to present. A chat message arrives at **8 minutes**; the last 2 minutes are for concluding. Teams that ran over were cut off mid-sentence (Zenith, Trailblazers).
- 5 minutes of questions from 3 Amazon scientists.
- Stated criteria: **novelty and approach; intelligent use of data; efficiency and scalability; presentation.**
- Judges' stated reasons for the 2025 top 3: ablation studies, data insights with concrete examples, "technical rigour", frugality (small model), scalability, working backwards from the customer.

Aim for **7 minutes 30 seconds** of talking. 9 slides plus backup slides. One speaker per block, hand-offs rehearsed.

| # | Slide | Time | Speaker | Content |
|---|---|---|---|---|
| 1 | Title | 0:15 | A | Team, college, members, **final score in big font**, one-line summary of the approach. |
| 2 | Problem and metric | 0:30 | A | Task in one sentence; what the metric punishes; constraints (licence, ≤8B, no external data). Why it matters to Amazon customers/sellers (cold-start pricing, flag mispriced listings). |
| 3 | Data insights | 1:15 | B | 3 observations → decisions, each with a chart or a real example row. Include one surprising finding. |
| 4 | Error analysis example | 0:45 | B | One concrete failure and fix, like the 2025 winner's "gluten-free" slide (actual 109, predicted 37 → fix). Judges named this as a reason for the win. |
| 5 | Architecture | 1:15 | C | One clean diagram: inputs → encoders (name, params, frozen/fine-tuned) → fusion → head → output. Loss and target on the slide. |
| 6 | Training and validation | 0:45 | C | k-fold scheme, LRs, schedule, EMA, hardware, time. CV vs public LB agreement. |
| 7 | **Ablation table** | 1:00 | D | Every step with its score and Δ. Judges praised this slide explicitly ("I really like the slide where you summarised all the experiments and the incremental gains"). Include 2 things that failed. |
| 8 | Efficiency and production | 0:45 | D | Params, GB, ms/item, full test inference time, hardware needed. Missing-image and rotated-image handling. Pack-size price parity. |
| 9 | Conclusion and future work | 0:30 | A | 3 takeaways, 2 future ideas you have partly tested (not a wish list). |

## Backup slides (show only if asked)

- Ensemble weights and per-model scores (a 2025 team did not know theirs — know yours by heart).
- Ablation: with vs without images; log vs raw target; loss choices.
- Per-category / per-price-bucket error table and a residual plot; worst-10 predictions.
- Price consistency check: 5 × pack-of-1 vs pack-of-5, same product across brands.
- Robustness test: rotated/cropped images, missing text fields.
- Compute log: GPU hours, cost, what you would do with more compute.

## Questions the 2025 judges asked (prepare an answer with a number for each)

- Why this model and not a smaller one? What does the extra complexity buy? (Deepak asked this of the team with two 3–7B VLMs.)
- Do you have an ablation for `<decision>`? Log vs linear target? With vs without images?
- Model size in GB, inference latency, what machine for production?
- What happens if images are rotated / missing in production?
- How do you make pack-of-5 and 5 × pack-of-1 prices consistent (no arbitrage)?
- Two brands of 500 ml milk: how do you stop ₹25 vs ₹100 predictions? Any post-hoc per-category analysis?
- Which layer did you take embeddings from? Were encoders fine-tuned or frozen?
- What exactly were your outliers, and why remove them? (Judge: high prices are natural in e-commerce.)
- Did you test your "real-world robustness" claim outside the challenge data?
- Which handcrafted features mattered most?
- How were ensemble weights learned?

## Delivery rules

- Rehearse 3 times with a timer; cut content, not speed.
- Cameras on for all members; test mic, screen share, slideshow mode 10 minutes before (one 2025 team lost time to AV problems).
- Answer the question asked in the first sentence, then give the number. If you don't know, say what you would measure and how.
- Don't call high-priced items "outliers" unless you prove they are errors.
