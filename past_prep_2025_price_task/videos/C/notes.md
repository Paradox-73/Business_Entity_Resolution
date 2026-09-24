# Video C — "AWS Best Practices for Amazon ML Challenge 2026" (21 Sep 2026)

Source: https://www.twitch.tv/videos/2880101957 (1h43m). Real content starts `[00:17:17]`; before that is a waiting screen.
- Host: Latish Kumar, recruiter, Amazon University Talent Acquisition (runs the ML Challenge).
- Presenter: Jatin Malhotra, AWS developer advocate.
- Transcript by Whisper `medium.en`. The rewards figures were checked against the slide at `[00:24:56]`.

## Challenge facts stated

- Registration "open up until tomorrow", i.e. 22 Sep `[00:17:31]`.
- On 25 Sep "a real-world dataset from Amazon drops **in your emails**" `[00:19:42]`.
- Length: the presenter said "**exactly 72 hours**" `[00:19:49]`. **Conflicts** with the Unstop round page (25 Sep 09:00 → 27 Sep 21:00 = 60 h). Check the email on 25 Sep; plan for 60 h to be safe.
- ~79,000 competing students; top 50 get interviews for Applied Scientist roles `[00:19:57]`.
- "The whole ML challenge runs on AWS" `[00:18:31]`; "free AWS credits for every participant who builds solutions" `[00:18:22]`. How challenge credits are delivered was **not** explained. The only credits shown were the standard AWS Free Tier $200 (below).

## Step 1 — AWS Builder Center profile (required for registration)

Builder Center is Amazon's free developer community site: workshops, sandbox accounts, blogs.

1. Go to **https://builder.aws.com** → **Sign in** (top right) `[00:25:50]`.
2. Pick a login: Amazon, GitHub, Apple, Google, or AWS Builder ID. He used Google; use a personal email.
3. First login asks for an **alias** `[00:26:44]`:
   - Rule as spoken: "between 3 to 19 characters, letters a–z, numbers 0–9" (transcript said "33 to 19"; 3–19 is the likely rule). `@` is not allowed.
   - **"This alias is also required for your ML challenge evaluation. If you haven't set your alias or sent it to the team, do it right away."** `[00:26:57]`
4. When asked "Are you a student?", answer **Yes** `[00:27:34]`.
5. **Verify student status** with any college credential / ID `[00:24:03]`. Unlocks (slide `[00:24:56]`, "$579 in free value, no credit card"):
   - 12 months of AWS Skill Builder Premium (worth $449).
   - Earn 7 badges → $10 AWS credits.
   - Earn 14 badges → $20 AWS credits.
   - Earn 21 badges → $100 AWS certification voucher.
6. Quick badges: upload a profile picture (badge appears in 1–2 min) `[00:29:27]`; log in on 7 days `[00:29:01]`.
7. **Free sandbox AWS accounts** from workshops: no credit card, **8 hours per week** per login, resets every Sunday; ready ~15 min after requesting `[00:22:09]`, `[00:28:08]`. Useful for practice, not for a 60-hour training run.

## Step 2 — AWS Free Tier account (2026 rules) `[00:30:25]`

- New account = **$100 credit at sign-up + $100 more** for 5 starter activities, $20 each `[01:00:41]`:
  - launch an EC2 instance
  - use a model in the Amazon Bedrock playground
  - set up a cost budget
  - create a web app with AWS Lambda
  - (5th not named clearly; the card list appears in the console home page)
  - Complete them while waiting for SageMaker to initialise.
- Credits: Console → **Billing and Cost Management → Credits** `[01:02:54]`.
- ~30 services are always free within limits (S3, Lambda, DynamoDB). **SageMaker is available on the free plan**; Bedrock Agents is paid-only `[00:34:09]`.
- Free-plan accounts are **not eligible for promotional credits** (e.g. Builder Center campaigns) `[00:33:47]`. Check before relying on badge credits.
- The free plan **auto-closes after 6 months** unless upgraded `[00:39:48]`.
- Sign-up steps `[00:43:25]`:
  1. aws sign-up page → email → verification code.
  2. Root credentials (password).
  3. Contact / address details.
  4. Payment: **UPI autopay** (needs a mandate of up to **₹15,000** in the account; ₹2 verification charge) or a credit card `[00:38:34]`. Not charged on the free plan.
  5. Identity: PAN, Aadhaar (via DigiLocker) or driving licence; name must match `[00:40:47]`. Then phone OTP.
- Turn on MFA with an authenticator app — he skipped it only for time `[00:36:29]`.
- Set a **billing alert / budget** so nothing surprises you `[00:32:14]`.
- Region: Mumbai (ap-south-1) or **us-east-1** (his preference: new services arrive there first) `[00:32:31]`.

## Step 3 — SageMaker workflow shown in the demo `[00:51:36]`

- Console → search **SageMaker AI** → **Set up SageMaker Studio** (quick setup). Creates the IAM role, S3 bucket and Studio for you.
- **Gotcha:** on a brand-new account the setup failed live; he said initialisation "might take 20 to 30 minutes" `[00:57:10]`. **Create the account and Studio before 25 Sep.**
- Studio → **JupyterLab** → Launch (default ml.t3.medium, 2 vCPU / 4 GB RAM — CPU only) → Python 3 notebook → `pip install sagemaker-core` (SDK v3) `[01:08:40]`.
- Upload the challenge data to your own S3 bucket `[01:13:55]`.
- Demo: built-in XGBoost on a public churn dataset.
  - Format rule for built-in XGBoost: **CSV, no header, target in the first column** `[01:20:16]`.
  - Split 67% train / 22% validation / 11% test, `random_state=42`.
  - Hyperparameters max_depth, eta 0.2, 100 rounds, `binary:logistic`. Test accuracy 91.6%.
- Predictions: a live endpoint, or **Batch Transform** (whole file at once; "often more cost effective" for competitions) `[01:30:35]`.
- Other tools named: SageMaker Autopilot (automatic model search for a baseline), Hyperparameter tuning jobs, SageMaker Experiments (tracking runs).
- **Delete endpoints after use**: ~$0.12/hour; one forgotten over a weekend ≈ $6, a month ≈ $90 `[01:39:33]`. **Stop the JupyterLab space** at the end of each day. Training jobs stop by themselves; model files in S3 are cheap.
- Metric advice: check exactly which metric the leaderboard uses; don't default to accuracy `[01:38:40]`.

## Gotchas and open points

- The 72 h vs 60 h conflict (above).
- **GPU was never mentioned.** Default Studio instance is CPU. New accounts usually have a GPU quota of 0 for SageMaker (ml.g4dn/ml.g5) and EC2 (G/P instances), and quota increases can take a day or more. Request increases **today** if you plan to train on AWS.
- $200 free-tier credit will not go far on GPUs; keep Kaggle/Colab as primary GPUs.
- The presenter's blog link was posted in the stream chat (not in the transcript); ask the organisers or check Builder Center.
