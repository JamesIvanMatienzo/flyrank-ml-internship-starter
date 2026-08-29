# Week-8 Demo Outline (5 Minutes)

*One chart. One honest result. One recommendation. Ready for the Week-8 showcase.*

---

### Slide 1 — The Question (45 sec)

> **"FlyRank manages content for 44+ client websites. Many pages rank on Google's first page but earn almost no clicks. Which pages should editors rewrite first — and how do you rank 52,000 pages by impact?"**

- Show: one row from the action queue — a page with 30,000 monthly impressions and 0.03% CTR at position 6.
- Hook: "At position 6, a typical page gets 0.36% CTR. This page gets 0.03%. That's roughly 100 lost clicks every single day, compounding."

---

### Slide 2 — The Method (1 min)

- **Label:** A page is an `opportunity` if its CTR is below its position tier's median AND it gets ≥ 1,000 impressions/month. Base rate: 33.7%.
- **5 honest features:** traffic volume (log), search position, engagement rate, word count, GA4 flag. No CTR — that would be using the answer key.
- **Split:** Grouped by client. Entire clients go to either train or test — never both. This tests whether the model works on a *new client it's never seen*, which is the actual deployment scenario.
- **Models:** Rule baseline → Logistic Regression → Random Forest. Each compared on the same test split.

---

### Slide 3 — The Chart (1 min 30 sec)

**Show: split comparison**

| Split | Precision@50 | ROC AUC |
|---|---|---|
| Random (inflated) | 94.0% | 0.879 |
| **Grouped client (honest)** | **48.0%** | **0.795** |
| Gap | +46.0 pp | +0.084 |

*Talking point:* "If I used a random split, I'd walk in here claiming 94% precision. The grouped split gives 48%. That 46-point gap is the model memorizing client quirks, not learning a real signal. Every number I report is the grouped number."

---

### Slide 4 — The Honest Result (1 min)

- Random Forest: ROC AUC = **0.796** on 9 completely unseen clients.
- That's only **0.018 behind the rule baseline** — which cheats by using the label metric directly as its core input.
- High-confidence queue: **7,809 pages**, 77% true opportunity rate vs. 20.5% holdout base rate → **3.8× lift**.
- Leakage injection test: adding `observed_ctr` jumps AUC from 0.795 → **1.000**. Confirmed the ban was correct.

---

### Slide 5 — Recommendation + Limits (45 sec)

**Recommendation:** Open `action_playbook_queue.csv`. Filter `confidence = High`. The top page has 30,000+ impressions, sits at position 6, and has near-zero CTR. Write a better title. That's it. The model told you which page to start with.

**Honest limits I'm stating myself before you ask:**
- No experiment was run. These pages *look worth reviewing* — the model cannot promise a rewrite will increase CTR.
- Queue is June 2026 only. Needs a monthly re-run.
- Pages below 500 impressions are not scored.

**Close:** *"The model's job isn't to predict clicks. It's to tell an editor which page to open first on Monday morning. That's a problem that scales across any portfolio."*

---

*Part of the FlyRank ML Internship — August 2026.*
*Repo: https://github.com/JamesIvanMatienzo/flyrank-ml-internship-starter*
