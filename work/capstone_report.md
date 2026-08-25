# Capstone Report — CTR / Engagement Opportunity Scoring

- **Author:** James Ivan Matienzo
- **Lane:** Lane 4 — CTR / Engagement Opportunity Scoring
- **Repo:** https://github.com/JamesIvanMatienzo/flyrank-ml-internship-starter
- **Date:** August 2026

> Copy this file to `work/capstone_report.md` and fill it in as you build. The eight
> sections mirror the Pass / Needs-Work rubric axes, so nothing here is optional.

## 1. Problem framing

**Decision supported:** This work supports a content editor's decision of **which pages to rewrite first** to recover lost search clicks. Given a portfolio of thousands of pages across dozens of client websites, editors cannot manually review every page — they need a ranked queue that surfaces the highest-impact opportunities at the top.

**Unit of analysis:** One content page per client, aggregated to monthly search performance over **June 2026** (the latest complete month in the warehouse). Each row represents a unique `(client_hash_id, content_hash_id)` pair with summed impressions, clicks, weighted average position, and session-weighted engagement rate for that month.

**Output:** A binary classification (`is_opportunity` = 0 or 1) with a predicted probability per page. Pages are ranked by predicted probability into a priority action queue — highest probability at the top. The queue includes reason codes explaining why each page was flagged.

**Human action:** A content editor or SEO strategist reads the ranked queue top-down, opens the highest-ranked pages, and rewrites the title tag, meta description, or on-page content to improve click-through rate. Each page review takes approximately 1–2 hours of editorial effort.

**Cost of a wrong call:**
- **False positive** (flagging a well-performing page): ~1–2 hours of editor time wasted per page reviewed. The page did not need a rewrite, but no permanent damage is done.
- **False negative** (missing a high-impression, low-CTR page): The page continues losing clicks every day it sits at a good search position with a poor snippet. This opportunity cost compounds over weeks and is invisible without the model.

The asymmetry means **Precision@K at small K** (top 20–50) is the right primary metric: the team has limited review capacity, so every queue slot must count.

**Why data/ML helps:** CTR expectations differ by position tier — a 0.2% CTR at positions 1–3 is alarming, but the same CTR at position 30+ is expected. A flat CTR threshold would generate false positives at deep positions and miss real problems at top positions. A position-adjusted scoring system (whether rule-based or learned) can account for position, volume, engagement, and content depth simultaneously — the pattern is real but too multi-dimensional for a single manual threshold.

## 2. Data safety

### Data sources used

1. **Starter CSV** (`data/raw/content_refresh_anonymized.csv`) — 30,000 rows × 44 columns, 32 pseudonymized clients. Used in `w01_research_question.ipynb` for initial exploration and lane selection. Not used for any modeling or evaluation.

2. **Warehouse dataset** (Hugging Face, gated) — `hf://datasets/FlyRank/internship-warehouse`. Used for all modeling work (w03–w05):
   - `fact_content_daily_performance_sample` (~11.7M rows, daily grain) — aggregated to monthly grain for June 2026, yielding **52,766 pages** after filtering for `impressions >= 500` and `avg_position > 0`.
   - `dim_content` (519,606 rows) — joined for `word_count`.
   - `dim_clients` (104 rows) — used for client-holdout split validation.

### Deliberately excluded columns

**Label leakage (would inflate scores dishonestly):**

| Excluded Field | Why |
|---|---|
| `observed_ctr` | Directly encodes the target outcome — `is_opportunity` is defined by comparing observed CTR to tier median CTR. Using it as a feature would let the model see the answer key. |
| `ctr_gap` | This IS the label definition (`ctr_gap > 0` → opportunity). Used only in label construction and the rule baseline, never as an ML model feature. |
| `trend_direction` | Post-hoc product trend flag derived from `trend_pct`. A trailing outcome that would leak future information. |
| `trend_pct` | Label-derived field from the starter pipeline. Excluded to prevent leakage. |

**ID fields (grouping only, never features):**

| Excluded Field | Why |
|---|---|
| `client_hash_id` | Pseudonymized client identifier. Used only for grouped train/test splits (GroupShuffleSplit) to prevent client-level memorization. Never passed to any model as a feature. |
| `content_hash_id` | Pseudonymized page identifier. Used only for joins and row identification. Never a model feature. |

**Privacy:**

| Excluded Category | Why |
|---|---|
| Client names, URLs, raw queries | No real client names, website URLs, or search queries appear anywhere in `work/`. All identifiers are pseudonymized hashes from the warehouse. |

All client and content identifiers are pseudonymized hashes. No real names, URLs, or queries appear anywhere in `work/`.

### Leakage audit

A dedicated leakage check was performed in `w03_feature_leakage_check.ipynb`. Each candidate feature was tested by computing its AUC against the label — any feature with suspiciously high AUC (near 1.0) was flagged and investigated. `observed_ctr` and `ctr_gap` were confirmed as leakers and permanently excluded from the model feature set.

## 3. Baseline

### The rule: a transparent, hand-coded scoring formula

The baseline is a **deterministic rule** that scores every page using four conditions and two multipliers. In plain words:

> "A page is worth reviewing if it gets enough traffic (≥ 1,000 impressions), ranks in a fixable position (position ≤ 20), and has a CTR below its position tier's median. The score is higher when the CTR gap is wider and the page has more traffic."

**Formula:**

```
baseline_score = visible × clickable × underperforming × ctr_gap × log_impressions
```

Where:
- `visible` = 1 if `total_impressions >= 1,000`, else 0
- `clickable` = 1 if `avg_position <= 20`, else 0
- `underperforming` = 1 if `ctr_gap > 0` (CTR below tier median), else 0
- `ctr_gap` = `tier_median_ctr - observed_ctr` (the size of the opportunity)
- `log_impressions` = `log1p(total_impressions)` (traffic volume, log-scaled to tame outliers)

Pages that fail any gate (not visible, not fixable position, not underperforming) receive a score of 0. The remaining pages are ranked by score descending.

Each scored page also receives **reason codes** — human-readable labels explaining why it was flagged (e.g., `high_traffic_underperformer`, `pos_11_20_opportunity`, `low_engagement`).

### Why it's a fair comparison

The baseline is evaluated on the **same data** (June 2026 warehouse, 52,766 pages), the **same label** (`is_opportunity`), and the **same metric** (Precision@K) as the ML models. The only difference is the scoring mechanism: the rule uses hand-coded thresholds, while the ML models learn from the training split.

### Baseline numbers

Evaluated on the full 52,766-page dataset (before train/test split):

| Metric | Value | Lift vs Random |
|---|---|---|
| Base rate (random guess) | 33.7% | 1.00× |
| Precision@10 | 100.0% | 2.97× |
| Precision@20 | 100.0% | 2.97× |
| Precision@50 | 100.0% | 2.97× |
| Precision@100 | 100.0% | 2.97× |
| Precision@200 | 100.0% | 2.97× |
| Precision@500 | 100.0% | 2.97× |
| Dummy baseline (majority class) | 66.3% | — |

The rule achieves **100% Precision@K across all K values** because it directly incorporates `ctr_gap` — the metric that defines the label. This is by design: the baseline is meant to be a strong, transparent reference point. The ML models must beat (or match) this score using only honest features that exclude `ctr_gap`.

**Scored pages:** 15,998 out of 52,766 (30.3%) received a score > 0 and were ranked into the action queue.

## 4. Model / analysis

**Target definition (one sentence):** A page is labeled `is_opportunity = 1` if its observed CTR falls below the median CTR for pages in the same position tier AND it receives at least 1,000 monthly impressions.

**Why this method fits the lane:** The task is binary classification with a ranking output — we need to score every page and surface the highest-probability opportunities at the top of an editorial queue. Logistic Regression provides a readable, coefficient-interpretable baseline; Random Forest handles the non-linear interaction between position tier and traffic volume without manual feature engineering.

**Models used:**
1. **Logistic Regression** (`sklearn`, `max_iter=1000`, `random_state=42`) — scaled features via `StandardScaler`
2. **Random Forest** (`n_estimators=200`, `max_depth=8`, `random_state=42`) — unscaled features, trained on the same split

**Exact feature list (5 features — all honest, no leakage):**

| Feature | Engineering | Reason kept |
|---|---|---|
| `log_impressions` | `log1p(total_impressions)` | Heavy right tail tamed; traffic volume is the strongest signal |
| `avg_position` | Impression-weighted average | Position tier drives CTR expectation |
| `engagement_rate` | `ga4_engaged_sessions / ga4_sessions × 100` | On-page quality signal |
| `word_count` | Raw from `dim_content`, 0-filled | Content depth proxy |
| `has_ga4_data` | Binary flag | Distinguishes pages with/without analytics coverage |

**Deliberately excluded:**

| Feature | Why excluded |
|---|---|
| `observed_ctr` | Directly encodes the label — injecting it jumps AUC to 1.000 (confirmed in ML-09) |
| `ctr_gap` | IS the label definition — this IS the answer key |
| `trend_direction` / `trend_pct` | Post-hoc trailing outcomes; leaked future signal |
| `client_hash_id` / `content_hash_id` | Grouping and join keys only, never features |

## 5. Evaluation

### Split Design

**GroupShuffleSplit by `client_hash_id`** — 80% train / 20% test, `random_state=42`.

- Train: 40,697 pages across 35 clients
- Test: 12,069 pages across 9 clients (entirely unseen during training)
- Test base rate: **20.5%** (lower than overall 33.7% — holdout clients happen to have fewer opportunities)

Whole clients go to either train or test, never both. This tests genuine generalization to new clients — the actual deployment scenario where a new client joins FlyRank.

### Results Table (same test split, base rate = 20.5%)

| Metric | Rule Baseline | Logistic Regression | Random Forest |
|---|---|---|---|
| Precision@10 | **100.0%** | 80.0% | 70.0% |
| Precision@20 | **100.0%** | 75.0% | 60.0% |
| Precision@50 | **100.0%** | 64.0% | 48.0% |
| Precision@100 | **100.0%** | 59.0% | 42.0% |
| ROC AUC | **0.814** | 0.750 | 0.796 |
| Lift vs random (P@10) | **4.88×** | 3.90× | 3.41× |

> The rule's 100% Precision@K is expected, not earned — it uses `ctr_gap` directly. ML models are banned from it and still reach ROC AUC = 0.796.

### Split Inflation Check (from ML-09)

Same Random Forest, two splits:

| Split | Precision@50 | ROC AUC |
|---|---|---|
| Random (inflated) | 94.0% | 0.879 |
| Grouped client (honest) | 48.0% | 0.795 |
| **Gap** | **+46.0 pp** | **+0.084** |

A random split nearly doubles Precision@50. All reported numbers are from the **grouped split**.

### Error Analysis

**False positives (model flags page that isn't an opportunity):** Concentrated in `pos_4_10` pages that have high impressions but average CTR — the model's strongest signal (`log_impressions`) overrides the weaker position-adjusted CTR signal for these cases.

**False negatives (model misses a true opportunity):** Most common in `pos_1_3` pages — positions 1–3 have very few examples in the training set (small tier), so the model under-flags top-slot underperformers. This is a known limitation of the class imbalance within the highest-value tier.

**Practical impact:** At the queue sizes editors actually use (top 20–50), the Precision@20 of 60% for RF means 12 of every 20 flagged pages are true opportunities — a meaningful improvement over the 20.5% base rate (3× lift).

## 6. Interpretation

### Feature Importances

**Random Forest (Gini importance):**

| Feature | Importance | Plain meaning |
|---|---|---|
| `log_impressions` | **0.747** | Traffic volume dominates — high-impression pages are far more likely to be flagged |
| `avg_position` | 0.118 | Rank matters, but less than volume |
| `engagement_rate` | 0.063 | Weak but consistent: lower engagement correlates with opportunity |
| `word_count` | 0.040 | Marginal — content length alone doesn't predict CTR gaps well |
| `has_ga4_data` | 0.032 | Clients without GA4 behave differently; flag separates them |

**Logistic Regression coefficients (scaled):**

| Feature | Coefficient | Direction |
|---|---|---|
| `log_impressions` | +1.034 | More traffic → more likely opportunity |
| `word_count` | +0.037 | Slightly longer pages → slightly more likely |
| `avg_position` | -0.033 | Better position → slightly less likely (higher-ranked pages have smaller gaps) |
| `engagement_rate` | -0.076 | Higher engagement → less likely opportunity |
| `has_ga4_data` | -0.173 | Pages with GA4 data slightly less likely (better-tracked clients tend to optimize more) |

### What the Model Found

The dominant finding is: **traffic volume (`log_impressions`) predicts opportunity label far more strongly than anything else.** This makes intuitive sense — high-impression pages have enough data to measure a CTR gap reliably, and the business impact of fixing them is larger.

### Surprises and Negative Results

- **`word_count` had near-zero effect.** The signal audit in ML-06 found no consistent pattern between page length and CTR underperformance across tiers. This is a negative result worth noting — a common SEO assumption ("longer pages perform better") does not hold in this data for CTR gap prediction.
- **`engagement_rate` is a weak signal.** Despite being theoretically sound (pages with poor on-page quality should have both low CTR and low engagement), its individual AUC against the label was only 0.527 — barely above random. The model uses it, but it's not driving decisions.
- **The model struggles most on `pos_1_3` pages.** With few training examples in this tier, the model systematically under-flags top-slot underperformers — which are arguably the highest-value fixes. A future improvement would oversample this tier or train a separate classifier for it.

## 7. Recommendation

### How a FlyRank Editor Uses This Tomorrow

1. Open `work/outputs/action_playbook_queue.csv`
2. Filter to `confidence = High` (7,809 pages, 77% true opportunity rate)
3. Sort by `rf_prob` descending
4. Open the top-ranked page in a browser — read its title and meta description
5. Ask: *"Does this title match what someone searching this topic actually wants?"*
6. If yes → rewrite title/meta; flag for Google re-crawl
7. Move to the next page; log actions taken

### Ranked Actions by Position Tier

| Action | Pages (High+Med) | When to apply |
|---|---|---|
| Rewrite title/meta (pos 4–10) | 14,135 | Page 1 placement with CTR below peers — snippet is the bottleneck |
| Push to Page 1 (pos 11–20) | 4,659 | Striking distance — title rewrite + 2–3 internal links from high-traffic pages |
| Fix snippet — top slot (pos 1–3) | 310 | #1–3 with below-peer CTR — consider structured data, FAQ schema |
| Monitor — limited upside (pos 21–50) | 2,186 | Deep position — ROI of rewrite is lower; revisit when position improves |

### Confidence and Limits

**Confidence:** High-confidence pages (RF prob ≥ 0.70) are true opportunities 77% of the time — a 3.8× lift over the 20.5% base rate on held-out clients.

**Explicit limits:**
- This is **decision-support, not automation.** Every flagged page requires a human to open it, read it, and apply editorial judgment before any change is made.
- The model **cannot guarantee** that a rewrite will increase CTR. It identifies pages that *look worth reviewing* based on observed patterns — not a controlled experiment.
- Scores are **from June 2026 only.** Re-run the pipeline monthly as new GSC data arrives.
- Pages with **< 500 impressions** are not scored — insufficient data.
- **Do not compare scores across clients** — clients have different industries, audiences, and traffic baselines.

## 8. Reproducibility

### Re-run from a Fresh Clone

```bash
# 1. Clone the repo
git clone https://github.com/JamesIvanMatienzo/flyrank-ml-internship-starter.git
cd flyrank-ml-internship-starter

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Hugging Face token
cp .env.example .env
# Edit .env and add: HF_TOKEN=your_token_here

# 4. Run notebooks in order (Jupyter or VS Code)
# w03_data_contract.ipynb         → verifies data access
# w03_feature_leakage_check.ipynb → builds feature vector, audits leakage
# w04_signal_audit.ipynb          → signal tests
# w04_baseline_score.ipynb        → rule baseline, outputs baseline_action_score.csv
# w05_model.ipynb                 → LR + RF models, outputs model results
# w06_validation_audit.ipynb      → honest split comparison, leakage injection
# w07_action_playbook.ipynb       → final queue, outputs action_playbook_queue.csv
# capstone.ipynb                  → full paper, all sections
```

### Random Seeds

All randomness is controlled by `SEED = 42` set at the top of every modeling cell:
- `GroupShuffleSplit(random_state=42)`
- `LogisticRegression(random_state=42)`
- `RandomForestClassifier(random_state=42, n_estimators=200, max_depth=8)`
- `np.random.seed(42)` set at the start of each notebook

### Key Dependencies

```
duckdb>=1.0
pandas>=2.0
numpy>=1.26
scikit-learn>=1.4
huggingface-hub>=0.23
python-dotenv>=1.0
```

Full environment: see `requirements.txt` in the repo root.

### Data Version

- Warehouse: `FlyRank/internship-warehouse`, build `v20260703` (frozen snapshot)
- Sample table: `fact_content_daily_performance_sample.parquet`
- Time window: `month = '2026-06'` (June 1–30, 2026)
- Filtered to: `gsc_impressions >= 500 AND avg_position > 0` → 52,766 pages, 44 clients

---

> **Claims checklist before submitting:** observed / measured / directional / decision-support
> **Metrics vs. base rate:** report your task's base rate (majority-class %) next to any
> precision@K or accuracy — a high score can just be a high base rate. AUC / lift over
> baseline are the honest discrimination numbers.
> language everywhere · no causal claims without an experiment or causal design · no
> "predicted Google's algorithm" · no client-identifying details · numbers in this report
> match a fresh re-run.
