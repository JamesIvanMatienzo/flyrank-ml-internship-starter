# Shareable Cuts

*Two ready-to-use formats. Copy as-is or edit to taste.*

---

## Cut 1 — Social Post (LinkedIn / X)

> For my ML internship capstone, I built a content triage model on real (pseudonymized) Google Search Console data.
>
> The problem: a page ranking #6 on Google should get ~0.36% click-through rate. Some pages at that position get 0.03%. That's hundreds of lost clicks per day — and without a system, no one flags it.
>
> My approach: define a position-adjusted CTR opportunity label across 52,766 pages and 44 client websites, then train a Random Forest on 5 honest features — deliberately excluding CTR itself to avoid data leakage.
>
> Key result: ROC AUC = 0.796 on 9 completely unseen clients. 7,809 pages in the high-confidence action queue with a 3.8× precision lift over guessing.
>
> The hardest part wasn't building the model — it was the validation design. A random split gave 94% precision. A grouped client split gave 48%. That 46-point gap is the difference between a result that travels and one that collapses when a new client shows up.
>
> Full write-up + code: https://github.com/JamesIvanMatienzo/flyrank-ml-internship-starter

---

## Cut 2 — Employer-Facing Summary (3 sentences)

> I built a machine-learning priority queue that identifies which content pages are underperforming their expected click-through rate given their Google search ranking position, and ranks them by impact for content editors to rewrite first. Working with 52,766 pages of real (pseudonymized) Google Search Console and GA4 data across 44 client websites, I trained a Random Forest classifier and validated it using a grouped client split — testing generalization to clients the model had never seen — achieving ROC AUC = 0.796 and a 3.8× precision lift on the holdout. The output is a ranked action queue of 7,809 high-confidence pages with editor-ready labels, backed by a validated leakage audit confirming that all reported metrics are honest and reproduce from scratch.

---

*Part of the FlyRank ML Internship — August 2026.*
*Repo: https://github.com/JamesIvanMatienzo/flyrank-ml-internship-starter*
