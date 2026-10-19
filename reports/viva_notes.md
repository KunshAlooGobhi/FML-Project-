# Viva notes (Phases 0 to 6)

Numbers quoted here come from `reports/`. If asked for a number you do not remember, say where it lives.

## Phase 0: Setup
- **Why a fixed seed (42) and pinned versions?** So anyone rerunning gets the same numbers. `requirements.txt` lists the exact versions.
- **Why the folder layout (`data/`, `src/`, `manifests/`, `reports/`)?** Raw data is never edited; every step is a script that reads one folder and writes another.
- **Why a manifest with SHA-256?** It proves which exact file was used. If the checksum changes, the file changed.

## Phase 1: Data
- **What are the two datasets and why not merge them?** The 1M file has user-product purchases with dates but no text. The products/reviews files have text but no user ids, and product ids do not link. Merging would be guesswork.
- **What is sparsity?** The share of empty cells in the user x item grid. Ours is 99.997%.
- **What is a k-core and why k = 2?** Keep only users and items with at least k interactions. The plan said 5, but only 5,094 of 603,815 users have 5+ purchases, so k = 5 would throw the data away.
- **Cleaning order?** Duplicates and nulls, standardise text, date checks, one row per (user, item), k-core, integer ids.

## Phase 2: Split, metrics, baselines
- **Why a time-based split, never random?** A random split lets the model train on the future and test on the past (leakage). Train is the earliest 70%, validation the next 15%, locked test the last 15%.
- **What is "locked test, scored once"?** Settings are chosen on validation, frozen and committed, then the test is scored one time. `final_eval.py` refuses to overwrite saved test results.
- **Warm / cold-item / cold-user?** Relative to train: warm = known user and item; cold-item = item never seen; cold-user = user never seen (about 20% of rows).
- **Explain the metrics.** Precision@K = hits / K. Recall@K = hits / relevant items. HitRate = at least one hit. MAP rewards hits near the top, averaged. NDCG discounts each hit by log2(rank+1), so a hit at rank 1 is worth more than at rank 10. Toy check: NDCG@5 = 0.7039 (`test_metrics.py`).
- **Coverage, novelty, diversity?** Coverage = share of the catalogue ever recommended. Novelty = mean of -log2(popularity). Diversity = 1 minus mean pairwise text similarity in a list.
- **Why popularity is the baseline to beat?** It is cheap and hard to beat. Tuned: all-time, 30/90/365-day windows, Bayesian rating.
- **Leakage guards?** Similarities, TF-IDF vocabulary, popularity counts and normalisation are fitted on train only; items a user already bought are never recommended.

## Phase 2b: The 1M file has no signal
- **What did you find?** Four checks match random purchasing: variance 11.08 vs mean 11.11 (Poisson), same-category repeat 0.2005 vs 0.2000 chance, same-brand 0.0829 vs 0.0833, early/late sales correlation 0.0011.
- **Why a simulated positive control?** To prove the pipeline can detect a pattern when one exists. It is labelled synthetic and never presented as real shoppers.

## Phase 3: Models
- **Item-kNN:** items are similar if bought by the same users (cosine). Shrinkage damps similarities built from few co-buyers. Keep top-k neighbours.
- **Biased SVD:** rating = global mean + user bias + item bias + user factors . item factors, fitted by SGD on ratings.
- **ALS / BPR:** both learn factors from interactions. ALS fits all cells with confidence weights; BPR learns that bought items should rank above unbought ones.
- **TF-IDF content:** words that are frequent in one product but rare overall get high weight; cosine similarity finds related products. Tuned on a proxy task (same leaf category), reaching NDCG@10 0.567 vs 0.038 random.
- **How were settings chosen?** Grid search on validation NDCG@10 (Table II); ties go to the first setting in grid order.

## Phase 4: Hybrid and cold start
- **How does the hybrid work?** Normalise each score by its row maximum, blend `(1-w) * collaborative + w * content`. The adaptive version uses content only for items with no history.
- **Result?** Validation-best global weight is 0. The hybrid only helps new items: with 200 hidden items, HitRate@10 goes from 0% (CF) to 4.2% (w = 0.1) and 6.5% (w = 0.3), at a cost of NDCG 0.189 to 0.170 on ordinary purchases.

## Phase 5: Frozen test and statistics
- **Paired bootstrap?** Resample users 1,000 times, recompute the mean per-user NDCG difference vs popularity, take the 2.5/97.5 percentiles as the 95% CI. Paired because both models are scored on the same users.
- **Bonferroni?** Multiply p by the number of comparisons (7 control, 5 on 1M). Improvement = CI excludes 0 and adjusted p < 0.05.
- **Control results (NDCG@10):** item-kNN 0.199, hybrid 0.194, ALS 0.193, BPR 0.147, popularity 0.063, SVD 0.057, content 0.014, random 0.003. kNN, ALS, BPR, hybrid significantly better; SVD and content worse.
- **1M results:** everything is about 0.0000 to 0.0002; no model is distinguishable from popularity or random. This is a property of the data, not a bug.
- **Why did rating-trained SVD lose?** RMSE and ranking are different objectives. SVD improved RMSE yet ranked below popularity.
- **E3 training size:** at 25% of data item-kNN (0.028) is below popularity (0.054); ALS stays above (0.070).
- **E7:** control order same on validation and test; 1M order reshuffled, which is what noise looks like.

## Phase 6: Write-up and demo
- **Honest limitations?** Synthetic control favours factor models; real Amazon files have no users, so content was tested on a proxy; offline evaluation; one seed; k-core 2; 5,000-user sample on 1M; neural CF not run.
- **Demo:** `python src/demo.py user CU42` (top-10 for a user, validation hits marked) and `python src/demo.py product polo golf` (real Amazon "customers also like").
- **One-sentence takeaway:** the method detects real signal where it exists and honestly reports none where it does not.

## Likely trick questions
- *"Your best model got 0.199, is that good?"* Only relative to popularity (0.063) on the simulated control; absolute numbers do not transfer to real shoppers.
- *"Why not merge the datasets?"* No shared keys; guessing would invent data.
- *"Why did the 1M file fail?"* It is random by the four diagnostics; we report popularity is not beaten.
- *"Did you peek at the test?"* No: settings were committed in `frozen_settings.json` before the single test run.
