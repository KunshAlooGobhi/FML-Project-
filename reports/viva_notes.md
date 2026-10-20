# Viva notes (Phases 0 to 7)

**Big picture in one breath:** the main result is on REAL Amazon Beauty reviews (Phase 7). The 1M Kaggle file and a simulated dataset are kept as sanity checks (Phases 1 to 6 explain the method on them).

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
- **Honest limitations?** See Phase 7 (Beauty is a pre-filtered 5-core, offline, one seed, extended grid, post-hoc dataset switch); the simulated control favours factor models; the Kaggle products file has no users so content was tested on a proxy there.
- **Demo:** `python src/demo.py user` (top-10 for a real Beauty shopper, later purchases ticked) and `python src/demo.py product mascara`; `streamlit run src/app.py` for the UI.
- **One-sentence takeaway:** the method detects real signal where it exists and honestly reports none where it does not.

## Likely trick questions
- *"Your best model got 0.0267, is that good?"* Relative to popularity (0.0059) it is 4.5 times better and statistically solid. The absolute value is small because there are 12,000 products and each shopper has only one or two later purchases; 0.199 on the simulated control is higher only because simulated structure is cleaner.
- *"Why not merge the datasets?"* No shared keys; guessing would invent data.
- *"Why did the 1M file fail?"* It is random by the four diagnostics; we report popularity is not beaten there.
- *"Did you pick Beauty because it gave better numbers?"* We picked it because it passes the signal checks and has real item text, after the 1M file failed; Beauty's test period was not scored until its settings were frozen. The switch is disclosed as post-hoc.
- *"Did you peek at the test?"* No: settings were committed in `frozen_settings.json` before the single test run.

## Phase 7: Real data (Amazon Beauty 5-core)
- **What is it?** Public Amazon review data: 22,363 users, 12,101 products, 198,502 reviews (2002 to 2014), 5-core (every user and item has >= 5 reviews), with titles, brands, categories, descriptions, prices. A review is treated as a purchase.
- **Why switch datasets?** The 1M file has no learnable pattern (four diagnostics match chance) and the other Kaggle files have no users. Beauty passes the same four checks: item-popularity variance/mean 34.0 (chance 1), early/late popularity correlation 0.30 (chance 0), same-category repeat 0.110 vs 0.021, same-brand 0.110 vs 0.033.
- **Same protocol?** Yes: time split (train to 2013-12-26, validation to 2014-04-08, test 2014-04-09 to 2014-07-23), settings frozen in `frozen_settings.json` before test scoring, test scored once, paired bootstrap with Bonferroni (7 comparisons).
- **Frozen settings:** popularity all-time; item-kNN k=200, shrink=100; ALS 256 factors, reg 0.001, alpha 40; BPR 256 factors, reg 0.01, lr 0.1; SVD 128 factors; content TF-IDF 20k features, (1,2)-grams; hybrid adaptive weight 0.5.
- **Why an extended grid?** Best validation settings sat on the edge of the Table II grids, so we added larger values (validation only). Documented deviation; ALS/BPR still sit on the new edge, so slightly larger models might gain a little.
- **Results (locked test NDCG@10):** hybrid 0.0267 > content 0.0227 > ALS 0.0224 > item-kNN 0.0207 > BPR 0.0157 > popularity 0.0059 > SVD 0.0016 > random 0.0004. HitRate@10: hybrid 8.3%, popularity 2.7%.
- **Hybrid vs components:** beats ALS by 0.0043 [0.0021, 0.0065] and content by 0.0040 [0.0019, 0.0061]; small but significant.
- **Cold items (E2):** 2,864 test rows are of products with no history. HitRate@10 on those purchases: hybrid 8.0%, content 3.6%, every collaborative model and popularity 0%. Cost to ordinary purchases: none (hybrid 0.0249 vs ALS 0.0253).
- **Cold users:** 40.8% of test rows (before the filter to warm users: 8,675 of 29,848 rows from users with no history). Only popularity can serve them (0.0168 vs random 0.0005).
- **E3:** at 25% of training data item-kNN 0.0042 and ALS 0.0034 are below popularity 0.0058; at 100% they are 0.0179 and 0.0171. Content is most data-efficient (0.0143 at 25%).
- **E4:** global blend w=0.5 gives 0.0216 vs item-kNN 0.0181 and content 0.0187: on real data content helps even for users with history (unlike the simulated control).
- **E7:** order shifts slightly (content/ALS/item-kNN swap) but the extremes are stable. Test values are higher for kNN/ALS/BPR because the final models are refit on train + validation.
- **Rating vs ranking:** SVD improves RMSE (1.105 vs global-mean 1.145) but ranks below popularity (0.0016 vs 0.0059).
- **Limitations in one line each:** pre-filtered 5-core (denser than a live shop), review is not purchase, data end 2014, offline only, absolute scores small, single seed, extended grid, dataset chosen post hoc, 40.8% cold-user rows served only by popularity.
- **UI/demo:** both use the frozen hybrid fitted on train only; ticks mark real later purchases (validation period, never the locked test). The UI shows mostly shoppers where it worked and occasionally one where it did not, by design for the presentation.
