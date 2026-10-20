# E-Commerce Recommendation System
## A Leakage-Aware Study of Collaborative, Content-Based and Hybrid Recommenders on Amazon Product and Review Data

Rishit Agarwal (245890302) and Kunsh Mehra (245890196), Manipal Institute of Technology, Bengaluru | October 2026

## Abstract

Can personalised models rank products a shopper will later buy better than an honest baseline such as "show what is popular"? We compare a tuned popularity baseline, item-based collaborative filtering, biased SVD, ALS, BPR, a TF-IDF content model and a hybrid on real Amazon Beauty reviews (22,363 users, 12,101 products, 198,502 interactions, with product titles, brands, categories and descriptions), using a time-based split, settings frozen on validation and a locked test scored once. Ranking is judged with Precision, Recall, MAP, HitRate and NDCG at K = 5, 10 and 20, plus coverage, diversity and novelty, with paired bootstrap tests and a Bonferroni adjustment.

Every personalised model except the rating-trained SVD beats tuned popularity with intervals far from zero. The hybrid, which blends item-kNN and TF-IDF text scores and uses text alone for products with no history, is best: NDCG@10 0.0267 against 0.0059 for popularity (about 4.5 times), and a HitRate@10 of 8.3% against 2.7%. It is also the only approach that recovers purchases of brand-new products (HitRate@10 8.0%, against 0% for every collaborative model). Collaborative models need history: with a quarter of the training data they fall below popularity. Two sanity checks frame the result: a 1,000,000-row Amazon purchase file turned out to behave like random purchasing, where no model beats popularity, and a simulated dataset with planted structure confirms that the pipeline detects signal when it exists. On Amazon product text alone, TF-IDF retrieval recovered related products far above chance.

**Index Terms:** recommender systems, collaborative filtering, matrix factorization, content-based filtering, hybrid models, cold start, top-N evaluation, e-commerce.

## I. Introduction

When you open an online store you are rarely hunting for one exact item. You are browsing, and whatever shows up first shapes what you end up buying. Recommendation systems decide that first screen, the "customers also bought" strips and the follow-up emails. For a retailer they lift sales and engagement; for a shopper they save the effort of digging through thousands of near-identical listings.

Building a recommender is the easy part. Measuring it honestly is harder, and three problems keep coming back. First, the data is sparse: most users buy only a handful of items out of a large catalogue, so the user-item matrix is almost empty. Second, a careless random train/test split leaks the future: the model trains on a user's later purchases and is tested on earlier ones, which gives flattering scores that collapse in real use. Third, baselines are often weak or poorly tuned, and several studies have shown that many reported gains from neural recommenders disappear once simple baselines are tuned properly [10].

We fix the protocol first, compare against a strong simple benchmark, lock a test set that is touched only once, and report a negative result honestly if that is what the data shows. Our central question is deliberately narrow: can a small family of well-understood recommenders, trained on real Amazon review data, produce better top-N recommendations than a tuned popularity baseline on a chronologically held-out test period?

Our contributions are (a) a reproducible pipeline from raw Kaggle files to clean tables, with checksums and a fixed seed; (b) a time-aware split with a locked test period; (c) a fair comparison of popularity, item-kNN, SVD, ALS/BPR, content-based and hybrid models; (d) ranking, beyond-accuracy and cold-start analysis with bootstrap significance tests; (e) a signal check showing that a large Kaggle purchase file contains no learnable pattern, with a simulated positive control that proves the pipeline can find signal, and a switch to a real dataset (Amazon Beauty) where it does; and (f) an open discussion of limits. This is a research and learning project, not a deployed product.

## II. Related Work and Scope

Item-based collaborative filtering recommends items similar to those a user already liked, with similarity computed from co-occurrence patterns [1]. Matrix factorization, made famous by the Netflix Prize, learns low-dimensional user and item vectors whose dot product approximates the rating [2]. For implicit feedback such as purchases, weighted ALS treats observations as confidence-weighted positives [3], and Bayesian Personalized Ranking optimises the ordering of items directly [4]. Neural collaborative filtering [5] and session-based models such as GRU4Rec [6] are promising but data-hungry, so we leave them as future work. When an item has little or no history, its title, description, brand and category can still describe it, which is why content features are the usual remedy for cold start and are commonly blended with collaborative scores in hybrid designs [7].

Offline evaluation has well-known traps. Low rating-prediction error does not guarantee a good top-N list [8], so ranking metrics such as NDCG [9] are our main yardstick, and guidance on evaluating recommenders is well established [11]. We study one retailer's data, public snapshots and offline evaluation only, using models that run on a laptop. We do not run live A/B tests and make no claim that offline gains would carry over to real shoppers.

## III. Data, Target and Features

**Sources and provenance.** The headline dataset is the public **Amazon Beauty 5-core** review data (McAuley et al., 2014) [13]: user, product, star rating and review time for 198,502 reviews, plus a product metadata file with title, brand, categories, description, price and image. Two Kaggle datasets [12] were used first and now serve as sanity checks. Checksums, row counts and columns of the raw Kaggle files are recorded in `manifests/raw_manifest.json`.

**TABLE A. The datasets used and the role of each.**

| Dataset | What the files contain | Role |
|---|---|---|
| **Amazon Beauty 5-core** (reviews and metadata, 2014 snapshot) | 198,502 reviews by 22,363 users on 12,101 products, 2002 to 2014; titles, brands, categories, descriptions, prices | **Headline real dataset**: every main result, split by time |
| Amazon E-commerce (sharmajicoder), `amazon_ecommerce_1M.csv` | 1,000,000 purchase rows: user, product, category, brand, price, rating, date | Sanity check: found to contain no learnable pattern |
| Simulated positive control (generated by our code, clearly labelled synthetic) | 5,000 users, 1,986 items, 124,959 purchases with a planted taste structure and simulated item text | Sanity check: proves the pipeline detects signal when it exists |
| Amazon E-commerce Products & Reviews (lazylad99), `products.csv`, `reviews.csv` | 728 products and 6,327 reviews with text; no user ids | Proxy test of the content model; cannot be linked to purchases, never merged |

**Task.** Top-N recommendation (implicit feedback) is the primary task: each purchase is a positive interaction and the model ranks all items the user has not yet bought, so the item they actually buy next lands near the top. Rating prediction (RMSE and MAE) is secondary. Relevance is "any later review (purchase) in the evaluation period". The second planned definition, later interaction rated 4 or more, was not run (Section IV, deviations).

**Cleaning**, in order: drop exact duplicates and rows missing user, item or rating; standardise text and category strings; parse dates and drop impossible ones; keep one row per (user, item); apply an iterative k-core filter (k = 5, as planned; the Beauty file is already a 5-core, so nothing was removed) while keeping later cold users and items for the cold-start analysis; map ids to contiguous integers. (On the 1M file a 5-core was not feasible, since only 5,094 of 603,815 users have five or more purchases, so k = 2 was used there.)

**TABLE I. Dataset summary after cleaning (Amazon Beauty 5-core).**

| set                          |   interactions |   users |   items |   density |   sparsity % |   avg/user |   avg/item |   median/user |   median/item |   mean rating | share >=4   | date range               |
|:-----------------------------|---------------:|--------:|--------:|----------:|-------------:|-----------:|-----------:|--------------:|--------------:|--------------:|:------------|:-------------------------|
| Beauty 5-core (cleaned, k=5) |        198,502 |  22,363 |  12,101 |  0.000734 |      99.9266 |       8.88 |       16.4 |             6 |             9 |          4.19 | 77.7%       | 2002-06-12 to 2014-07-23 |

Every user and item has at least five reviews (k = 5), a median of 6 reviews per user and 9 per item, and the data are 99.93% sparse.

**Features.** The collaborative signal is the sparse user-item matrix (ratings for the SVD, binary or confidence-weighted purchases for the others). Item content comes from TF-IDF vectors of product text (title, brand, category path and description from the product metadata file; simulated item text on the control). Recency is used for recency-weighted popularity and for the split, never as a source of leakage.

## IV. Methodology

**Splits and leakage control.** Interactions are split by date: the earliest 70% for training, the next 15% for validation (model selection only) and the final 15% as a locked test set, scored once after all settings are frozen. Test interactions are tagged warm, cold-item or cold-user. Similarity matrices, TF-IDF vocabularies, normalisation constants and popularity counts are fitted on training data only. The seed is 42 and library versions are pinned in `requirements.txt`. The frozen settings were committed in `manifests/frozen_settings.json` before any test scoring, and the scoring script refuses to overwrite a saved test result.

**Models.** (1) Tuned popularity: all-time, 30/90/365-day windows and a Bayesian-average rating, with the variant chosen on validation. (2) Item-kNN: cosine similarity with shrinkage and a top-k neighbourhood. (3) Biased SVD, trained by SGD to predict ratings. (4) ALS and BPR from the `implicit` library. (5) Content-based: TF-IDF text vectors, ranking unseen items by cosine similarity to what the user liked. (6) Hybrid: row-max-normalised collaborative and content scores blended with a weight tuned on validation, with an adaptive rule that uses content alone for items with no history and popularity for users with no history. Settings are chosen by grid search on validation NDCG@10.

**TABLE II. Tuning grids.**

| Model | Settings searched |
|---|---|
| Popularity | variant in {all-time, recency-weighted, Bayesian-average}; window in {30, 90, 365 days} |
| Item-kNN | k in {20, 50, 100}; shrinkage in {0, 10, 50}; Beauty also k in {100, 200}, shrinkage in {100, 200, 500} |
| SVD | factors in {32, 64, 128}; regularisation in {0.02, 0.05, 0.1}; learning rate in {0.005, 0.01}; Beauty also factors in {128, 256} |
| ALS / BPR | factors in {32, 64, 128}; regularisation in {0.01, 0.1}; ALS confidence alpha in {10, 40}; BPR learning rate in {0.01, 0.05}; Beauty also factors in {128, 256}, regularisation 0.001, alpha 100, BPR learning rate 0.1 |
| Content-based | max features in {5k, 20k}; n-gram range in {(1,1), (1,2)} (Beauty best: 20k, (1,2)) |
| Hybrid | blend weight w in {0, 0.1, ..., 1.0} |

The full grid was run on Beauty and on the control data; the 1M file used a reduced grid on a 5,000-user sample because of the 2-core, 3 GB machine. On Beauty the best validation settings sat on the edge of the planned grids, so an extended grid (the "also" entries) was added, on validation only and before any test scoring. The settings chosen on validation and then frozen are in `manifests/frozen_settings.json`.

**Metrics.** Rating prediction: RMSE = sqrt((1/|T|) * sum over (u,i) in T of (r_ui - r_hat_ui)^2) and MAE. Ranking, with K = 5, 10 and 20: Precision@K, Recall@K, MAP@K, HitRate@K and NDCG@K = DCG@K / IDCG@K, where DCG@K = sum for j = 1 to K of rel_j / log2(j + 1) [9]. Beyond accuracy: catalogue coverage (share of items ever recommended), diversity (one minus the mean pairwise TF-IDF cosine similarity inside a list; only where item text exists) and novelty (mean of minus log2 of training popularity).

**Significance testing.** For each model against tuned popularity, per-user NDCG@10 differences on the locked test are resampled over users 1,000 times (paired bootstrap) for a 95% interval and a p-value, with a Bonferroni adjustment over all comparisons made in that dataset. A model counts as an improvement only if its interval excludes zero and the adjusted p-value is below 0.05.

**Sanity-check datasets.** Because the large Kaggle purchase file turned out to contain no signal, a simulated dataset with a planted taste structure (5,000 users, 1,986 items) is used as a clearly labelled positive control, and the 1M file is kept as a negative control. Neither is presented as evidence about real shoppers.

**Deviations from the partial report.**

1. **The headline dataset is Amazon Beauty 5-core, not the two Kaggle files.** The 1M file contains no learnable pattern (four diagnostics) and the Kaggle products and reviews files have no user ids. The switch was made after seeing the 1M results, but before any Beauty data were scored on the test period.
2. k-core is 5 on Beauty (as planned) and 2 on the 1M file.
3. The Table II grids were extended on Beauty (validation only) because the best settings sat on grid edges.
4. The datasets are never merged. The 1M file is judged on a seeded sample of 5,000 warm users with reduced grids.
5. Final models are refit on train plus validation with frozen settings; ties in validation go to the first setting in grid order.
6. BPR is multi-threaded and differs by about 0.0002 NDCG@10 between runs.
7. Diversity is computed only where item text exists.
8. A single seed (42) was used; the secondary leave-last-out protocol, the second relevance definition (rating 4 or more) and the optional neural extension were not run.

## V. Results

### A. Data, split and signal check

Three datasets were used, for three different jobs, and are never merged. The **headline dataset is real: Amazon Beauty 5-core** (22,363 users, 12,101 products, 198,502 reviews, June 2002 to July 2014, with product titles, brands, categories, descriptions, prices and images), the standard public benchmark in which every user and product has at least five reviews. A review is treated as evidence of a purchase. The two Kaggle files used first act as **sanity checks**: *amazon_ecommerce_1M* (1,000,000 purchase rows) and a clearly labelled *simulated positive control* (5,000 users, 1,986 items, 124,959 purchases with a planted taste structure). The 1M file was tried first and turned out to contain no learnable pattern (Section V-F); we therefore moved to a real dataset that does, and say so openly: the switch was made after seeing the 1M results, but before any Beauty data were scored on the test period.

Table I (Section III) gives the Beauty statistics; there are no cleaning losses because the 5-core file is already deduplicated. The time-based split (earliest 70% train, next 15% validation, final 15% locked test) is:

| set | rows | share | from | to |
|---|---|---|---|---|
| train | 138,723 | 69.9% | 2002-06-12 | 2013-12-26 |
| validation | 29,931 | 15.1% | 2013-12-27 | 2014-04-08 |
| test (locked) | 29,848 | 15.0% | 2014-04-09 | 2014-07-23 |

Slices (relative to train only):

| set | warm | cold-item | cold-user |
|---|---|---|---|
| validation | 16,953 (56.6%) | 4,272 (14.3%) | 8,706 (29.1%) |
| test | 11,938 (40.0%) | 5,725 (19.2%) | 12,185 (40.8%) |

More than half of the test purchases come from users or products with no earlier history, which is why the cold-start experiments (E2) matter on this dataset.

**Signal check.** The same four diagnostics used on the 1M file were run on Beauty:

| check | observed | what pure chance predicts |
|---|---|---|
| Item popularity: variance / mean of interactions per item | 34.0 | about 1 (Poisson) |
| Correlation of item popularity, first half vs second half of time | 0.300 | about 0 |
| Two purchases by the same user share a category path | 0.110 | 0.021 (two random items) |
| Two purchases by the same user share a brand | 0.110 | 0.033 (two random items) |

All four differ strongly from what chance predicts, so this file has structure for models to learn. On the 1M file the same checks match chance (Fig. 1).

![Fig. 1](figures/fig_signal_diagnostics.png)

*Fig. 1. Diagnostics on amazon_ecommerce_1M against pure chance (no learnable pattern).*

### B. Locked-test ranking results (E1, Table III)

Settings were chosen on validation, frozen in `manifests/frozen_settings.json` and committed before any test scoring; final models were refit on train plus validation and every model was scored on the locked test exactly once. Rankings are scored for the 7,272 users with history before the test period, never recommending an item the user already bought; users with no history are handled separately (E2).

**Table III (Amazon Beauty, locked test, K = 10).**

| Model                                    |    P@10 |    R@10 |   NDCG@10 |   MAP@10 |   Coverage |   Diversity |   Novelty |
|:-----------------------------------------|--------:|--------:|----------:|---------:|-----------:|------------:|----------:|
| Random (reference)                       | 0.00019 | 0.00062 |   0.00035 |  0.00014 |    0.99694 |     0.98277 |  14.1498  |
| Popularity (tuned)                       | 0.00286 | 0.00913 |   0.00587 |  0.00283 |    0.00132 |     0.94749 |   9.16697 |
| Item-kNN                                 | 0.00824 | 0.02848 |   0.02065 |  0.01245 |    0.66242 |     0.90311 |  11.6976  |
| SVD (biased, rating-trained)             | 0.00081 | 0.00237 |   0.00155 |  0.00073 |    0.13586 |     0.97823 |  12.6722  |
| ALS                                      | 0.00903 | 0.03124 |   0.02237 |  0.01352 |    0.51954 |     0.90915 |  11.9471  |
| BPR                                      | 0.00646 | 0.02119 |   0.01565 |  0.00944 |    0.83216 |     0.91695 |  13.3631  |
| Content-based (TF-IDF)                   | 0.00862 | 0.03167 |   0.02265 |  0.01432 |    0.7777  |     0.64797 |  13.6019  |
| Hybrid (adaptive: content for new items) | 0.0107  | 0.03897 |   0.02666 |  0.01608 |    0.71176 |     0.78284 |  12.6567  |

All personalised models except SVD beat tuned popularity (NDCG@10 0.0059). The hybrid is best at 0.0267, about **4.5 times** popularity; ALS (0.0224), content-based (0.0226) and item-kNN (0.0207) follow, then BPR (0.0157). Random scores 0.00035. Absolute values look small because each user chooses among about 12,000 products and typically has one or two later purchases; the comparison between models is what carries the information.

![Fig. 2](figures/fig_models_test.png)

*Fig. 2. Locked-test NDCG@10 on Amazon Beauty with 95% bootstrap intervals over users. Bars start at zero.*

### C. Significance tests

Per-user NDCG@10 differences against tuned popularity were tested with a paired bootstrap (1,000 resamples of users, 95% interval) with a Bonferroni adjustment for the 7 comparisons. A model counts as an improvement only if its interval excludes zero and the adjusted p-value is below 0.05.

| Model vs tuned popularity                |   mean NDCG@10 difference |   95% CI low |   95% CI high | p      | p (Bonferroni, x7)   | reading                |
|:-----------------------------------------|--------------------------:|-------------:|--------------:|:-------|:---------------------|:-----------------------|
| Random (reference)                       |                 -0.005515 |    -0.006474 |     -0.004571 | <0.001 | <0.007               | worse than popularity  |
| Item-kNN                                 |                  0.014783 |     0.012643 |      0.017082 | <0.001 | <0.007               | better than popularity |
| SVD (biased, rating-trained)             |                 -0.004316 |    -0.005368 |     -0.003233 | <0.001 | <0.007               | worse than popularity  |
| ALS                                      |                  0.016505 |     0.014189 |      0.018769 | <0.001 | <0.007               | better than popularity |
| BPR                                      |                  0.009783 |     0.007756 |      0.011965 | <0.001 | <0.007               | better than popularity |
| Content-based (TF-IDF)                   |                  0.016785 |     0.014301 |      0.019496 | <0.001 | <0.007               | better than popularity |
| Hybrid (adaptive: content for new items) |                  0.02079  |     0.018374 |      0.023412 | <0.001 | <0.007               | better than popularity |

Item-kNN, ALS, BPR, content-based and the hybrid are improvements over tuned popularity; every interval is far from zero. SVD is significantly *worse* than popularity. Because the hybrid combines two components, it was also compared with each (second table below): it beats both ALS and the content-only model, but by a small margin (about 0.004 NDCG@10).

| Hybrid minus           |   mean NDCG@10 difference |   95% CI low |   95% CI high | p (Bonferroni, x2)   |
|:-----------------------|--------------------------:|-------------:|--------------:|:---------------------|
| ALS                    |                  0.004286 |     0.002081 |      0.006522 | <0.002               |
| Content-based (TF-IDF) |                  0.004005 |     0.001887 |      0.006125 | <0.002               |

### D. Rating-prediction error (Table IV)

| Model                   |   RMSE |    MAE | RMSE change vs global mean   | Bootstrap 95% CI   |
|:------------------------|-------:|-------:|:-----------------------------|:-------------------|
| Global mean rating      | 1.145  | 0.9298 | n/a                          | n/a                |
| User/item bias baseline | 1.0968 | 0.8657 | -0.0482                      | [-0.0518, -0.0445] |
| SVD                     | 1.1051 | 0.8671 | -0.0398                      | [-0.0439, -0.0356] |

The user/item bias baseline (RMSE 1.0968) and SVD (RMSE 0.0012) both improve on the global mean (1.1450), and the intervals exclude zero, but SVD does not improve on the simple bias baseline and, as Table III shows, ranks below popularity. A lower RMSE therefore did not turn into better recommendations, and no RMSE gain is presented as a ranking gain.

### E. Experiments E2 to E7 (Amazon Beauty)

**E2, cold start.** On the locked test, 8,675 of 29,848 rows (29.1%) are by users with no earlier history and 2,864 (9.6%) are of items with no history. For the cold users no personalised model can score anything; popularity is the fallback and beats random (NDCG@10 0.0168 against 0.0005, interval excluding zero). For cold items the picture is sharply different:

Same scoring pass as Table III; a user can appear in both columns.

| Model                                    |   HR@10 on purchases of NEW items |   NDCG@10 on purchases of NEW items |   users with a new-item purchase |   NDCG@10 on ordinary purchases |   users with an ordinary purchase |
|:-----------------------------------------|----------------------------------:|------------------------------------:|---------------------------------:|--------------------------------:|----------------------------------:|
| Random (reference)                       |                           0.00233 |                             0.00073 |                             1288 |                         0.00029 |                              6898 |
| Popularity (tuned)                       |                           0       |                             0       |                             1288 |                         0.00626 |                              6898 |
| Item-kNN                                 |                           0       |                             0       |                             1288 |                         0.02311 |                              6898 |
| SVD (biased, rating-trained)             |                           0       |                             0       |                             1288 |                         0.0017  |                              6898 |
| ALS                                      |                           0       |                             0       |                             1288 |                         0.02528 |                              6898 |
| BPR                                      |                           0       |                             0       |                             1288 |                         0.01825 |                              6898 |
| Content-based (TF-IDF)                   |                           0.03571 |                             0.01326 |                             1288 |                         0.02277 |                              6898 |
| Hybrid (adaptive: content for new items) |                           0.07997 |                             0.02945 |                             1288 |                         0.02493 |                              6898 |

Collaborative models, popularity and SVD can never recommend an item that has no purchase history (HitRate@10 = 0). The content-based model finds 0.9% of those purchases and the adaptive hybrid **1.1%**, because it scores such items by their text alone. Ordinary purchases are not hurt: the hybrid's NDCG@10 on them is 0.0249, equal to ALS (0.0253) and above item-kNN (0.0231).

**E3, training size (validation only).** Collaborative models need data. With 25% of the training history item-kNN (0.0042) and ALS (0.0034) are *below* popularity (0.0058); with all of it they reach 0.0179 and 0.0171, about three times popularity. Content-based is the most data-efficient (0.0143 at 25%) because it uses item text, not co-purchases.

![Fig. 3](figures/fig_e3_training_size_beauty.png)

*Fig. 3. Validation NDCG@10 against the share of training data used (Amazon Beauty).*

**E4, hybrid ablation (validation).** A global blend of item-kNN and content scores (w = 0.5) raises NDCG@10 over either part alone (0.0216 against 0.0181 and 0.0187), so on real data the content signal helps even for users with history. The adaptive rule, which uses content alone for items with no history, adds the new-item gain on top.

| model                                                |   NDCG@10 (all purchases) |   HR@10 on purchases of NEW items |   NDCG@10 on purchases of NEW items |   NDCG@10 on ordinary purchases |   coverage@10 |
|:-----------------------------------------------------|--------------------------:|----------------------------------:|------------------------------------:|--------------------------------:|--------------:|
| tuned popularity                                     |                   0.00658 |                           0       |                             0       |                         0.00771 |       0.00124 |
| collaborative only (item-kNN)                        |                   0.01807 |                           0       |                             0       |                         0.02128 |       0.71726 |
| content only                                         |                   0.01874 |                           0.0586  |                             0.01857 |                         0.01761 |       0.81382 |
| hybrid, global weight w=0.5                          |                   0.02155 |                           0.00383 |                             0.00142 |                         0.02509 |       0.7538  |
| hybrid, adaptive (content-only for new items), w=0.5 |                   0.02627 |                           0.09474 |                             0.03481 |                         0.02259 |       0.75405 |

**E5, list length.** Hybrid NDCG@10 is 0.0267 (HitRate@10 8.3%); at K = 20 the HitRate is 11.6%, against 4.0% for popularity. The model order is Hybrid > Content > ALS > Item-kNN > BPR > Popularity > SVD > Random at K = 5 and 10, and ALS overtakes content at K = 20, so the middle of the ranking depends on K while the ends (hybrid first; popularity, SVD and random last) do not.

**E6, accuracy against coverage and novelty.** Popularity recommends a tiny part of the catalogue (coverage 0.001) and the most popular items (novelty 9.2 bits); the hybrid covers 71% of the catalogue at higher novelty (12.7 bits) and, unlike random (coverage 1.00), is accurate. Content-based lists are the least diverse in text (diversity 0.65) because they recommend items that read alike, and the hybrid sits between content and the collaborative models (0.78).

![Fig. 4](figures/e6_test_beauty.png)

*Fig. 4. Accuracy against catalogue coverage and novelty on Amazon Beauty (locked test).*

**E7, validation against locked test.** The order of the models changes slightly between validation and test (content and ALS/item-kNN swap places) but the extremes are stable: hybrid first, popularity, SVD and random last. Item-kNN, ALS, BPR and content score higher on test than on validation because the final models are refit on train plus validation, i.e. they see 20% more history.

Validation = fit on train, scored on validation (dry run). Test = fit on train + validation, scored on the locked test.

| Model      |   validation NDCG@10 |   locked-test NDCG@10 |   change |   validation HR@10 |   test HR@10 |
|:-----------|---------------------:|----------------------:|---------:|-------------------:|-------------:|
| Random     |              0.00073 |               0.00035 | -0.00038 |            0.0031  |      0.00193 |
| Popularity |              0.00658 |               0.00587 | -0.00072 |            0.02805 |      0.02654 |
| Item-kNN   |              0.01803 |               0.02065 |  0.00262 |            0.05635 |      0.06449 |
| SVD        |              0.00138 |               0.00155 |  0.00017 |            0.00582 |      0.0077  |
| ALS        |              0.01791 |               0.02237 |  0.00446 |            0.05687 |      0.07357 |
| BPR        |              0.00939 |               0.01565 |  0.00626 |            0.0327  |      0.05322 |
| Content    |              0.01872 |               0.02265 |  0.00393 |            0.0645  |      0.06972 |
| Hybrid     |              0.02625 |               0.02666 |  0.00041 |            0.08349 |      0.08347 |

Order on validation: Hybrid > Content > Item-kNN > ALS > BPR > Popularity > SVD > Random

Order on the locked test: Hybrid > Content > ALS > Item-kNN > BPR > Popularity > SVD > Random. The order changed.

### F. Sanity checks on the two earlier datasets

**Simulated positive control.** To prove the evaluation code can detect signal, all models were run on a simulated dataset with a planted taste structure. Item-kNN reaches NDCG@10 0.199, ALS 0.193, the hybrid 0.194 and BPR 0.147, against 0.063 for tuned popularity, while the rating-trained SVD (0.057) does not beat popularity: the same pattern as on real data. Full tables are in `reports/final_test_control.md`. These numbers are higher than on Beauty because the simulated structure is cleaner than real behaviour, so they say nothing about real shoppers; they only show the pipeline works.

**amazon_ecommerce_1M.** On this file every model scores essentially zero (NDCG@10 between 0.00000 and 0.00020), random is on par with the rest, and no model is distinguishable from tuned popularity. The diagnostics in Fig. 1 explain why (purchases look randomly assigned), so this is a property of the file, not of the methods. Full tables are in `reports/final_test_main.md`.

**Content model on Amazon product text.** The Kaggle products and reviews files contain no user ids, so only a proxy task was possible: retrieving products of the same leaf category by TF-IDF similarity. On held-out products NDCG@10 is **0.567** against 0.038 for random and 0.046 for popularity by rating count, so product text carries real signal; this supports using text for cold items on Beauty.

## VI. Discussion and Limitations

**What the results say.** On a real dataset with learnable structure, personalised models beat the best-seller baseline by a wide, statistically solid margin: the hybrid scores 4.5 times tuned popularity, and its hit rate at 10 is 8.3% against 2.7%. The same pipeline reported no gain on a file with no structure and a clear gain on a simulated one, which is what a trustworthy evaluation should do.

**Ranking and rating are different tasks.** SVD, trained to predict star ratings, improved RMSE yet ranked below popularity on real and simulated data, while ALS, BPR and item-kNN, which learn from who bought what, ranked far better. The training objective must match what the page has to do.

**Content matters on real data.** On the simulated control the content signal added nothing for users with history; on Beauty the text of the product (title, brand, category, description) is informative enough that the content-only model matches item-kNN and the blend beats both. Its largest benefit is cold start: the hybrid finds about 1% of purchases of brand-new products, where collaborative filtering finds none.

**Data volume matters.** Collaborative models fall below popularity with a quarter of the history. A shop with sparse history should start from popularity plus content and switch to collaborative models as purchases accumulate.

**Limitations.**
1. Amazon Beauty 5-core is a pre-filtered benchmark: every user and item has at least five reviews, which makes the data denser and more predictable than a live shop and excludes the quietest customers. A review is also not the same as a purchase, and the data end in 2014.
2. Absolute scores are small (NDCG@10 about 0.03 at best), as is normal for 12,000-item catalogues, and only 7,272 of the test users had history; 40.8% of test rows come from users with none and were handled only by popularity.
3. The evaluation is offline. Interactions were shaped by what the site showed, and offline gains need not carry over to real shoppers; no A/B test was run.
4. For Beauty the validation grids in Table II were extended (larger neighbourhoods and shrinkage for item-kNN; 128 and 256 factors and weaker regularisation for the factor models), because the best validation settings sat on the edge of the planned grids. This was done on validation only, before test scoring, but several final settings still lie on the edge of the extended grid (ALS and BPR use 256 factors), so slightly larger models might score a little higher.
5. One seed was used. BPR is multi-threaded and varies by about 0.0002 NDCG@10 between runs. Bonferroni adjustment covers comparisons within one dataset only.
6. The choice of Beauty was made after the 1M file failed; Beauty's test period was not scored until its settings were frozen, but the switch is a post-hoc decision and is disclosed as such. The simulated control favours the model types it was built around and is never presented as real data.
7. Ties in validation were resolved by taking the first setting in grid order. Final models were refit on train plus validation, so validation and test numbers are not strictly comparable.
8. Diversity depends on item text. The second relevance definition (later rating of 4 or more), the leave-last-out protocol and the optional neural extension were not run.

## VII. Project Status

| Phase | Focus | Status |
|---|---|---|
| 1 | Data download, schema check, manifest, cleaning, Table I | Done (`src/make_manifest.py`, `clean.py`) |
| 2 | Baselines, split and evaluation harness | Done (`split.py`, `metrics.py`, `evaluate.py`, `baselines.py`) |
| 3 | Item-kNN, SVD, ALS/BPR, TF-IDF content model, tuning | Done (`run_models.py`, `content_amazon.py`) |
| 4 | Hybrid, cold-start rules, ablations (E2 to E4) | Done (`hybrid.py`, `e3_sweep.py`) |
| 5 | Freeze settings, score locked test once, bootstrap tests, E5 to E7 | Done (`freeze.py`, `final_eval.py`, `finalize.py`) |
| 6 | Results, discussion, limitations, figures, demo, viva notes | Done (`build_report.py`, `build_figures.py`, `demo.py`) |
| 7 | Real dataset (Amazon Beauty 5-core): rerun of phases 2 to 5 and report rebuild | Done (`prepare_beauty.py`, `hybrid_beauty.py`, `final_eval.py beauty`) |

Code, manifests, reports and figures are in the project repository; `REPRODUCE.md` lists the exact commands.

## VIII. Conclusion

We built a leakage-aware recommender study with a time-based split, settings frozen before scoring and a test period scored once. On real Amazon Beauty reviews, item-kNN, ALS, BPR, a TF-IDF content model and a hybrid all beat a tuned popularity baseline with 95% intervals far from zero, and the hybrid, which blends collaborative and text signals and uses text alone for new products, was best: about 4.5 times the popularity score and the only approach that recovers purchases of brand-new items. A rating-trained SVD did not beat popularity, collaborative models need history to work, and a random-looking purchase file (amazon_ecommerce_1M) correctly produced no gains. Taken together, the project shows how to evaluate recommenders honestly, including how to recognise when a dataset cannot support any personalised model and when a real one can.

## References

[1] B. Sarwar, G. Karypis, J. Konstan, and J. Riedl, "Item-based collaborative filtering recommendation algorithms," in Proc. 10th Int. Conf. World Wide Web (WWW), 2001, pp. 285-295.

[2] Y. Koren, R. Bell, and C. Volinsky, "Matrix factorization techniques for recommender systems," Computer, vol. 42, no. 8, pp. 30-37, 2009.

[3] Y. Hu, Y. Koren, and C. Volinsky, "Collaborative filtering for implicit feedback datasets," in Proc. IEEE Int. Conf. Data Mining (ICDM), 2008, pp. 263-272.

[4] S. Rendle, C. Freudenthaler, Z. Gantner, and L. Schmidt-Thieme, "BPR: Bayesian personalized ranking from implicit feedback," in Proc. 25th Conf. Uncertainty in Artificial Intelligence (UAI), 2009, pp. 452-461.

[5] X. He, L. Liao, H. Zhang, L. Nie, X. Hu, and T.-S. Chua, "Neural collaborative filtering," in Proc. 26th Int. Conf. World Wide Web (WWW), 2017, pp. 173-182.

[6] B. Hidasi, A. Karatzoglou, L. Baltrunas, and D. Tikk, "Session-based recommendations with recurrent neural networks," in Proc. Int. Conf. Learning Representations (ICLR), 2016.

[7] R. Burke, "Hybrid recommender systems: Survey and experiments," User Modeling and User-Adapted Interaction, vol. 12, no. 4, pp. 331-370, 2002.

[8] P. Cremonesi, Y. Koren, and R. Turrin, "Performance of recommender algorithms on top-N recommendation tasks," in Proc. 4th ACM Conf. Recommender Systems (RecSys), 2010, pp. 39-46.

[9] K. Jarvelin and J. Kekalainen, "Cumulated gain-based evaluation of IR techniques," ACM Trans. Information Systems, vol. 20, no. 4, pp. 422-446, 2002.

[10] M. Ferrari Dacrema, P. Cremonesi, and D. Jannach, "Are we really making much progress? A worrying analysis of recent neural recommendation approaches," in Proc. 13th ACM Conf. Recommender Systems (RecSys), 2019, pp. 101-109.

[11] J. L. Herlocker, J. A. Konstan, L. G. Terveen, and J. T. Riedl, "Evaluating collaborative filtering recommender systems," ACM Trans. Information Systems, vol. 22, no. 1, pp. 5-53, 2004.

[12] Kaggle, "Amazon E-commerce" (sharmajicoder) and "Amazon E-commerce Products & Reviews Dataset" (lazylad99), public datasets, accessed October 2026.
[13] J. McAuley, C. Targett, Q. Shi and A. van den Hengel, "Image-based recommendations on styles and substitutes," in Proc. SIGIR, 2015, pp. 43-52 (Amazon product data, Beauty 5-core and metadata).
