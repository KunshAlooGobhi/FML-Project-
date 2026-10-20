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

@@TABLE1@@

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
