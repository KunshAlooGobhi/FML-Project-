# E-Commerce Recommendation System (Fundamentals of Machine Learning project)

A leakage-aware study of collaborative, content-based and hybrid recommenders on Amazon product and review data.
Authors: Rishit Agarwal (245890302), Kunsh Mehra (245890196). Full report: `reports/final_report.md` (and `.docx`). Every number comes from code in `src/` and files in `reports/`.

## Headline result (real data: Amazon Beauty 5-core, locked test, NDCG@10)
22,363 users, 12,101 products, 198,502 reviews (2002 to 2014). Time-based split 70/15/15; settings frozen on validation and committed before the test was scored once.

| Model | NDCG@10 | HitRate@10 |
|---|---|---|
| **Hybrid** (item-kNN + TF-IDF text, text only for new items) | **0.0267** | **8.3%** |
| Content-based (TF-IDF) | 0.0227 | 7.0% |
| ALS | 0.0224 | 7.4% |
| Item-kNN | 0.0207 | 6.4% |
| BPR | 0.0157 | 5.3% |
| Tuned popularity (baseline) | 0.0059 | 2.7% |
| SVD (rating-trained) | 0.0016 | 0.8% |
| Random | 0.0004 | 0.2% |

Hybrid is about 4.5x popularity; item-kNN, ALS, BPR, content and hybrid beat popularity with 95% bootstrap CIs far from zero (Bonferroni, 7 comparisons); SVD is significantly worse. On purchases of brand-new products the hybrid finds 8.0% (HitRate@10) and every collaborative model 0%. With 25% of the training data item-kNN and ALS fall below popularity. Details: `reports/final_test_beauty.md`, `reports/hybrid_validation_beauty.md`, `reports/e3_training_size_beauty.md`.

## Sanity checks (kept on purpose)
- `amazon_ecommerce_1M.csv` (Kaggle, 1M rows) behaves like random purchasing (`reports/signal_diagnostics.md`): no model beats tuned popularity (`reports/final_test_main.md`).
- A SIMULATED positive control (`src/make_control_data.py`, clearly synthetic) shows the pipeline detects signal when it exists (`reports/final_test_control.md`).
- Kaggle `products.csv` / `reviews.csv` (no user ids) test the content model on a proxy task (`reports/content_amazon.md`).

## Data (not committed)
Beauty: `reviews_Beauty_5.json.gz` and `meta_Beauty.json.gz` (Amazon product data, McAuley et al.), plus the three Kaggle files, in the folder that contains `recsys-project/`. Checksums of the Kaggle files: `manifests/raw_manifest.json`.

## Deviations from the partial report (all documented)
1. The headline dataset is Amazon Beauty 5-core; the Kaggle files turned out to have no learnable pattern (or no users). The switch was made after the 1M results but before any Beauty test scoring.
2. k-core is 5 on Beauty, 2 on the 1M file (only 5,094 of 603,815 users have >= 5 purchases).
3. On Beauty the Table II grids were extended (validation only) because best settings sat on grid edges.
4. Datasets are never merged. The 1M file is judged on a seeded sample of 5,000 warm users with reduced grids.
5. Final models are refit on train + validation with frozen settings; ties in validation go to the first setting in grid order.
6. BPR (`implicit`, multi-threaded) varies by about 0.0002 NDCG@10 between runs; single seed (42).
7. Diversity needs item text. Rating >= 4 relevance, leave-last-out and the neural extension were not run.

## Run
See `REPRODUCE.md` for every command. Quick start:
```
pip install -r requirements.txt
streamlit run src/app.py        # UI on real Beauty shoppers
python src/demo.py user         # terminal demo: python src/demo.py product mascara
```
