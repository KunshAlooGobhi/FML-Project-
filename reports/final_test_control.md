# LOCKED TEST: control (from src/final_eval.py + src/finalize.py)

Settings frozen in manifests/frozen_settings.json. Warm users with history in the fit data: 4,912. Fit data = train + validation; scored on the locked test.

## Table III: ranking results, K = 10 (E1)

| Model                                    |    P@10 |    R@10 |   NDCG@10 |   MAP@10 |   Coverage |   Diversity |   Novelty |
|:-----------------------------------------|--------:|--------:|----------:|---------:|-----------:|------------:|----------:|
| Random (reference)                       | 0.00183 | 0.00424 |   0.00307 |  0.00122 |     1      |     0.98558 |  12.1436  |
| Popularity (tuned)                       | 0.03039 | 0.07994 |   0.06285 |  0.03207 |     0.0095 |     0.97372 |   7.32537 |
| Item-kNN                                 | 0.08673 | 0.23096 |   0.19883 |  0.12157 |     0.126  |     0.95889 |   8.20844 |
| SVD (biased, rating-trained)             | 0.02726 | 0.07278 |   0.05715 |  0.02919 |     0.016  |     0.98315 |   7.6036  |
| ALS                                      | 0.08565 | 0.22627 |   0.19294 |  0.11678 |     0.27   |     0.96576 |   8.73704 |
| BPR                                      | 0.06338 | 0.16962 |   0.14687 |  0.08664 |     0.507  |     0.95248 |  10.0925  |
| Content-based (TF-IDF)                   | 0.00735 | 0.01943 |   0.01369 |  0.00622 |     0.892  |     0.89012 |  12.1633  |
| Hybrid (adaptive: content for new items) | 0.08369 | 0.22283 |   0.19367 |  0.11833 |     0.139  |     0.95468 |   8.7207  |

## Paired bootstrap on per-user NDCG@10 (1000 resamples of users; 7 comparisons made; Bonferroni-adjusted)

A model counts as an improvement only if its 95% interval excludes zero AND the Bonferroni-adjusted p < 0.05. With 1,000 resamples the smallest reportable p is below 1/1,000.

| Model vs tuned popularity                |   mean NDCG@10 difference |   95% CI low |   95% CI high | p      | p (Bonferroni, x7)   | reading                |
|:-----------------------------------------|--------------------------:|-------------:|--------------:|:-------|:---------------------|:-----------------------|
| Random (reference)                       |                 -0.05978  |    -0.063198 |     -0.056255 | <0.001 | <0.007               | worse than popularity  |
| Item-kNN                                 |                  0.135978 |     0.130655 |      0.14157  | <0.001 | <0.007               | better than popularity |
| SVD (biased, rating-trained)             |                 -0.005698 |    -0.007543 |     -0.003947 | <0.001 | <0.007               | worse than popularity  |
| ALS                                      |                  0.130093 |     0.124279 |      0.136098 | <0.001 | <0.007               | better than popularity |
| BPR                                      |                  0.084026 |     0.078016 |      0.090337 | <0.001 | <0.007               | better than popularity |
| Content-based (TF-IDF)                   |                 -0.049162 |    -0.052885 |     -0.045198 | <0.001 | <0.007               | worse than popularity  |
| Hybrid (adaptive: content for new items) |                  0.130824 |     0.125376 |      0.1363   | <0.001 | <0.007               | better than popularity |

## E5: list-length sweep (K = 5, 10, 20)

| Model      |   NDCG@5 |   NDCG@10 |   NDCG@20 |    HR@5 |   HR@10 |   HR@20 |
|:-----------|---------:|----------:|----------:|--------:|--------:|--------:|
| Random     |  0.00211 |   0.00307 |   0.00496 | 0.00936 | 0.01812 | 0.03685 |
| Popularity |  0.05068 |   0.06285 |   0.08403 | 0.17386 | 0.2612  | 0.39719 |
| Item-kNN   |  0.17294 |   0.19883 |   0.23155 | 0.45297 | 0.57573 | 0.69422 |
| SVD        |  0.04436 |   0.05715 |   0.07522 | 0.1478  | 0.23738 | 0.35322 |
| ALS        |  0.165   |   0.19294 |   0.22587 | 0.44646 | 0.56759 | 0.68546 |
| BPR        |  0.12768 |   0.14687 |   0.17405 | 0.3581  | 0.46926 | 0.58836 |
| Content    |  0.00909 |   0.01369 |   0.02007 | 0.034   | 0.07125 | 0.12968 |
| Hybrid     |  0.16935 |   0.19367 |   0.22513 | 0.44707 | 0.56637 | 0.68424 |

Model order by NDCG@5 / @10 / @20: Item-kNN > Hybrid > ALS > BPR > Popularity > SVD > Content > Random | Item-kNN > Hybrid > ALS > BPR > Popularity > SVD > Content > Random | Item-kNN > ALS > Hybrid > BPR > Popularity > SVD > Content > Random. The order changes with K.

## Table IV: rating-prediction error on the locked test (18,792 rows)

| Model                   |   RMSE |    MAE | RMSE change vs global mean   | Bootstrap 95% CI   |
|:------------------------|-------:|-------:|:-----------------------------|:-------------------|
| Global mean rating      | 0.6728 | 0.6009 | n/a                          | n/a                |
| User/item bias baseline | 0.6334 | 0.5325 | -0.0394                      | [-0.0427, -0.0364] |
| SVD                     | 0.636  | 0.5344 | -0.0368                      | [-0.0404, -0.0333] |

## E2: cold slices

Rows by slice relative to the fit data: {'warm': 18776, 'cold-item': 16, 'cold-user': 0}.

No cold users in this dataset/split.

## E7: validation vs locked test

Validation = fit on train, scored on validation (dry run). Test = fit on train + validation, scored on the locked test.

| Model      |   validation NDCG@10 |   locked-test NDCG@10 |   change |   validation HR@10 |   test HR@10 |
|:-----------|---------------------:|----------------------:|---------:|-------------------:|-------------:|
| Random     |              0.00299 |               0.00307 |  7e-05   |            0.01856 |      0.01812 |
| Popularity |              0.05919 |               0.06285 |  0.00366 |            0.25209 |      0.2612  |
| Item-kNN   |              0.18255 |               0.19883 |  0.01628 |            0.55537 |      0.57573 |
| SVD        |              0.05273 |               0.05715 |  0.00442 |            0.21987 |      0.23738 |
| ALS        |              0.16864 |               0.19294 |  0.02431 |            0.53539 |      0.56759 |
| BPR        |              0.13672 |               0.14687 |  0.01015 |            0.45401 |      0.46926 |
| Content    |              0.01295 |               0.01369 |  0.00074 |            0.06975 |      0.07125 |
| Hybrid     |              0.17856 |               0.19367 |  0.01511 |            0.54456 |      0.56637 |

Order on validation: Item-kNN > Hybrid > ALS > BPR > Popularity > SVD > Content > Random

Order on the locked test: Item-kNN > Hybrid > ALS > BPR > Popularity > SVD > Content > Random. The order is unchanged.

## E6: accuracy vs coverage and novelty

![E6 scatter](figures/e6_test_control.png)

