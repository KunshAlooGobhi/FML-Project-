# LOCKED TEST: beauty (from src/final_eval.py + src/finalize.py)

Settings frozen in manifests/frozen_settings.json. Warm users with history in the fit data: 7,272. Fit data = train + validation; scored on the locked test.

## Table III: ranking results, K = 10 (E1)

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

## Paired bootstrap on per-user NDCG@10 (1000 resamples of users; 7 comparisons made; Bonferroni-adjusted)

A model counts as an improvement only if its 95% interval excludes zero AND the Bonferroni-adjusted p < 0.05. With 1,000 resamples the smallest reportable p is below 1/1,000.

| Model vs tuned popularity                |   mean NDCG@10 difference |   95% CI low |   95% CI high | p      | p (Bonferroni, x7)   | reading                |
|:-----------------------------------------|--------------------------:|-------------:|--------------:|:-------|:---------------------|:-----------------------|
| Random (reference)                       |                 -0.005515 |    -0.006474 |     -0.004571 | <0.001 | <0.007               | worse than popularity  |
| Item-kNN                                 |                  0.014783 |     0.012643 |      0.017082 | <0.001 | <0.007               | better than popularity |
| SVD (biased, rating-trained)             |                 -0.004316 |    -0.005368 |     -0.003233 | <0.001 | <0.007               | worse than popularity  |
| ALS                                      |                  0.016505 |     0.014189 |      0.018769 | <0.001 | <0.007               | better than popularity |
| BPR                                      |                  0.009783 |     0.007756 |      0.011965 | <0.001 | <0.007               | better than popularity |
| Content-based (TF-IDF)                   |                  0.016785 |     0.014301 |      0.019496 | <0.001 | <0.007               | better than popularity |
| Hybrid (adaptive: content for new items) |                  0.02079  |     0.018374 |      0.023412 | <0.001 | <0.007               | better than popularity |

## E5: list-length sweep (K = 5, 10, 20)

| Model      |   NDCG@5 |   NDCG@10 |   NDCG@20 |    HR@5 |   HR@10 |   HR@20 |
|:-----------|---------:|----------:|----------:|--------:|--------:|--------:|
| Random     |  0.00023 |   0.00035 |   0.00056 | 0.00096 | 0.00193 | 0.00413 |
| Popularity |  0.00457 |   0.00587 |   0.00773 | 0.01636 | 0.02654 | 0.04043 |
| Item-kNN   |  0.01726 |   0.02065 |   0.02438 | 0.04538 | 0.06449 | 0.09021 |
| SVD        |  0.00118 |   0.00155 |   0.00227 | 0.00454 | 0.0077  | 0.01444 |
| ALS        |  0.01866 |   0.02237 |   0.02735 | 0.04895 | 0.07357 | 0.10547 |
| BPR        |  0.01304 |   0.01565 |   0.0199  | 0.03438 | 0.05322 | 0.08072 |
| Content    |  0.01915 |   0.02265 |   0.02731 | 0.04772 | 0.06972 | 0.0986  |
| Hybrid     |  0.0221  |   0.02666 |   0.03215 | 0.05734 | 0.08347 | 0.11592 |

Model order by NDCG@5 / @10 / @20: Hybrid > Content > ALS > Item-kNN > BPR > Popularity > SVD > Random | Hybrid > Content > ALS > Item-kNN > BPR > Popularity > SVD > Random | Hybrid > ALS > Content > Item-kNN > BPR > Popularity > SVD > Random. The order changes with K.

## Table IV: rating-prediction error on the locked test (29,848 rows)

| Model                   |   RMSE |    MAE | RMSE change vs global mean   | Bootstrap 95% CI   |
|:------------------------|-------:|-------:|:-----------------------------|:-------------------|
| Global mean rating      | 1.145  | 0.9298 | n/a                          | n/a                |
| User/item bias baseline | 1.0968 | 0.8657 | -0.0482                      | [-0.0518, -0.0445] |
| SVD                     | 1.1051 | 0.8671 | -0.0398                      | [-0.0439, -0.0356] |

## E2: cold slices

Rows by slice relative to the fit data: {'warm': 18309, 'cold-item': 2864, 'cold-user': 8675}.

Cold users (no history; 1,085 total, sample of 1,085); personalised models cannot score them and fall back to popularity.

| Model              |   NDCG@10 |   HR@10 |
|:-------------------|----------:|--------:|
| Random             |    0.0005 | 0.00369 |
| Popularity (tuned) |    0.0168 | 0.09954 |

Popularity minus random on cold users: 0.016298, 95% CI [0.012774, 0.020045], p = 0.001.

## E2 (new items): purchases of items with no history in the fit data

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

## Hybrid against its own components (extra comparison, 2 tests)

| Hybrid minus           |   mean NDCG@10 difference |   95% CI low |   95% CI high | p (Bonferroni, x2)   |
|:-----------------------|--------------------------:|-------------:|--------------:|:---------------------|
| ALS                    |                  0.004286 |     0.002081 |      0.006522 | <0.002               |
| Content-based (TF-IDF) |                  0.004005 |     0.001887 |      0.006125 | <0.002               |

## E7: validation vs locked test

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

## E6: accuracy vs coverage and novelty

![E6 scatter](figures/e6_test_beauty.png)

