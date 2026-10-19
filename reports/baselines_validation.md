# Baselines on VALIDATION (from src/baselines.py)

Validation users with >=1 purchase: 88,655 (74,322 seen in train). Best by NDCG@10: **random**.

| model           |   P@10 |    R@10 |   HR@10 |   MAP@10 |   NDCG@5 |   NDCG@10 |   NDCG@20 |   coverage@10 |   NDCG@10_warm_users |   NDCG@10_cold_users |   NDCG@10_rating>=4 |
|:----------------|-------:|--------:|--------:|---------:|---------:|----------:|----------:|--------------:|---------------------:|---------------------:|--------------------:|
| random          |  1e-05 | 0.00011 | 0.00014 |    4e-05 |    5e-05 |     6e-05 |    0.0001 |       0.00012 |                7e-05 |                2e-05 |               5e-05 |
| popularity_all  |  1e-05 | 0.00012 | 0.00012 |    3e-05 |    3e-05 |     5e-05 |    8e-05  |       0.00012 |                5e-05 |                4e-05 |               5e-05 |
| popularity_30d  |  1e-05 | 7e-05   | 9e-05   |    2e-05 |    2e-05 |     4e-05 |    6e-05  |       0.00012 |                3e-05 |                5e-05 |               5e-05 |
| popularity_90d  |  1e-05 | 7e-05   | 8e-05   |    1e-05 |    1e-05 |     3e-05 |    6e-05  |       0.00012 |                2e-05 |                3e-05 |               1e-05 |
| popularity_365d |  2e-05 | 0.00012 | 0.00016 |    3e-05 |    4e-05 |     6e-05 |    0.0001 |       0.00012 |                6e-05 |                6e-05 |               6e-05 |
| bayes_rating    |  0     | 4e-05   | 5e-05   |    1e-05 |    2e-05 |     2e-05 |    5e-05  |       0.00012 |                2e-05 |                3e-05 |               3e-05 |

Rating prediction (validation): global mean RMSE 0.5187, MAE 0.4177; item-mean RMSE 0.5796, MAE 0.4644.
