# Tuned models on VALIDATION, warm users (main; from src/run_models.py)

Best setting per model by validation NDCG@10.

| model            | best_setting                  |   P@10 |   R@10 |   HR@10 |   MAP@10 |   NDCG@5 |   NDCG@10 |   NDCG@20 |   coverage@10 |      RMSE |       MAE |   users |
|:-----------------|:------------------------------|-------:|-------:|--------:|---------:|---------:|----------:|----------:|--------------:|----------:|----------:|--------:|
| random           |                               |  4e-05 | 0.0004 |  0.0004 |  0.00013 |  0.00013 |   0.0002  |   0.0002  |       0.42687 | nan       | nan       |    5000 |
| pop:bayes_rating |                               |  0     | 0      |  0      |  0       |  0       |   0       |   0       |       0.00011 | nan       | nan       |    5000 |
| itemknn          | k=100, shrink=50              |  2e-05 | 0.0002 |  0.0002 |  0.0001  |  0.00013 |   0.00013 |   0.00013 |       0.31461 | nan       | nan       |    5000 |
| svd              | factors=32, reg=0.1, lr=0.01  |  0     | 0      |  0      |  0       |  0       |   0       |   0       |       0.00755 |   0.54223 |   0.43557 |    5000 |
| als              | factors=32, reg=0.1, alpha=40 |  0     | 0      |  0      |  0       |  0       |   0       |   5e-05   |       0.05402 | nan       | nan       |    5000 |
| bpr              | factors=32, reg=0.1, lr=0.05  |  0     | 0      |  0      |  0       |  0       |   0       |   0       |       0.06551 | nan       | nan       |    5000 |
