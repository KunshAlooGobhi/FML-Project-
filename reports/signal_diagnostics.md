# Signal diagnostics for amazon_ecommerce_1M (from src/diagnose_signal.py)

| test | observed | what pure chance predicts |
|---|---|---|
| variance of item purchase counts | 11.08 | 11.11 (Poisson: variance = mean) |
| user repeats previous category | 0.2005 | 0.2000 |
| user repeats previous brand | 0.0829 | 0.0833 |
| corr(item purchases in train period, in later period) | 0.0011 | 0 |
| avg within-item rating std vs overall | 0.507 vs 0.519 | equal if rating is not a product property |

Conclusion: every statistic matches random assignment. Popularity, user taste and item rating carry no signal.
