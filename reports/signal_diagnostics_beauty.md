# Signal diagnostics: Amazon Beauty 5-core (from src/diagnose_beauty.py)

| check | observed | what pure chance predicts |
|---|---|---|
| Item popularity: variance / mean of interactions per item | 34.0 | about 1 (Poisson) |
| Correlation of item popularity, first half vs second half of time | 0.300 | about 0 |
| Two purchases by the same user share a category path | 0.110 | 0.021 (two random items) |
| Two purchases by the same user share a brand | 0.110 | 0.033 (two random items) |

All four differ strongly from chance, so this file contains learnable structure (unlike amazon_ecommerce_1M).
