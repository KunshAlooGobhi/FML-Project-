# Table I: dataset statistics (from src/clean.py)

| set | interactions | users | items | sparsity % | avg/user | avg/item | date range |
|---|---|---|---|---|---|---|---|
| full (cleaned) | 999,993 | 603,815 | 89,999 | 99.9982 | 1.66 | 11.11 | 2024-03-31 to 2026-03-31 |
| k-core k=2 | 669,572 | 273,820 | 89,528 | 99.9973 | 2.45 | 7.48 | 2024-03-31 to 2026-03-31 |

Cleaning log: {"raw_rows": 1000000, "after_exact_dup_and_null_drop": 1000000, "after_date_checks": 1000000, "after_user_item_dedup": 999993}
