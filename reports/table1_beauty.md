# Table I: Amazon Beauty 5-core (from src/prepare_beauty.py)

| set                          |   interactions |   users |   items |   density |   sparsity % |   avg/user |   avg/item |   median/user |   median/item |   mean rating | share >=4   | date range               |
|:-----------------------------|---------------:|--------:|--------:|----------:|-------------:|-----------:|-----------:|--------------:|--------------:|--------------:|:------------|:-------------------------|
| Beauty 5-core (cleaned, k=5) |        198,502 |  22,363 |  12,101 |  0.000734 |      99.9266 |       8.88 |       16.4 |             6 |             9 |          4.19 | 77.7%       | 2002-06-12 to 2014-07-23 |

Cleaning log: {"raw_rows": 198502, "after_null_and_exact_dup_drop": 198502, "after_date_checks": 198502, "after_user_item_dedup": 198502, "after_kcore_5": 198502, "items_with_metadata_text": 12101, "items": 12101}
