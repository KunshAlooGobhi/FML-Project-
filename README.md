# E-Commerce Recommendation System (Fundamentals of Machine Learning project)

A leakage-aware study of collaborative, content-based and hybrid recommenders on Amazon product and review data.

## Data (not committed; see `manifests/raw_manifest.json` for checksums)
- `amazon_ecommerce_1M.csv` - 1M purchases, 603,815 users, 89,999 products, dates 2024-03-31 to 2026-03-31. Appears synthetic.
- `products.csv`, `reviews.csv` - 728 real Amazon clothing products and 6,327 reviews. No user IDs; product IDs cannot be linked to the 1M file.

## Layout
`data/raw`, `data/processed`, `notebooks/`, `src/`, `manifests/`, `reports/`

## Documented deviations from the partial report
- k-core is **k=2**, not 5 (only 5,094 users have >=5 purchases). Cold-start tests keep the full data.
- The two datasets are used for different roles and are never merged.

## Run
```
cd src && python make_manifest.py
```
