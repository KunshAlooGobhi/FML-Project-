"""Phase 1 / Step 2: clean the 1M purchase file (order follows the report).
1 drop exact duplicates + rows missing user/item/rating
2 standardise text   3 parse dates   4 one row per (user, item)
5 k-core (iterative)   6 map IDs to contiguous integers
Outputs: data/processed/interactions_full.csv.gz (cold-start tests) and interactions_k2.csv.gz (main experiments)
plus reports/table1_dataset_stats.md (Table I) - every number comes from this run."""
import json
import pandas as pd
from config import RAW_DIR, PROCESSED_DIR, ROOT, K_CORE

def kcore(df, k):
    """Repeatedly drop users and items with fewer than k interactions until stable."""
    while True:
        u = df.groupby("user_id")["product_id"].transform("size")
        i = df.groupby("product_id")["user_id"].transform("size")
        keep = (u >= k) & (i >= k)
        if keep.all():
            return df
        df = df[keep]

def stats(df, name):
    n_u, n_i, n = df.user_id.nunique(), df.product_id.nunique(), len(df)
    return {"set": name, "interactions": n, "users": n_u, "items": n_i,
            "sparsity_%": round(100 * (1 - n / (n_u * n_i)), 4),
            "avg_per_user": round(n / n_u, 2), "avg_per_item": round(n / n_i, 2),
            "date_min": str(df.purchase_date.min().date()), "date_max": str(df.purchase_date.max().date())}

def main():
    log = {}
    df = pd.read_csv(RAW_DIR / "amazon_ecommerce_1M.csv")
    log["raw_rows"] = len(df)
    df = df.drop_duplicates()
    df = df.dropna(subset=["user_id", "product_id", "rating"])
    log["after_exact_dup_and_null_drop"] = len(df)

    for c in ["category", "subcategory", "brand", "location", "device", "payment_method", "delivery_status"]:
        df[c] = df[c].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()

    df["purchase_date"] = pd.to_datetime(df["purchase_date"], errors="coerce")
    df = df.dropna(subset=["purchase_date"])
    df = df[(df.purchase_date >= "2000-01-01") & (df.purchase_date <= "2026-12-31")]  # impossible dates
    log["after_date_checks"] = len(df)

    # one row per (user, product): keep the most recent purchase
    df = df.sort_values("purchase_date").drop_duplicates(["user_id", "product_id"], keep="last")
    log["after_user_item_dedup"] = len(df)

    full = df.copy()
    core = kcore(df, K_CORE).copy()

    for d in (full, core):  # contiguous integer IDs are what matrix models need
        d["u"] = d.user_id.astype("category").cat.codes
        d["i"] = d.product_id.astype("category").cat.codes

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    full.to_csv(PROCESSED_DIR / "interactions_full.csv.gz", index=False)
    core.to_csv(PROCESSED_DIR / "interactions_k2.csv.gz", index=False)

    rows = [stats(full, "full (cleaned)"), stats(core, f"k-core k={K_CORE}")]
    md = ["# Table I: dataset statistics (from src/clean.py)\n",
          "| set | interactions | users | items | sparsity % | avg/user | avg/item | date range |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['set']} | {r['interactions']:,} | {r['users']:,} | {r['items']:,} | {r['sparsity_%']} | {r['avg_per_user']} | {r['avg_per_item']} | {r['date_min']} to {r['date_max']} |")
    md.append("\nCleaning log: " + json.dumps(log))
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "table1_dataset_stats.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

if __name__ == "__main__":
    main()
