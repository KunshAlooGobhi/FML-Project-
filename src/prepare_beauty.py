"""Real-data pipeline, step 1: Amazon 'Beauty' reviews, 5-core (McAuley et al., Stanford SNAP / UCSD) -> the table layout the other datasets use.
Each review is one positive interaction (user, product, rating, review date). Cleaning follows the report: drop rows missing user/item/rating,
one row per (user, item) keeping the LATEST review, k-core k=5 (the file is already 5-core; re-applied as a check), contiguous integer ids.
Item text for the content model comes from the product metadata file (title, brand, categories, description): static attributes, no review text, so no future leakage.
Writes data/processed/interactions_beauty.csv.gz, beauty_items.csv.gz and reports/table1_beauty.md."""
import gzip, json, ast
import numpy as np, pandas as pd
from config import RAW_DIR, PROCESSED_DIR, ROOT

K = 5
def load_reviews():
    rows = []
    with gzip.open(RAW_DIR / "reviews_Beauty_5.json.gz", "rt") as f:
        for l in f:
            d = json.loads(l); rows.append((d.get("reviewerID"), d.get("asin"), d.get("overall"), d.get("unixReviewTime")))
    return pd.DataFrame(rows, columns=["user_id", "product_id", "rating", "ts"])

def load_meta(wanted):
    out = {}
    with gzip.open(RAW_DIR / "meta_Beauty.json.gz", "rt") as f:
        for l in f:
            d = ast.literal_eval(l)
            if d.get("asin") in wanted:
                cats = " ".join(" ".join(c) for c in d.get("categories", []))
                out[d["asin"]] = dict(title=d.get("title", "") or "", brand=d.get("brand", "") or "", price=d.get("price", np.nan), categories=cats,
                                      description=d.get("description", "") or "", image=d.get("imUrl", "") or "")
    return out

def kcore(df, k):
    while True:
        uc, ic = df.user_id.value_counts(), df.product_id.value_counts()
        keep = df.user_id.map(uc).ge(k) & df.product_id.map(ic).ge(k)
        if keep.all(): return df
        df = df[keep]

def stats(name, d):
    u, i, n = d.user_id.nunique(), d.product_id.nunique(), len(d)
    return {"set": name, "interactions": f"{n:,}", "users": f"{u:,}", "items": f"{i:,}", "density": f"{n/(u*i):.2e}", "sparsity %": f"{100*(1-n/(u*i)):.4f}",
            "avg/user": f"{n/u:.2f}", "avg/item": f"{n/i:.2f}", "median/user": int(d.groupby('user_id').size().median()), "median/item": int(d.groupby('product_id').size().median()),
            "mean rating": f"{d.rating.mean():.3f}", "share >=4": f"{(d.rating>=4).mean()*100:.1f}%", "date range": f"{d.purchase_date.min().date()} to {d.purchase_date.max().date()}"}

def main():
    raw = load_reviews(); log = {"raw_rows": len(raw)}
    df = raw.dropna(subset=["user_id", "product_id", "rating", "ts"]).drop_duplicates(); log["after_null_and_exact_dup_drop"] = len(df)
    df["purchase_date"] = pd.to_datetime(df.ts, unit="s"); df = df[(df.purchase_date >= "1995-01-01") & (df.purchase_date <= "2015-12-31")]; log["after_date_checks"] = len(df)
    df = df.sort_values("purchase_date", kind="stable").drop_duplicates(["user_id", "product_id"], keep="last"); log["after_user_item_dedup"] = len(df)
    df = kcore(df.drop(columns="ts"), K); log[f"after_kcore_{K}"] = len(df)
    df["u"] = df.user_id.astype("category").cat.codes; cat = df.product_id.astype("category"); df["i"] = cat.cat.codes
    df.to_csv(PROCESSED_DIR / "interactions_beauty.csv.gz", index=False)
    asins = list(cat.cat.categories); meta = load_meta(set(asins))
    items = pd.DataFrame([{"i": k, "product_id": a, **meta.get(a, dict(title="", brand="", price=np.nan, categories="", description="", image=""))} for k, a in enumerate(asins)])
    items["text"] = (items.title + " " + items.brand + " " + items.categories + " " + items.description).str.lower().str.replace(r"<[^>]+>", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    items.to_csv(PROCESSED_DIR / "beauty_items.csv.gz", index=False)
    log["items_with_metadata_text"] = int((items.text.str.len() > 0).sum()); log["items"] = len(items)
    t = pd.DataFrame([stats(f"Beauty 5-core (cleaned, k={K})", df)])
    (ROOT / "reports" / "table1_beauty.md").write_text("# Table I: Amazon Beauty 5-core (from src/prepare_beauty.py)\n\n" + t.to_markdown(index=False) + f"\n\nCleaning log: {json.dumps(log)}\n")
    print(t.T.to_string()); print(log)

if __name__ == "__main__":
    main()
