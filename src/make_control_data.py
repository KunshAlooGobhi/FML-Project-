"""POSITIVE CONTROL (simulated, clearly NOT real data): purchases generated WITH a planted pattern.
Purpose: prove the split / metrics / models can find signal when it exists. The 1M file has none.
Planted structure: 10 taste groups (users in a group like similar items) + a few globally popular items."""
import numpy as np, pandas as pd
from config import SEED, PROCESSED_DIR

def main(n_users=5000, n_items=2000, dim=8, groups=10):
    rng = np.random.default_rng(SEED)
    centers = rng.normal(size=(groups, dim))
    U = centers[rng.integers(0, groups, n_users)] + 0.5 * rng.normal(size=(n_users, dim))   # user taste
    V = rng.normal(size=(n_items, dim))                                                       # item "style"
    pop = rng.normal(0, 1.0, n_items)                                                         # item popularity (log scale)
    score = 0.8 * U @ V.T + pop
    rows = []
    t0, span = pd.Timestamp("2024-03-31"), 730
    for u in range(n_users):
        p = np.exp(score[u] - score[u].max()); p /= p.sum()
        k = 10 + rng.poisson(15)
        items = rng.choice(n_items, size=k, replace=False, p=p)
        z = (score[u, items] - score[u].mean()) / score[u].std()
        rating = np.clip(np.round(3.5 + 0.5 * z + rng.normal(0, 0.6, k)), 1, 5)
        days = rng.integers(0, span + 1, k)
        rows += [(f"CU{u}", f"CP{i}", u, i, r, t0 + pd.Timedelta(days=int(d))) for i, r, d in zip(items, rating, days)]
    df = pd.DataFrame(rows, columns=["user_id", "product_id", "u", "i", "rating", "purchase_date"])
    df.to_csv(PROCESSED_DIR / "control_interactions.csv.gz", index=False)
    print(f"control data: {len(df):,} interactions, {df.u.nunique():,} users, {df.i.nunique():,} items (SIMULATED)")

if __name__ == "__main__":
    main()
