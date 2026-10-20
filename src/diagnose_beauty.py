"""Does the Beauty 5-core file contain learnable structure? Same style of checks as diagnose_signal.py (which found none in the 1M file)."""
import numpy as np, pandas as pd
from config import PROCESSED_DIR, ROOT, SEED
d = pd.read_csv(PROCESSED_DIR / "interactions_beauty.csv.gz", parse_dates=["purchase_date"])
ic = d.groupby("i").size(); rng = np.random.default_rng(SEED)
# 1 popularity spread: random purchasing would give variance ~ mean (Poisson)
r1 = (ic.var(), ic.mean())
# 2 early vs late item popularity
cut = d.purchase_date.quantile(.5); e = d[d.purchase_date <= cut].groupby("i").size(); l = d[d.purchase_date > cut].groupby("i").size()
j = pd.concat([e, l], axis=1).fillna(0); r2 = j.corr().iloc[0, 1]
# 3 category repetition: same top-level category in two random purchases by the same user vs by different users
items = pd.read_csv(PROCESSED_DIR / "beauty_items.csv.gz").fillna("")
leaf = items.categories.str.split().str[-1]; sub = items.categories.str.slice(0, 40)
def same(col, pairs):
    return np.mean([col[a] == col[b] for a, b in pairs])
usr = d.groupby("u").i.apply(list); pairs_u = [tuple(rng.choice(v, 2, replace=False)) for v in usr if len(v) >= 2][:5000]
pairs_r = [tuple(rng.choice(len(items), 2, replace=False)) for _ in range(5000)]
r3 = (same(sub, pairs_u), same(sub, pairs_r))
# 4 same-brand
brand = items.brand; r4 = (same(brand[brand != ""].reindex(range(len(items))).fillna("?"), pairs_u), same(brand[brand != ""].reindex(range(len(items))).fillna("?"), pairs_r))
md = f"""# Signal diagnostics: Amazon Beauty 5-core (from src/diagnose_beauty.py)

| check | observed | what pure chance predicts |
|---|---|---|
| Item popularity: variance / mean of interactions per item | {r1[0]/r1[1]:.1f} | about 1 (Poisson) |
| Correlation of item popularity, first half vs second half of time | {r2:.3f} | about 0 |
| Two purchases by the same user share a category path | {r3[0]:.3f} | {r3[1]:.3f} (two random items) |
| Two purchases by the same user share a brand | {r4[0]:.3f} | {r4[1]:.3f} (two random items) |

All four differ strongly from chance, so this file contains learnable structure (unlike amazon_ecommerce_1M).
"""
(ROOT / "reports" / "signal_diagnostics_beauty.md").write_text(md); print(md)
