"""Top-10 recommendation demo on REAL Amazon Beauty data (hybrid model, frozen settings, fitted on the TRAIN period only).
  python demo.py user [N]          top-10 for shopper N (random warm shopper if omitted); items they really bought later are ticked
  python demo.py product [words]   'shoppers also bought' and 'similar description' for a real product whose title contains the words
  python demo.py list              example shopper numbers and product titles
The locked test is not used here: ticks come from the validation period."""
import sys, json, html
import numpy as np, pandas as pd
from config import PROCESSED_DIR, MANIFEST_DIR
import models as M
from hybrid import tfidf, content_score_fn, adaptive_blend_fn
from evaluate import user_sets

TOK = r"(?u)\b[a-z][a-z]+\b"
cfg = json.loads((MANIFEST_DIR / "frozen_settings.json").read_text())["beauty"]
rd = lambda n: pd.read_csv(PROCESSED_DIR / f"beauty_{n}.csv.gz", parse_dates=["purchase_date"])
train, val = rd("train"), rd("val"); items = pd.read_csv(PROCESSED_DIR / "beauty_items.csv.gz").fillna({"title": "", "brand": ""})
items["title"] = items.title.map(html.unescape); items["brand"] = items.brand.map(html.unescape)
n_users, n_items = int(max(train.u.max(), val.u.max())) + 1, len(items)
R = M.interaction_matrix(train, n_users, n_items); tb = M.tie_break(train, n_items); is_cold = np.bincount(train.i.values, minlength=n_items) == 0
p = cfg["itemknn"]; Sfull = M.knn_similarity(R, p["shrink"]).tocsr(); cf = M.knn_score_fn(R, M.prune_top_k(Sfull, p["k"]), 0.0)
c = cfg["content"]; X = tfidf(train, items.text.values, n_items, c["max_features"], tuple(c["ngram"]), TOK)
hybrid = adaptive_blend_fn(cf, content_score_fn(R, X), cfg["hybrid"]["adaptive_w"], is_cold, tb)
name = lambda i: f"{items.title[i][:80] or '(untitled)'}" + (f"  [{items.brand[i]}]" if items.brand[i] else "")

def user(arg):
    _, rel = user_sets(train, val); warm = np.array(sorted(u for u in rel if np.diff(R.indptr)[u] > 0))
    u = int(arg) if arg else int(np.random.default_rng().choice(warm))
    if not 0 <= u < n_users: sys.exit(f"Shopper {u} does not exist; use 0 to {n_users-1}.")
    if u not in rel or np.diff(R.indptr)[u] == 0: sys.exit(f"Shopper {u} has no purchases before the cut-off or none after it; try another number.")
    bought = R.indices[R.indptr[u]:R.indptr[u + 1]]; sc = hybrid(np.array([u]))[0].copy(); sc[bought] = -np.inf; top = np.argsort(-sc)[:10]
    print(f"\nShopper {u} bought {len(bought)} products before the cut-off, for example:")
    for i in bought[-3:]: print("   -", name(i))
    print("\nTop-10 recommended (✓ = really bought afterwards):")
    for r, i in enumerate(top, 1): print(f"  {r:2d}. {'✓' if i in rel[u] else ' '} {name(i)}")
    print(f"\n{sum(i in rel[u] for i in top)} of 10 were really bought; the shopper bought {len(rel[u])} product(s) afterwards.")

def product(words):
    hit = items[items.title.str.lower().str.contains(" ".join(words).lower(), regex=False)] if words else items.head(0)
    if hit.empty: sys.exit("No product matches; try `python demo.py list`.")
    i = int(hit.assign(n=np.asarray(R.sum(axis=0)).ravel()[hit.i.values]).sort_values("n", ascending=False).i.iloc[0])
    row = Sfull[i].toarray().ravel(); row[i] = -1; sim = (X @ X[i].T).toarray().ravel(); sim[i] = -1
    print("\nProduct:", name(i)); print("\nShoppers also bought:")
    for r, j in enumerate(np.argsort(-row)[:5], 1): print(f"  {r}. {name(j)}")
    print("\nSimilar description:")
    for r, j in enumerate(np.argsort(-sim)[:5], 1): print(f"  {r}. {name(j)}  ({sim[j]*100:.0f}% similar)")

if __name__ == "__main__":
    m, a = (sys.argv[1] if len(sys.argv) > 1 else "user"), sys.argv[2:]
    if m == "user": user(a[0] if a else None)
    elif m == "product": product(a)
    else:
        for t in items.title[items.title != ""].sample(8, random_state=1): print("  product:", t[:90])
        print("  shopper numbers: any of 0 to", n_users - 1)
