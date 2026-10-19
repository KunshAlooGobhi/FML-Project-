"""Top-10 recommendation demo.
  python demo.py user [CU123]       personalised top-10 for a user of the SIMULATED control data (random user if omitted)
  python demo.py product [words]    'customers also like' for a REAL Amazon product (title contains the words)
  python demo.py list               show example titles / user ids
The user demo fits on the TRAIN period only and marks items the user really bought later (validation period) with a tick.
The locked test is not used. Control data is simulated, so 'style groups' below are simulated item labels."""
import sys, json, re, collections
import numpy as np, pandas as pd
from config import PROCESSED_DIR, MANIFEST_DIR, SEED
import models as M
from baselines import build_rankings

def group_of(text):
    c = collections.Counter(re.match(r"g(\d+)w", t).group(1) for t in text.split() if re.match(r"g\d+w", t)); return int(c.most_common(1)[0][0]) if c else -1

def user_demo(user):
    cfg = json.loads((MANIFEST_DIR / "frozen_settings.json").read_text())["control"]
    rd = lambda n: pd.read_csv(PROCESSED_DIR / f"control_{n}.csv.gz", parse_dates=["purchase_date"])
    train, val = rd("train"), rd("val"); texts = pd.read_csv(PROCESSED_DIR / "control_items.csv.gz").text.values
    grp = np.array([group_of(t) for t in texts]); n_users, n_items = int(max(train.u.max(), val.u.max())) + 1, len(texts)
    R = M.interaction_matrix(train, n_users, n_items); tb = M.tie_break(train, n_items)
    warm = np.flatnonzero(np.diff(R.indptr) > 0)
    u = int(str(user).replace("CU", "")) if user else int(np.random.default_rng().choice(warm))
    if not 0 <= u < R.shape[0]: sys.exit(f"CU{u} does not exist; valid ids are CU0 to CU{R.shape[0]-1}.")
    if np.diff(R.indptr)[u] == 0: sys.exit(f"CU{u} has no purchase history in the training period (cold user): only popularity could serve them.")
    p = cfg["itemknn"]; sc = M.knn_score_fn(R, M.prune_top_k(M.knn_similarity(R, p["shrink"]), p["k"]), tb)(np.array([u]))[0].copy()
    bought = R.indices[R.indptr[u]:R.indptr[u + 1]]; sc[bought] = -np.inf; top = np.argsort(-sc)[:10]
    later = set(val[val.u == u].i); pop = build_rankings(train)[cfg["popularity"]["variant"]]; pop_top = [i for i in pop if i not in set(bought)][:10]
    mix = collections.Counter(grp[bought]); tot = sum(mix.values())
    print(f"\nUser CU{u}: {len(bought)} purchases in the training period.")
    print("Taste summary (share of purchases by simulated style group): " + ", ".join(f"group {g}: {100*c/tot:.0f}%" for g, c in mix.most_common(4)))
    print("\nTop-10 for this user (item-kNN)                    | Same list for everyone (tuned popularity)")
    for r in range(10):
        a, b = top[r], pop_top[r]
        print(f" {r+1:>2}. item CP{a:<5} group {grp[a]:<2} {'✓ bought later' if a in later else '               '}  |  item CP{b:<5} group {grp[b]:<2} {'✓ bought later' if b in later else ''}")
    ha, hb = sum(i in later for i in top), sum(i in later for i in pop_top)
    print(f"\nItems this user really bought in the next (validation) period: {len(later)}. Hits in top-10: personalised {ha}, popularity {hb}.")
    print("(✓ = the user did buy it later. Simulated data: this shows how the method works, not real shoppers.)")

def product_demo(words, n=5):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from content_amazon import build
    p = build(["title", "about", "desc", "reviews"])
    if not words: sys.exit("Give a few words of a product title, e.g.:  python demo.py product 'polo golf'")
    hit = p[p.title.str.lower().str.contains(words.lower(), regex=False)]
    if hit.empty: sys.exit(f"No product title contains '{words}'. Try: python demo.py list")
    q = hit.index[0]; X = TfidfVectorizer(stop_words="english", sublinear_tf=True, max_features=5000, ngram_range=(1, 2)).fit_transform(p.text)
    sim = (X @ X.T).toarray()[q]; sim[q] = -1; top = np.argsort(-sim)[:n]
    price = lambda i: f"${p.price_value[i]:.2f}" if pd.notna(p.price_value[i]) else "price n/a"
    print(f"\nYou looked at: {p.title[q][:90]}  [{p.leaf[q]}, {price(q)}]\n\nCustomers who viewed this may also like (text similarity, real Amazon products):")
    for r, i in enumerate(top, 1): print(f" {r}. {p.title[i][:90]}  [{p.leaf[i]}, {price(i)}]  similarity {sim[i]:.2f}")

if __name__ == "__main__":
    a = sys.argv[1:]; mode = a[0] if a else "user"
    if mode == "user": user_demo(a[1] if len(a) > 1 else None)
    elif mode == "product": product_demo(" ".join(a[1:]))
    elif mode == "list":
        from content_amazon import build
        print("Example product words:", "; ".join(build(["title"]).title.sample(5, random_state=SEED).str[:45]))
        tr = pd.read_csv(PROCESSED_DIR / "control_train.csv.gz"); print("Example control users:", ", ".join(f"CU{u}" for u in tr.u.drop_duplicates().sample(5, random_state=SEED)))
    else: sys.exit(__doc__)
