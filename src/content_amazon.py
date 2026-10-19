"""Phase 3: TF-IDF content model on the real Amazon products (+ review text).
There are NO users in these files, so purchase-based evaluation is impossible. Proxy task instead:
'find products similar to product X'; relevant = other products in the SAME leaf category (e.g. Jeans).
The category trail and brand are NOT in the model's text, so the model cannot read the answer.
Tune on half of the query products, report on the other (held-out) half. Table II grid."""
import itertools, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from config import RAW_DIR, ROOT, SEED
from metrics import precision_at_k, hit_rate_at_k, average_precision_at_k, ndcg_at_k

def build(sources):
    p = pd.read_csv(RAW_DIR / "products.csv"); r = pd.read_csv(RAW_DIR / "reviews.csv")
    rev = r.dropna(subset=["cleaned_review_text"]).groupby("productASIN").cleaned_review_text.apply(" ".join)
    parts = {"title": p.title.fillna(""), "about": p.about_item.fillna(""), "desc": p.product_description.fillna(""),
             "reviews": p.asin.map(rev).fillna("")}
    p["text"] = pd.concat([parts[s] for s in sources], axis=1).agg(" ".join, axis=1)
    p["leaf"] = p.breadcrumbs.fillna("").str.split("›").str[-1].str.strip()
    return p

def score_queries(sim, leaf, queries, ranking_fn):
    rows = []
    for q in queries:
        relevant = set(np.flatnonzero((leaf == leaf[q]) & (np.arange(len(leaf)) != q)))
        recs = ranking_fn(q)
        rows.append([ndcg_at_k(recs, relevant, 10), precision_at_k(recs, relevant, 10),
                     hit_rate_at_k(recs, relevant, 10), average_precision_at_k(recs, relevant, 10)])
    return np.mean(rows, axis=0)

def main():
    rng = np.random.default_rng(SEED)
    sources_full = ["title", "about", "desc", "reviews"]
    p = build(sources_full); leaf = p.leaf.values
    counts = p.leaf.value_counts(); eligible = np.flatnonzero(p.leaf.map(counts).ge(2).values)
    perm = rng.permutation(eligible); half = len(perm) // 2
    tune_q, test_q = np.sort(perm[:half]), np.sort(perm[half:])

    def run(sources, max_f, ngram, queries):
        d = build(sources)
        X = TfidfVectorizer(stop_words="english", sublinear_tf=True, max_features=max_f, ngram_range=ngram).fit_transform(d.text)
        sim = (X @ X.T).toarray(); np.fill_diagonal(sim, -1)
        order = lambda q: np.argsort(-sim[q], kind="stable")[:10].tolist()
        return score_queries(sim, leaf, queries, order), X.shape[1]

    rows = []
    for mf, ng in itertools.product([5000, 20000], [(1, 1), (1, 2)]):
        s, vocab = run(sources_full, mf, ng, tune_q)
        rows.append({"max_features": mf, "ngram": str(ng), "vocab_size": vocab, "tune_NDCG@10": s[0]})
    grid = pd.DataFrame(rows); best = grid.sort_values("tune_NDCG@10").iloc[-1]
    mf, ng = int(best.max_features), eval(best.ngram)

    # held-out report + baselines on the SAME held-out queries
    s, _ = run(sources_full, mf, ng, test_q)
    rnd = lambda q: rng.choice(np.delete(np.arange(len(p)), q), 10, replace=False).tolist()
    rc = pd.to_numeric(p.rating_count.astype(str).str.replace(r"[^0-9]", "", regex=True), errors="coerce").fillna(0).values   # "74,633 ratings" -> 74633
    pop_order = np.argsort(-rc, kind="stable")
    pop = lambda q: [i for i in pop_order if i != q][:10]
    sr = score_queries(None, leaf, test_q, rnd); sp = score_queries(None, leaf, test_q, pop)
    res = pd.DataFrame([["random", *sr], ["popularity (rating_count)", *sp], [f"TF-IDF content (max_features={mf}, ngram={ng})", *s]],
                       columns=["model", "NDCG@10", "P@10", "HR@10", "MAP@10"])
    # which text carries the signal? (best setting, held-out queries)
    abl = []
    for name, src in [("title only", ["title"]), ("title + about", ["title", "about"]), ("title + about + description", ["title", "about", "desc"]),
                      ("all text + review text", sources_full)]:
        a, _ = run(src, mf, ng, test_q); abl.append([name, *a])
    abl = pd.DataFrame(abl, columns=["text used", "NDCG@10", "P@10", "HR@10", "MAP@10"])
    # demo: 'customers also like'
    d = build(sources_full)
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True, max_features=mf, ngram_range=ng).fit_transform(d.text)
    sim = (X @ X.T).toarray(); np.fill_diagonal(sim, -1)
    demo = []
    for q in test_q[:3]:
        demo.append(f"**{p.title[q][:70]}** ({leaf[q]})\n" + "\n".join(f"  - {p.title[j][:70]} ({leaf[j]})" for j in np.argsort(-sim[q])[:5]))
    md = ["# Content model on Amazon products (from src/content_amazon.py)\n",
          f"Queries: {len(eligible)} products in a leaf category shared with another product. Tuned on {len(tune_q)}, reported on {len(test_q)} held-out queries. "
          "Relevant = other products in the same leaf category (proxy; NOT user purchases - these files contain no users).\n",
          "## Tuning grid (Table II)\n", grid.round(4).to_markdown(index=False), "\n## Held-out result\n", res.round(4).to_markdown(index=False),
          "\n## Which text carries the signal\n", abl.round(4).to_markdown(index=False), "\n## Demo: similar products\n", "\n\n".join(demo)]
    (ROOT / "reports" / "content_amazon.md").write_text("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
