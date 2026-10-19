"""Phase 4 (SIMULATED control data only): content model, hybrid blend, ablation, cold-item stress test.
Real Amazon text and real purchases cannot be linked (different ids, and the review file has no users),
so the hybrid mechanism is demonstrated on the control data with simulated item text. Validation only."""
import itertools, numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from config import PROCESSED_DIR, ROOT, SEED
from evaluate import user_sets, warm_eval_users, evaluate_batch
import models as M

KEYS = ["P@10", "R@10", "HR@10", "MAP@10", "NDCG@10", "NDCG@20", "coverage@10", "users"]

def content_score_fn(R, X):
    """User profile = sum of the TF-IDF vectors of items they bought; score = cosine(profile, item)."""
    def fn(users):
        prof = (R[users] @ X).tocsr()
        norm = np.sqrt(np.asarray(prof.multiply(prof).sum(1)).ravel()); norm[norm == 0] = 1
        return ((sp.diags(1 / norm) @ prof) @ X.T).toarray().astype(np.float32)
    return fn

def blend_fn(cf_fn, ct_fn, w, tb):
    def fn(users):
        a, b = cf_fn(users), ct_fn(users)
        a = a / np.maximum(a.max(1, keepdims=True), 1e-9); b = b / np.maximum(b.max(1, keepdims=True), 1e-9)   # row-max scale to [0,1]
        return (1 - w) * a + w * b + tb
    return fn

def adaptive_blend_fn(cf_fn, ct_fn, w, is_cold, tb):
    """Per-item weight: items with NO training history (is_cold) are scored by content alone (w_i = 1);
    all other items use the global weight w. This is the 'content for cold items' design from the report."""
    wi = np.where(is_cold, 1.0, w).astype(np.float32)
    def fn(users):
        a, b = cf_fn(users), ct_fn(users)
        a = a / np.maximum(a.max(1, keepdims=True), 1e-9); b = b / np.maximum(b.max(1, keepdims=True), 1e-9)
        return (1 - wi) * a + wi * b + tb
    return fn

def tfidf(train, texts, n_items, max_f, ngram):
    v = TfidfVectorizer(max_features=max_f, ngram_range=ngram, sublinear_tf=True, token_pattern=r"\S+")
    v.fit(texts[np.unique(train.i.values)])                       # vocabulary fitted on items seen in TRAIN only
    return v.transform(texts[:n_items])                            # any item (even unseen) can be encoded from its text

def main():
    rd = lambda n: pd.read_csv(PROCESSED_DIR / f"control_{n}.csv.gz", parse_dates=["purchase_date"])
    train, val = rd("train"), rd("val")
    texts = pd.read_csv(PROCESSED_DIR / "control_items.csv.gz").text.values
    n_users, n_items = int(max(train.u.max(), val.u.max())) + 1, int(max(train.i.max(), val.i.max())) + 1
    R = M.interaction_matrix(train, n_users, n_items); tb = M.tie_break(train, n_items)
    _, rel = user_sets(train, val); users = warm_eval_users(R, rel, None, SEED)
    ev = lambda fn, R_=R, rel_=rel, u_=users: evaluate_batch(fn, R_, rel_, u_, n_items)[1]

    # 1) content-only model, Table II grid
    rows = []
    for mf, ng in itertools.product([5000, 20000], [(1, 1), (1, 2)]):
        s = ev(content_score_fn(R, tfidf(train, texts, n_items, mf, ng)))
        rows.append({"max_features": mf, "ngram": str(ng), **{k: s[k] for k in KEYS}})
    cg = pd.DataFrame(rows); bc = cg.sort_values("NDCG@10").iloc[-1]; mf, ng = int(bc.max_features), eval(bc.ngram)
    X = tfidf(train, texts, n_items, mf, ng)

    # 2) hybrid: item-kNN (k=100, shrink=0: best CF on validation) + content, blend weight grid
    S = M.prune_top_k(M.knn_similarity(R, 0), 100); cf = M.knn_score_fn(R, S, 0.0); ct = content_score_fn(R, X)
    rows = []
    for w in np.round(np.arange(0, 1.01, 0.1), 1):
        s = ev(blend_fn(cf, ct, w, tb)); rows.append({"w": w, **{k: s[k] for k in KEYS}})
    hg = pd.DataFrame(rows); bh = hg.sort_values("NDCG@10").iloc[-1]; wbest = float(bh.w)
    pop = pd.read_csv(ROOT / "reports" / "models_control_pop.csv"); pop = pop[pop.model != "random"].sort_values("NDCG@10").iloc[-1]
    cold_mask = np.bincount(train.i.values, minlength=n_items) == 0
    sa = ev(adaptive_blend_fn(cf, ct, wbest, cold_mask, tb))
    abl = pd.DataFrame([["tuned popularity", pop["NDCG@10"], pop["HR@10"], pop["coverage@10"]],
                        ["content only (w=1)", hg[hg.w == 1.0]["NDCG@10"].iloc[0], hg[hg.w == 1.0]["HR@10"].iloc[0], hg[hg.w == 1.0]["coverage@10"].iloc[0]],
                        ["collaborative only (w=0)", hg[hg.w == 0.0]["NDCG@10"].iloc[0], hg[hg.w == 0.0]["HR@10"].iloc[0], hg[hg.w == 0.0]["coverage@10"].iloc[0]],
                        [f"hybrid, global weight w={wbest}", bh["NDCG@10"], bh["HR@10"], bh["coverage@10"]],
                        [f"hybrid, adaptive (content-only for items with no history), w={wbest}", sa["NDCG@10"], sa["HR@10"], sa["coverage@10"]]],
                       columns=["model", "NDCG@10", "HR@10", "coverage@10"])

    # 3) cold-item stress test: pretend a random 10% of items are NEW (remove all their training rows)
    rng = np.random.default_rng(SEED); cold = rng.choice(n_items, int(0.1 * n_items), replace=False)
    tr2 = train[~train.i.isin(cold)]; R2 = M.interaction_matrix(tr2, n_users, n_items); tb2 = M.tie_break(tr2, n_items)
    val_c = val[val.i.isin(cold)]; _, rel_c = user_sets(tr2, val_c); users_c = warm_eval_users(R2, rel_c, None, SEED)
    X2 = tfidf(tr2, texts, n_items, mf, ng)
    cf2 = M.knn_score_fn(R2, M.prune_top_k(M.knn_similarity(R2, 0), 100), 0.0); ct2 = content_score_fn(R2, X2)
    rows = []
    for w in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
        s = ev(blend_fn(cf2, ct2, w, tb2), R2, rel_c, users_c); rows.append({"w": w, **{k: s[k] for k in ("HR@10", "NDCG@10", "R@10", "users")}})
    is_cold2 = np.bincount(tr2.i.values, minlength=n_items) == 0
    for w in (0.0, 0.1, 0.2, 0.3):
        s = ev(adaptive_blend_fn(cf2, ct2, w, is_cold2, tb2), R2, rel_c, users_c)
        rows.append({"w": f"adaptive, w={w}", **{k: s[k] for k in ("HR@10", "NDCG@10", "R@10", "users")}})
    # warm-user quality of the same adaptive hybrid on this reduced training set (does injecting new items hurt?)
    _, rel_w = user_sets(tr2, val[~val.i.isin(cold)]); users_w = warm_eval_users(R2, rel_w, None, SEED)
    warm_chk = pd.DataFrame([
        ["collaborative only (w=0)", ev(blend_fn(cf2, ct2, 0.0, tb2), R2, rel_w, users_w)["NDCG@10"]],
        ["adaptive hybrid, w=0.1", ev(adaptive_blend_fn(cf2, ct2, 0.1, is_cold2, tb2), R2, rel_w, users_w)["NDCG@10"]]],
        columns=["model", "NDCG@10 on purchases of NON-new items"])
    cold_t = pd.DataFrame(rows)

    md = ["# Phase 4 on the SIMULATED control data (from src/hybrid.py)\n",
          "Item text is simulated; results show the mechanism, not a real-world effect. Validation only, warm users.\n",
          "## Content-only model, Table II grid\n", cg.round(5).to_markdown(index=False),
          f"\n## Hybrid blend weight (item-kNN k=100 + content max_features={mf}, ngram={ng})\n", hg.round(5).to_markdown(index=False),
          "\n## Ablation (E4)\n", abl.round(5).to_markdown(index=False),
          f"\n## Cold-item stress test: {len(cold)} items hidden from training as 'new arrivals'; {len(val_c)} validation purchases of them, {len(users_c)} users\n",
          cold_t.round(5).to_markdown(index=False),
          "\n### Does the adaptive hybrid hurt ordinary (non-new-item) recommendations?\n", warm_chk.round(5).to_markdown(index=False)]
    (ROOT / "reports" / "hybrid_validation_control.md").write_text("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
