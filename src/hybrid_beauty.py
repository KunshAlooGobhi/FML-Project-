"""Phase 4 on REAL data (Amazon Beauty 5-core + real product metadata text). Validation only.
Content model (Table II grid), hybrid blend weight, adaptive hybrid (content-only for items with no training history),
and E2/E4 slices: purchases of NEW items (no training history) vs ordinary purchases.
Two stages (each call must stay under 3 minutes on the 2-core VM):  python hybrid_beauty.py grid   then   python hybrid_beauty.py abl
Writes reports/hybrid_validation_beauty.md and reports/hybrid_beauty_choice.json."""
import sys, time, pickle, itertools, json
import numpy as np, pandas as pd
from config import PROCESSED_DIR, ROOT, SEED
from evaluate import user_sets, warm_eval_users, evaluate_batch
import models as M
from hybrid import tfidf, content_score_fn, blend_fn, adaptive_blend_fn, KEYS
from baselines import build_rankings

TOK = r"(?u)\b[a-z][a-z]+\b"; T0 = time.time()
WS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
CF = {"k": 200, "shrink": 100}      # best item-kNN on validation (reports/models_validation_beauty.md)
def log(m): print(f"[{time.time()-T0:5.0f}s] {m}", flush=True)

def cached(fn):
    """Every grid point re-uses the same collaborative / content scores for the same batch of users."""
    store = {}
    def g(ub):
        key = ub.tobytes()
        if key not in store:
            if len(store) >= 25: store.pop(next(iter(store)))
            store[key] = fn(ub)
        return store[key]
    return g

def main(stage="grid"):
    rd = lambda n: pd.read_csv(PROCESSED_DIR / f"beauty_{n}.csv.gz", parse_dates=["purchase_date"])
    train, val = rd("train"), rd("val"); texts = pd.read_csv(PROCESSED_DIR / "beauty_items.csv.gz").text.fillna("").values
    n_users, n_items = int(max(train.u.max(), val.u.max())) + 1, int(max(train.i.max(), val.i.max())) + 1
    R = M.interaction_matrix(train, n_users, n_items); tb = M.tie_break(train, n_items)
    _, rel = user_sets(train, val); users = warm_eval_users(R, rel, None, SEED)
    is_cold = np.bincount(train.i.values, minlength=n_items) == 0
    ev = lambda fn, rel_=rel, u_=users: evaluate_batch(fn, R, rel_, u_, n_items)[1]
    cf = cached(M.knn_score_fn(R, M.prune_top_k(M.knn_similarity(R, CF["shrink"]), CF["k"]), 0.0))
    PK = ROOT / "reports" / "hybrid_beauty_grid.pkl"

    if stage == "grid":
        rows = []
        for mf, ng in itertools.product([5000, 20000], [(1, 1), (1, 2)]):
            s = ev(content_score_fn(R, tfidf(train, texts, n_items, mf, ng, TOK))); rows.append({"max_features": mf, "ngram": str(ng), **{k: s[k] for k in KEYS}})
        cg = pd.DataFrame(rows); bc = cg.sort_values("NDCG@10").iloc[-1]; mf, ng = int(bc.max_features), eval(bc.ngram); log("content grid done")
        ct = cached(content_score_fn(R, tfidf(train, texts, n_items, mf, ng, TOK)))
        hg = pd.DataFrame([{"w": w, **{k: ev(blend_fn(cf, ct, w, tb))[k] for k in KEYS}} for w in WS]); wbest = float(hg.sort_values("NDCG@10").iloc[-1].w); log("hybrid grid done")
        ag = pd.DataFrame([{"adaptive_w": w, **{k: ev(adaptive_blend_fn(cf, ct, w, is_cold, tb))[k] for k in KEYS}} for w in WS]); abest = float(ag.sort_values("NDCG@10").iloc[-1].adaptive_w)
        pickle.dump(dict(cg=cg, hg=hg, ag=ag, wbest=wbest, abest=abest, mf=mf, ng=ng), open(PK, "wb")); log("grid stage saved"); return

    g = pickle.load(open(PK, "rb")); cg, hg, ag, wbest, abest, mf, ng = (g[k] for k in ("cg", "hg", "ag", "wbest", "abest", "mf", "ng"))
    ct = cached(content_score_fn(R, tfidf(train, texts, n_items, mf, ng, TOK)))
    rel_new = {u: {i for i in r if is_cold[i]} for u, r in rel.items()}; rel_new = {u: r for u, r in rel_new.items() if r}
    rel_old = {u: {i for i in r if not is_cold[i]} for u, r in rel.items()}; rel_old = {u: r for u, r in rel_old.items() if r}
    u_new = np.array([u for u in users if u in rel_new]); u_old = np.array([u for u in users if u in rel_old])
    sv = np.full(n_items, -1.0, dtype=np.float32); rk = build_rankings(train)["popularity_all"]; sv[np.asarray(rk)] = np.linspace(1, 0, len(rk), dtype=np.float32)
    variants = [("tuned popularity", lambda ub: np.tile(sv, (len(ub), 1))), ("collaborative only (item-kNN)", blend_fn(cf, ct, 0.0, tb)), ("content only", blend_fn(cf, ct, 1.0, tb)),
                (f"hybrid, global weight w={wbest}", blend_fn(cf, ct, wbest, tb)),
                (f"hybrid, adaptive (content-only for new items), w={abest}", adaptive_blend_fn(cf, ct, abest, is_cold, tb))]
    abl = []
    for name, fn in variants:
        a, n_, o = ev(fn), ev(fn, rel_new, u_new), ev(fn, rel_old, u_old)
        abl.append({"model": name, "NDCG@10 (all purchases)": a["NDCG@10"], "HR@10 on purchases of NEW items": n_["HR@10"], "NDCG@10 on purchases of NEW items": n_["NDCG@10"],
                    "NDCG@10 on ordinary purchases": o["NDCG@10"], "coverage@10": a["coverage@10"]}); log(name)
    abl = pd.DataFrame(abl)
    md = ["# Phase 4 on REAL data: Amazon Beauty 5-core with real product text (from src/hybrid_beauty.py)\n",
          f"Validation only. Item text = title, brand, category path, description from the product metadata file. {int(is_cold.sum()):,} of {n_items:,} items have no training history; "
          f"{sum(len(v) for v in rel_new.values()):,} validation purchases ({len(u_new):,} users) are of such new items.\n",
          "## Content-only model, Table II grid\n", cg.round(5).to_markdown(index=False),
          f"\n## Hybrid blend weight (item-kNN k={CF['k']}, shrink={CF['shrink']} + content max_features={mf}, ngram={ng})\n", hg.round(5).to_markdown(index=False),
          "\n## Adaptive hybrid (content-only for items with no training history), weight for all other items\n", ag.round(5).to_markdown(index=False),
          "\n## Ablation and new-item slice (E2, E4)\n", abl.round(5).to_markdown(index=False)]
    (ROOT / "reports" / "hybrid_validation_beauty.md").write_text("\n".join(md) + "\n")
    (ROOT / "reports" / "hybrid_beauty_choice.json").write_text(json.dumps({"max_features": mf, "ngram": list(ng), "w_validation_best": wbest, "adaptive_w": abest, "cf": CF}, indent=1))
    print("\n".join(md))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "grid")
