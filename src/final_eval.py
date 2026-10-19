"""Phase 5: frozen settings -> final fit -> score. One model per call; each result is saved.
  python final_eval.py <control|main> <val|test> <step> [step ...]
steps: random popularity itemknn svd als bpr content hybrid_adaptive cold finalize
split 'val' = DRY RUN (fit on train, score validation; also feeds E7). split 'test' = LOCKED TEST (fit on train+validation;
refuses to overwrite any saved model result, so every model is scored on the test exactly once)."""
import sys, json, pickle
import numpy as np, pandas as pd
from config import PROCESSED_DIR, ROOT, SEED, MANIFEST_DIR
from evaluate import user_sets, warm_eval_users, evaluate_batch
import models as M
from baselines import build_rankings
from hybrid import tfidf, content_score_fn, adaptive_blend_fn

FROZEN = json.loads((MANIFEST_DIR / "frozen_settings.json").read_text())

def out_dir(dataset, split):
    d = ROOT / "reports" / f"final_{split}_{dataset}"; d.mkdir(parents=True, exist_ok=True); return d

class Ctx:
    def __init__(self, dataset, split):
        self.dataset, self.split, self.cfg = dataset, split, FROZEN[dataset]
        prefix = "control_" if dataset == "control" else ""
        rd = lambda n: pd.read_csv(PROCESSED_DIR / f"{prefix}{n}.csv.gz", parse_dates=["purchase_date"])
        train, val, test = rd("train"), rd("val"), rd("test")
        self.fit, self.ev = (train, val) if split == "val" else (pd.concat([train, val], ignore_index=True), test)
        self.n_users = int(max(d.u.max() for d in (train, val, test))) + 1
        self.n_items = int(max(d.i.max() for d in (train, val, test))) + 1
        self.R = M.interaction_matrix(self.fit, self.n_users, self.n_items); self.tb = M.tie_break(self.fit, self.n_items)
        _, self.rel = user_sets(self.fit, self.ev)
        self.users = warm_eval_users(self.R, self.rel, self.cfg["eval_users_max"], SEED)
        counts = np.bincount(self.fit.i.values, minlength=self.n_items)
        self.is_cold = counts == 0
        self.pop_share = (counts + 1) / (len(self.fit) + self.n_items)      # for novelty
        self.texts = pd.read_csv(PROCESSED_DIR / "control_items.csv.gz").text.values if dataset == "control" else None

def score_fn_for(name, c):
    cfg = c.cfg; extra = {}
    if name == "random":
        rng = np.random.default_rng(SEED); fn = lambda ub: rng.random((len(ub), c.n_items), dtype=np.float32)
    elif name == "popularity":
        ranking = build_rankings(c.fit)[cfg["popularity"]["variant"]]
        sv = np.full(c.n_items, -1.0, dtype=np.float32); sv[np.asarray(ranking)] = np.linspace(1, 0, len(ranking), dtype=np.float32)
        fn = lambda ub: np.tile(sv, (len(ub), 1))
    elif name == "itemknn":
        p = cfg["itemknn"]; fn = M.knn_score_fn(c.R, M.prune_top_k(M.knn_similarity(c.R, p["shrink"]), p["k"]), c.tb)
    elif name == "svd":
        p = cfg["svd"]; m = M.fit_svd(c.fit, c.n_users, c.n_items, p["factors"], p["reg"], p["lr"]); fn = M.svd_score_fn(m, c.tb)
        e = c.ev.rating.values - M.svd_predict(m, c.ev.u.values, c.ev.i.values); extra["rating_err"] = e
    elif name == "als":
        p = cfg["als"]; fn = M.factor_score_fn(M.fit_als(c.R, p["factors"], p["reg"], p["alpha"]), c.tb)
    elif name == "bpr":
        p = cfg["bpr"]; fn = M.factor_score_fn(M.fit_bpr(c.R, p["factors"], p["reg"], p["lr"]), c.tb)
    elif name in ("content", "hybrid_adaptive"):
        p = cfg["content"]; X = tfidf(c.fit, c.texts, c.n_items, p["max_features"], tuple(p["ngram"])); ct = content_score_fn(c.R, X)
        if name == "content":
            fn = lambda ub: ct(ub) + c.tb
        else:
            cf = M.knn_score_fn(c.R, M.prune_top_k(M.knn_similarity(c.R, cfg["itemknn"]["shrink"]), cfg["itemknn"]["k"]), 0.0)
            fn = adaptive_blend_fn(cf, ct, cfg["hybrid"]["adaptive_w"], c.is_cold, c.tb)
    return fn, extra

def diversity_novelty(recs, c, sample=1000):
    nov = np.mean([np.mean(-np.log2(c.pop_share[r[:10]])) for r in recs.values()])
    div = np.nan
    if c.texts is not None:
        p = c.cfg["content"]; X = tfidf(c.fit, c.texts, c.n_items, p["max_features"], tuple(p["ngram"]))
        us = list(recs)[:sample]; vals = []
        for u in us:
            V = X[recs[u][:10]].toarray(); S = V @ V.T; k = len(V)
            vals.append(1 - (S.sum() - np.trace(S)) / (k * (k - 1)))
        div = float(np.mean(vals))
    return float(nov), div

def run_step(dataset, split, name):
    d = out_dir(dataset, split); f = d / f"{name}.pkl"
    if split == "test" and f.exists():
        sys.exit(f"REFUSED: {name} was already scored on the locked test ({f.name}). Each model is scored once.")
    c = Ctx(dataset, split)
    if name == "cold":                                     # E2: cold users (no history in the fit data), popularity vs random
        has = np.diff(c.R.indptr) > 0
        cu = np.array(sorted(u for u in c.rel if u >= c.R.shape[0] or not has[u]))
        cu = np.sort(np.random.default_rng(SEED).choice(cu, min(len(cu), 5000), replace=False)) if len(cu) else cu
        res = {"n_cold_users_total": int(sum(1 for u in c.rel if not has[u])), "users": cu}
        for m in (("random", "popularity") if len(cu) else ()):
            fn, _ = score_fn_for(m, c); pu, s = evaluate_batch(fn, c.R, c.rel, cu, c.n_items); res[m] = {"per_user": pu[["u", "NDCG@10", "HR@10"]], "summary": s}
        slices = c.ev.assign(known_user=c.ev.u.isin(set(c.fit.u)), known_item=c.ev.i.isin(set(c.fit.i)))
        res["row_slices"] = {"warm": int((slices.known_user & slices.known_item).sum()), "cold-item": int((slices.known_user & ~slices.known_item).sum()), "cold-user": int((~slices.known_user).sum())}
    else:
        fn, extra = score_fn_for(name, c)
        pu, s, recs = evaluate_batch(fn, c.R, c.rel, c.users, c.n_items, return_recs=True)
        nov, div = diversity_novelty(recs, c); s["novelty"], s["diversity"] = nov, div
        res = {"per_user": pu, "summary": s, **extra}
    pickle.dump(res, open(f, "wb"))
    print(f"[{dataset}/{split}] {name}: saved. " + (f"NDCG@10={res['summary']['NDCG@10']:.5f} HR@10={res['summary']['HR@10']:.4f} users={res['summary']['users']}" if "summary" in res and name != "cold" else ""))

if __name__ == "__main__":
    ds, sp, *steps = sys.argv[1:]
    if "finalize" in steps:
        from finalize import finalize; finalize(ds, sp); steps = [s for s in steps if s != "finalize"]
    for st in steps:
        run_step(ds, sp, st)
