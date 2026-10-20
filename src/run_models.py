"""Phase 3: tune each model on VALIDATION (best NDCG@10), grids from Table II of the report.
Usage: python run_models.py <prefix> <model> [max_users] [full|small]
  model in: pop, knn, svd, als, bpr, summary.  Locked test is never touched here."""
import sys, time, itertools, json
import numpy as np, pandas as pd
from config import PROCESSED_DIR, ROOT, SEED
from evaluate import user_sets, warm_eval_users, evaluate_batch
import models as M
from baselines import build_rankings

GRIDS = {
    "full": dict(knn=dict(k=[20, 50, 100], shrink=[0, 10, 50]),
                 svd=dict(factors=[32, 64, 128], reg=[0.02, 0.05, 0.1], lr=[0.005, 0.01]),
                 als=dict(factors=[32, 64, 128], reg=[0.01, 0.1], alpha=[10, 40]),
                 bpr=dict(factors=[32, 64, 128], reg=[0.01, 0.1], lr=[0.01, 0.05])),
    # reduced grids for the big (synthetic) 1M file on a 2-core / 3 GB machine; documented in the report
    "small": dict(knn=dict(k=[20, 50, 100], shrink=[0, 10, 50]),
                  svd=dict(factors=[32], reg=[0.02, 0.1], lr=[0.005, 0.01]),
                  als=dict(factors=[32], reg=[0.01, 0.1], alpha=[10, 40]),
                  bpr=dict(factors=[32], reg=[0.01, 0.1], lr=[0.01, 0.05])),
}
# extended grid (added after seeing that several best Table II settings sat on the grid edge); VALIDATION only, documented as a deviation
GRIDS["ext"] = dict(knn=dict(k=[100, 200], shrink=[100, 200, 500]),
                    svd=dict(factors=[128, 256], reg=[0.02, 0.05], lr=[0.005]),
                    als=dict(factors=[128, 256], reg=[0.01, 0.001], alpha=[40, 100]),
                    bpr=dict(factors=[128, 256], reg=[0.01, 0.001], lr=[0.05, 0.1]))
KEYS = ["P@10", "R@10", "HR@10", "MAP@10", "NDCG@5", "NDCG@10", "NDCG@20", "coverage@10", "users"]

def combos(g):
    return [dict(zip(g, v)) for v in itertools.product(*g.values())]

def main(prefix, model, max_users=None, grid="full"):
    rd = lambda n: pd.read_csv(PROCESSED_DIR / f"{prefix}{n}.csv.gz", parse_dates=["purchase_date"])
    train, val = rd("train"), rd("val")
    n_users, n_items = int(max(train.u.max(), val.u.max())) + 1, int(max(train.i.max(), val.i.max())) + 1
    R = M.interaction_matrix(train, n_users, n_items); tb = M.tie_break(train, n_items)
    _, rel = user_sets(train, val)
    users = warm_eval_users(R, rel, max_users, SEED)
    out = ROOT / "reports"; tag = prefix.strip("_") or "main"
    print(f"[{tag}/{model}] eval users: {len(users):,}, train rows {len(train):,}", flush=True)
    rows = []
    def score(name, cfg, fn, extra=None):
        t = time.time(); _, s = evaluate_batch(fn, R, rel, users, n_items)
        rows.append({"model": name, **cfg, **{k: s[k] for k in KEYS}, **(extra or {}), "secs": round(time.time() - t, 1)})
        print(f"  {name} {cfg} NDCG@10={s['NDCG@10']:.5f} HR@10={s['HR@10']:.4f}", flush=True)

    g = GRIDS[grid]
    if model == "pop":
        rng = np.random.default_rng(SEED)
        score("random", {}, lambda ub: rng.random((len(ub), n_items), dtype=np.float32))
        for name, ranking in build_rankings(train).items():
            if name == "random": continue
            sv = np.full(n_items, -1.0, dtype=np.float32)
            sv[np.asarray(ranking)] = np.linspace(1, 0, len(ranking), dtype=np.float32)   # rank -> score
            score(f"pop:{name}", {}, lambda ub, sv=sv: np.tile(sv, (len(ub), 1)))
    elif model == "knn":
        for shrink in g["knn"]["shrink"]:
            S = M.knn_similarity(R, shrink)
            for k in g["knn"]["k"]:
                score("itemknn", {"k": k, "shrink": shrink}, M.knn_score_fn(R, M.prune_top_k(S, k), tb))
    elif model == "svd":
        for cfg in combos(g["svd"]):
            m = M.fit_svd(train, n_users, n_items, cfg["factors"], cfg["reg"], cfg["lr"])
            pv = M.svd_predict(m, val.u.values, val.i.values); e = val.rating.values - pv
            score("svd", cfg, M.svd_score_fn(m, tb), {"RMSE": float(np.sqrt((e ** 2).mean())), "MAE": float(np.abs(e).mean())})
    elif model in ("als", "bpr"):
        for cfg in combos(g[model]):
            m = M.fit_als(R, cfg["factors"], cfg["reg"], cfg["alpha"]) if model == "als" else M.fit_bpr(R, cfg["factors"], cfg["reg"], cfg["lr"])
            score(model, cfg, M.factor_score_fn(m, tb))
    if rows:
        pd.DataFrame(rows).to_csv(out / f"models_{tag}_{model}{'_ext' if grid == 'ext' else ''}.csv", index=False)

def summary(prefix):
    tag = prefix.strip("_") or "main"; out = ROOT / "reports"; best = []
    for model in ("pop", "knn", "svd", "als", "bpr"):
        f = out / f"models_{tag}_{model}.csv"
        if not f.exists(): continue
        d = pd.read_csv(f); fe = out / f"models_{tag}_{model}_ext.csv"
        if fe.exists(): d = pd.concat([d, pd.read_csv(fe)], ignore_index=True)
        if model == "pop":
            best += [d[d.model == "random"].iloc[0], d[d.model != "random"].sort_values("NDCG@10").iloc[-1]]
        else:
            best.append(d.sort_values("NDCG@10").iloc[-1])
    b = pd.DataFrame(best).reset_index(drop=True)
    cfgcols = [c for c in b.columns if c not in KEYS + ["model", "RMSE", "MAE", "secs"]]
    b["best_setting"] = b[cfgcols].apply(lambda r: ", ".join(f"{c}={r[c]:g}" for c in cfgcols if pd.notna(r[c])), axis=1)
    cols = ["model", "best_setting", "P@10", "R@10", "HR@10", "MAP@10", "NDCG@5", "NDCG@10", "NDCG@20", "coverage@10", "RMSE", "MAE", "users"]
    md = f"# Tuned models on VALIDATION, warm users ({tag}; from src/run_models.py)\n\nBest setting per model by validation NDCG@10.\n\n" + b[cols].round(5).to_markdown(index=False) + "\n"
    (out / f"models_validation_{tag}.md").write_text(md); print(md)

if __name__ == "__main__":
    a = sys.argv[1:]
    prefix = "" if a[0] in ("main", "") else a[0]
    if a[1] == "summary":
        summary(prefix)
    else:
        main(prefix, a[1], int(a[2]) if len(a) > 2 and a[2] != "0" else None, a[3] if len(a) > 3 else "full")
