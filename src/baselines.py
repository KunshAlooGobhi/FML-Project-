"""Phase 2 / Step 2: baselines scored on VALIDATION only (locked test untouched).
All statistics (counts, mean ratings, windows) are computed from TRAIN only - no leakage."""
import numpy as np, pandas as pd
from config import PROCESSED_DIR, ROOT, SEED
from evaluate import user_sets, evaluate, ranked_list_recommender

def build_rankings(train):
    end = train.purchase_date.max()
    rng = np.random.default_rng(SEED)

    rankings = {"random": rng.permutation(train.i.unique())}
    rankings["popularity_all"] = train.i.value_counts().index
    for days in (30, 90, 365):                                    # recency-weighted: only recent purchases count
        recent = train[train.purchase_date > end - pd.Timedelta(days=days)]
        rankings[f"popularity_{days}d"] = recent.i.value_counts().index
    g = train.groupby("i").rating.agg(["sum", "count"])           # Bayesian average: shrink small-sample means to global mean
    m, mu = 20, train.rating.mean()
    rankings["bayes_rating"] = ((g["sum"] + m * mu) / (g["count"] + m)).sort_values(ascending=False).index

    return rankings

def main(prefix=""):
    rd = lambda n: pd.read_csv(PROCESSED_DIR / f"{prefix}{n}.csv.gz", parse_dates=["purchase_date"])
    train, val = rd("train"), rd("val")
    n_items = pd.concat([train.i, val.i]).nunique()
    rankings = build_rankings(train)
    mu = train.rating.mean()

    seen, rel_any = user_sets(train, val)
    _, rel_good = user_sets(train, val, min_rating=4.0)
    results = []
    for name, ranking in rankings.items():
        rec = ranked_list_recommender(ranking)
        pu, s = evaluate(rec, seen, rel_any, n_items)
        _, s4 = evaluate(rec, seen, rel_good, n_items)
        results.append({"model": name, **{k: s[k] for k in ("P@10", "R@10", "HR@10", "MAP@10", "NDCG@5", "NDCG@10", "NDCG@20", "coverage@10")},
                        "NDCG@10_warm_users": pu[pu.warm_user]["NDCG@10"].mean(),
                        "NDCG@10_cold_users": pu[~pu.warm_user]["NDCG@10"].mean(),
                        "NDCG@10_rating>=4": s4["NDCG@10"]})
        print("done", name, flush=True)
    res = pd.DataFrame(results).set_index("model")
    best = res["NDCG@10"].idxmax()

    # rating-prediction baselines (RMSE/MAE) on validation
    ib = train.groupby("i").rating.mean()
    pred_item = val.i.map(ib).fillna(mu)
    rm = lambda p: float(np.sqrt(((val.rating - p) ** 2).mean()))
    ma = lambda p: float((val.rating - p).abs().mean())

    (ROOT / "reports").mkdir(exist_ok=True)
    res.round(5).to_csv(ROOT / "reports" / f"baselines_validation{'_' + prefix.strip('_') if prefix else ''}.csv")
    md = ["# Baselines on VALIDATION (from src/baselines.py)\n",
          f"Validation users with >=1 purchase: {len(rel_any):,} ({sum(u in seen for u in rel_any):,} seen in train). Best by NDCG@10: **{best}**.\n",
          res.round(5).to_markdown(), "",
          f"Rating prediction (validation): global mean RMSE {rm(mu):.4f}, MAE {ma(mu):.4f}; item-mean RMSE {rm(pred_item):.4f}, MAE {ma(pred_item):.4f}."]
    (ROOT / "reports" / f"baselines_validation{'_' + prefix.strip('_') if prefix else ''}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

if __name__ == "__main__":
    main()
