"""Freeze every hyperparameter from VALIDATION results (no test data is read here).
Rule for ties (e.g. the 1M file, where all validation scores are ~0): the first setting in grid order."""
import json, numpy as np, pandas as pd
from config import PROCESSED_DIR, ROOT, MANIFEST_DIR

def first_best(df):
    return df[df["NDCG@10"] == df["NDCG@10"].max()].iloc[0]

def clean(row, keys, ints=()):
    return {k: (int(row[k]) if k in ints else float(row[k])) for k in keys}

def bias_lambda(prefix):
    train = pd.read_csv(PROCESSED_DIR / f"{prefix}train.csv.gz"); val = pd.read_csv(PROCESSED_DIR / f"{prefix}val.csv.gz")
    mu = train.rating.mean(); best = None
    gi = train.groupby("i").rating.agg(["sum", "count"])
    for lam in (5, 10, 25, 50):
        bi = (gi["sum"] - gi["count"] * mu) / (gi["count"] + lam)
        r = train.rating - mu - train.i.map(bi)
        gu = r.groupby(train.u).agg(["sum", "count"]); bu = gu["sum"] / (gu["count"] + lam)
        pred = mu + val.i.map(bi).fillna(0) + val.u.map(bu).fillna(0)
        rmse = float(np.sqrt(((val.rating - pred) ** 2).mean()))
        if best is None or rmse < best[1]: best = (lam, rmse)
    return {"lambda": best[0], "validation_rmse": round(best[1], 5)}

def main():
    out = {"_note": "Frozen from validation results only. Ties -> first setting in grid order. Final models are refit on train+validation.",
           "_sources": "reports/models_*_*.csv, reports/hybrid_validation_control.md"}
    for tag, prefix, max_users in (("control", "control_", None), ("main", "", 5000)):
        R = ROOT / "reports"; d = {"eval_users_max": max_users}
        pop = pd.read_csv(R / f"models_{tag}_pop.csv"); pop = pop[pop.model != "random"]
        d["popularity"] = {"variant": first_best(pop).model.split(":")[1]}
        d["itemknn"] = clean(first_best(pd.read_csv(R / f"models_{tag}_knn.csv")), ["k", "shrink"], ints=("k", "shrink"))
        d["svd"] = clean(first_best(pd.read_csv(R / f"models_{tag}_svd.csv")), ["factors", "reg", "lr"], ints=("factors",))
        d["als"] = clean(first_best(pd.read_csv(R / f"models_{tag}_als.csv")), ["factors", "reg", "alpha"], ints=("factors",))
        d["bpr"] = clean(first_best(pd.read_csv(R / f"models_{tag}_bpr.csv")), ["factors", "reg", "lr"], ints=("factors",))
        d["bias_baseline"] = bias_lambda(prefix)
        if tag == "control":   # from reports/hybrid_validation_control.md
            d["content"] = {"max_features": 5000, "ngram": [1, 2]}
            d["hybrid"] = {"w_validation_best": 0.0, "adaptive_w": 0.1,
                           "note": "validation-best global weight is 0 (= item-kNN); the adaptive variant (content-only for items with no history, w=0.1) is the declared cold-start design"}
        out[tag] = d
    MANIFEST_DIR.mkdir(exist_ok=True)
    (MANIFEST_DIR / "frozen_settings.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
