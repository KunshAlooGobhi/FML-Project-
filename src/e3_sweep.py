"""E3: training-size sweep on the CONTROL data, scored on VALIDATION only (the locked test is not used).
Random subsets (seed 42) of 25 / 50 / 100% of the training interactions; frozen settings; all fractions are evaluated
on the same users (those with history even in the 25% subset), so the curves are comparable."""
import json, numpy as np, pandas as pd
from config import PROCESSED_DIR, ROOT, SEED, MANIFEST_DIR
from evaluate import user_sets, warm_eval_users, evaluate_batch
import models as M
from baselines import build_rankings

def main():
    cfg = json.loads((MANIFEST_DIR / "frozen_settings.json").read_text())["control"]
    rd = lambda n: pd.read_csv(PROCESSED_DIR / f"control_{n}.csv.gz", parse_dates=["purchase_date"])
    train, val = rd("train"), rd("val")
    n_users, n_items = int(max(train.u.max(), val.u.max())) + 1, int(max(train.i.max(), val.i.max())) + 1
    perm = np.random.default_rng(SEED).permutation(len(train)); subs = {f: train.iloc[np.sort(perm[:int(len(train) * f)])] for f in (0.25, 0.5, 1.0)}
    R25 = M.interaction_matrix(subs[0.25], n_users, n_items); _, rel = user_sets(subs[0.25], val); users = warm_eval_users(R25, rel)
    rows = []
    for f, tr in subs.items():
        R = M.interaction_matrix(tr, n_users, n_items); tb = M.tie_break(tr, n_items)
        rank = build_rankings(tr)[cfg["popularity"]["variant"]]; sv = np.full(n_items, -1.0, dtype=np.float32); sv[np.asarray(rank)] = np.linspace(1, 0, len(rank), dtype=np.float32)
        p = cfg["itemknn"]; a = cfg["als"]
        fns = {"Popularity": lambda ub: np.tile(sv, (len(ub), 1)),
               "Item-kNN": M.knn_score_fn(R, M.prune_top_k(M.knn_similarity(R, p["shrink"]), p["k"]), tb),
               "ALS": M.factor_score_fn(M.fit_als(R, a["factors"], a["reg"], a["alpha"]), tb)}
        for name, fn in fns.items():
            _, s = evaluate_batch(fn, R, rel, users, n_items); rows.append({"training data": f"{int(f*100)}%", "interactions": len(tr), "model": name, "NDCG@10": s["NDCG@10"], "HR@10": s["HR@10"]})
    d = pd.DataFrame(rows); wide = d.pivot(index="model", columns="training data", values="NDCG@10")[["25%", "50%", "100%"]]
    md = ["# E3: training-size sweep (control data, validation only; from src/e3_sweep.py)\n", f"Evaluated on the same {len(users):,} users in every row.\n",
          "NDCG@10 by share of training data used:\n", wide.round(5).to_markdown(), "\nDetail:\n", d.round(5).to_markdown(index=False)]
    (ROOT / "reports" / "e3_training_size_control.md").write_text("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
