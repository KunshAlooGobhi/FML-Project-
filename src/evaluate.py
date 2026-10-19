"""Reusable evaluation harness: every model (baseline or advanced) is scored by this same code."""
import numpy as np, pandas as pd
from metrics import precision_at_k, recall_at_k, hit_rate_at_k, average_precision_at_k, ndcg_at_k

KS = (5, 10, 20)

def user_sets(train, ev, min_rating=None):
    """seen[u] = items u bought in TRAIN (never re-recommended); rel[u] = items u bought in the eval period."""
    seen = train.groupby("u")["i"].apply(set).to_dict()
    rel_df = ev if min_rating is None else ev[ev.rating >= min_rating]
    return seen, rel_df.groupby("u")["i"].apply(set).to_dict()

def evaluate(rec_fn, seen, rel, n_items, ks=KS):
    """rec_fn(u, seen_items, n) -> ranked list of n item ids. Returns (per-user DataFrame, summary dict)."""
    rows, shown = [], set()
    for u, relevant in rel.items():
        recs = rec_fn(u, seen.get(u, set()), max(ks))
        row = {"u": u, "warm_user": u in seen}
        for k in ks:
            row[f"P@{k}"] = precision_at_k(recs, relevant, k)
            row[f"R@{k}"] = recall_at_k(recs, relevant, k)
            row[f"HR@{k}"] = hit_rate_at_k(recs, relevant, k)
            row[f"MAP@{k}"] = average_precision_at_k(recs, relevant, k)
            row[f"NDCG@{k}"] = ndcg_at_k(recs, relevant, k)
        shown.update(recs[:10])
        rows.append(row)
    per_user = pd.DataFrame(rows)
    summary = per_user.drop(columns=["u", "warm_user"]).mean().to_dict()
    summary["coverage@10"] = len(shown) / n_items            # share of catalogue ever recommended
    summary["users"] = len(per_user)
    return per_user, summary

def ranked_list_recommender(ranking):
    """Same ranked list for everyone, minus what the user already bought."""
    ranking = list(ranking)
    def rec(u, seen_items, n):
        out = []
        for i in ranking:
            if i not in seen_items:
                out.append(i)
                if len(out) == n:
                    break
        return out
    return rec


# ---------- batch evaluation for score-matrix models (kNN, SVD, ALS, BPR) ----------
def warm_eval_users(R, rel, max_users=None, seed=42):
    """Users who have training history (R = train user-item csr) and >=1 validation purchase.
    If max_users is set, a seeded random sample - identical for every model, so comparisons stay fair."""
    has_hist = np.diff(R.indptr) > 0
    users = np.array(sorted(u for u in rel if u < R.shape[0] and has_hist[u]))
    if max_users and len(users) > max_users:
        users = np.sort(np.random.default_rng(seed).choice(users, max_users, replace=False))
    return users

def evaluate_batch(score_fn, R, rel, users, n_items, ks=KS, batch=400, return_recs=False):
    """score_fn(user_array) -> (len(users), n_items) float32 scores. Seen (train) items are masked out."""
    maxk, rows, shown, recs_all = max(ks), [], set(), {}
    for s in range(0, len(users), batch):
        ub = users[s:s + batch]
        sc = np.asarray(score_fn(ub), dtype=np.float32)
        for r, u in enumerate(ub):
            sc[r, R.indices[R.indptr[u]:R.indptr[u + 1]]] = -np.inf
        part = np.argpartition(-sc, maxk - 1, axis=1)[:, :maxk]
        order = np.argsort(-np.take_along_axis(sc, part, 1), axis=1)
        top = np.take_along_axis(part, order, 1)
        for r, u in enumerate(ub):
            recs, relevant = top[r].tolist(), rel[u]
            row = {"u": u}
            for k in ks:
                row[f"P@{k}"] = precision_at_k(recs, relevant, k)
                row[f"R@{k}"] = recall_at_k(recs, relevant, k)
                row[f"HR@{k}"] = hit_rate_at_k(recs, relevant, k)
                row[f"MAP@{k}"] = average_precision_at_k(recs, relevant, k)
                row[f"NDCG@{k}"] = ndcg_at_k(recs, relevant, k)
            shown.update(recs[:10]); recs_all[int(u)] = recs
            rows.append(row)
    per_user = pd.DataFrame(rows)
    summary = per_user.drop(columns="u").mean().to_dict()
    summary["coverage@10"] = len(shown) / n_items
    summary["users"] = len(per_user)
    return (per_user, summary, recs_all) if return_recs else (per_user, summary)
