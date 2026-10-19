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
