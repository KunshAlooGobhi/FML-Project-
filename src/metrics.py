"""Ranking metrics for one user. `recs` = ranked list of recommended item ids (best first),
`relevant` = set of items the user actually interacted with in the evaluation period."""
import math

def precision_at_k(recs, relevant, k):
    return sum(1 for r in recs[:k] if r in relevant) / k

def recall_at_k(recs, relevant, k):
    return sum(1 for r in recs[:k] if r in relevant) / len(relevant) if relevant else 0.0

def hit_rate_at_k(recs, relevant, k):
    return 1.0 if any(r in relevant for r in recs[:k]) else 0.0

def average_precision_at_k(recs, relevant, k):
    hits, total = 0, 0.0
    for rank, r in enumerate(recs[:k], start=1):
        if r in relevant:
            hits += 1
            total += hits / rank          # precision at each hit position
    return total / min(len(relevant), k) if relevant else 0.0

def ndcg_at_k(recs, relevant, k):
    """DCG gives 1/log2(rank+1) credit per hit (top ranks count more); IDCG is the best possible DCG."""
    dcg = sum(1 / math.log2(rank + 1) for rank, r in enumerate(recs[:k], start=1) if r in relevant)
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / idcg if idcg else 0.0
