"""Toy example worked out by hand: we recommend [a,b,c,d,e]; user really bought {a,c,f}."""
import math
from metrics import *
recs, rel, k = list("abcde"), {"a", "c", "f"}, 5
assert math.isclose(precision_at_k(recs, rel, k), 2 / 5)                 # a and c hit -> 2 of 5
assert math.isclose(recall_at_k(recs, rel, k), 2 / 3)                    # caught 2 of the 3 bought
assert hit_rate_at_k(recs, rel, k) == 1.0
assert math.isclose(average_precision_at_k(recs, rel, k), (1/1 + 2/3) / 3)   # hits at rank 1 and 3
dcg, idcg = 1 + 1/2, 1 + 1/math.log2(3) + 1/2                            # hits at ranks 1,3 vs ideal ranks 1,2,3
assert math.isclose(ndcg_at_k(recs, rel, k), dcg / idcg)
assert ndcg_at_k(list("xyz"), rel, 3) == 0.0                              # no hits -> 0
print("all metric checks passed; NDCG@5 =", round(ndcg_at_k(recs, rel, k), 4))
