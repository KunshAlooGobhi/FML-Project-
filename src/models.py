"""Model implementations. Each fit_* returns a score_fn(user_array) -> (n_users_in_batch, n_items) matrix."""
import numpy as np, scipy.sparse as sp

def interaction_matrix(df, n_users, n_items):
    return sp.csr_matrix((np.ones(len(df), dtype=np.float32), (df.u.values, df.i.values)), shape=(n_users, n_items))

def tie_break(train, n_items):
    """Tiny popularity bonus in [0, 1e-6): makes ties deterministic and sensible (documented in the report)."""
    c = np.bincount(train.i.values, minlength=n_items).astype(np.float32)
    return 1e-6 * c / (c.max() + 1)

# ---------- item-kNN: cosine similarity with shrinkage, top-k neighbours ----------
def knn_similarity(R, shrink):
    """sim_ij = co(i,j) / (sqrt(n_i * n_j) + shrink); shrink damps similarities backed by few co-purchases."""
    co = (R.T @ R).tocoo()
    n = np.asarray(R.sum(axis=0)).ravel()
    keep = co.row != co.col
    row, col, v = co.row[keep], co.col[keep], co.data[keep]
    return sp.csr_matrix((v / (np.sqrt(n[row] * n[col]) + shrink), (row, col)), shape=co.shape)

def prune_top_k(S, k):
    S = S.tocsr(); data, idx, ptr = [], [], [0]
    for r in range(S.shape[0]):
        a, b = S.indptr[r], S.indptr[r + 1]
        d, c = S.data[a:b], S.indices[a:b]
        if len(d) > k:
            top = np.argpartition(-d, k - 1)[:k]; d, c = d[top], c[top]
        data.append(d); idx.append(c); ptr.append(ptr[-1] + len(d))
    return sp.csr_matrix((np.concatenate(data), np.concatenate(idx), np.array(ptr)), shape=S.shape)

def knn_score_fn(R, S, tb):
    return lambda users: (R[users] @ S).toarray() + tb

# ---------- biased SVD (FunkSVD-style, trained by SGD to predict ratings) ----------
def fit_svd(train, n_users, n_items, factors, reg, lr, epochs=20, seed=42, batch=2048):
    rng = np.random.default_rng(seed)
    u, i, r = train.u.values, train.i.values, train.rating.values.astype(np.float64)
    mu = r.mean(); bu = np.zeros(n_users); bi = np.zeros(n_items)
    P = 0.1 * rng.standard_normal((n_users, factors)); Q = 0.1 * rng.standard_normal((n_items, factors))
    for _ in range(epochs):
        perm = rng.permutation(len(r))
        for s in range(0, len(r), batch):
            ix = perm[s:s + batch]; uu, ii = u[ix], i[ix]
            pu, qi = P[uu], Q[ii]
            err = r[ix] - (mu + bu[uu] + bi[ii] + (pu * qi).sum(1))
            np.add.at(bu, uu, lr * (err - reg * bu[uu])); np.add.at(bi, ii, lr * (err - reg * bi[ii]))
            np.add.at(P, uu, lr * (err[:, None] * qi - reg * pu)); np.add.at(Q, ii, lr * (err[:, None] * pu - reg * qi))
    return dict(mu=mu, bu=bu, bi=bi, P=P, Q=Q)

def svd_predict(m, u, i):
    return np.clip(m["mu"] + m["bu"][u] + m["bi"][i] + (m["P"][u] * m["Q"][i]).sum(1), 1, 5)

def svd_score_fn(m, tb):
    Qt = m["Q"].T.astype(np.float32)
    return lambda users: (m["P"][users].astype(np.float32) @ Qt) + m["bi"].astype(np.float32) + tb

# ---------- implicit-feedback models via the `implicit` library ----------
def fit_als(R, factors, reg, alpha, iterations=15, seed=42):
    from implicit.cpu.als import AlternatingLeastSquares
    m = AlternatingLeastSquares(factors=factors, regularization=reg, alpha=alpha, iterations=iterations, random_state=seed)
    m.fit(R, show_progress=False); return m

def fit_bpr(R, factors, reg, lr, iterations=100, seed=42):
    from implicit.cpu.bpr import BayesianPersonalizedRanking
    m = BayesianPersonalizedRanking(factors=factors, learning_rate=lr, regularization=reg, iterations=iterations, random_state=seed)
    m.fit(R, show_progress=False); return m

def factor_score_fn(m, tb):
    U, V = np.asarray(m.user_factors, dtype=np.float32), np.asarray(m.item_factors, dtype=np.float32)
    return lambda users: U[users] @ V.T + tb
