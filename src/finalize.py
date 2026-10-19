"""Merge saved per-model results into Tables III/IV, bootstrap tests, E2, E5, E6, E7. Reads saved results only (no refitting)."""
import pickle, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import ROOT, SEED
from final_eval import out_dir, Ctx, FROZEN

ORDER = ["random", "popularity", "itemknn", "svd", "als", "bpr", "content", "hybrid_adaptive"]
LABEL = {"random": "Random (reference)", "popularity": "Popularity (tuned)", "itemknn": "Item-kNN", "svd": "SVD (biased, rating-trained)",
         "als": "ALS", "bpr": "BPR", "content": "Content-based (TF-IDF)", "hybrid_adaptive": "Hybrid (adaptive: content for new items)"}
SHORT = {"random": "Random", "popularity": "Popularity", "itemknn": "Item-kNN", "svd": "SVD", "als": "ALS", "bpr": "BPR", "content": "Content", "hybrid_adaptive": "Hybrid"}
B = 1000
OFFSET = {"itemknn": (-7, 7, "right"), "hybrid_adaptive": (8, 7, "left"), "als": (8, -13, "left"), "svd": (8, -13, "left"), "popularity": (8, 6, "left"),
          "bpr": (8, 6, "left"), "content": (-8, 7, "right"), "random": (8, -13, "left")}

def paired_bootstrap(a, b, m):
    """Paired bootstrap over users of mean(a-b). Returns diff, 95% CI, p, Bonferroni-adjusted p."""
    rng = np.random.default_rng(SEED); d = np.asarray(a) - np.asarray(b); n = len(d)
    boot = d[rng.integers(0, n, (B, n))].mean(1)
    p = min(1.0, max(2 * min((boot <= 0).mean(), (boot >= 0).mean()), 1 / B))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return d.mean(), lo, hi, p, min(1.0, p * m)

def verdict(diff, lo, hi, padj):
    if lo > 0 and padj < 0.05: return "better than popularity"
    if hi < 0 and padj < 0.05: return "worse than popularity"
    return "not distinguishable"

def md_table(df, fmt=5):
    return df.round(fmt).to_markdown(index=False)

def load(dataset, split):
    d = out_dir(dataset, split); res = {}
    for n in ORDER:
        if (d / f"{n}.pkl").exists(): res[n] = pickle.load(open(d / f"{n}.pkl", "rb"))
    cold = pickle.load(open(d / "cold.pkl", "rb")) if (d / "cold.pkl").exists() else None
    return res, cold

def bias_predict(fit, ev, lam):
    mu = fit.rating.mean(); g = fit.groupby("i").rating.agg(["sum", "count"])
    bi = (g["sum"] - g["count"] * mu) / (g["count"] + lam)
    gu = (fit.rating - mu - fit.i.map(bi)).groupby(fit.u).agg(["sum", "count"]); bu = gu["sum"] / (gu["count"] + lam)
    return mu + ev.i.map(bi).fillna(0).values + ev.u.map(bu).fillna(0).values

def figure(res, dataset, split):
    surf, ink, ink2 = "#fcfcfb", "#0b0b0b", "#52514e"
    role_col = {"random": "#8c8b86", "popularity": "#2a78d6"}; other = "#eb6834"          # reference palette slots 1-2 + neutral
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=surf)
    for ax, (xk, xl) in zip(axes, (("coverage@10", "Catalogue coverage @10 (share of items ever recommended)"), ("novelty", "Novelty (mean -log2 popularity of recommended items)"))):
        ax.set_facecolor(surf)
        for n, r in res.items():
            s = r["summary"]; x, y = s[xk] if xk != "novelty" else s["novelty"], s["NDCG@10"]
            ax.scatter(x, y, s=80, color=role_col.get(n, other), edgecolor=surf, linewidth=2, zorder=3)
            dx, dy, ha = OFFSET[n]
            ax.annotate(SHORT[n], (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=9, color=ink, ha=ha)
        ax.set_xlabel(xl, color=ink2, fontsize=9); ax.set_ylabel("NDCG@10 (accuracy)", color=ink2, fontsize=9)
        ax.grid(True, color="#e3e2de", linewidth=0.8, zorder=0); ax.tick_params(colors=ink2, labelsize=8)
        for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
        for sp_ in ("left", "bottom"): ax.spines[sp_].set_color("#c9c8c2")
    fig.suptitle(f"E6: accuracy vs coverage and novelty ({dataset}, {'locked test' if split == 'test' else 'validation dry run'})", color=ink, fontsize=11, x=0.01, ha="left")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=c, markeredgecolor=surf, markersize=8, label=l) for c, l in
                        ((role_col["random"], "random reference"), (role_col["popularity"], "tuned popularity"), (other, "personalised / content models"))],
               loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=ink2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94)); (ROOT / "reports" / "figures").mkdir(exist_ok=True)
    p = ROOT / "reports" / "figures" / f"e6_{split}_{dataset}.png"; fig.savefig(p, dpi=150, facecolor=surf); plt.close(fig); return p

def finalize(dataset, split):
    res, cold = load(dataset, split); title = "LOCKED TEST" if split == "test" else "VALIDATION DRY RUN (fit on train)"
    pop = res["popularity"]["per_user"]["NDCG@10"].values
    for n in res: assert (res[n]["per_user"].u.values == res["popularity"]["per_user"].u.values).all()
    md = [f"# {title}: {dataset} (from src/final_eval.py + src/finalize.py)\n",
          f"Settings frozen in manifests/frozen_settings.json. Warm users with history in the fit data: {int(res['popularity']['summary']['users']):,}. Fit data = " + ("train + validation; scored on the locked test." if split == "test" else "train; scored on validation.") + "\n"]
    t3 = pd.DataFrame([{"Model": LABEL[n], "P@10": r["summary"]["P@10"], "R@10": r["summary"]["R@10"], "NDCG@10": r["summary"]["NDCG@10"], "MAP@10": r["summary"]["MAP@10"],
                        "Coverage": r["summary"]["coverage@10"], "Diversity": r["summary"]["diversity"], "Novelty": r["summary"]["novelty"]} for n, r in res.items()])
    md += ["## Table III: ranking results, K = 10 (E1)\n", md_table(t3.assign(Diversity=t3.Diversity.map(lambda v: "n/a (no item text)" if pd.isna(v) else f"{v:.5f}"))), ""]
    others = [n for n in res if n != "popularity"]; m = len(others); rows = []
    for n in others:
        diff, lo, hi, p, padj = paired_bootstrap(res[n]["per_user"]["NDCG@10"].values, pop, m)
        rows.append({"Model vs tuned popularity": LABEL[n], "mean NDCG@10 difference": diff, "95% CI low": lo, "95% CI high": hi, "p": (f"<{1/B:.3f}" if p <= 1 / B else f"{p:.3f}"), f"p (Bonferroni, x{m})": (f"<{min(1, m / B):.3f}" if p <= 1 / B else f"{padj:.3f}"), "reading": verdict(diff, lo, hi, padj)})
    md += [f"## Paired bootstrap on per-user NDCG@10 ({B} resamples of users; {m} comparisons made; Bonferroni-adjusted)\n",
           "A model counts as an improvement only if its 95% interval excludes zero AND the Bonferroni-adjusted p < 0.05. With 1,000 resamples the smallest reportable p is below 1/1,000.\n", md_table(pd.DataFrame(rows), 6), ""]
    e5 = pd.DataFrame([{"Model": SHORT[n], **{f"NDCG@{k}": r["summary"][f"NDCG@{k}"] for k in (5, 10, 20)}, **{f"HR@{k}": r["summary"][f"HR@{k}"] for k in (5, 10, 20)}} for n, r in res.items()])
    rk = {k: list(e5.sort_values(f"NDCG@{k}", ascending=False).Model) for k in (5, 10, 20)}
    md += ["## E5: list-length sweep (K = 5, 10, 20)\n", md_table(e5), "",
           "Model order by NDCG@5 / @10 / @20: " + " | ".join(" > ".join(rk[k]) for k in (5, 10, 20)) + (". The order is identical at every K.\n" if rk[5] == rk[10] == rk[20] else ". The order changes with K.\n")]
    c = Ctx(dataset, split); fit, ev = c.fit, c.ev
    se_g = (ev.rating.values - fit.rating.mean()) ** 2; ae_g = np.abs(ev.rating.values - fit.rating.mean())
    pb = bias_predict(fit, ev, FROZEN[dataset]["bias_baseline"]["lambda"]); err = {"Global mean rating": ev.rating.values - fit.rating.mean(), "User/item bias baseline": ev.rating.values - pb}
    if "svd" in res: err["SVD"] = res["svd"]["rating_err"]
    rng = np.random.default_rng(SEED); n_r = len(ev); idxs = [rng.integers(0, n_r, n_r) for _ in range(B)]
    base_boot = np.array([np.sqrt(se_g[i].mean()) for i in idxs]); rows = []
    for name, e in err.items():
        rmse = float(np.sqrt((e ** 2).mean())); row = {"Model": name, "RMSE": rmse, "MAE": float(np.abs(e).mean())}
        if name == "Global mean rating": row.update({"RMSE change vs global mean": "n/a", "Bootstrap 95% CI": "n/a"})
        else:
            boot = np.array([np.sqrt((e[i] ** 2).mean()) for i in idxs]) - base_boot; lo, hi = np.percentile(boot, [2.5, 97.5])
            row.update({"RMSE change vs global mean": f"{rmse - float(np.sqrt(se_g.mean())):+.4f}", "Bootstrap 95% CI": f"[{lo:.4f}, {hi:.4f}]"})
        rows.append(row)
    md += [f"## Table IV: rating-prediction error on {'the locked test' if split == 'test' else 'validation'} ({n_r:,} rows)\n", md_table(pd.DataFrame(rows), 4), ""]
    if cold is not None:
        md += ["## E2: cold slices\n", f"Rows by slice relative to the fit data: {cold['row_slices']}.\n"]
        if "random" in cold:
            a, b = cold["popularity"]["per_user"]["NDCG@10"].values, cold["random"]["per_user"]["NDCG@10"].values
            diff, lo, hi, p, padj = paired_bootstrap(a, b, 1)
            md += [f"Cold users (no history; {cold['n_cold_users_total']:,} total, sample of {len(cold['users']):,}); personalised models cannot score them and fall back to popularity.\n",
                   md_table(pd.DataFrame([{"Model": "Random", "NDCG@10": b.mean(), "HR@10": cold['random']['summary']['HR@10']}, {"Model": "Popularity (tuned)", "NDCG@10": a.mean(), "HR@10": cold['popularity']['summary']['HR@10']}])),
                   f"\nPopularity minus random on cold users: {diff:.6f}, 95% CI [{lo:.6f}, {hi:.6f}], p = {p:.3f}.\n"]
        else: md += ["No cold users in this dataset/split.\n"]
    if split == "test":
        vres, _ = load(dataset, "val")
        if vres:
            rows = []
            for n, r in res.items():
                if n in vres: rows.append({"Model": SHORT[n], "validation NDCG@10": vres[n]["summary"]["NDCG@10"], "locked-test NDCG@10": r["summary"]["NDCG@10"],
                                           "change": r["summary"]["NDCG@10"] - vres[n]["summary"]["NDCG@10"], "validation HR@10": vres[n]["summary"]["HR@10"], "test HR@10": r["summary"]["HR@10"]})
            e7 = pd.DataFrame(rows); rv = list(e7.sort_values("validation NDCG@10", ascending=False).Model); rt = list(e7.sort_values("locked-test NDCG@10", ascending=False).Model)
            md += ["## E7: validation vs locked test\n", "Validation = fit on train, scored on validation (dry run). Test = fit on train + validation, scored on the locked test.\n", md_table(e7), "",
                   "Order on validation: " + " > ".join(rv) + "\n\nOrder on the locked test: " + " > ".join(rt) + (". The order is unchanged.\n" if rv == rt else ". The order changed.\n")]
    fig = figure(res, dataset, split); md += [f"## E6: accuracy vs coverage and novelty\n", f"![E6 scatter]({fig.relative_to(ROOT / 'reports').as_posix()})\n"]
    (ROOT / "reports" / f"final_{split}_{dataset}.md").write_text("\n".join(md) + "\n"); print("\n".join(md))
