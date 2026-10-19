"""Report figures (reference data-viz palette; every number comes from saved results, nothing typed in by hand)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from config import ROOT, SEED, PROCESSED_DIR
from finalize import load, SHORT, ORDER

SURF, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de", "#c9c8c2"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8c8b86"          # reference palette slots 1, 2 + neutral
FIG = ROOT / "reports" / "figures"; FIG.mkdir(exist_ok=True)

def style(ax):
    ax.set_facecolor(SURF); ax.grid(True, axis="x", color=GRID, linewidth=0.8, zorder=0); ax.tick_params(colors=INK2, labelsize=8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(AXIS)

def boot_ci(x, B=1000):
    rng = np.random.default_rng(SEED); m = x[rng.integers(0, len(x), (B, len(x)))].mean(1); return np.percentile(m, [2.5, 97.5])

def fig_models():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=SURF)
    for ax, (ds, title) in zip(axes, (("control", "Control (simulated, positive control)"), ("main", "amazon_ecommerce_1M (5,000 users)"))):
        res, _ = load(ds, "test"); names = sorted(res, key=lambda n: res[n]["summary"]["NDCG@10"])
        mean = [res[n]["summary"]["NDCG@10"] for n in names]; ci = [boot_ci(res[n]["per_user"]["NDCG@10"].values) for n in names]
        cols = [GREY if n == "random" else BLUE if n == "popularity" else ORANGE for n in names]
        y = np.arange(len(names)); style(ax)
        ax.barh(y, mean, color=cols, height=0.55, zorder=3, edgecolor=SURF, linewidth=2)
        ax.errorbar(mean, y, xerr=[[m - c[0] for m, c in zip(mean, ci)], [c[1] - m for m, c in zip(mean, ci)]], fmt="none", ecolor=INK, elinewidth=1.2, capsize=3, zorder=4)
        ax.set_yticks(y); ax.set_yticklabels([SHORT[n] for n in names], color=INK, fontsize=9); ax.set_xlim(left=0)
        ax.set_xlabel("NDCG@10 on the locked test (bars start at zero; whiskers = 95% bootstrap CI over users)", color=INK2, fontsize=8)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    fig.legend(handles=[Line2D([], [], marker="s", ls="", color=c, markersize=9, label=l) for c, l in ((GREY, "random reference"), (BLUE, "tuned popularity"), (ORANGE, "personalised / content models"))],
               loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle("Locked-test accuracy: real signal in the control data, none in the 1M file", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94)); fig.savefig(FIG / "fig_models_test.png", dpi=150, facecolor=SURF); plt.close(fig)

def fig_e3():
    d = pd.read_csv(ROOT / "reports" / "e3_training_size_control.md", sep="|", skiprows=lambda i: False, engine="python", on_bad_lines="skip") if False else None
    import re
    txt = (ROOT / "reports" / "e3_training_size_control.md").read_text(); rows = {}
    for line in txt.split("Detail:")[0].splitlines():
        m = re.match(r"\|\s*(ALS|Item-kNN|Popularity)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", line)
        if m: rows[m.group(1)] = [float(m.group(i)) for i in (2, 3, 4)]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=SURF); ax.set_facecolor(SURF); x = [25, 50, 100]
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0); ax.tick_params(colors=INK2, labelsize=8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(AXIS)
    for name, col in (("Popularity", BLUE), ("Item-kNN", ORANGE), ("ALS", "#1baf7a")):
        ax.plot(x, rows[name], color=col, linewidth=2, marker="o", markersize=7, markeredgecolor=SURF, markeredgewidth=2, zorder=3)
        ax.annotate(name, (100, rows[name][2]), xytext=(8, 0), textcoords="offset points", color=INK, fontsize=9, va="center")
    ax.set_xticks(x); ax.set_xticklabels(["25%", "50%", "100%"]); ax.set_xlim(20, 118)
    ax.set_xlabel("Share of training data used", color=INK2, fontsize=9); ax.set_ylabel("NDCG@10 (validation)", color=INK2, fontsize=9)
    ax.set_title("E3: item-kNN needs data; with 25% it falls below popularity", color=INK, fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(FIG / "fig_e3_training_size.png", dpi=150, facecolor=SURF); plt.close(fig)

def fig_signal():
    df = pd.read_csv(PROCESSED_DIR / "interactions_full.csv.gz", parse_dates=["purchase_date"])
    c = df.groupby("i").size(); d = df.sort_values(["u", "purchase_date"]); d["pc"], d["pb"] = d.groupby("u").category.shift(), d.groupby("u").brand.shift(); x = d.dropna(subset=["pc"])
    pc, pb = df.category.value_counts(normalize=True), df.brand.value_counts(normalize=True)
    cut = df.purchase_date.quantile(0.7); a = df[df.purchase_date < cut].groupby("i").size(); b = df[df.purchase_date >= cut].groupby("i").size()
    corr = pd.concat([a, b], axis=1).fillna(0).corr().iloc[0, 1]
    tests = [("Variance of purchases\nper product", c.var(), c.mean(), "{:.2f}"), ("User repeats previous\ncategory", (x.category == x.pc).mean(), (pc ** 2).sum(), "{:.3f}"),
             ("User repeats previous\nbrand", (x.brand == x.pb).mean(), (pb ** 2).sum(), "{:.3f}"), ("Correlation: early vs later\nproduct sales", corr, 0.0, "{:.3f}")]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.6), facecolor=SURF)
    for ax, (t, obs, exp, f) in zip(axes, tests):
        ax.set_facecolor(SURF); ax.bar([0, 1], [obs, exp], color=[ORANGE, GREY], width=0.55, edgecolor=SURF, linewidth=2, zorder=3)
        for xi, v in ((0, obs), (1, exp)): ax.annotate(f.format(v), (xi, v), xytext=(0, 4), textcoords="offset points", ha="center", color=INK, fontsize=9)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["observed", "pure chance"], color=INK2, fontsize=8); ax.set_title(t, color=INK, fontsize=9, loc="left")
        ax.tick_params(axis="y", colors=INK2, labelsize=8); ax.set_ylim(0, max(obs, exp, 0.01) * 1.25); ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color(AXIS)
    fig.suptitle("amazon_ecommerce_1M behaves like random purchases: observed equals chance on all four tests", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(FIG / "fig_signal_diagnostics.png", dpi=150, facecolor=SURF); plt.close(fig)

if __name__ == "__main__":
    fig_models(); fig_e3(); fig_signal(); print("figures written:", sorted(p.name for p in FIG.glob("fig_*.png")))
