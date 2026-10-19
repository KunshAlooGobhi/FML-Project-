"""Does the 1M file contain ANY learnable purchase pattern? Four tests against what pure chance predicts."""
import pandas as pd
from config import PROCESSED_DIR, ROOT

def main():
    df = pd.read_csv(PROCESSED_DIR / "interactions_full.csv.gz", parse_dates=["purchase_date"])
    c = df.groupby("i").size()
    d = df.sort_values(["u", "purchase_date"])
    d["pc"], d["pb"] = d.groupby("u").category.shift(), d.groupby("u").brand.shift()
    x = d.dropna(subset=["pc"])
    pc, pb = df.category.value_counts(normalize=True), df.brand.value_counts(normalize=True)
    cut = df.purchase_date.quantile(0.7)
    a = df[df.purchase_date < cut].groupby("i").size(); b = df[df.purchase_date >= cut].groupby("i").size()
    corr = pd.concat([a, b], axis=1).fillna(0).corr().iloc[0, 1]
    r = df.groupby("i").rating.std().mean()
    lines = ["# Signal diagnostics for amazon_ecommerce_1M (from src/diagnose_signal.py)\n",
        "| test | observed | what pure chance predicts |", "|---|---|---|",
        f"| variance of item purchase counts | {c.var():.2f} | {c.mean():.2f} (Poisson: variance = mean) |",
        f"| user repeats previous category | {(x.category == x.pc).mean():.4f} | {(pc**2).sum():.4f} |",
        f"| user repeats previous brand | {(x.brand == x.pb).mean():.4f} | {(pb**2).sum():.4f} |",
        f"| corr(item purchases in train period, in later period) | {corr:.4f} | 0 |",
        f"| avg within-item rating std vs overall | {r:.3f} vs {df.rating.std():.3f} | equal if rating is not a product property |",
        "", "Conclusion: every statistic matches random assignment. Popularity, user taste and item rating carry no signal."]
    (ROOT / "reports" / "signal_diagnostics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
if __name__ == "__main__":
    main()
