"""Table I in the exact layout of the partial report (density, mean rating, share >= 4, medians)."""
import pandas as pd
from config import PROCESSED_DIR, ROOT
rows = {}
for name, f in (("Full data", "interactions_full"), ("k-core (k = 2)", "interactions_k2")):
    d = pd.read_csv(PROCESSED_DIR / f"{f}.csv.gz")
    u, i, n = d.user_id.nunique(), d.product_id.nunique(), len(d)
    rows[name] = {
        "Users / items / interactions": f"{u:,} / {i:,} / {n:,}",
        "Density (interactions / (users x items))": f"{n/(u*i):.2e}",
        "Mean rating / share of ratings of 4 or more": f"{d.rating.mean():.3f} / {(d.rating>=4).mean()*100:.1f}%",
        "Median interactions per user / per item": f"{d.groupby('user_id').size().median():.0f} / {d.groupby('product_id').size().median():.0f}",
    }
t = pd.DataFrame(rows)
md = "# Table I in the partial-report layout (from src/table1_full.py)\n\n" + t.reset_index().rename(columns={"index": "Quantity"}).to_markdown(index=False) + "\n"
(ROOT / "reports" / "table1_report_layout.md").write_text(md); print(md)
