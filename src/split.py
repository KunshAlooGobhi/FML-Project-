"""Phase 2 / Step 1: time-based 70/15/15 split of the k=2 interactions (never random).
Cut points are DATES, so no purchase in an earlier set is later than any purchase in a later set.
Test/validation rows are tagged warm / cold-item / cold-user against the TRAIN set only."""
import pandas as pd
from config import PROCESSED_DIR, ROOT, SPLIT

def tag(rows, train):
    """Label each held-out row by what the model could have learned from TRAIN alone."""
    user_seen, item_seen = rows.u.isin(set(train.u)), rows.i.isin(set(train.i))
    label = ["cold-user" if not u else ("cold-item" if not i else "warm")
             for u, i in zip(user_seen, item_seen)]
    return rows.assign(slice=label)

def main():
    df = pd.read_csv(PROCESSED_DIR / "interactions_k2.csv.gz", parse_dates=["purchase_date"])
    df = df.sort_values("purchase_date", kind="stable").reset_index(drop=True)
    t1 = df.purchase_date.iloc[int(len(df) * SPLIT[0])]
    t2 = df.purchase_date.iloc[int(len(df) * (SPLIT[0] + SPLIT[1]))]
    train = df[df.purchase_date < t1]
    val = df[(df.purchase_date >= t1) & (df.purchase_date < t2)]
    test = df[df.purchase_date >= t2]
    assert train.purchase_date.max() < val.purchase_date.min() <= val.purchase_date.max() < test.purchase_date.min()
    val, test = tag(val, train), tag(test, train)
    for name, d in (("train", train), ("val", val), ("test", test)):
        d.to_csv(PROCESSED_DIR / f"{name}.csv.gz", index=False)
    lines = ["# Split summary (time-based; from src/split.py)\n",
             "| set | rows | share | from | to |", "|---|---|---|---|---|"]
    for name, d in (("train", train), ("validation", val), ("test (locked)", test)):
        lines.append(f"| {name} | {len(d):,} | {len(d)/len(df):.1%} | {d.purchase_date.min().date()} | {d.purchase_date.max().date()} |")
    lines += ["", "Slices (relative to train only):", "", "| set | warm | cold-item | cold-user |", "|---|---|---|---|"]
    for name, d in (("validation", val), ("test", test)):
        c = d.slice.value_counts()
        lines.append(f"| {name} | {c.get('warm',0):,} ({c.get('warm',0)/len(d):.1%}) | {c.get('cold-item',0):,} ({c.get('cold-item',0)/len(d):.1%}) | {c.get('cold-user',0):,} ({c.get('cold-user',0)/len(d):.1%}) |")
    (ROOT / "reports" / "split_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
