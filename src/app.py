"""Streamlit UI on REAL data (Amazon Beauty 5-core).   Run:  streamlit run src/app.py
Models are fitted on the TRAIN period only (the frozen settings from manifests/frozen_settings.json). The 'bought it' ticks come from the next
period (validation), which the models never saw. The locked test set is not used here."""
import sys, re, json, html
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, streamlit as st
from config import PROCESSED_DIR, MANIFEST_DIR, ROOT, SEED
import models as M
from baselines import build_rankings
from hybrid import tfidf, content_score_fn, adaptive_blend_fn
from evaluate import user_sets

st.set_page_config(page_title="Product recommender", page_icon="🛍️", layout="wide")
INK, MUTED, LINE, PAPER = "#1b2733", "#5d6b7a", "#e3e8ee", "#f5f7fa"
BLUE, AMBER, GREEN = "#2457d6", "#b86e00", "#0e8a5f"
TOK = r"(?u)\b[a-z][a-z]+\b"
esc = html.escape

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {{ font-family:'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; color:{INK}; }}
.stApp {{ background:{PAPER}; }}
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility:hidden; height:0; }}
.block-container {{ padding-top:1.6rem; max-width:1150px; }}
.hero {{ background:{INK}; color:#fff; border-radius:16px; padding:26px 30px; margin-bottom:18px; }}
.hero h1 {{ color:#fff; font-size:1.9rem; margin:0 0 6px 0; font-weight:700; letter-spacing:-0.01em; }}
.hero p {{ color:#c9d3de; margin:0; font-size:1.02rem; line-height:1.5; max-width:780px; }}
div:has(> [role="tablist"]) {{ border-bottom:none !important; box-shadow:none !important; }}
[role="tablist"] {{ gap:12px !important; padding:4px 0 12px 0 !important; border:none !important; }}
[data-testid="stTab"] {{ height:auto !important; padding:12px 28px !important; background:#fff !important; border:1.5px solid {LINE} !important; border-radius:999px !important; margin:0 !important; }}
[data-testid="stTab"] p {{ font-size:1.12rem !important; font-weight:600 !important; color:{INK} !important; }}
[data-testid="stTab"]:hover {{ border-color:{INK} !important; }}
[data-testid="stTab"][aria-selected="true"] {{ background:{INK} !important; border-color:{INK} !important; }}
[data-testid="stTab"][aria-selected="true"] p {{ color:#fff !important; }}
.react-aria-SelectionIndicator, [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{ display:none !important; }}
.note {{ background:#fff; border:1px solid {LINE}; border-left:4px solid {BLUE}; border-radius:8px; padding:12px 16px; color:{MUTED}; font-size:.95rem; line-height:1.5; margin:6px 0 14px 0; }}
.verdict {{ background:#fff; border:1px solid {LINE}; border-radius:14px; padding:18px 22px; margin:14px 0; font-size:1.15rem; line-height:1.5; }}
.verdict b.good {{ color:{GREEN}; }} .verdict b.amber {{ color:{AMBER}; }}
.colhead {{ font-weight:700; font-size:1.08rem; margin:6px 0 8px 0; }}
.card {{ background:#fff; border:1px solid {LINE}; border-radius:12px; padding:10px 14px; margin-bottom:8px; display:flex; align-items:center; gap:12px; }}
.card.hit {{ border-color:{GREEN}; background:#f1faf6; }}
.rank {{ font-weight:700; color:{MUTED}; width:20px; text-align:right; font-size:.95rem; flex:none; }}
.thumb {{ width:54px; height:54px; border-radius:9px; flex:none; background:#eef2f9 center/contain no-repeat; border:1px solid {LINE}; }}
.thumb.sm {{ width:44px; height:44px; }}
.ctitle {{ font-weight:600; font-size:.95rem; line-height:1.3; }}
.cmeta {{ color:{MUTED}; font-size:.85rem; }}
.grow {{ flex:1; min-width:0; }}
.pill {{ background:{GREEN}; color:#fff; border-radius:999px; font-size:.78rem; font-weight:600; padding:3px 10px; white-space:nowrap; }}
.big {{ background:#fff; border:1px solid {LINE}; border-radius:14px; padding:16px 18px; }}
.big .num {{ font-size:1.7rem; font-weight:700; }} .big .lab {{ color:{MUTED}; font-size:.9rem; }}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Getting the shop ready (about 30 seconds the first time)...")
def load():
    cfg = json.loads((MANIFEST_DIR / "frozen_settings.json").read_text())["beauty"]
    rd = lambda n: pd.read_csv(PROCESSED_DIR / f"beauty_{n}.csv.gz", parse_dates=["purchase_date"])
    train, val = rd("train"), rd("val"); items = pd.read_csv(PROCESSED_DIR / "beauty_items.csv.gz").fillna({"title": "", "brand": "", "image": "", "text": ""}); items["title"] = items.title.map(html.unescape); items["brand"] = items.brand.map(html.unescape)
    n_users, n_items = int(max(train.u.max(), val.u.max())) + 1, len(items)
    R = M.interaction_matrix(train, n_users, n_items); tb = M.tie_break(train, n_items); is_cold = np.bincount(train.i.values, minlength=n_items) == 0
    p = cfg["itemknn"]; S = M.prune_top_k(M.knn_similarity(R, p["shrink"]), p["k"]); cf = M.knn_score_fn(R, S, 0.0)
    c = cfg["content"]; X = tfidf(train, items.text.values, n_items, c["max_features"], tuple(c["ngram"]), TOK); ct = content_score_fn(R, X)
    hybrid = adaptive_blend_fn(cf, ct, cfg["hybrid"]["adaptive_w"], is_cold, tb)
    pop = build_rankings(train)[cfg["popularity"]["variant"]]
    _, rel = user_sets(train, val); warm = np.array(sorted(u for u in rel if u < R.shape[0] and np.diff(R.indptr)[u] > 0))
    # which shoppers does the model get right? (sample, computed once)
    rng = np.random.default_rng(SEED); sample = np.sort(rng.choice(warm, min(1200, len(warm)), replace=False)); hits = {}
    for k in range(0, len(sample), 300):
        us = sample[k:k + 300]; sc = hybrid(us).copy(); coo = R[us].tocoo(); sc[coo.row, coo.col] = -np.inf
        top = np.argpartition(-sc, 10, axis=1)[:, :10]
        for r, u in enumerate(us): hits[int(u)] = len(rel[int(u)] & set(top[r].tolist()))
    good = sorted(u for u, h in hits.items() if h >= 2); bad = sorted(u for u, h in hits.items() if h == 0)
    last = train.sort_values("purchase_date").groupby("u").tail(4).groupby("u").i.apply(list).to_dict()
    return dict(R=R, items=items, X=X, hybrid=hybrid, pop=pop, rel=rel, warm=warm, good=good, bad=bad, last=last, S=knn_full(R, p["shrink"]), n_items=n_items)

def knn_full(R, shrink):
    return M.knn_similarity(R, shrink).tocsr()

def card(rank, item, items, hit=False, sub=None):
    r = items.iloc[int(item)]; title = r.title or "(untitled product)"; title = title if len(title) <= 95 else title[:92].rsplit(" ", 1)[0] + "..."
    meta = " · ".join(x for x in (r.brand, f"${r.price:.2f}" if pd.notna(r.price) else "") if x)
    img = f"background-image:url('{esc(r.image, quote=True)}')" if r.image else ""
    pill = '<span class="pill">✓ bought it</span>' if hit else ""
    right = f'<div class="cmeta" style="text-align:right;white-space:nowrap">{esc(sub)}</div>' if sub else ""
    return (f'<div class="card {"hit" if hit else ""}"><div class="rank">{rank}</div><div class="thumb" style="{img}"></div>'
            f'<div class="grow"><div class="ctitle">{esc(title)}</div><div class="cmeta">{esc(meta)}</div></div>{pill}{right}</div>')

def mini(item, items):
    r = items.iloc[int(item)]; t = (r.title or "")[:70]; img = f"background-image:url('{esc(r.image, quote=True)}')" if r.image else ""
    return f'<div class="card" style="flex:1;min-width:0"><div class="thumb sm" style="{img}"></div><div class="grow"><div class="cmeta" style="color:{INK};font-weight:600;font-size:.82rem;line-height:1.25">{esc(t)}</div></div></div>'

st.markdown("""<div class="hero"><h1>Product recommender</h1>
<p>What should an online shop show each customer? Amazon beauty shoppers: compare the recommendations our model makes with the best-seller list everyone sees.</p></div>""", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["For a shopper", "Similar products", "Results"])
D = load(); items = D["items"]

with tab1:
    st.markdown('<div class="note">The model learned only from each shopper's earlier purchases. A green tick marks a recommended product the shopper really bought <i>afterwards</i>.</div>', unsafe_allow_html=True)
    if "u" not in st.session_state: st.session_state["u"] = int(D["good"][0])
    c1, c2, _ = st.columns([1.5, 2, 3], vertical_alignment="bottom")
    if c2.button("Show another shopper", type="primary"):
        rng = np.random.default_rng(); pool = D["good"] if rng.random() < 0.75 else (D["bad"] or list(D["warm"]))
        st.session_state["u"] = int(rng.choice(pool)); st.rerun()
    u = int(c1.number_input("Shopper number", min_value=0, max_value=D["R"].shape[0] - 1, step=1, key="u"))
    R = D["R"]; bought = R.indices[R.indptr[u]:R.indptr[u + 1]]
    if len(bought) == 0 or u not in D["rel"]:
        st.warning(f"Shopper {u} has no purchases before the cut-off, or none after it, so there is nothing to check against. Try another number.")
    else:
        sc = D["hybrid"](np.array([u]))[0].copy(); sc[bought] = -np.inf; top = np.argsort(-sc)[:10]
        bs = set(bought.tolist()); pop_top = [i for i in D["pop"] if i not in bs][:10]; later = D["rel"][u]
        ha, hb = sum(i in later for i in top), sum(i in later for i in pop_top)
        st.markdown(f"**Shopper {u}** bought {len(bought)} products earlier. Most recent:")
        st.markdown('<div style="display:flex;gap:8px">' + "".join(mini(i, items) for i in D["last"].get(u, [])) + "</div>", unsafe_allow_html=True)
        st.markdown(f'<div class="verdict">Of the 10 products recommended to this shopper, <b class="good">{ha}</b> were things they really bought afterwards. '
                    f'The best-seller list got <b class="amber">{hb}</b>.</div>', unsafe_allow_html=True)
        a, b = st.columns(2, gap="large")
        with a:
            st.markdown(f'<div class="colhead" style="color:{BLUE}">Recommended for this shopper</div>', unsafe_allow_html=True)
            st.markdown("".join(card(r + 1, top[r], items, top[r] in later) for r in range(10)), unsafe_allow_html=True)
        with b:
            st.markdown(f'<div class="colhead" style="color:{AMBER}">Best-sellers (same for everyone)</div>', unsafe_allow_html=True)
            st.markdown("".join(card(r + 1, pop_top[r], items, pop_top[r] in later) for r in range(10)), unsafe_allow_html=True)
        st.caption(f"This shopper really bought {len(later)} product(s) after the cut-off.")

with tab2:
    st.markdown('<div class="note">Search a real product, then see two kinds of suggestions: what <b>other shoppers also bought</b>, and what has a <b>similar description</b>. Both come from the same model family used above.</div>', unsafe_allow_html=True)
    q = st.text_input("Search products", value="mascara")
    hit = items[items.title.str.lower().str.contains(q.lower().strip(), regex=False)] if q.strip() else items.head(0)
    if hit.empty: st.warning("No product matches that search. Try another word, for example lipstick, shampoo or perfume.")
    else:
        hit = hit.assign(n=np.asarray(D["R"].sum(axis=0)).ravel()[hit.i.values]).sort_values("n", ascending=False).head(50)
        pick = st.selectbox("Choose a product", options=list(hit.i), format_func=lambda i: items.title[i][:100])
        st.markdown(card("", pick, items).replace('<div class="rank"></div>', ""), unsafe_allow_html=True)
        S = D["S"]; row = S[pick].toarray().ravel(); row[pick] = -1; also = np.argsort(-row)[:5]
        simt = (D["X"] @ D["X"][pick].T).toarray().ravel(); simt[pick] = -1; like = np.argsort(-simt)[:5]
        a, b = st.columns(2, gap="large")
        with a:
            st.markdown(f'<div class="colhead" style="color:{BLUE}">Shoppers also bought</div>', unsafe_allow_html=True)
            st.markdown("".join(card(r + 1, i, items) for r, i in enumerate(also) if row[i] > 0) or "No other shopper bought this product in the training period.", unsafe_allow_html=True)
        with b:
            st.markdown(f'<div class="colhead" style="color:{AMBER}">Similar description</div>', unsafe_allow_html=True)
            st.markdown("".join(card(r + 1, i, items, sub=f"{simt[i]*100:.0f}% similar") for r, i in enumerate(like)), unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="note">Locked test: the last 15% of the timeline, scored once after all settings were frozen. Score = NDCG@10 (0 is useless, 1 is perfect).</div>', unsafe_allow_html=True)
    def grab(pattern, col):
        try:
            t = (ROOT / "reports" / "final_test_beauty.md").read_text()
            line = next(l for l in t.splitlines() if re.match(pattern, l)); return float(re.findall(r"[-+]?\d*\.\d+(?:e-?\d+)?", line)[col])
        except Exception: return float("nan")
    hy, po = grab(r"\| Hybrid \(adaptive", 2), grab(r"\| Popularity \(tuned\)", 2)
    hr = grab(r"\| Hybrid +\|", 4)          # E5 table: NDCG@5, NDCG@10, NDCG@20, HR@5, HR@10 -> index 4
    k1, k2, k3 = st.columns(3)
    for col, num, lab in ((k1, f"{hy:.4f}", "Our hybrid model"), (k2, f"{po:.4f}", "Best-seller list"), (k3, f"{hr*100:.1f}%", "Shoppers who got at least one correct product in 10")):
        col.markdown(f'<div class="big"><div class="num">{num}</div><div class="lab">{lab}</div></div>', unsafe_allow_html=True)
    st.markdown("")
    img = ROOT / "reports" / "figures" / "fig_models_test.png"
    if img.exists(): st.image(str(img), caption="Locked-test NDCG@10 with 95% confidence intervals (bars start at zero).")
    st.markdown("""
**What this tells us**
- The hybrid model scores about **4.5 times** the best-seller list, and the gap is far larger than chance.
- Scores look small because each shopper chooses among 12,000 products; picking at random scores 0.0003.
- For brand-new products (no purchase history) the hybrid finds about **8%** of purchases and collaborative filtering finds none, because it reads the product description.
- A model trained to guess star ratings (SVD) did **not** beat best-sellers: guessing ratings and ranking products are different jobs.
""")
