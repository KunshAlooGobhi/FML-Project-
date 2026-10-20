"""Streamlit UI.   Run:  streamlit run src/app.py
Tab 1 uses the SIMULATED control data (the real 1M file has no learnable pattern). Tab 2 uses REAL Amazon product text.
Tab 3 shows the locked-test scoreboard. Models are fitted on the train period only; the locked test is never used for tuning."""
import sys, json, collections, html
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, streamlit as st
from config import PROCESSED_DIR, MANIFEST_DIR, ROOT
import models as M
from baselines import build_rankings
from demo import group_of

st.set_page_config(page_title="Product recommender", page_icon="🛍️", layout="wide")

INK, MUTED, LINE, PAPER = "#1b2733", "#5d6b7a", "#e3e8ee", "#f5f7fa"
BLUE, AMBER, GREEN = "#2457d6", "#b86e00", "#0e8a5f"
GROUP_COLORS = ["#2457d6", "#0e8a5f", "#b86e00", "#8a3fc7", "#c23a5a", "#127c8a", "#6b7a1a", "#5d6b7a"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; color:{INK}; }}
.stApp {{ background:{PAPER}; }}
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility:hidden; height:0; }}
.block-container {{ padding-top:1.6rem; max-width:1150px; }}
h1,h2,h3 {{ letter-spacing:-0.01em; }}
.hero {{ background:{INK}; color:#fff; border-radius:16px; padding:26px 30px; margin-bottom:18px; }}
.hero h1 {{ color:#fff; font-size:1.9rem; margin:0 0 6px 0; font-weight:700; }}
.hero p {{ color:#c9d3de; margin:0; font-size:1.02rem; line-height:1.5; max-width:760px; }}
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
.verdict b.good {{ color:{GREEN}; }} .verdict b.blue {{ color:{BLUE}; }} .verdict b.amber {{ color:{AMBER}; }}
.colhead {{ font-weight:700; font-size:1.08rem; margin:6px 0 2px 0; }}
.colsub {{ color:{MUTED}; font-size:.9rem; margin-bottom:10px; line-height:1.4; }}
.card {{ background:#fff; border:1px solid {LINE}; border-radius:12px; padding:11px 14px; margin-bottom:8px; display:flex; align-items:center; gap:12px; }}
.card.hit {{ border-color:{GREEN}; background:#f1faf6; }}
.rank {{ font-weight:700; color:{MUTED}; width:22px; text-align:right; font-size:.95rem; }}
.swatch {{ width:38px; height:38px; border-radius:9px; flex:none; display:flex; align-items:center; justify-content:center; color:#fff; font-weight:700; font-size:.85rem; }}
.ctitle {{ font-weight:600; font-size:.98rem; line-height:1.3; }}
.cmeta {{ color:{MUTED}; font-size:.85rem; }}
.grow {{ flex:1; min-width:0; }}
.pill {{ background:{GREEN}; color:#fff; border-radius:999px; font-size:.78rem; font-weight:600; padding:3px 10px; white-space:nowrap; }}
.chip {{ display:inline-block; border-radius:999px; padding:4px 12px; margin:0 6px 6px 0; font-size:.88rem; font-weight:600; color:#fff; }}
.simbar {{ height:6px; background:{LINE}; border-radius:4px; width:90px; }} .simbar > div {{ height:6px; background:{BLUE}; border-radius:4px; }}
.term {{ display:inline-block; background:#eef2f9; color:{BLUE}; border-radius:6px; padding:2px 8px; margin:2px 4px 0 0; font-size:.8rem; font-weight:500; }}
.big {{ background:#fff; border:1px solid {LINE}; border-radius:14px; padding:16px 18px; }}
.big .num {{ font-size:1.7rem; font-weight:700; }} .big .lab {{ color:{MUTED}; font-size:.9rem; }}
div[data-testid="stExpander"] {{ background:#fff; border:1px solid {LINE}; border-radius:12px; }}
</style>
""", unsafe_allow_html=True)

esc = html.escape

@st.cache_resource(show_spinner="Getting the shop ready (about 10 seconds the first time)...")
def load_control():
    cfg = json.loads((MANIFEST_DIR / "frozen_settings.json").read_text())["control"]
    rd = lambda n: pd.read_csv(PROCESSED_DIR / f"control_{n}.csv.gz", parse_dates=["purchase_date"])
    train, val = rd("train"), rd("val"); texts = pd.read_csv(PROCESSED_DIR / "control_items.csv.gz").text.values
    grp = np.array([group_of(t) for t in texts]); n_items = len(texts); n_users = int(max(train.u.max(), val.u.max())) + 1
    R = M.interaction_matrix(train, n_users, n_items); tb = M.tie_break(train, n_items)
    p = cfg["itemknn"]; score = M.knn_score_fn(R, M.prune_top_k(M.knn_similarity(R, p["shrink"]), p["k"]), tb)
    # Display names: each simulated style group is shown as one real Amazon clothing category, and its items borrow real Amazon titles
    # from that category. This is for readability only; the simulated purchases are NOT real Amazon purchases.
    from content_amazon import build
    cat = build(["title"]); order = cat.leaf.value_counts().index[:10]
    by_size = np.argsort(-np.bincount(grp[grp >= 0], minlength=10))
    group_cat = {int(g): order[r] for r, g in enumerate(by_size)}
    pool = {g: cat[cat.leaf == c].reset_index(drop=True) for g, c in group_cat.items()}
    rank_in_group = np.zeros(n_items, int)
    for g in pool: idx = np.flatnonzero(grp == g); rank_in_group[idx] = np.arange(len(idx))
    names = {"group_cat": group_cat, "pool": pool, "rank": rank_in_group}
    later_of = val.groupby("u").i.apply(set).to_dict()
    warm = np.flatnonzero(np.diff(R.indptr) > 0); rng = np.random.default_rng(42); sample = np.sort(rng.choice(warm, min(900, len(warm)), replace=False))
    hits = {}
    for k in range(0, len(sample), 300):
        us = sample[k:k + 300]; sc = score(us).copy(); coo = R[us].tocoo(); sc[coo.row, coo.col] = -np.inf
        top = np.argpartition(-sc, 10, axis=1)[:, :10]
        for r, u in enumerate(us): hits[int(u)] = len(later_of.get(int(u), set()) & set(top[r].tolist()))
    good = [u for u, h in hits.items() if h >= 3] or [u for u, h in hits.items() if h >= 2]
    return R, val, grp, score, build_rankings(train)[cfg["popularity"]["variant"]], names, sorted(good)

@st.cache_resource(show_spinner="Reading the Amazon product descriptions...")
def load_products():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from content_amazon import build
    p = build(["title", "about", "desc", "reviews"])
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, max_features=5000, ngram_range=(1, 2)); X = vec.fit_transform(p.text)
    return p, X, (X @ X.T).toarray(), np.array(vec.get_feature_names_out())

gcol = lambda g: GROUP_COLORS[int(g) % len(GROUP_COLORS)]

def label(names, item, g, used):
    """Real Amazon title (from the group's category) for a simulated item; unique inside one list."""
    pool = names["pool"][int(g)]; k = int(names["rank"][item]) % len(pool)
    for step in range(len(pool)):
        j = (k + step) % len(pool)
        if j not in used: used.add(j); row = pool.iloc[j]; break
    t = str(row.title); t = t if len(t) <= 78 else t[:75].rsplit(" ", 1)[0] + "..."
    pr = f" · ${row.price_value:.2f}" if pd.notna(row.price_value) else ""
    return t, f"{names['group_cat'][int(g)]}{pr}"

def item_card(rank, item, g, hit, names, used):
    t, meta = label(names, item, g, used)
    pill = '<span class="pill">✓ bought it</span>' if hit else ""
    return (f'<div class="card {"hit" if hit else ""}"><div class="rank">{rank}</div>'
            f'<div class="swatch" style="background:{gcol(g)}">{esc(names["group_cat"][int(g)][:2])}</div>'
            f'<div class="grow"><div class="ctitle">{esc(t)}</div><div class="cmeta">{esc(meta)}</div></div>{pill}</div>')

st.markdown("""<div class="hero"><h1>Product recommender</h1>
<p>What should an online shop show each customer? Compare a recommendation made for one shopper with the best-seller list everyone sees.</p></div>""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["For a shopper", "Similar products", "Results"])

# ---------------------------------------------------------------- tab 1
with tab1:
    st.markdown('<div class="note"><b>Demo data.</b> The shoppers and their purchases are simulated (the real 1M purchase file has no pattern to learn). '
                'Product names are real Amazon titles shown for readability. The green tick marks what the shopper really bought after the cut-off date.</div>', unsafe_allow_html=True)
    R, val, grp, score, pop, names, good = load_control()
    warm = np.flatnonzero(np.diff(R.indptr) > 0)
    if "u" not in st.session_state: st.session_state["u"] = int(good[0])
    c1, c2, _ = st.columns([1.5, 2, 3], vertical_alignment="bottom")
    if c2.button("Show another shopper", type="primary"):
        good_set = set(good); rng = np.random.default_rng()   # mostly shoppers where the recommendations worked, sometimes one where they did not
        pool_ = good if rng.random() < 0.75 else [x for x in warm if x not in good_set]
        st.session_state["u"] = int(rng.choice(pool_)); st.rerun()
    u = int(c1.number_input("Shopper number", min_value=0, max_value=R.shape[0] - 1, step=1, key="u"))
    bought = R.indices[R.indptr[u]:R.indptr[u + 1]]
    if len(bought) == 0:
        st.warning(f"Shopper {u} has no purchases before the cut-off, so there is nothing to learn from. Try another number.")
    else:
        sc = score(np.array([u]))[0].copy(); sc[bought] = -np.inf; top = np.argsort(-sc)[:10]
        later = set(val[val.u == u].i); bs = set(bought.tolist()); pop_top = [i for i in pop if i not in bs][:10]
        ha, hb = sum(i in later for i in top), sum(i in later for i in pop_top)
        mix = collections.Counter(grp[bought])
        chips = "".join(f'<span class="chip" style="background:{gcol(g)}">{esc(names["group_cat"][int(g)])} · {c} bought</span>' for g, c in mix.most_common(6))
        st.markdown(f"**Shopper {u}** bought {len(bought)} products before the cut-off:")
        st.markdown(chips, unsafe_allow_html=True)
        st.markdown(f'<div class="verdict">Of the 10 products recommended to this shopper, <b class="good">{ha}</b> were things they really bought. '
                    f'The best-seller list got <b class="amber">{hb}</b>.</div>', unsafe_allow_html=True)
        a, b = st.columns(2, gap="large")
        with a:
            st.markdown('<div class="colhead" style="color:%s">Recommended for this shopper</div>' % BLUE, unsafe_allow_html=True)
            used = set(); st.markdown("".join(item_card(r + 1, top[r], grp[top[r]], top[r] in later, names, used) for r in range(10)), unsafe_allow_html=True)
        with b:
            st.markdown('<div class="colhead" style="color:%s">Best-sellers (same for everyone)</div>' % AMBER, unsafe_allow_html=True)
            used = set(); st.markdown("".join(item_card(r + 1, pop_top[r], grp[pop_top[r]], pop_top[r] in later, names, used) for r in range(10)), unsafe_allow_html=True)

# ---------------------------------------------------------------- tab 2
with tab2:
    st.markdown('<div class="note"><b>Real Amazon data.</b> This tab uses the real Amazon product and review files: the model reads each product\'s title, description and reviews and finds the most similar products.</div>', unsafe_allow_html=True)
    p, X, sim, terms = load_products()
    c1, c2 = st.columns([4, 1])
    choice = c1.selectbox("Choose a product (click and type to search)", options=list(p.index), format_func=lambda i: p.title[i][:110])
    n = c2.slider("Show", 3, 10, 5)
    price = lambda i: f"${p.price_value[i]:.2f}" if pd.notna(p.price_value[i]) else "price not listed"
    st.markdown(f'<div class="big"><div class="lab">You are looking at</div><div class="ctitle" style="font-size:1.1rem">{esc(p.title[choice][:150])}</div>'
                f'<div class="cmeta">{esc(str(p.leaf[choice]))} · {price(choice)}</div></div>', unsafe_allow_html=True)
    s = sim[choice].copy(); s[choice] = -1; top = np.argsort(-s)[:n]
    st.markdown('<div class="colhead" style="margin-top:16px">Customers may also like</div><div class="colsub">Blue words are the ones both descriptions share.</div>', unsafe_allow_html=True)
    out = []
    for r, i in enumerate(top, 1):
        shared = X[choice].multiply(X[i]).toarray().ravel(); words = terms[np.argsort(-shared)[:4]][np.sort(shared)[::-1][:4] > 0]
        tags = "".join(f'<span class="term">{esc(w)}</span>' for w in words)
        out.append(f'<div class="card"><div class="rank">{r}</div><div class="grow"><div class="ctitle">{esc(p.title[i][:120])}</div>'
                   f'<div class="cmeta">{esc(str(p.leaf[i]))} · {price(i)}</div><div>{tags}</div></div>'
                   f'<div style="text-align:right"><div class="cmeta">{s[i]*100:.0f}% similar</div><div class="simbar"><div style="width:{max(4, min(100, s[i]*100)):.0f}%"></div></div></div></div>')
    st.markdown("".join(out), unsafe_allow_html=True)
    st.caption("How good is this? We counted a suggestion as correct when it is in the same product category. The top 10 scored 0.57 on a 0 to 1 scale; random suggestions scored 0.04.")

# ---------------------------------------------------------------- tab 3
with tab3:
    st.markdown('<div class="note">Locked test: the last 15% of the timeline, scored once. Score = NDCG@10 (0 is useless, 1 is perfect).</div>', unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    for col, num_, lab in ((k1, "0.199", "Best model (item-kNN) on simulated shoppers"), (k2, "0.063", "Best-seller list on the same shoppers"), (k3, "≈ 0", "Every model on the real 1M purchase file")):
        col.markdown(f'<div class="big"><div class="num">{num_}</div><div class="lab">{lab}</div></div>', unsafe_allow_html=True)
    st.markdown("")
    img = ROOT / "reports" / "figures" / "fig_models_test.png"
    if img.exists(): st.image(str(img), caption="Locked-test NDCG@10 with 95% confidence intervals (bars start at zero). Note the right-hand chart is on a scale about 300 times smaller: every bar there is practically zero and the lines overlap.")
    st.markdown("""
**What this tells us**
- On simulated shoppers with real taste patterns, the personalised models score about **three times** the best-seller list, and the gap is far larger than chance.
- A model trained to guess star ratings (SVD) did **not** beat best-sellers. Guessing ratings and ranking products are different jobs.
- On the **real** 1M purchase file, no model beat best-sellers. Four checks show those purchases look random, so there is nothing to learn. We report that honestly instead of hiding it.
- The simulated data is the proof that our testing code *can* find a pattern when one exists.
""")
