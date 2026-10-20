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
.stTabs [data-baseweb="tab-list"] {{ gap:6px; border-bottom:1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ font-weight:600; font-size:1rem; padding:10px 18px; color:{MUTED}; }}
.stTabs [aria-selected="true"] {{ color:{INK}; }}
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
    return R, val, grp, score, build_rankings(train)[cfg["popularity"]["variant"]]

@st.cache_resource(show_spinner="Reading the Amazon product descriptions...")
def load_products():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from content_amazon import build
    p = build(["title", "about", "desc", "reviews"])
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, max_features=5000, ngram_range=(1, 2)); X = vec.fit_transform(p.text)
    return p, X, (X @ X.T).toarray(), np.array(vec.get_feature_names_out())

gcol = lambda g: GROUP_COLORS[int(g) % len(GROUP_COLORS)]

def item_card(rank, item, g, hit):
    pill = '<span class="pill">✓ bought it later</span>' if hit else ""
    return (f'<div class="card {"hit" if hit else ""}"><div class="rank">{rank}</div>'
            f'<div class="swatch" style="background:{gcol(g)}">S{g}</div>'
            f'<div class="grow"><div class="ctitle">Product {item}</div><div class="cmeta">Style {g}</div></div>{pill}</div>')

st.markdown("""<div class="hero"><h1>Product recommender</h1>
<p>An online shop can show thousands of products, so it has to choose. This page lets you try the three kinds of
recommender we built and tested, and see in plain terms how well each one works.</p></div>""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["For a shopper", "Similar products", "How well does it work?"])

# ---------------------------------------------------------------- tab 1
with tab1:
    st.markdown(f'<div class="note"><b>Simulated shoppers.</b> This tab uses made-up customers whose tastes we planted ourselves, '
                f'because the real purchase file turned out to contain no pattern to learn (see the third tab). '
                f'It shows how the method works, not how real people shop.</div>', unsafe_allow_html=True)
    with st.expander("How does it work? (30 seconds)"):
        st.markdown("""
1. **Look at the past.** We only use what each shopper bought *before* a cut-off date.
2. **Find patterns.** Products that the same people tend to buy together count as "similar".
3. **Recommend.** For one shopper we pick the unbought products most similar to what they already bought.
4. **Check against the future.** We then look at what they really bought *after* the cut-off. A green tick means the model guessed right.

The comparison list is what a shop with no personalisation does: show the best-sellers to everyone.""")
    R, val, grp, score, pop = load_control()
    warm = np.flatnonzero(np.diff(R.indptr) > 0)
    c1, c2, _ = st.columns([2, 2, 3])
    num = c1.number_input("Shopper number", min_value=0, max_value=R.shape[0] - 1, value=int(st.session_state.get("u", warm[42])), step=1)
    c2.write(""); c2.write("")
    if c2.button("Pick a random shopper"):
        st.session_state["u"] = int(np.random.default_rng().choice(warm)); st.rerun()
    u = int(num)
    bought = R.indices[R.indptr[u]:R.indptr[u + 1]]
    if len(bought) == 0:
        st.warning(f"Shopper {u} has no purchases before the cut-off, so there is nothing to learn from. A real shop would show best-sellers to a new visitor. Try another number.")
    else:
        sc = score(np.array([u]))[0].copy(); sc[bought] = -np.inf; top = np.argsort(-sc)[:10]
        later = set(val[val.u == u].i); bs = set(bought.tolist()); pop_top = [i for i in pop if i not in bs][:10]
        ha, hb = sum(i in later for i in top), sum(i in later for i in pop_top)
        mix = collections.Counter(grp[bought])
        chips = "".join(f'<span class="chip" style="background:{gcol(g)}">Style {g} · {c} bought</span>' for g, c in mix.most_common(6))
        st.markdown(f"**Shopper {u} bought {len(bought)} products before the cut-off.** Their taste:")
        st.markdown(chips, unsafe_allow_html=True)
        better = ha > hb
        verdict = (f'Of the 10 products we recommended to this shopper, <b class="good">{ha}</b> were things they really went on to buy. '
                   f'The best-seller list got <b class="amber">{hb}</b>. ') + ("Personalising helped here." if better else ("Same result here; the model wins on average across many shoppers, not every single one." if ha == hb else "The best-sellers did better for this shopper; the model wins on average across many shoppers, not every single one."))
        st.markdown(f'<div class="verdict">{verdict}</div>', unsafe_allow_html=True)
        a, b = st.columns(2, gap="large")
        with a:
            st.markdown('<div class="colhead" style="color:%s">Recommended for this shopper</div><div class="colsub">Picked from what this person bought.</div>' % BLUE, unsafe_allow_html=True)
            st.markdown("".join(item_card(r + 1, top[r], grp[top[r]], top[r] in later) for r in range(10)), unsafe_allow_html=True)
        with b:
            st.markdown('<div class="colhead" style="color:%s">Best-sellers (shown to everyone)</div><div class="colsub">The same list for every shopper. This is the baseline we have to beat.</div>' % AMBER, unsafe_allow_html=True)
            st.markdown("".join(item_card(r + 1, pop_top[r], grp[pop_top[r]], pop_top[r] in later) for r in range(10)), unsafe_allow_html=True)
        st.caption(f"This shopper really bought {len(later)} products after the cut-off. Coloured squares are the product's style group (simulated).")

# ---------------------------------------------------------------- tab 2
with tab2:
    st.markdown('<div class="note"><b>Real Amazon products.</b> No shopper data here. The model reads each product\'s title, description and reviews '
                'and finds products described with similar words. This is how a shop fills the "you may also like" strip on a product page.</div>', unsafe_allow_html=True)
    p, X, sim, terms = load_products()
    c1, c2 = st.columns([4, 1])
    choice = c1.selectbox("Choose a product (click and type to search)", options=list(p.index), format_func=lambda i: p.title[i][:110])
    n = c2.slider("Show", 3, 10, 5)
    price = lambda i: f"${p.price_value[i]:.2f}" if pd.notna(p.price_value[i]) else "price not listed"
    st.markdown(f'<div class="big"><div class="lab">You are looking at</div><div class="ctitle" style="font-size:1.1rem">{esc(p.title[choice][:150])}</div>'
                f'<div class="cmeta">{esc(str(p.leaf[choice]))} · {price(choice)}</div></div>', unsafe_allow_html=True)
    s = sim[choice].copy(); s[choice] = -1; top = np.argsort(-s)[:n]
    st.markdown('<div class="colhead" style="margin-top:16px">Customers may also like</div><div class="colsub">Sorted by how similar the words are. The blue words are the ones the two descriptions share most.</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="note">We kept the last 15% of the timeline locked away and scored every model on it <b>once</b>, after all settings were frozen. '
                'The score is NDCG@10: it rewards putting the products a shopper really bought near the top of the 10 recommended. 0 is useless, 1 is perfect.</div>', unsafe_allow_html=True)
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
