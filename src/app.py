"""Streamlit UI for the recommendation demo.   Run:  streamlit run src/app.py
Tab 1 uses the SIMULATED control data (the real 1M file has no learnable pattern). Tab 2 uses REAL Amazon product text.
Models are fitted on the train period only; the locked test is never used."""
import sys, json, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, streamlit as st
from config import PROCESSED_DIR, MANIFEST_DIR
import models as M
from baselines import build_rankings
from demo import group_of

st.set_page_config(page_title="Recommender demo", layout="wide")

@st.cache_resource(show_spinner="Fitting item-kNN on the training period...")
def load_control():
    cfg = json.loads((MANIFEST_DIR / "frozen_settings.json").read_text())["control"]
    rd = lambda n: pd.read_csv(PROCESSED_DIR / f"control_{n}.csv.gz", parse_dates=["purchase_date"])
    train, val = rd("train"), rd("val"); texts = pd.read_csv(PROCESSED_DIR / "control_items.csv.gz").text.values
    grp = np.array([group_of(t) for t in texts]); n_items = len(texts); n_users = int(max(train.u.max(), val.u.max())) + 1
    R = M.interaction_matrix(train, n_users, n_items); tb = M.tie_break(train, n_items)
    p = cfg["itemknn"]; score = M.knn_score_fn(R, M.prune_top_k(M.knn_similarity(R, p["shrink"]), p["k"]), tb)
    pop = build_rankings(train)[cfg["popularity"]["variant"]]
    return R, val, grp, score, pop

@st.cache_resource(show_spinner="Building the text model on real Amazon products...")
def load_products():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from content_amazon import build
    p = build(["title", "about", "desc", "reviews"])
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True, max_features=5000, ngram_range=(1, 2)).fit_transform(p.text)
    return p, (X @ X.T).toarray()

st.title("E-commerce recommender demo")
st.caption("Leakage-aware study of collaborative, content-based and hybrid recommenders. Fitted on the training period only.")
tab1, tab2 = st.tabs(["Recommend for a user (simulated)", "Similar products (real Amazon)"])

with tab1:
    st.info("This tab uses a SIMULATED shopper dataset with planted tastes. The real 1M purchase file contains no learnable pattern "
            "(see the report), so it cannot demonstrate personalisation. This shows how the method works, not real shoppers.")
    R, val, grp, score, pop = load_control()
    warm = np.flatnonzero(np.diff(R.indptr) > 0)
    c1, c2 = st.columns([3, 1])
    uid = c1.number_input("User number (CU...)", min_value=0, max_value=R.shape[0] - 1, value=int(warm[42]), step=1)
    if c2.button("Random user"):
        uid = int(np.random.default_rng().choice(warm)); st.session_state["u"] = uid
    u = int(st.session_state.pop("u", uid))
    bought = R.indices[R.indptr[u]:R.indptr[u + 1]]
    if len(bought) == 0:
        st.warning(f"CU{u} has no purchases in the training period (a cold user). Only the popularity list could serve them.")
    else:
        sc = score(np.array([u]))[0].copy(); sc[bought] = -np.inf; top = np.argsort(-sc)[:10]
        later = set(val[val.u == u].i); bs = set(bought); pop_top = [i for i in pop if i not in bs][:10]
        mix = collections.Counter(grp[bought]); tot = sum(mix.values())
        st.subheader(f"User CU{u}: {len(bought)} purchases in training")
        st.bar_chart(pd.Series({f"group {g}": c / tot for g, c in sorted(mix.items())}, name="share of purchases"), height=180)
        frame = lambda ids: pd.DataFrame({"rank": range(1, 11), "item": [f"CP{i}" for i in ids], "style group": grp[ids],
                                          "bought later?": ["✓ yes" if i in later else "" for i in ids]}).set_index("rank")
        a, b = st.columns(2)
        a.markdown("**Top 10 for this user (item-kNN)**"); a.dataframe(frame(top), width="stretch")
        b.markdown("**Same list for everyone (tuned popularity)**"); b.dataframe(frame(np.array(pop_top)), width="stretch")
        st.metric("Hits in the user's next period (top 10)", f"personalised {sum(i in later for i in top)}  |  popularity {sum(i in later for i in pop_top)}",
                  help=f"The user really bought {len(later)} items in the validation period.")

with tab2:
    p, sim = load_products()
    st.write("Pick a real Amazon product; the model finds products with the most similar text (title, description, reviews).")
    choice = st.selectbox("Product", options=list(p.index), format_func=lambda i: p.title[i][:100], index=0)
    n = st.slider("How many to show", 3, 10, 5)
    s = sim[choice].copy(); s[choice] = -1; top = np.argsort(-s)[:n]
    price = lambda i: f"${p.price_value[i]:.2f}" if pd.notna(p.price_value[i]) else "n/a"
    st.markdown(f"**You looked at:** {p.title[choice][:120]}  \n*{p.leaf[choice]}, {price(choice)}*")
    st.dataframe(pd.DataFrame({"product": [p.title[i][:100] for i in top], "category": [p.leaf[i] for i in top],
                               "price": [price(i) for i in top], "similarity": [round(float(s[i]), 2) for i in top]}), width="stretch", hide_index=True)
    st.caption("Held-out check on this model: NDCG@10 0.567 vs 0.038 for random, using 'same leaf category' as the proxy for relevant.")
