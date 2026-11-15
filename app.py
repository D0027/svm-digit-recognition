"""
Handwritten Digit Recognition with SVM : interactive Streamlit app
Author: Deepak Yadav (D0027)
Dataset: https://archive.ics.uci.edu/dataset/80 (also bundled in scikit-learn)
"""
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageOps
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

st.set_page_config(page_title="SVM Digit Lab", page_icon="✍️", layout="wide")

ACCENT = "#7C5CFF"
CYAN = "#22D3EE"
PINK = "#F472B6"

# ------------------------------------------------------------------ styling
st.markdown(
    f"""
<style>
.block-container {{padding-top: 1.6rem; max-width: 1250px;}}
.hero {{
  background: linear-gradient(135deg, rgba(124,92,255,.35), rgba(34,211,238,.18));
  border: 1px solid rgba(124,92,255,.45); border-radius: 20px;
  padding: 28px 34px; margin-bottom: 18px;
}}
.hero h1 {{margin:0; font-size: 2.2rem;
  background: linear-gradient(90deg,#fff,{CYAN}); -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;}}
.hero p {{margin:6px 0 0; color:#B8C1DB; font-size:1.02rem;}}
.pill {{display:inline-block; padding:3px 12px; margin:10px 6px 0 0; border-radius:999px;
  background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.14);
  font-size:.78rem; color:#D5DBF0;}}
.card {{background:#141B2D; border:1px solid #243049; border-radius:16px; padding:18px 20px;}}
.bigdigit {{font-size:5.2rem; font-weight:800; line-height:1; text-align:center;
  background: linear-gradient(135deg,{ACCENT},{CYAN}); -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;}}
.conf {{text-align:center; color:#B8C1DB; margin-top:4px;}}
div[data-testid="stMetric"] {{background:#141B2D; border:1px solid #243049;
  border-radius:14px; padding:12px 16px;}}
button[data-baseweb="tab"] {{font-size:1rem; font-weight:600;}}
footer {{visibility:hidden;}}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ data & model
@st.cache_data(show_spinner=False)
def get_data():
    d = load_digits()
    X, y = d.data, d.target
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X, y, X_tr, X_te, y_tr, y_te


def build_model(kernel="rbf", C=1.0, gamma=0.1, probability=True):
    return make_pipeline(
        MinMaxScaler(),
        SVC(kernel=kernel, C=C, gamma=gamma, probability=probability, random_state=42),
    )


@st.cache_resource(show_spinner="Training the SVM...")
def get_final_model():
    X, y, X_tr, X_te, y_tr, y_te = get_data()
    model = build_model("rbf", 1.0, 0.1).fit(X_tr, y_tr)
    return model


@st.cache_data(show_spinner=False)
def get_final_metrics():
    X, y, X_tr, X_te, y_tr, y_te = get_data()
    model = get_final_model()
    pred = model.predict(X_te)
    cv = cross_val_score(build_model(probability=False), X_tr, y_tr, cv=5).mean()
    return {
        "test": accuracy_score(y_te, pred),
        "cv": cv,
        "pred": pred,
        "n_sv": int(model[-1].n_support_.sum()),
    }


@st.cache_data(show_spinner="Running grid search (one time)...")
def get_grid():
    X, y, X_tr, X_te, y_tr, y_te = get_data()
    Cs = [0.1, 1, 10, 100]
    gammas = [0.001, 0.01, 0.1, 1]
    z = np.zeros((len(gammas), len(Cs)))
    for i, g in enumerate(gammas):
        for j, c in enumerate(Cs):
            z[i, j] = cross_val_score(
                build_model("rbf", c, g, probability=False), X_tr, y_tr, cv=3
            ).mean()
    return Cs, gammas, z


@st.cache_data(show_spinner=False)
def get_pca():
    X, y, *_ = get_data()
    p = PCA(n_components=2, random_state=42).fit(X)
    return p.transform(X), p.explained_variance_ratio_.sum()


def dark(fig, h=380):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=h,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ------------------------------------------------------------------ helpers
def show_digit(arr, h=260):
    fig = px.imshow(
        np.asarray(arr).reshape(8, 8),
        color_continuous_scale="Purples",
        zmin=0,
        zmax=16,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_coloraxes(showscale=False)
    fig.update_layout(height=h, margin=dict(l=0, r=0, t=0, b=0),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def preprocess_upload(img: Image.Image, invert: bool):
    """Photo/drawing -> 8x8 vector in the 0..16 range, like the training data."""
    g = ImageOps.grayscale(img.convert("RGBA").convert("RGB"))
    a = np.asarray(g, dtype=float)
    if invert:
        a = 255 - a
    a = a - a.min()
    if a.max() > 0:
        a = a / a.max()
    ys, xs = np.where(a > 0.25)
    if len(xs) == 0:
        return np.zeros(64)
    a = a[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]  # crop to the digit
    h, w = a.shape
    s = int(max(h, w) * 1.05)  # pad to square with a margin, like the dataset
    canvas = np.zeros((s, s))
    canvas[(s - h) // 2: (s - h) // 2 + h, (s - w) // 2: (s - w) // 2 + w] = a
    small = Image.fromarray((canvas * 255).astype("uint8")).resize(
        (8, 8), Image.LANCZOS
    )
    v = np.asarray(small, dtype=float) / 255 * 16
    return np.clip(v, 0, 16).flatten()


def prob_chart(proba, pred):
    colors = [ACCENT if i == pred else "#3A4668" for i in range(10)]
    fig = go.Figure(go.Bar(x=list(range(10)), y=proba, marker_color=colors,
                           text=[f"{p:.0%}" if p > 0.02 else "" for p in proba],
                           textposition="outside"))
    fig.update_xaxes(dtick=1, title="Digit")
    fig.update_yaxes(range=[0, 1.12], tickformat=".0%", title="Probability")
    fig.update_layout(title="Class probabilities")
    return dark(fig, 320)


def result_panel(vec, model):
    proba = model.predict_proba(vec.reshape(1, -1))[0]
    pred = int(np.argmax(proba))
    top3 = np.argsort(proba)[::-1][:3]
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(
            f'<div class="card"><div class="bigdigit">{pred}</div>'
            f'<div class="conf">confidence <b>{proba[pred]:.1%}</b></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Top 3**")
        for k in top3:
            st.progress(float(proba[k]), text=f"Digit {k}: {proba[k]:.1%}")
    with c2:
        st.plotly_chart(prob_chart(proba, pred), width="stretch")
    return pred, proba


# ------------------------------------------------------------------ load
X, y, X_tr, X_te, y_tr, y_te = get_data()
model = get_final_model()
M = get_final_metrics()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### ⚙️ Final model")
    st.code("MinMaxScaler\nSVC(kernel='rbf',\n    C=1, gamma=0.1)", language="python")
    st.metric("CV accuracy", f"{M['cv']:.3f}")
    st.metric("Test accuracy", f"{M['test']:.3f}")
    st.metric("Support vectors", f"{M['n_sv']} / {len(X_tr)}")
    st.divider()
    st.markdown(
        "**Dataset:** [UCI Optical Digits](https://archive.ics.uci.edu/dataset/80)  \n"
        "**Code:** [GitHub D0027](https://github.com/D0027/svm-digit-recognition)  \n"
        "**Portfolio:** [d0027.github.io](https://d0027.github.io)"
    )

# ------------------------------------------------------------------ hero
st.markdown(
    f"""
<div class="hero">
  <h1>✍️ SVM Digit Lab</h1>
  <p>Draw, upload or pick a digit and watch a Support Vector Machine read it live.
     Then tune <b>C</b> and <b>gamma</b> yourself and see overfitting happen.</p>
  <span class="pill">RBF kernel</span><span class="pill">1,797 images · 8×8</span>
  <span class="pill">CV {M['cv']:.3f}</span><span class="pill">Test {M['test']:.3f}</span>
  <span class="pill">scikit-learn · Streamlit</span>
</div>
""",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🔮 Predict", "📊 Performance", "🎛️ Playground", "🗺️ Data Explorer", "📘 About"]
)

# ================================================================== TAB 1
with tab1:
    mode = st.radio(
        "Input method",
        ["🎲 Test sample", "🖌️ Pixel painter", "📤 Upload image"],
        horizontal=True,
    )

    if mode == "🎲 Test sample":
        c_a, c_b = st.columns([1, 2])
        with c_a:
            only_wrong = st.toggle("Only show digits the model gets wrong", False)
            idxs = np.where(M["pred"] != y_te)[0] if only_wrong else np.arange(len(y_te))
            if "ridx" not in st.session_state:
                st.session_state.ridx = 0
            if st.button("🎲 Random sample", width="stretch"):
                st.session_state.ridx = int(np.random.randint(len(idxs)))
            pos = st.slider("Sample number", 0, len(idxs) - 1,
                            min(st.session_state.ridx, len(idxs) - 1))
            i = idxs[pos]
            st.plotly_chart(show_digit(X_te[i]), width="stretch")
            st.caption(f"True label: **{y_te[i]}**")
        with c_b:
            pred, _ = result_panel(X_te[i], model)
            if pred == y_te[i]:
                st.success(f"Correct: predicted {pred}, true label {y_te[i]}")
            else:
                st.error(f"Wrong: predicted {pred}, true label {y_te[i]}")

    elif mode == "🖌️ Pixel painter":
        st.caption("Edit the 8×8 grid (0 = blank, 16 = full ink) and the prediction "
                   "updates instantly. Start blank or from a sample and modify it.")
        if "pk" not in st.session_state:
            st.session_state.pk = 0
            st.session_state.grid = np.zeros((8, 8), dtype=int)
        b1, b2, b3 = st.columns(3)
        with b1:
            start = st.selectbox("Start from digit", ["Blank"] + [str(d) for d in range(10)])
        with b2:
            st.write("")
            if st.button("Load / reset grid", width="stretch"):
                if start == "Blank":
                    st.session_state.grid = np.zeros((8, 8), dtype=int)
                else:
                    j = int(np.where(y_te == int(start))[0][0])
                    st.session_state.grid = X_te[j].reshape(8, 8).astype(int)
                st.session_state.pk += 1
        with b3:
            st.write("")
            if st.button("🧹 Clear", width="stretch"):
                st.session_state.grid = np.zeros((8, 8), dtype=int)
                st.session_state.pk += 1
        g1, g2 = st.columns([1, 2])
        with g1:
            df = pd.DataFrame(st.session_state.grid, columns=[f"c{i}" for i in range(8)])
            edited = st.data_editor(
                df,
                key=f"grid{st.session_state.pk}",
                column_config={c: st.column_config.NumberColumn(c, min_value=0, max_value=16, step=1, width="small")
                               for c in df.columns},
                hide_index=True,
                width="stretch",
            )
            vec = np.clip(edited.to_numpy(dtype=float), 0, 16).flatten()
            st.plotly_chart(show_digit(vec, 220), width="stretch")
        with g2:
            if vec.sum() == 0:
                st.info("Grid is blank. Fill some pixels to get a prediction.")
            else:
                result_panel(vec, model)

    else:
        st.caption("Upload a photo or screenshot of ONE digit (dark ink on light paper "
                   "or light on dark). It is cropped, padded and shrunk to 8×8.")
        u1, u2 = st.columns([1, 2])
        with u1:
            f = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp"])
            inv = st.toggle("Dark ink on light background (invert)", True)
        if f is not None:
            img = Image.open(f)
            vec = preprocess_upload(img, inv)
            with u1:
                st.image(img, caption="Your image", width="stretch")
                st.markdown("**What the model sees (8×8)**")
                st.plotly_chart(show_digit(vec, 220), width="stretch")
            with u2:
                result_panel(vec, model)
            st.caption("Note: the model is trained on tiny 8×8 scans, so real-world "
                       "photos can be less accurate than the 99% test score.")
        else:
            with u2:
                st.info("Upload an image to begin.")

    with st.expander("🧪 Robustness test: add noise to a digit"):
        r1, r2 = st.columns([1, 2])
        with r1:
            dsel = st.selectbox("Digit", list(range(10)), key="noise_digit")
            lvl = st.slider("Noise level", 0.0, 8.0, 2.0, 0.5)
            seed = st.number_input("Seed", 0, 999, 1)
        j = int(np.where(y_te == dsel)[0][0])
        rng = np.random.default_rng(int(seed))
        noisy = np.clip(X_te[j] + rng.normal(0, lvl, 64), 0, 16)
        with r1:
            st.plotly_chart(show_digit(noisy, 220), width="stretch")
        with r2:
            result_panel(noisy, model)

# ================================================================== TAB 2
with tab2:
    k1, k2, k3, k4 = st.columns(4)
    errs = int((M["pred"] != y_te).sum())
    k1.metric("Test accuracy", f"{M['test']:.2%}")
    k2.metric("5-fold CV", f"{M['cv']:.2%}")
    k3.metric("Errors", f"{errs} / {len(y_te)}")
    k4.metric("Support vectors", f"{M['n_sv']}", f"{M['n_sv']/len(X_tr):.0%} of train")

    left, right = st.columns(2)
    with left:
        cm = confusion_matrix(y_te, M["pred"])
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Purples",
                        labels=dict(x="Predicted", y="True"),
                        x=list(range(10)), y=list(range(10)),
                        title="Confusion matrix")
        fig.update_xaxes(dtick=1)
        fig.update_yaxes(dtick=1)
        st.plotly_chart(dark(fig, 430), width="stretch")
    with right:
        rep = pd.DataFrame(classification_report(y_te, M["pred"], output_dict=True)).T
        rep = rep.loc[[str(i) for i in range(10)], ["precision", "recall", "f1-score"]]
        fig = go.Figure()
        for col, colr in zip(rep.columns, [ACCENT, CYAN, PINK]):
            fig.add_bar(name=col, x=rep.index, y=rep[col], marker_color=colr)
        fig.update_layout(barmode="group", title="Per-class precision / recall / F1")
        fig.update_yaxes(range=[0.9, 1.01])
        st.plotly_chart(dark(fig, 430), width="stretch")

    st.markdown("#### ❌ Misclassified test digits")
    wrong = np.where(M["pred"] != y_te)[0]
    if len(wrong) == 0:
        st.success("No mistakes!")
    else:
        cols = st.columns(min(len(wrong), 6))
        for c, w in zip(cols, wrong[:6]):
            with c:
                st.plotly_chart(show_digit(X_te[w], 150), width="stretch")
                st.caption(f"true **{y_te[w]}** → pred **{M['pred'][w]}**")

    st.markdown("#### 🔧 Grid search: C vs gamma (3-fold CV)")
    Cs, gammas, z = get_grid()
    fig = px.imshow(z, x=[str(c) for c in Cs], y=[str(g) for g in gammas],
                    text_auto=".3f", color_continuous_scale="Viridis",
                    labels=dict(x="C", y="gamma", color="CV acc"),
                    title="Good values form a diagonal ridge")
    st.plotly_chart(dark(fig, 400), width="stretch")

# ================================================================== TAB 3
with tab3:
    st.markdown("Change the hyperparameters, train a fresh SVM, and compare it with the final model.")
    p1, p2 = st.columns([1, 2])
    with p1:
        kern = st.selectbox("Kernel", ["rbf", "poly", "linear", "sigmoid"])
        logC = st.slider("log10(C)", -2.0, 3.0, 0.0, 0.25)
        logG = st.slider("log10(gamma)", -4.0, 1.0, -1.0, 0.25,
                         disabled=(kern == "linear"))
        C, gam = 10 ** logC, 10 ** logG
        st.caption(f"C = **{C:.3g}**, gamma = **{gam:.3g}**")
        go_btn = st.button("🚀 Train this model", type="primary", width="stretch")
    with p2:
        if go_btn:
            t0 = time.time()
            m = build_model(kern, C, gam, probability=False).fit(X_tr, y_tr)
            dt = time.time() - t0
            tr = m.score(X_tr, y_tr)
            te = m.score(X_te, y_te)
            cv = cross_val_score(build_model(kern, C, gam, probability=False),
                                 X_tr, y_tr, cv=5).mean()
            nsv = int(m[-1].n_support_.sum())
            a, b, c, d = st.columns(4)
            a.metric("Train", f"{tr:.3f}")
            b.metric("CV", f"{cv:.3f}", f"{cv - M['cv']:+.3f} vs final")
            c.metric("Test", f"{te:.3f}")
            d.metric("Support vectors", nsv)
            gap = tr - cv
            if cv < 0.5:
                st.error("Severe overfitting: every point only 'sees' itself (gamma too large).")
            elif gap > 0.03:
                st.warning(f"Overfitting: train-CV gap is {gap:.3f}.")
            elif tr < 0.95:
                st.warning("Underfitting: even the training score is low.")
            else:
                st.success("Healthy fit. Train and CV are close.")
            fig = go.Figure(go.Bar(x=["Train", "CV", "Test"], y=[tr, cv, te],
                                   marker_color=[CYAN, ACCENT, PINK],
                                   text=[f"{v:.3f}" for v in (tr, cv, te)],
                                   textposition="outside"))
            fig.update_yaxes(range=[0, 1.1])
            fig.update_layout(title=f"{kern} kernel · trained in {dt*1000:.0f} ms")
            st.plotly_chart(dark(fig, 340), width="stretch")
        else:
            st.info("Pick settings on the left and press **Train this model**. "
                    "Try gamma = 10 (log10 = 1) to see overfitting.")

# ================================================================== TAB 4
with tab4:
    coords, var = get_pca()
    e1, e2 = st.columns([3, 2])
    with e1:
        dfp = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "digit": y.astype(str)})
        fig = px.scatter(dfp, x="PC1", y="PC2", color="digit",
                         category_orders={"digit": [str(i) for i in range(10)]},
                         title=f"PCA projection of all 1,797 digits ({var:.0%} variance in 2 components)",
                         opacity=0.8)
        st.plotly_chart(dark(fig, 480), width="stretch")
    with e2:
        st.markdown("**Average image per digit**")
        d = st.select_slider("Digit", options=list(range(10)), value=0)
        st.plotly_chart(show_digit(X[y == d].mean(axis=0), 300), width="stretch")
        st.caption(f"{int((y == d).sum())} images of digit {d}")
    st.markdown("**Mean pixel intensity per position**")
    fig = px.imshow(X.mean(axis=0).reshape(8, 8), color_continuous_scale="Magma")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    st.plotly_chart(dark(fig, 300), width="stretch")
    st.caption("Border pixels are almost always blank, which is why StandardScaler "
               "(which inflates them) did worse than MinMax in the notebook.")

# ================================================================== TAB 5
with tab5:
    a1, a2 = st.columns(2)
    with a1:
        st.markdown(
            """
### How it works
1. **Margin:** an SVM picks the boundary with the widest gap between classes.
2. **Support vectors:** only the points on or inside the margin define the boundary.
3. **Kernel trick:** the RBF kernel learns curved boundaries without building the high-dimensional space.
4. **C** trades margin width for training errors; **gamma** sets how far one point's influence reaches.
5. **10 classes:** scikit-learn's `SVC` trains 45 one-vs-one classifiers and votes.

### Key results
| Finding | Value |
|---|---|
| Final model CV / test | 0.988 / 0.992 |
| Circles data: linear vs RBF | 0.53 vs 1.00 |
| gamma = 10 | train 1.00, CV 0.145 |
| MinMax vs StandardScaler | 0.988 vs 0.981 |
"""
        )
    with a2:
        st.markdown(
            """
### Stack
`Python` · `scikit-learn` · `Streamlit` · `Plotly` · `Pillow` · `NumPy` · `Pandas`

### Links
- 📓 [GitHub repository](https://github.com/D0027/svm-digit-recognition)
- 📦 [Dataset (UCI Optical Digits)](https://archive.ics.uci.edu/dataset/80)
- 🌐 [Portfolio](https://d0027.github.io)
- 💼 [LinkedIn](https://linkedin.com/in/deepakyadav027)

### Limits
The model learned from 8×8 scans, so photos of real handwriting may be less
accurate than the 99% test score suggests. SVM training time also grows roughly
like n², so it suits small and medium data.

Built by **Deepak Yadav**.
"""
        )
