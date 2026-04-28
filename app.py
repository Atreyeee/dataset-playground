import streamlit as st
import pandas as pd
import numpy as np

from utils.load_data import load_csv
from utils.summary import get_summary
from utils.visualization import plot_distribution, plot_correlation, plot_scatter, plot_boxplot
from utils.ml_model import run_ml_model

# ─────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="DataLens — Dataset Playground",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
#  Global CSS — dark professional theme
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Root & base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #0d0f14;
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #111318 !important;
    border-right: 1px solid #1e2130 !important;
}
[data-testid="stSidebar"] .block-container { padding: 2rem 1.2rem; }

/* ── Hide default header ── */
header[data-testid="stHeader"] { display: none; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1280px; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #161922;
    border: 1px solid #1e2130;
    border-radius: 12px;
    padding: 1rem 1.25rem;
}
[data-testid="stMetricLabel"] { font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.08em; color: #6b7280 !important; }
[data-testid="stMetricValue"] { font-size: 28px !important; font-weight: 600 !important; color: #f1f5f9 !important; }
[data-testid="stMetricDelta"] { font-size: 12px !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid #1e2130; border-radius: 8px; overflow: hidden; }

/* ── Buttons ── */
.stButton > button {
    background: #2563eb;
    color: #fff;
    border: none;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 14px;
    padding: 0.5rem 1.5rem;
    transition: background 0.2s;
}
.stButton > button:hover { background: #1d4ed8; border: none; }

/* ── Selectbox / inputs ── */
[data-baseweb="select"] > div {
    background: #161922 !important;
    border: 1px solid #1e2130 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
.stSelectbox label, .stMultiSelect label { font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: #6b7280; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #161922;
    border: 1.5px dashed #2a3042;
    border-radius: 12px;
    padding: 1rem;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #161922;
    border: 1px solid #1e2130 !important;
    border-radius: 10px;
}

/* ── Tabs ── */
[data-baseweb="tab-list"] { background: transparent; border-bottom: 1px solid #1e2130; gap: 0; }
[data-baseweb="tab"] {
    background: transparent !important;
    color: #6b7280 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.2rem !important;
    border-bottom: 2px solid transparent !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: #60a5fa !important;
    border-bottom: 2px solid #2563eb !important;
}

/* ── Divider ── */
hr { border-color: #1e2130; }

/* ── Success / error ── */
.stSuccess { background: #0a2e1a !important; border: 1px solid #166534 !important; color: #4ade80 !important; border-radius: 8px !important; }
.stError   { background: #2a0a0a !important; border: 1px solid #7f1d1d !important; color: #f87171 !important; border-radius: 8px !important; }
.stWarning { background: #1c1a08 !important; border: 1px solid #713f12 !important; color: #fbbf24 !important; border-radius: 8px !important; }
.stInfo    { background: #0a1628 !important; border: 1px solid #1e3a5f !important; color: #60a5fa !important; border-radius: 8px !important; }

/* ── Section header pill ── */
.section-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: #161922; border: 1px solid #1e2130;
    border-radius: 20px; padding: 4px 14px;
    font-size: 11px; font-weight: 500; letter-spacing: 0.08em;
    text-transform: uppercase; color: #6b7280;
    margin-bottom: 0.75rem;
}
.section-pill .dot { width: 6px; height: 6px; border-radius: 50%; background: #2563eb; }

/* ── Page title ── */
.page-title {
    font-size: 28px; font-weight: 600; color: #f1f5f9;
    letter-spacing: -0.02em; margin-bottom: 0.25rem;
}
.page-sub { font-size: 14px; color: #6b7280; margin-bottom: 2rem; }

/* ── Stat badge ── */
.stat-badge {
    display: inline-block;
    background: #0f2040; color: #60a5fa;
    border: 1px solid #1e3a5f;
    border-radius: 6px; padding: 2px 10px;
    font-size: 12px; font-family: 'DM Mono', monospace;
    font-weight: 500;
}

/* ── Score card ── */
.score-card {
    background: linear-gradient(135deg, #0f2040 0%, #0a1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.score-card .score-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; color: #6b7280; margin-bottom: 0.5rem; }
.score-card .score-value { font-size: 48px; font-weight: 600; color: #60a5fa; font-family: 'DM Mono', monospace; }
.score-card .score-type  { font-size: 13px; color: #94a3b8; margin-top: 0.25rem; }

/* ── Feature importance bar ── */
.fi-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; font-size: 13px; }
.fi-name { width: 140px; color: #94a3b8; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-family: 'DM Mono', monospace; font-size: 12px; }
.fi-bar-bg { flex: 1; background: #161922; border-radius: 4px; height: 8px; }
.fi-bar { background: linear-gradient(90deg, #2563eb, #60a5fa); border-radius: 4px; height: 8px; }
.fi-val { width: 42px; text-align: right; color: #6b7280; font-size: 12px; font-family: 'DM Mono', monospace; }

/* ── Logo / brand ── */
.brand { font-size: 18px; font-weight: 600; color: #f1f5f9; letter-spacing: -0.01em; }
.brand span { color: #2563eb; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────
if "data" not in st.session_state:
    st.session_state.data = None
if "filename" not in st.session_state:
    st.session_state.filename = None


# ─────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand">Data<span>Lens</span></div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:12px;color:#374151;margin-bottom:1.5rem;">Dataset Exploration Platform</p>', unsafe_allow_html=True)
    st.divider()

    menu = st.radio(
        "Navigation",
        ["◈  Overview", "📋  Data Table", "📊  Visualize", "🤖  ML Model"],
        label_visibility="collapsed"
    )
    page = menu.split("  ")[1]

    st.divider()

    # Dataset status
    if st.session_state.data is not None:
        df = st.session_state.data
        st.markdown(f'<p style="font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:#6b7280;margin-bottom:.5rem;">Active Dataset</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:13px;color:#e2e8f0;font-weight:500;margin-bottom:.25rem;">{st.session_state.filename}</p>', unsafe_allow_html=True)
        st.markdown(f'<span class="stat-badge">{df.shape[0]:,} rows × {df.shape[1]} cols</span>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⟳  Clear dataset", use_container_width=True):
            st.session_state.data = None
            st.session_state.filename = None
            st.rerun()
    else:
        st.markdown('<p style="font-size:12px;color:#374151;">No dataset loaded.<br>Go to Overview to upload.</p>', unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<p style="font-size:11px;color:#374151;">DataLens v2.0 · Built with Streamlit</p>', unsafe_allow_html=True)


# ─────────────────────────────────────────
#  Helper
# ─────────────────────────────────────────
def no_data_warning():
    st.markdown('<br>', unsafe_allow_html=True)
    st.info("📂  No dataset loaded yet. Head to **Overview** to upload a CSV.")


def section_header(label: str):
    st.markdown(f'<div class="section-pill"><span class="dot"></span>{label}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────
#  PAGE: Overview (Upload + Summary)
# ─────────────────────────────────────────
if page == "Overview":
    st.markdown('<div class="page-title">Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Upload a CSV and inspect your dataset at a glance.</div>', unsafe_allow_html=True)

    col_upload, col_info = st.columns([1.2, 1], gap="large")

    with col_upload:
        section_header("Upload")
        file = st.file_uploader("Drop a CSV file here", type=["csv"], label_visibility="collapsed")

        if file:
            df, err = load_csv(file)
            if df is not None:
                st.session_state.data = df
                st.session_state.filename = file.name
                st.success(f"✓  Loaded **{file.name}** — {df.shape[0]:,} rows, {df.shape[1]} columns")
            else:
                st.error(f"Failed to load file: {err}")

    with col_info:
        section_header("Quick tips")
        st.markdown("""
<ul style="font-size:13px;color:#6b7280;line-height:2;padding-left:1.2rem;">
  <li>CSV files up to 200 MB are supported</li>
  <li>Mixed numeric &amp; categorical columns work fine</li>
  <li>Missing values are handled automatically in ML</li>
  <li>Use the sidebar to navigate between views</li>
</ul>""", unsafe_allow_html=True)

    if st.session_state.data is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        df = st.session_state.data
        summary = get_summary(df)

        # ── Metric row ──
        section_header("Dataset stats")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", f"{summary['shape'][0]:,}")
        m2.metric("Columns", str(summary['shape'][1]))
        m3.metric("Missing values", str(int(summary['missing'].sum())))
        m4.metric("Numeric cols", str(len(df.select_dtypes(include='number').columns)))

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Column types & missing ──
        col_a, col_b = st.columns(2, gap="large")

        with col_a:
            section_header("Column types")
            type_df = pd.DataFrame({
                "Column": summary["dtypes"].index,
                "Type": summary["dtypes"].values.astype(str),
            })
            st.dataframe(type_df, use_container_width=True, hide_index=True, height=250)

        with col_b:
            section_header("Missing values")
            miss = summary["missing"]
            miss_df = pd.DataFrame({
                "Column": miss.index,
                "Missing": miss.values,
                "% Missing": (miss.values / df.shape[0] * 100).round(1),
            })
            miss_df = miss_df.sort_values("Missing", ascending=False)
            st.dataframe(miss_df, use_container_width=True, hide_index=True, height=250)

        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Descriptive statistics")
        desc = df.describe(include='all').T.round(3)
        st.dataframe(desc, use_container_width=True, height=300)


# ─────────────────────────────────────────
#  PAGE: Data Table
# ─────────────────────────────────────────
elif page == "Data Table":
    st.markdown('<div class="page-title">Data Table</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Browse, filter, and search your raw data.</div>', unsafe_allow_html=True)

    if st.session_state.data is None:
        no_data_warning()
    else:
        df = st.session_state.data

        # ── Filter controls ──
        section_header("Filters")
        fcol1, fcol2, fcol3 = st.columns([1.5, 1.5, 1])

        with fcol1:
            search_col = st.selectbox("Filter by column", ["— none —"] + list(df.columns))
        with fcol2:
            search_val = st.text_input("Contains value", placeholder="e.g. male, 30, true …")
        with fcol3:
            n_rows = st.selectbox("Rows to show", [50, 100, 500, "All"], index=0)

        filtered = df.copy()
        if search_col != "— none —" and search_val:
            mask = filtered[search_col].astype(str).str.contains(search_val, case=False, na=False)
            filtered = filtered[mask]

        display_df = filtered if n_rows == "All" else filtered.head(int(n_rows))

        st.markdown("<br>", unsafe_allow_html=True)
        section_header(f"{len(filtered):,} rows matched")
        st.dataframe(display_df, use_container_width=True, height=480, hide_index=False)

        # ── Download filtered ──
        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇  Download filtered CSV",
            data=csv_bytes,
            file_name="filtered_data.csv",
            mime="text/csv",
        )


# ─────────────────────────────────────────
#  PAGE: Visualize
# ─────────────────────────────────────────
elif page == "Visualize":
    st.markdown('<div class="page-title">Visualize</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Explore distributions, correlations, and relationships.</div>', unsafe_allow_html=True)

    if st.session_state.data is None:
        no_data_warning()
    else:
        df = st.session_state.data
        num_cols = list(df.select_dtypes(include='number').columns)
        cat_cols = list(df.select_dtypes(include='object').columns)
        all_cols = list(df.columns)

        tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Correlation", "Scatter", "Box / Violin"])

        # ── Tab 1: Distribution ──
        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            d_col1, d_col2 = st.columns([1, 3])
            with d_col1:
                dist_col = st.selectbox("Column", all_cols, key="dist_col")
                if dist_col in num_cols:
                    bins = st.slider("Bins", 5, 100, 30)
                    show_kde = st.checkbox("Overlay KDE", value=True)
                else:
                    bins, show_kde = 30, False
                    st.info("Categorical column — showing value counts")
            with d_col2:
                fig = plot_distribution(df, dist_col, bins=bins, show_kde=show_kde)
                st.pyplot(fig)

        # ── Tab 2: Correlation heatmap ──
        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            if len(num_cols) < 2:
                st.warning("Need at least 2 numeric columns for a correlation heatmap.")
            else:
                h_col1, h_col2 = st.columns([1, 3])
                with h_col1:
                    selected_cols = st.multiselect("Columns", num_cols, default=num_cols[:min(8, len(num_cols))])
                    method = st.selectbox("Method", ["pearson", "spearman", "kendall"])
                with h_col2:
                    if len(selected_cols) >= 2:
                        fig = plot_correlation(df, selected_cols, method=method)
                        st.pyplot(fig)
                    else:
                        st.info("Select at least 2 columns.")

        # ── Tab 3: Scatter ──
        with tab3:
            st.markdown("<br>", unsafe_allow_html=True)
            if len(num_cols) < 2:
                st.warning("Need at least 2 numeric columns for a scatter plot.")
            else:
                s1, s2, s3 = st.columns(3)
                with s1:
                    x_col = st.selectbox("X axis", num_cols, index=0, key="sx")
                with s2:
                    y_col = st.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1), key="sy")
                with s3:
                    hue_col = st.selectbox("Color by", ["— none —"] + cat_cols, key="shue")
                hue = None if hue_col == "— none —" else hue_col
                fig = plot_scatter(df, x_col, y_col, hue=hue)
                st.pyplot(fig)

        # ── Tab 4: Box / Violin ──
        with tab4:
            st.markdown("<br>", unsafe_allow_html=True)
            b1, b2, b3 = st.columns(3)
            with b1:
                bv_col = st.selectbox("Numeric column", num_cols, key="bvc")
            with b2:
                grp_col = st.selectbox("Group by", ["— none —"] + cat_cols, key="bgc")
            with b3:
                bv_type = st.selectbox("Chart type", ["Box", "Violin"])
            grp = None if grp_col == "— none —" else grp_col
            fig = plot_boxplot(df, bv_col, group=grp, kind=bv_type.lower())
            st.pyplot(fig)


# ─────────────────────────────────────────
#  PAGE: ML Model
# ─────────────────────────────────────────
elif page == "ML Model":
    st.markdown('<div class="page-title">ML Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Train a baseline model, inspect metrics and feature importance.</div>', unsafe_allow_html=True)

    if st.session_state.data is None:
        no_data_warning()
    else:
        df = st.session_state.data

        section_header("Configuration")
        ml1, ml2, ml3 = st.columns(3)

        with ml1:
            target = st.selectbox("Target column", df.columns)
        with ml2:
            test_size = st.slider("Test split %", 10, 40, 20)
        with ml3:
            scale = st.checkbox("Scale features", value=True)

        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("▶  Train model", use_container_width=False)

        if run:
            with st.spinner("Training…"):
                results = run_ml_model(df, target, test_size=test_size / 100, scale_features=scale)

            if results.get("error"):
                st.error(f"Training failed: {results['error']}")
            else:
                st.markdown("<br>", unsafe_allow_html=True)
                section_header("Results")

                # ── Score card ──
                r1, r2, r3 = st.columns([1, 1, 2])
                metric_name = "Accuracy" if results["problem"] == "Classification" else "R² Score"
                score_val = results["score"]

                with r1:
                    st.markdown(f"""
<div class="score-card">
  <div class="score-label">{metric_name}</div>
  <div class="score-value">{score_val:.2%}</div>
  <div class="score-type">{results['problem']}</div>
</div>""", unsafe_allow_html=True)

                with r2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.metric("Model", results["model_name"])
                    st.metric("Train rows", f"{results['train_rows']:,}")
                    st.metric("Test rows", f"{results['test_rows']:,}")

                with r3:
                    if results.get("extra_metrics"):
                        section_header("Extra metrics")
                        em_df = pd.DataFrame(list(results["extra_metrics"].items()), columns=["Metric", "Value"])
                        st.dataframe(em_df, use_container_width=True, hide_index=True)

                # ── Feature importance ──
                if results.get("feature_importance") is not None:
                    st.markdown("<br>", unsafe_allow_html=True)
                    section_header("Feature importance")
                    fi = results["feature_importance"]
                    max_fi = fi["Importance"].max() or 1
                    for _, row in fi.head(15).iterrows():
                        pct = row["Importance"] / max_fi * 100
                        st.markdown(f"""
<div class="fi-row">
  <div class="fi-name">{row['Feature']}</div>
  <div class="fi-bar-bg"><div class="fi-bar" style="width:{pct:.1f}%"></div></div>
  <div class="fi-val">{row['Importance']:.3f}</div>
</div>""", unsafe_allow_html=True)

                # ── Confusion matrix / residuals ──
                if results.get("confusion_matrix") is not None:
                    st.markdown("<br>", unsafe_allow_html=True)
                    section_header("Confusion matrix")
                    cm_fig = results["confusion_matrix"]
                    st.pyplot(cm_fig)

                if results.get("residual_plot") is not None:
                    st.markdown("<br>", unsafe_allow_html=True)
                    section_header("Residuals vs predicted")
                    st.pyplot(results["residual_plot"])
