import streamlit as st
import pandas as pd
import numpy as np

from load_data import load_csv
from summary import get_summary
from visualization import plot_distribution, plot_correlation, plot_scatter, plot_boxplot
from ml_model import run_ml_model

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
#  Global CSS — Bento Grid / Aurora theme
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── CSS Variables ── */
:root {
    --bg:          #050810;
    --surface:     #0b0f1a;
    --surface2:    #0f1520;
    --border:      rgba(99,179,237,0.10);
    --border2:     rgba(99,179,237,0.18);
    --accent:      #63b3ed;
    --accent2:     #f6ad55;
    --accent3:     #68d391;
    --text-primary:#e8f0fe;
    --text-muted:  #4a5568;
    --text-dim:    #2d3748;
    --glow:        rgba(99,179,237,0.15);
    --glow2:       rgba(246,173,85,0.12);
}

/* ── Base ── */
@font-face { }
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
.stApp {
    background: var(--bg);
    color: var(--text-primary);
}

/* ── Aurora background mesh ── */
.stApp::before {
    content: '';
    position: fixed;
    top: -40%;
    left: -20%;
    width: 70%;
    height: 70%;
    background: radial-gradient(ellipse, rgba(99,179,237,0.06) 0%, transparent 65%);
    pointer-events: none;
    z-index: 0;
    animation: aurora1 12s ease-in-out infinite alternate;
}
.stApp::after {
    content: '';
    position: fixed;
    bottom: -30%;
    right: -15%;
    width: 60%;
    height: 60%;
    background: radial-gradient(ellipse, rgba(246,173,85,0.05) 0%, transparent 65%);
    pointer-events: none;
    z-index: 0;
    animation: aurora2 15s ease-in-out infinite alternate;
}
@keyframes aurora1 {
    0%   { transform: translate(0,0) scale(1); }
    100% { transform: translate(5%,8%) scale(1.1); }
}
@keyframes aurora2 {
    0%   { transform: translate(0,0) scale(1); }
    100% { transform: translate(-6%,-5%) scale(1.08); }
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border2) !important;
    backdrop-filter: blur(12px);
}
[data-testid="stSidebar"] .block-container {
    padding: 2.2rem 1.4rem;
}

/* ── Hide default header ── */
header[data-testid="stHeader"] { display: none; }
.block-container {
    padding: 2.2rem 3rem 5rem !important;
    max-width: 1320px;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.25s, box-shadow 0.25s;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(99,179,237,0.35);
    box-shadow: 0 0 28px var(--glow);
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    opacity: 0.6;
}
[data-testid="stMetricLabel"] {
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--text-muted) !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    font-size: 32px !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: -0.02em;
}
[data-testid="stMetricDelta"] { font-size: 12px !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border2) !important;
    border-radius: 12px;
    overflow: hidden;
    background: var(--surface2);
}

/* ── Buttons ── */
.stButton > button {
    background: transparent;
    color: var(--accent);
    border: 1px solid var(--border2);
    border-radius: 10px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 13px;
    letter-spacing: 0.04em;
    padding: 0.55rem 1.6rem;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}
.stButton > button::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(99,179,237,0.08), transparent);
    opacity: 0;
    transition: opacity 0.2s;
}
.stButton > button:hover {
    border-color: var(--accent);
    box-shadow: 0 0 20px var(--glow), inset 0 0 20px rgba(99,179,237,0.04);
    color: #fff;
}
.stButton > button:hover::before { opacity: 1; }

/* ── Selectbox / inputs ── */
[data-baseweb="select"] > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'Syne', sans-serif !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-baseweb="select"] > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(99,179,237,0.12) !important;
}
.stSelectbox label, .stMultiSelect label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-muted) !important;
    font-weight: 600;
}
.stTextInput > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}
.stTextInput > div > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(99,179,237,0.12) !important;
}
.stTextInput label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-muted) !important;
    font-weight: 600;
}

/* ── Sliders ── */
[data-testid="stSlider"] [data-baseweb="slider"] {
    padding: 0 4px;
}
[data-testid="stSlider"] label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-muted) !important;
    font-weight: 600;
}

/* ── Checkbox ── */
.stCheckbox label span {
    font-size: 13px !important;
    color: var(--text-primary) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--surface2);
    border: 1.5px dashed rgba(99,179,237,0.22);
    border-radius: 16px;
    padding: 1.5rem;
    transition: border-color 0.25s, box-shadow 0.25s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,179,237,0.45);
    box-shadow: 0 0 30px var(--glow);
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--surface2);
    border: 1px solid var(--border2) !important;
    border-radius: 12px;
}

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid var(--border);
    gap: 0;
}
[data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 0.7rem 1.4rem !important;
    border-bottom: 2px solid transparent !important;
    font-family: 'Syne', sans-serif !important;
    transition: color 0.2s !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}
[data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
}

/* ── Divider ── */
hr { border-color: var(--border); }

/* ── Alerts ── */
.stSuccess {
    background: rgba(104,211,145,0.06) !important;
    border: 1px solid rgba(104,211,145,0.25) !important;
    color: #68d391 !important;
    border-radius: 10px !important;
}
.stError {
    background: rgba(245,101,101,0.06) !important;
    border: 1px solid rgba(245,101,101,0.25) !important;
    color: #fc8181 !important;
    border-radius: 10px !important;
}
.stWarning {
    background: rgba(246,173,85,0.06) !important;
    border: 1px solid rgba(246,173,85,0.25) !important;
    color: #f6ad55 !important;
    border-radius: 10px !important;
}
.stInfo {
    background: rgba(99,179,237,0.06) !important;
    border: 1px solid rgba(99,179,237,0.22) !important;
    color: var(--accent) !important;
    border-radius: 10px !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] > div {
    border-color: var(--accent) transparent transparent transparent !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: transparent;
    color: var(--accent3);
    border: 1px solid rgba(104,211,145,0.28);
    border-radius: 10px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 13px;
    padding: 0.55rem 1.6rem;
    transition: all 0.2s;
}
.stDownloadButton > button:hover {
    border-color: var(--accent3);
    box-shadow: 0 0 20px rgba(104,211,145,0.15);
    color: #fff;
}

/* ── Page title ── */
.page-title {
    font-size: 36px;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.03em;
    margin-bottom: 0.2rem;
    line-height: 1.1;
    position: relative;
    display: inline-block;
}
.page-title::after {
    content: '';
    display: block;
    width: 32px;
    height: 3px;
    background: var(--accent);
    border-radius: 2px;
    margin-top: 8px;
    box-shadow: 0 0 12px var(--accent);
}
.page-sub {
    font-size: 13px;
    color: var(--text-muted);
    margin-top: 0.75rem;
    margin-bottom: 2.2rem;
    letter-spacing: 0.02em;
    font-weight: 400;
}

/* ── Section label ── */
.section-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: 6px;
    padding: 3px 12px 3px 8px;
    font-size: 9.5px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.9rem;
    font-family: 'JetBrains Mono', monospace;
}
.section-pill .dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 6px var(--accent);
    animation: pulse 2.4s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.7); }
}

/* ── Stat badge ── */
.stat-badge {
    display: inline-block;
    background: rgba(99,179,237,0.08);
    color: var(--accent);
    border: 1px solid rgba(99,179,237,0.22);
    border-radius: 6px;
    padding: 3px 12px;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    letter-spacing: 0.04em;
}

/* ── Score card ── */
.score-card {
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: 18px;
    padding: 2rem 2.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.score-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
}
.score-card::after {
    content: '';
    position: absolute;
    bottom: -40%; left: 50%;
    transform: translateX(-50%);
    width: 140px; height: 140px;
    background: radial-gradient(circle, rgba(99,179,237,0.10), transparent 70%);
    pointer-events: none;
}
.score-card .score-label {
    font-size: 9.5px;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: var(--text-muted);
    margin-bottom: 0.6rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}
.score-card .score-value {
    font-size: 52px;
    font-weight: 700;
    color: var(--accent);
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.03em;
    text-shadow: 0 0 40px rgba(99,179,237,0.4);
}
.score-card .score-type {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 0.4rem;
    letter-spacing: 0.06em;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Feature importance bar ── */
.fi-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
    font-size: 12px;
    padding: 6px 0;
    border-bottom: 1px solid var(--border);
}
.fi-row:last-child { border-bottom: none; }
.fi-name {
    width: 150px;
    color: var(--text-muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    flex-shrink: 0;
}
.fi-bar-bg {
    flex: 1;
    background: var(--surface);
    border-radius: 3px;
    height: 6px;
    overflow: hidden;
}
.fi-bar {
    background: linear-gradient(90deg, var(--accent), rgba(99,179,237,0.4));
    border-radius: 3px;
    height: 6px;
    box-shadow: 2px 0 8px rgba(99,179,237,0.3);
    transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
}
.fi-val {
    width: 48px;
    text-align: right;
    color: var(--text-muted);
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    flex-shrink: 0;
}

/* ── Brand / Logo ── */
.brand {
    font-size: 20px;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    line-height: 1;
}
.brand span { color: var(--accent); }
.brand-dot {
    display: inline-block;
    width: 6px; height: 6px;
    background: var(--accent2);
    border-radius: 50%;
    margin-left: 2px;
    vertical-align: middle;
    position: relative;
    top: -2px;
    box-shadow: 0 0 8px var(--accent2);
}

/* ── Nav radio buttons ── */
[data-testid="stRadio"] > label {
    font-size: 10px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.14em !important;
    color: var(--text-dim) !important;
    font-weight: 700 !important;
    margin-bottom: 0.5rem !important;
}
[data-testid="stRadio"] > div {
    gap: 4px !important;
    display: flex !important;
    flex-direction: column !important;
}
[data-testid="stRadio"] > div label {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    display: flex !important;
    align-items: center !important;
}
[data-testid="stRadio"] > div label:hover {
    background: var(--surface2) !important;
    color: var(--text-primary) !important;
    border-color: var(--border) !important;
}
[data-testid="stRadio"] > div [aria-checked="true"] {
    background: rgba(99,179,237,0.08) !important;
    border-color: rgba(99,179,237,0.25) !important;
    color: var(--accent) !important;
}

/* ── Quick tips list ── */
.tip-list {
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.tip-list li {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    font-size: 13px;
    color: var(--text-muted);
    line-height: 1.5;
}
.tip-list li::before {
    content: '→';
    color: var(--accent);
    font-size: 12px;
    flex-shrink: 0;
    margin-top: 1px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Sidebar status card ── */
.status-card {
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: 12px;
    padding: 1rem 1.1rem;
    margin-bottom: 1rem;
}
.status-label {
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: var(--text-dim);
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 6px;
}
.status-name {
    font-size: 13px;
    color: var(--text-primary);
    font-weight: 600;
    margin-bottom: 6px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,179,237,0.3); }
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
    st.markdown(
        '<div class="brand">Data<span>Lens</span><span class="brand-dot"></span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:11px;color:#2d3748;margin-top:4px;margin-bottom:1.5rem;'
        'letter-spacing:0.06em;font-family:\'JetBrains Mono\',monospace;">DATASET EXPLORATION PLATFORM</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    menu = st.radio(
        "Navigation",
        ["◈  Overview", "📋  Data Table", "📊  Visualize", "🤖  ML Model"],
        label_visibility="collapsed",
    )
    page = menu.split("  ")[1]

    st.divider()

    # Dataset status
    if st.session_state.data is not None:
        df = st.session_state.data
        st.markdown(
            f'<div class="status-card">'
            f'<div class="status-label">Active Dataset</div>'
            f'<div class="status-name">{st.session_state.filename}</div>'
            f'<span class="stat-badge">{df.shape[0]:,} × {df.shape[1]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("⟳  Clear dataset", use_container_width=True):
            st.session_state.data = None
            st.session_state.filename = None
            st.rerun()
    else:
        st.markdown(
            '<p style="font-size:12px;color:#2d3748;line-height:1.7;">'
            'No dataset loaded.<br>Go to <b style="color:#4a5568;">Overview</b> to upload.</p>',
            unsafe_allow_html=True,
        )

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:10px;color:#1e2533;font-family:\'JetBrains Mono\',monospace;'
        'letter-spacing:0.06em;">DATALENS v2.0 · STREAMLIT</p>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────
def no_data_warning():
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("📂  No dataset loaded yet. Head to **Overview** to upload a CSV.")


def section_header(label: str):
    st.markdown(
        f'<div class="section-pill"><span class="dot"></span>{label}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────
#  PAGE: Overview
# ─────────────────────────────────────────
if page == "Overview":
    st.markdown('<div class="page-title">Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Upload a CSV and inspect your dataset at a glance.</div>',
        unsafe_allow_html=True,
    )

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
<ul class="tip-list">
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
    st.markdown(
        '<div class="page-sub">Browse, filter, and search your raw data.</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.data is None:
        no_data_warning()
    else:
        df = st.session_state.data

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
    st.markdown(
        '<div class="page-sub">Explore distributions, correlations, and relationships.</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.data is None:
        no_data_warning()
    else:
        df = st.session_state.data
        num_cols = list(df.select_dtypes(include='number').columns)
        cat_cols = list(df.select_dtypes(include='object').columns)
        all_cols = list(df.columns)

        tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Correlation", "Scatter", "Box / Violin"])

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
    st.markdown(
        '<div class="page-sub">Train a baseline model, inspect metrics and feature importance.</div>',
        unsafe_allow_html=True,
    )

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
                        em_df = pd.DataFrame(
                            list(results["extra_metrics"].items()),
                            columns=["Metric", "Value"],
                        )
                        st.dataframe(em_df, use_container_width=True, hide_index=True)

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

                if results.get("confusion_matrix") is not None:
                    st.markdown("<br>", unsafe_allow_html=True)
                    section_header("Confusion matrix")
                    st.pyplot(results["confusion_matrix"])

                if results.get("residual_plot") is not None:
                    st.markdown("<br>", unsafe_allow_html=True)
                    section_header("Residuals vs predicted")
                    st.pyplot(results["residual_plot"])
