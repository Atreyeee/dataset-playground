import streamlit as st
import pandas as pd
import numpy as np

from utils.load_data import load_csv, load_sample_dataset
from utils.summary import get_summary, generate_insights
from utils.visualization import plot_distribution, plot_correlation, plot_scatter, plot_boxplot, plot_bar, plot_line, plot_pie, plot_histogram, plot_pairplot
from utils.ml_model import run_ml_model

st.set_page_config(page_title="DataLens — Dataset Playground", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset & Base ─────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #111827;
}

/* ── Top Toolbar (Deploy + 3-dot menu) ───────────────────── */
[data-testid="stToolbar"],
header[data-testid="stHeader"],
.stAppDeployButton,
[data-testid="stHeader"] {
    background-color: #FACC15 !important;
}

header[data-testid="stHeader"] {
    background: #FACC15 !important;
    border-bottom: 1px solid #EAB308 !important;
}

/* Toolbar buttons inside header */
[data-testid="stToolbar"] button,
[data-testid="stToolbarActions"] button,
[data-testid="stDecoration"] {
    background: transparent !important;
    color: #111827 !important;
}

[data-testid="stDecoration"] {
    background: #FACC15 !important;
}

/* The top status/deploy bar */
.stAppDeployButton > button {
    background: #111827 !important;
    color: #FACC15 !important;
    border: none !important;
    box-shadow: none !important;
}

.stAppDeployButton > button:hover {
    background: #374151 !important;
    transform: none !important;
}

/* Background */
.stApp {
    background-color: #F3F4F8;
}

/* ── Sidebar ──────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E5E7EB;
}

[data-testid="stSidebar"] > div:first-child {
    padding: 0;
}

/* Hide default sidebar header */
[data-testid="stSidebarNav"] { display: none; }

/* Sidebar radio — navigation pills */
[data-testid="stSidebar"] .stRadio > div {
    gap: 4px;
    flex-direction: column;
}

[data-testid="stSidebar"] .stRadio label {
    background: transparent;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 14px;
    font-weight: 500;
    color: #6B7280;
    cursor: pointer;
    transition: all 0.15s ease;
    border: none !important;
    width: 100%;
    display: flex;
    align-items: center;
    gap: 8px;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background: #EFF6FF;
    color: #2563EB;
}

[data-testid="stSidebar"] .stRadio [aria-checked="true"] + label,
[data-testid="stSidebar"] .stRadio input:checked + div label {
    background: #EFF6FF;
    color: #2563EB;
}

[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] [aria-checked="true"] ~ div {
    color: #2563EB;
    font-weight: 600;
}

/* ── Brand Logo ───────────────────────────────────────────── */
.brand {
    font-size: 17px;
    font-weight: 700;
    color: #111827;
    letter-spacing: -0.5px;
    padding: 28px 20px 4px 20px;
    line-height: 1;
}

.brand span { color: #2563EB; }
.brand-dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #FACC15;
    margin-left: 2px;
    vertical-align: middle;
    margin-bottom: 3px;
}

/* ── Status Card (Sidebar) ────────────────────────────────── */
.status-card {
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 14px 16px;
    margin: 0 12px 12px 12px;
}

.status-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #9CA3AF;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.status-name {
    font-size: 13px;
    font-weight: 600;
    color: #111827;
    margin-bottom: 8px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.stat-badge {
    display: inline-block;
    background: #EFF6FF;
    color: #2563EB;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 20px;
    font-family: 'DM Mono', monospace;
}

/* ── Page Title ───────────────────────────────────────────── */
.page-header {
    margin-bottom: 32px;
    padding-bottom: 20px;
    border-bottom: 1px solid #E5E7EB;
}

.page-title {
    font-size: 28px;
    font-weight: 700;
    color: #111827;
    letter-spacing: -0.5px;
    line-height: 1.2;
    margin-bottom: 6px;
}

.page-subtitle {
    font-size: 14px;
    color: #6B7280;
    font-weight: 400;
}

/* ── Section Pill ─────────────────────────────────────────── */
.section-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    font-weight: 600;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 14px;
}

.dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    background: #FACC15;
    border-radius: 2px;
}

/* ── Cards ────────────────────────────────────────────────── */
.card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

/* ── Metric Cards ─────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 20px 22px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

[data-testid="stMetricLabel"] {
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #9CA3AF !important;
}

[data-testid="stMetricValue"] {
    font-size: 28px !important;
    font-weight: 700 !important;
    color: #111827 !important;
    font-family: 'DM Mono', monospace !important;
    letter-spacing: -1px;
}

/* ── Score Card (ML) ──────────────────────────────────────── */
.score-card {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
    border-radius: 14px;
    padding: 32px 24px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(37,99,235,0.25);
    color: white;
}

.score-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    opacity: 0.8;
    margin-bottom: 12px;
}

.score-value {
    font-size: 52px;
    font-weight: 700;
    font-family: 'DM Mono', monospace;
    line-height: 1;
    margin-bottom: 10px;
    letter-spacing: -2px;
}

.score-type {
    font-size: 12px;
    opacity: 0.75;
    font-weight: 500;
    background: rgba(255,255,255,0.15);
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
}

/* ── Buttons ──────────────────────────────────────────────── */
.stButton > button {
    background: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.01em;
    transition: all 0.15s ease !important;
    box-shadow: 0 1px 3px rgba(37,99,235,0.3) !important;
    cursor: pointer;
}

.stButton > button:hover {
    background: #1D4ED8 !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.35) !important;
    transform: translateY(-1px);
}

.stButton > button:active {
    transform: translateY(0px) !important;
}

/* Secondary / clear button */
.stButton > button[kind="secondary"],
button[data-testid*="clear"] {
    background: #FFFFFF !important;
    color: #374151 !important;
    border: 1px solid #D1D5DB !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
}

.stButton > button[kind="secondary"]:hover {
    background: #F9FAFB !important;
    border-color: #9CA3AF !important;
    transform: none;
}

/* ── File Uploader ────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 2px dashed #D1D5DB;
    border-radius: 12px;
    padding: 20px;
    transition: border-color 0.2s;
}

[data-testid="stFileUploader"]:hover {
    border-color: #2563EB;
}

/* ── Select / Input ───────────────────────────────────────── */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    border-radius: 8px !important;
    border-color: #D1D5DB !important;
    background: #FFFFFF !important;
    font-size: 14px !important;
}

[data-baseweb="select"]:focus-within > div,
[data-baseweb="input"]:focus-within > div {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
}

/* ── Dataframe / Table ────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden;
    border: 1px solid #E5E7EB !important;
}

/* ── Alert / Info boxes ───────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 4px !important;
    font-size: 14px;
}

.stSuccess {
    background: #F0FDF4 !important;
    border-color: #22C55E !important;
    color: #166534 !important;
}

.stInfo {
    background: #EFF6FF !important;
    border-color: #2563EB !important;
    color: #1E40AF !important;
}

.stError {
    background: #FEF2F2 !important;
    border-color: #EF4444 !important;
}

/* ── Spinner ──────────────────────────────────────────────── */
[data-testid="stSpinner"] {
    color: #2563EB !important;
}

/* ── Divider ──────────────────────────────────────────────── */
hr {
    border: none;
    border-top: 1px solid #E5E7EB;
    margin: 16px 0;
}

/* ── Empty State ──────────────────────────────────────────── */
.empty-state {
    background: #FFFFFF;
    border: 1px dashed #D1D5DB;
    border-radius: 14px;
    padding: 56px 40px;
    text-align: center;
    margin-top: 16px;
}

.empty-state-icon {
    font-size: 36px;
    margin-bottom: 16px;
}

.empty-state-title {
    font-size: 18px;
    font-weight: 600;
    color: #374151;
    margin-bottom: 8px;
}

.empty-state-desc {
    font-size: 14px;
    color: #9CA3AF;
    max-width: 360px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Chart wrapper ────────────────────────────────────────── */
.chart-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    margin-top: 20px;
}

.chart-title {
    font-size: 14px;
    font-weight: 600;
    color: #374151;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid #F3F4F6;
}

/* ── Pyplot wrapper ───────────────────────────────────────── */
[data-testid="stImage"], .stPyplot {
    border-radius: 10px;
    overflow: hidden;
}

/* ── Main content padding ─────────────────────────────────── */
.block-container {
    padding: 52px 40px 40px 40px !important;
    max-width: 1280px;
}

/* ── Selectbox label ──────────────────────────────────────── */
.stSelectbox label, .stFileUploader label {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #374151 !important;
    margin-bottom: 6px !important;
}

/* ── Sidebar nav label hidden ─────────────────────────────── */
[data-testid="stSidebar"] .stRadio > label {
    display: none;
}

/* ── Sidebar padding ──────────────────────────────────────── */
[data-testid="stSidebar"] > div > div {
    padding: 0 8px;
}

/* ── Highlight max cell in dataframe ─────────────────────── */
.stDataFrame [data-testid="stDataFrameResizable"] {
    font-family: 'DM Mono', monospace;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────
if "data" not in st.session_state:
    st.session_state.data = None
if "filename" not in st.session_state:
    st.session_state.filename = None

# ─── Helpers ───────────────────────────────────────────────────────────────────
def no_data_warning():
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">📂</div>
        <div class="empty-state-title">No dataset loaded yet</div>
        <div class="empty-state-desc">Head to <strong>Overview</strong> to upload a CSV file or load the sample Titanic dataset to get started.</div>
    </div>
    """, unsafe_allow_html=True)

def section_header(label: str):
    st.markdown(f'<div class="section-pill"><span class="dot"></span>{label}</div>', unsafe_allow_html=True)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand">Dataset<span> Playground</span><span class="brand-dot"></span></div>', unsafe_allow_html=True)
    st.markdown("<div style='padding: 0 12px;margin-bottom:8px;margin-top:20px;font-size:11px;font-weight:600;color:#9CA3AF;letter-spacing:0.08em;text-transform:uppercase;'>Navigation</div>", unsafe_allow_html=True)

    menu = st.radio("Navigation", ["◈  Overview", "📋  Data Table", "📊  Visualize", "🤖  ML Model"], label_visibility="collapsed")
    page = menu.split("  ")[1]
    st.divider()

    if st.session_state.data is not None:
        df = st.session_state.data
        st.markdown(f'<div class="status-card"><div class="status-label">Active Dataset</div><div class="status-name">{st.session_state.filename}</div><span class="stat-badge">{df.shape[0]:,} × {df.shape[1]}</span></div>', unsafe_allow_html=True)
        if st.button("⟳  Clear dataset", use_container_width=True):
            st.session_state.data = None
            st.session_state.filename = None
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Overview
# ─────────────────────────────────────────────────────────────────────────────
if page == "Overview":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Overview</div>
        <div class="page-subtitle">Upload your dataset and explore key statistics at a glance.</div>
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_info = st.columns([1.2, 1], gap="large")

    with col_upload:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        section_header("Upload Dataset")
        file = st.file_uploader("Drop a CSV file here", type=["csv"], label_visibility="collapsed")
        if file:
            df, err = load_csv(file)
            if df is not None:
                st.session_state.data = df
                st.session_state.filename = file.name
                st.success(f"✓  Loaded **{file.name}**")
            else:
                st.error(err)

        st.markdown("<div style='margin-top:16px; margin-bottom:8px; font-size:13px; color:#9CA3AF; text-align:center;'>— or —</div>", unsafe_allow_html=True)
        if st.button("Load Sample Dataset (Titanic)", use_container_width=True):
            df, err = load_sample_dataset()
            if df is not None:
                st.session_state.data = df
                st.session_state.filename = "titanic.csv"
                st.success("✓  Sample dataset loaded!")
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.data is not None:
        df = st.session_state.data

        # Metrics Row
        st.markdown("<div style='margin: 28px 0 16px 0;'>", unsafe_allow_html=True)
        section_header("Dataset Summary")
        summary = get_summary(df)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Rows", f"{summary['shape'][0]:,}")
        m2.metric("Columns", str(summary['shape'][1]))
        m3.metric("Missing Values", str(int(summary['missing'].sum())))
        m4.metric("Numeric Columns", str(len(df.select_dtypes(include='number').columns)))
        st.markdown("</div>", unsafe_allow_html=True)

        # Insights
        st.markdown("<div style='margin-top: 28px;'>", unsafe_allow_html=True)
        section_header("Auto Insights")
        insights = generate_insights(df)
        if insights:
            for insight in insights:
                st.info(insight)
        else:
            st.success("✓  Data looks clean — no major missing values or high skewness detected.")
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        with col_info:
            st.markdown("""
            <div style="background:#FFFFFF;border:1px solid #E5E7EB;border-radius:12px;padding:28px 24px;box-shadow:0 1px 3px rgba(0,0,0,0.05);">
                <div style="font-size:13px;font-weight:600;color:#9CA3AF;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:16px;">How it works</div>
                <div style="display:flex;flex-direction:column;gap:14px;">
                    <div style="display:flex;gap:12px;align-items:flex-start;">
                        <div style="background:#EFF6FF;color:#2563EB;width:28px;height:28px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px;flex-shrink:0;">1</div>
                        <div><div style="font-size:14px;font-weight:600;color:#111827;">Upload a CSV</div><div style="font-size:13px;color:#6B7280;margin-top:2px;">Drag & drop your dataset or use the sample.</div></div>
                    </div>
                    <div style="display:flex;gap:12px;align-items:flex-start;">
                        <div style="background:#EFF6FF;color:#2563EB;width:28px;height:28px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px;flex-shrink:0;">2</div>
                        <div><div style="font-size:14px;font-weight:600;color:#111827;">Explore & Clean</div><div style="font-size:13px;color:#6B7280;margin-top:2px;">View, filter, and clean data in the Data Table tab.</div></div>
                    </div>
                    <div style="display:flex;gap:12px;align-items:flex-start;">
                        <div style="background:#EFF6FF;color:#2563EB;width:28px;height:28px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px;flex-shrink:0;">3</div>
                        <div><div style="font-size:14px;font-weight:600;color:#111827;">Visualize</div><div style="font-size:13px;color:#6B7280;margin-top:2px;">Generate 9 chart types from your data instantly.</div></div>
                    </div>
                    <div style="display:flex;gap:12px;align-items:flex-start;">
                        <div style="background:#EFF6FF;color:#2563EB;width:28px;height:28px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px;flex-shrink:0;">4</div>
                        <div><div style="font-size:14px;font-weight:600;color:#111827;">Run ML Models</div><div style="font-size:13px;color:#6B7280;margin-top:2px;">Auto-train and compare multiple models in one click.</div></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Data Table
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Data Table":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Data Table</div>
        <div class="page-subtitle">Inspect, clean, and manage your dataset.</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.data is None:
        no_data_warning()
    else:
        df = st.session_state.data

        # Cleaning tools in a card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        section_header("Data Cleaning Tools")
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            if st.button("Drop Duplicates", use_container_width=True):
                st.session_state.data = df.drop_duplicates()
                st.success("✓  Duplicates removed!")
                st.rerun()
        with c2:
            if st.button("Fill Missing (Mean/Mode)", use_container_width=True):
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].mean())
                    else:
                        df[col] = df[col].fillna(df[col].mode()[0])
                st.session_state.data = df
                st.success("✓  Missing values filled!")
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Table
        st.markdown("<div style='margin-top: 8px;'>", unsafe_allow_html=True)
        section_header(f"Dataset — {df.shape[0]:,} rows × {df.shape[1]} columns")
        st.markdown('<div style="background:#FFFFFF;border:1px solid #E5E7EB;border-radius:12px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.05);">', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, height=480)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Visualize
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Visualize":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Visualize</div>
        <div class="page-subtitle">Generate charts and uncover patterns in your data.</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.data is None:
        no_data_warning()
    else:
        df = st.session_state.data
        num_cols = list(df.select_dtypes(include='number').columns)
        cat_cols = list(df.select_dtypes(include='object').columns)
        all_cols = list(df.columns)

        # Controls card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        section_header("Chart Configuration")

        plot_type = st.selectbox("Select Plot Type", [
            "Distribution", "Correlation", "Scatter", "Box / Violin", "Bar", "Line", "Pie", "Histogram", "Pairplot"
        ])

        fig = None

        if plot_type == "Distribution":
            col = st.selectbox("Column", all_cols)
            if st.button("Generate Chart", use_container_width=False):
                fig = plot_distribution(df, col)

        elif plot_type == "Correlation":
            if len(num_cols) < 2:
                st.warning("⚠️  Need 2 or more numeric columns to generate a correlation matrix.")
            elif st.button("Generate Chart"):
                fig = plot_correlation(df, num_cols)

        elif plot_type == "Scatter":
            sc1, sc2, sc3 = st.columns(3)
            with sc1: x = st.selectbox("X-Axis", num_cols, index=0)
            with sc2: y = st.selectbox("Y-Axis", num_cols, index=min(1, len(num_cols)-1))
            with sc3: hue = st.selectbox("Color By", ["— none —"] + cat_cols)
            if st.button("Generate Chart"):
                fig = plot_scatter(df, x, y, hue)

        elif plot_type == "Box / Violin":
            bv1, bv2, bv3 = st.columns(3)
            with bv1: col = st.selectbox("Numeric Column", num_cols)
            with bv2: grp = st.selectbox("Group By (Optional)", ["None"] + cat_cols)
            with bv3: kind = st.selectbox("Type", ["Box", "Violin"]).lower()
            if st.button("Generate Chart"):
                fig = plot_boxplot(df, col, grp if grp != "None" else None, kind)

        elif plot_type == "Bar":
            ba1, ba2 = st.columns(2)
            with ba1: cat = st.selectbox("Category Column", cat_cols if cat_cols else all_cols)
            with ba2: num = st.selectbox("Numeric Column (Optional)", ["None"] + num_cols)
            if st.button("Generate Chart"):
                fig = plot_bar(df, cat, num if num != "None" else None)

        elif plot_type == "Line":
            li1, li2 = st.columns(2)
            with li1: x = st.selectbox("X-Axis", all_cols)
            with li2: y = st.selectbox("Y-Axis", num_cols)
            if st.button("Generate Chart"):
                fig = plot_line(df, x, y)

        elif plot_type == "Pie":
            cat = st.selectbox("Categorical Column", cat_cols if cat_cols else all_cols)
            if st.button("Generate Chart"):
                fig = plot_pie(df, cat)

        elif plot_type == "Histogram":
            col = st.selectbox("Numeric Column", num_cols)
            if st.button("Generate Chart"):
                fig = plot_histogram(df, col)

        elif plot_type == "Pairplot":
            st.markdown("<div style='font-size:13px;color:#6B7280;margin-bottom:12px;'>Generates a pairplot for all numeric columns. May take a few seconds.</div>", unsafe_allow_html=True)
            if st.button("Generate Chart"):
                fig = plot_pairplot(df)

        st.markdown('</div>', unsafe_allow_html=True)

        # Chart output card
        if fig:
            st.markdown(f"""
            <div class="chart-card">
                <div class="chart-title">📊 {plot_type} Chart</div>
            """, unsafe_allow_html=True)
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ML Model
# ─────────────────────────────────────────────────────────────────────────────
elif page == "ML Model":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">ML Model</div>
        <div class="page-subtitle">Auto-train and compare multiple machine learning models on your dataset.</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.data is None:
        no_data_warning()
    else:
        df = st.session_state.data

        st.markdown('<div class="card">', unsafe_allow_html=True)
        section_header("Model Configuration")
        target = st.selectbox("Target Column", df.columns)
        st.markdown("<div style='margin-top:4px;font-size:12px;color:#9CA3AF;'>Select the column you want to predict. The model type (classification or regression) is determined automatically.</div>", unsafe_allow_html=True)
        run = st.button("▶  Train & Compare Models", use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)

        if run:
            with st.spinner("Training models — this may take a moment..."):
                results = run_ml_model(df, target)

            if results.get("error"):
                st.error(f"Training failed: {results['error']}")
            else:
                st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
                section_header("Results")
                metric_name = "Accuracy" if results["problem"] == "Classification" else "R² Score"

                r1, r2 = st.columns([1, 2], gap="large")

                with r1:
                    st.markdown(f"""
                    <div class="score-card">
                        <div class="score-label">Best Model · {results['model_name']}</div>
                        <div class="score-value">{results['score']:.2%}</div>
                        <div class="score-type">{results['problem']} · {metric_name}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with r2:
                    st.markdown("""
                    <div style="font-size:14px;font-weight:600;color:#374151;margin-bottom:12px;">Model Comparison Scoreboard</div>
                    """, unsafe_allow_html=True)
                    scores_df = pd.DataFrame(list(results["all_model_scores"].items()), columns=["Model", metric_name])
                    st.markdown('<div style="background:#FFFFFF;border:1px solid #E5E7EB;border-radius:10px;overflow:hidden;">', unsafe_allow_html=True)
                    st.dataframe(
                        scores_df.style.highlight_max(subset=[metric_name], color='#DCFCE7'),
                        use_container_width=True,
                        hide_index=True
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
                c1, c2 = st.columns(2, gap="large")

                with c1:
                    if results.get("confusion_matrix"):
                        st.markdown('<div class="chart-card"><div class="chart-title">Confusion Matrix</div>', unsafe_allow_html=True)
                        st.pyplot(results["confusion_matrix"])
                        st.markdown('</div>', unsafe_allow_html=True)
                    elif results.get("residual_plot"):
                        st.markdown('<div class="chart-card"><div class="chart-title">Residual Plot</div>', unsafe_allow_html=True)
                        st.pyplot(results["residual_plot"])
                        st.markdown('</div>', unsafe_allow_html=True)

                with c2:
                    if results.get("feature_importance") is not None:
                        st.markdown('<div class="chart-card"><div class="chart-title">Feature Importance (Top 10)</div>', unsafe_allow_html=True)
                        st.dataframe(
                            results["feature_importance"].head(10),
                            use_container_width=True,
                            hide_index=True
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
