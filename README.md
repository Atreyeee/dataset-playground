# ◈ DataLens — Dataset Exploration Platform

A professional, dark-themed dataset playground built with Python & Streamlit.

## Features

| Page | What it does |
|------|-------------|
| **Overview** | Upload CSV, see shape / types / missing values / descriptive stats |
| **Data Table** | Filter rows by any column, search values, download filtered CSV |
| **Visualize** | Distribution · Correlation heatmap · Scatter (with trend line) · Box/Violin |
| **ML Model** | Auto-selects Random Forest or Linear/Logistic Regression · Feature importance · Confusion matrix / Residuals |

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech stack

- **Python 3.10+**
- Streamlit · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · SciPy

## What's new vs v1

- Dark professional theme with DM Sans / DM Mono fonts
- `load_csv` now surfaces real error messages (encoding fallback included)
- ML backend auto-upgrades to Random Forest when data is sufficient
- Feature importance bars rendered inline
- Confusion matrix & residuals plot included
- Scatter plots include an auto-fitted trend line
- KDE overlay on histograms via SciPy
- Correlation heatmap with method selector (Pearson / Spearman / Kendall)
- Fixed `__init__.py` naming bug from v1
- `plt.close()` called after every figure to prevent memory leaks
