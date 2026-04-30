import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

BG, SURFACE, BORDER, TEXT_PRI, TEXT_MUT = "#0d0f14", "#161922", "#1e2130", "#e2e8f0", "#6b7280"
ACCENT, ACCENT2, GRID = "#2563eb", "#60a5fa", "#1e2130"
PALETTE = ["#2563eb", "#60a5fa", "#34d399", "#f59e0b", "#f472b6", "#a78bfa", "#fb923c", "#22d3ee"]

def _apply_dark(fig, ax_list=None):
    fig.patch.set_facecolor(BG)
    axes = ax_list or fig.axes
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=TEXT_MUT, labelsize=10)
        ax.xaxis.label.set_color(TEXT_MUT)
        ax.yaxis.label.set_color(TEXT_MUT)
        ax.title.set_color(TEXT_PRI)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.grid(color=GRID, linewidth=0.5, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
    return fig

def plot_distribution(df, column, bins=30, show_kde=True):
    fig, ax = plt.subplots(figsize=(9, 4))
    if df[column].dtype == "object" or df[column].nunique() <= 20:
        vc = df[column].value_counts().head(20)
        ax.barh(vc.index.astype(str), vc.values, color=ACCENT)
        ax.invert_yaxis()
    else:
        sns.histplot(df[column].dropna(), bins=bins, kde=show_kde, color=ACCENT, ax=ax)
    ax.set_title(f"{column} — Distribution")
    return _apply_dark(fig)

def plot_correlation(df, columns, method="pearson"):
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[columns].corr(method=method)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Heatmap")
    return _apply_dark(fig)

def plot_scatter(df, x, y, hue=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    hue_data = df[hue] if hue and hue != "— none —" else None
    sns.scatterplot(data=df, x=x, y=y, hue=hue_data, palette=PALETTE[:df[hue].nunique()] if hue_data is not None else None, ax=ax, s=60)
    ax.set_title(f"Scatter: {x} vs {y}")
    return _apply_dark(fig)

def plot_boxplot(df, column, group=None, kind="box"):
    fig, ax = plt.subplots(figsize=(9, 5))
    if kind == "violin":
        sns.violinplot(data=df, x=group, y=column, ax=ax, color=ACCENT)
    else:
        sns.boxplot(data=df, x=group, y=column, ax=ax, color=ACCENT2)
    ax.set_title(f"{column} grouped by {group}" if group else f"Boxplot of {column}")
    return _apply_dark(fig)

def plot_bar(df, cat_col, num_col=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    if num_col and num_col != "None":
        sns.barplot(data=df, x=cat_col, y=num_col, ax=ax, palette="Blues_d")
    else:
        df[cat_col].value_counts().head(15).plot(kind='bar', color=ACCENT, ax=ax)
    ax.set_title(f"Bar Chart: {cat_col}")
    return _apply_dark(fig)

def plot_line(df, x, y):
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=df, x=x, y=y, marker="o", color=ACCENT, ax=ax)
    ax.set_title(f"Line Plot: {x} vs {y}")
    return _apply_dark(fig)

def plot_pie(df, cat_col):
    fig, ax = plt.subplots(figsize=(7, 7))
    counts = df[cat_col].value_counts().head(10)
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=PALETTE, startangle=90, textprops={'color': TEXT_PRI})
    ax.set_title(f"Pie Chart: {cat_col}", color=TEXT_PRI)
    fig.patch.set_facecolor(BG)
    return fig

def plot_histogram(df, col):
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df[col].dropna(), bins=30, color=ACCENT2, ax=ax)
    ax.set_title(f"Histogram: {col}")
    return _apply_dark(fig)

def plot_pairplot(df):
    num_df = df.select_dtypes(include=np.number).dropna().iloc[:, :5] # Limit to 5 for performance
    grid = sns.pairplot(num_df, corner=True, plot_kws={'color': ACCENT})
    grid.fig.patch.set_facecolor(BG)
    for ax in grid.axes.flatten():
        if ax:
            ax.set_facecolor(SURFACE)
            ax.tick_params(colors=TEXT_MUT)
            ax.xaxis.label.set_color(TEXT_MUT)
            ax.yaxis.label.set_color(TEXT_MUT)
    return grid.fig
