import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np

# ── Dark theme palette ──────────────────────────────────────────────────────
BG       = "#0d0f14"
SURFACE  = "#161922"
BORDER   = "#1e2130"
TEXT_PRI = "#e2e8f0"
TEXT_MUT = "#6b7280"
ACCENT   = "#2563eb"
ACCENT2  = "#60a5fa"
GRID     = "#1e2130"

PALETTE = [
    "#2563eb", "#60a5fa", "#34d399", "#f59e0b",
    "#f472b6", "#a78bfa", "#fb923c", "#22d3ee",
]


def _apply_dark(fig, ax_list=None):
    """Apply the dark theme to a figure and its axes."""
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


def plot_distribution(df: pd.DataFrame, column: str, bins: int = 30, show_kde: bool = True):
    """Histogram (numeric) or horizontal bar chart (categorical)."""
    fig, ax = plt.subplots(figsize=(9, 4))

    if df[column].dtype == "object" or df[column].nunique() <= 20:
        vc = df[column].value_counts().head(20)
        bars = ax.barh(vc.index.astype(str), vc.values, color=ACCENT, alpha=0.85, height=0.65)
        ax.invert_yaxis()
        ax.set_xlabel("Count")
        ax.set_title(f"{column}  —  value counts")
        # value labels
        for bar in bars:
            w = bar.get_width()
            ax.text(w * 1.01, bar.get_y() + bar.get_height() / 2,
                    f"{int(w):,}", va="center", fontsize=9, color=TEXT_MUT)
    else:
        ax.hist(df[column].dropna(), bins=bins, color=ACCENT, alpha=0.75, edgecolor=BG, linewidth=0.5)
        if show_kde:
            ax2 = ax.twinx()
            ax2.set_facecolor(SURFACE)
            ax2.tick_params(colors=TEXT_MUT, labelsize=9)
            ax2.yaxis.label.set_color(TEXT_MUT)
            for spine in ax2.spines.values():
                spine.set_edgecolor(BORDER)
            kde_data = df[column].dropna()
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(kde_data)
            xs = np.linspace(kde_data.min(), kde_data.max(), 300)
            ax2.plot(xs, kde(xs), color=ACCENT2, linewidth=2)
            ax2.set_ylabel("Density", color=TEXT_MUT)
            ax2.yaxis.set_tick_params(labelcolor=TEXT_MUT)
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        ax.set_title(f"{column}  —  distribution")

    plt.tight_layout()
    _apply_dark(fig)
    return fig


def plot_correlation(df: pd.DataFrame, columns: list, method: str = "pearson"):
    """Heatmap of correlation matrix."""
    corr = df[columns].corr(method=method)
    n = len(columns)
    size = max(6, min(12, n * 0.9))
    fig, ax = plt.subplots(figsize=(size, size * 0.85))

    # custom diverging cmap on dark bg
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(
        corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
        annot=True, fmt=".2f", linewidths=0.4,
        linecolor=BG, square=True, ax=ax,
        annot_kws={"size": 9, "color": TEXT_PRI},
        cbar_kws={"shrink": 0.6},
    )
    ax.set_title(f"{method.capitalize()} correlation", fontsize=13)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    plt.tight_layout()
    _apply_dark(fig)
    return fig


def plot_scatter(df: pd.DataFrame, x: str, y: str, hue: str | None = None):
    """Scatter plot with optional hue."""
    fig, ax = plt.subplots(figsize=(9, 5))

    if hue and hue in df.columns:
        categories = df[hue].dropna().unique()
        for i, cat in enumerate(categories):
            mask = df[hue] == cat
            ax.scatter(df.loc[mask, x], df.loc[mask, y],
                       label=str(cat), color=PALETTE[i % len(PALETTE)],
                       alpha=0.65, s=22, edgecolors="none")
        ax.legend(title=hue, fontsize=9, title_fontsize=9,
                  facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT_PRI)
    else:
        ax.scatter(df[x], df[y], color=ACCENT, alpha=0.55, s=20, edgecolors="none")

    # trend line
    try:
        z = np.polyfit(df[x].dropna(), df[y].dropna(), 1)
        p = np.poly1d(z)
        xs = np.linspace(df[x].min(), df[x].max(), 200)
        ax.plot(xs, p(xs), color=ACCENT2, linewidth=1.5, linestyle="--", alpha=0.7)
    except Exception:
        pass

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{x}  ×  {y}")
    plt.tight_layout()
    _apply_dark(fig)
    return fig


def plot_boxplot(df: pd.DataFrame, column: str, group: str | None = None, kind: str = "box"):
    """Box or violin plot, optionally grouped."""
    fig, ax = plt.subplots(figsize=(9, 5))

    plot_data = df[[column] + ([group] if group else [])].dropna()

    if kind == "violin":
        if group:
            sns.violinplot(data=plot_data, x=group, y=column, ax=ax,
                           palette=PALETTE[:plot_data[group].nunique()],
                           inner="box", linewidth=0.8)
        else:
            sns.violinplot(data=plot_data, y=column, ax=ax, color=ACCENT,
                           inner="box", linewidth=0.8)
    else:
        bp_props = dict(
            boxprops=dict(color=ACCENT2, facecolor=SURFACE),
            medianprops=dict(color=ACCENT, linewidth=2),
            whiskerprops=dict(color=TEXT_MUT),
            capprops=dict(color=TEXT_MUT),
            flierprops=dict(marker="o", color=ACCENT2, alpha=0.4, markersize=4),
        )
        if group:
            groups = [plot_data[column][plot_data[group] == g] for g in plot_data[group].unique()]
            labels = list(plot_data[group].unique())
            ax.boxplot(groups, labels=labels, patch_artist=True, **bp_props)
        else:
            ax.boxplot(plot_data[column], patch_artist=True, **bp_props)

    ax.set_title(f"{column}" + (f" grouped by {group}" if group else ""))
    plt.tight_layout()
    _apply_dark(fig)
    return fig
