import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, r2_score, mean_absolute_error,
    mean_squared_error, f1_score, precision_score,
    recall_score, confusion_matrix,
)

# colours shared with visualization.py
BG      = "#0d0f14"
SURFACE = "#161922"
BORDER  = "#1e2130"
TEXT_PRI = "#e2e8f0"
TEXT_MUT = "#6b7280"
ACCENT  = "#2563eb"
ACCENT2 = "#60a5fa"


def _apply_dark(fig):
    fig.patch.set_facecolor(BG)
    for ax in fig.axes:
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=TEXT_MUT, labelsize=10)
        ax.xaxis.label.set_color(TEXT_MUT)
        ax.yaxis.label.set_color(TEXT_MUT)
        ax.title.set_color(TEXT_PRI)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.grid(color=BORDER, linewidth=0.5, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
    return fig


def run_ml_model(df: pd.DataFrame, target: str,
                 test_size: float = 0.2,
                 scale_features: bool = True) -> dict:
    """
    Train an auto-selected ML model and return a rich results dict.

    Returns a dict with keys:
        problem, model_name, score, train_rows, test_rows,
        extra_metrics, feature_importance, confusion_matrix,
        residual_plot, error
    """
    result = {
        "problem": None, "model_name": None, "score": None,
        "train_rows": None, "test_rows": None,
        "extra_metrics": None, "feature_importance": None,
        "confusion_matrix": None, "residual_plot": None,
        "error": None,
    }

    try:
        # ── 1. Prep ──────────────────────────────────────────────────────────
        df = df.copy().dropna()
        if df.empty:
            result["error"] = "No rows remain after dropping missing values."
            return result

        y_raw = df[target]
        X_raw = df.drop(columns=[target])

        # encode categoricals in X
        X = pd.get_dummies(X_raw, drop_first=True)
        feature_names = list(X.columns)

        # detect problem type
        if y_raw.dtype == "object" or y_raw.nunique() <= 20:
            problem = "Classification"
            le = LabelEncoder()
            y = le.fit_transform(y_raw.astype(str))
        else:
            problem = "Regression"
            y = y_raw.values

        result["problem"] = problem

        # ── 2. Split ─────────────────────────────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y, test_size=test_size, random_state=42
        )
        result["train_rows"] = len(X_train)
        result["test_rows"] = len(X_test)

        # ── 3. Scale ─────────────────────────────────────────────────────────
        if scale_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

        # ── 4. Model selection ───────────────────────────────────────────────
        if problem == "Classification":
            # prefer RF if enough data, else logistic
            if len(X_train) >= 100:
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                model_name = "Random Forest Classifier"
            else:
                model = LogisticRegression(max_iter=500, random_state=42)
                model_name = "Logistic Regression"
        else:
            if len(X_train) >= 100:
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model_name = "Random Forest Regressor"
            else:
                model = LinearRegression()
                model_name = "Linear Regression"

        result["model_name"] = model_name

        # ── 5. Train & evaluate ──────────────────────────────────────────────
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if problem == "Classification":
            score = accuracy_score(y_test, preds)
            result["score"] = score

            avg = "binary" if len(np.unique(y)) == 2 else "weighted"
            extra = {
                "F1 score":  round(f1_score(y_test, preds, average=avg, zero_division=0), 4),
                "Precision": round(precision_score(y_test, preds, average=avg, zero_division=0), 4),
                "Recall":    round(recall_score(y_test, preds, average=avg, zero_division=0), 4),
            }
            result["extra_metrics"] = extra

            # confusion matrix figure
            cm = confusion_matrix(y_test, preds)
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                linewidths=0.5, linecolor=BG,
                annot_kws={"size": 12, "color": TEXT_PRI},
                cbar_kws={"shrink": 0.7},
            )
            ax_cm.set_xlabel("Predicted", color=TEXT_MUT)
            ax_cm.set_ylabel("Actual", color=TEXT_MUT)
            ax_cm.set_title("Confusion matrix", color=TEXT_PRI)
            _apply_dark(fig_cm)
            plt.tight_layout()
            result["confusion_matrix"] = fig_cm

        else:
            score = r2_score(y_test, preds)
            result["score"] = score

            mae  = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            extra = {
                "MAE":  round(mae, 4),
                "RMSE": round(rmse, 4),
            }
            result["extra_metrics"] = extra

            # residuals plot
            residuals = y_test - preds
            fig_res, ax_res = plt.subplots(figsize=(8, 4))
            ax_res.scatter(preds, residuals, alpha=0.5, s=18, color=ACCENT, edgecolors="none")
            ax_res.axhline(0, color=ACCENT2, linewidth=1.5, linestyle="--")
            ax_res.set_xlabel("Predicted")
            ax_res.set_ylabel("Residual")
            ax_res.set_title("Residuals vs Predicted")
            _apply_dark(fig_res)
            plt.tight_layout()
            result["residual_plot"] = fig_res

        # ── 6. Feature importance ────────────────────────────────────────────
        if hasattr(model, "feature_importances_"):
            fi = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_,
            }).sort_values("Importance", ascending=False).reset_index(drop=True)
            result["feature_importance"] = fi
        elif hasattr(model, "coef_"):
            coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
            fi = pd.DataFrame({
                "Feature": feature_names,
                "Importance": np.abs(coefs),
            }).sort_values("Importance", ascending=False).reset_index(drop=True)
            result["feature_importance"] = fi

        return result

    except Exception as e:
        result["error"] = str(e)
        return result
