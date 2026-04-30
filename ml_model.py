import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix

BG, SURFACE, BORDER, TEXT_PRI, TEXT_MUT = "#0d0f14", "#161922", "#1e2130", "#e2e8f0", "#6b7280"
ACCENT, ACCENT2 = "#2563eb", "#60a5fa"

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
    return fig

def run_ml_model(df: pd.DataFrame, target: str, test_size: float = 0.2, scale_features: bool = True) -> dict:
    result = {"problem": None, "model_name": None, "score": None, "train_rows": None, "test_rows": None,
              "feature_importance": None, "confusion_matrix": None, "residual_plot": None, "error": None, "all_model_scores": {}}

    try:
        df = df.copy().dropna()
        if df.empty:
            result["error"] = "No rows remain after dropping missing values."
            return result

        y_raw = df[target]
        X_raw = df.drop(columns=[target])
        X = pd.get_dummies(X_raw, drop_first=True)
        feature_names = list(X.columns)

        if y_raw.dtype == "object" or y_raw.nunique() <= 15:
            problem = "Classification"
            le = LabelEncoder()
            y = le.fit_transform(y_raw.astype(str))
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Random Forest Classifier": RandomForestClassifier(random_state=42),
                "Decision Tree Classifier": DecisionTreeClassifier(random_state=42)
            }
        else:
            problem = "Regression"
            y = y_raw.values
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest Regressor": RandomForestRegressor(random_state=42),
                "Decision Tree Regressor": DecisionTreeRegressor(random_state=42)
            }

        result["problem"] = problem
        X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=test_size, random_state=42)
        result["train_rows"], result["test_rows"] = len(X_train), len(X_test)

        if scale_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

        best_score = -float('inf')
        best_model = None

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = accuracy_score(y_test, preds) if problem == "Classification" else r2_score(y_test, preds)
            result["all_model_scores"][name] = round(score, 4)

            if score > best_score:
                best_score = score
                best_model = model
                result["model_name"] = name
                result["score"] = score
                best_preds = preds

        # Visualizations
        if problem == "Classification":
            cm = confusion_matrix(y_test, best_preds)
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm, annot_kws={"size": 12, "color": BG})
            ax_cm.set_title("Confusion Matrix", color=TEXT_PRI)
            result["confusion_matrix"] = _apply_dark(fig_cm)
        else:
            residuals = y_test - best_preds
            fig_res, ax_res = plt.subplots(figsize=(8, 4))
            ax_res.scatter(best_preds, residuals, alpha=0.5, color=ACCENT)
            ax_res.axhline(0, color=ACCENT2, linestyle="--")
            ax_res.set_title("Residuals vs Predicted", color=TEXT_PRI)
            result["residual_plot"] = _apply_dark(fig_res)

        # Feature Importance
        if hasattr(best_model, "feature_importances_"):
            result["feature_importance"] = pd.DataFrame({"Feature": feature_names, "Importance": best_model.feature_importances_}).sort_values("Importance", ascending=False)
        elif hasattr(best_model, "coef_"):
            coefs = best_model.coef_[0] if best_model.coef_.ndim > 1 else best_model.coef_
            result["feature_importance"] = pd.DataFrame({"Feature": feature_names, "Importance": np.abs(coefs)}).sort_values("Importance", ascending=False)

        return result
    except Exception as e:
        result["error"] = str(e)
        return result
