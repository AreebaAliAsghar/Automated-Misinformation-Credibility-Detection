import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# Semantic ML model import
try:
    from predict_semantic_ml import predict_semantic_news
except Exception:
    predict_semantic_news = None

# Deep learning model import
from predict_deep import predict_news_deep


def get_prediction(text, model_type):
    if model_type == "Deep Learning Model":
        return predict_news_deep(text)

    elif model_type == "Semantic ML Model":
        if predict_semantic_news is None:
            raise ImportError(
                "predict_semantic_ml.py was not found or has an import error. "
                "Make sure the semantic ML prediction file exists inside the src folder."
            )
        return predict_semantic_news(text)

    else:
        raise ValueError("Invalid model type selected.")


def get_category_style(category):
    styles = {
        "Likely Real": {
            "box_bg": "#ecfdf5",
            "box_text": "#065f46",
            "border": "#10b981",
            "bar": "#10b981"
        },
        "Mostly Credible": {
            "box_bg": "#eff6ff",
            "box_text": "#1e40af",
            "border": "#3b82f6",
            "bar": "#3b82f6"
        },
        "Uncertain / Needs Verification": {
            "box_bg": "#fffbeb",
            "box_text": "#92400e",
            "border": "#f59e0b",
            "bar": "#f59e0b"
        },
        "Suspicious": {
            "box_bg": "#fff7ed",
            "box_text": "#9a3412",
            "border": "#f97316",
            "bar": "#f97316"
        },
        "Likely Fake": {
            "box_bg": "#fef2f2",
            "box_text": "#991b1b",
            "border": "#ef4444",
            "bar": "#ef4444"
        }
    }

    return styles.get(category, styles["Uncertain / Needs Verification"])