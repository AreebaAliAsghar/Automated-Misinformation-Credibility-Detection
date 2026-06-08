import joblib
from sentence_transformers import SentenceTransformer

from linguistic_report import generate_linguistic_report, calculate_linguistic_risk


MODEL_PATH = "models/semantic_ml_model.pkl"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


RISK_MAP = {
    "true": 0.00,
    "mostly-true": 0.20,
    "half-true": 0.40,
    "barely-true": 0.60,
    "false": 0.80,
    "pants-fire": 1.00
}


def get_final_category(final_risk):
    if final_risk <= 0.25:
        return "Likely Real"
    elif final_risk <= 0.45:
        return "Mostly Credible"
    elif final_risk <= 0.60:
        return "Uncertain / Needs Verification"
    elif final_risk <= 0.80:
        return "Suspicious"
    else:
        return "Likely Fake"


def calculate_soft_model_risk(probability_dict):
    model_risk = 0

    for label, prob in probability_dict.items():
        label = str(label).lower()
        model_risk += float(prob) * RISK_MAP.get(label, 0.50)

    return model_risk


def load_semantic_model():
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    classifier = joblib.load(MODEL_PATH)

    return embedding_model, classifier


def predict_semantic_news(text):
    embedding_model, classifier = load_semantic_model()

    text_embedding = embedding_model.encode(
        [text],
        convert_to_numpy=True
    )

    probabilities = classifier.predict_proba(text_embedding)[0]
    class_names = classifier.classes_

    probability_dict = {
        str(label): float(prob)
        for label, prob in zip(class_names, probabilities)
    }

    predicted_label = str(classifier.predict(text_embedding)[0])
    model_confidence = float(probability_dict[predicted_label])

    model_risk = calculate_soft_model_risk(probability_dict)
    credibility_score = 1 - model_risk

    linguistic_risk = calculate_linguistic_risk(text)

    final_risk = (model_risk * 0.90) + (linguistic_risk * 0.10)

    final_category = get_final_category(final_risk)

    if model_confidence < 0.30:
        final_category = "Uncertain / Needs Verification"

    report = generate_linguistic_report(text)

    return {
        "model_used": "Sentence-BERT Embeddings + Softmax Logistic Regression",
        "predicted_label": predicted_label,
        "model_confidence": model_confidence,
        "probability_dict": probability_dict,
        "credibility_score": credibility_score,
        "model_risk": model_risk,
        "linguistic_risk": linguistic_risk,
        "final_risk": final_risk,
        "final_category": final_category,
        "report": report
    }