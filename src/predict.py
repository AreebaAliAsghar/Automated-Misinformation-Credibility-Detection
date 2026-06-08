import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import clean_text
from linguistic_report import generate_linguistic_report, calculate_linguistic_risk
from credibility_score import get_credibility_score, get_risk_score, get_final_category

def load_model_and_vectorizer():
    model = joblib.load("models/ml_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    return model, vectorizer

def predict_news(text):
    model, vectorizer = load_model_and_vectorizer()

    cleaned_text = clean_text(text)
    text_tfidf = vectorizer.transform([cleaned_text])

    probabilities = model.predict_proba(text_tfidf)[0]
    class_names = model.classes_

    probability_dict = {
        str(label): float(prob)
        for label, prob in zip(class_names, probabilities)
    }

    predicted_label = str(model.predict(text_tfidf)[0])
    model_confidence = probability_dict[predicted_label]

    credibility_score = get_credibility_score(predicted_label)
    model_risk = get_risk_score(predicted_label)
    linguistic_risk = calculate_linguistic_risk(text)

    final_risk = (model_risk * 0.85) + (linguistic_risk * 0.15)
    final_category = get_final_category(final_risk)

    report = generate_linguistic_report(text)

    return {
        "model_used": "TF-IDF + Multinomial Logistic Regression",
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

if __name__ == "__main__":
    sample_text = input("Enter news statement: ")
    result = predict_news(sample_text)
    print(result)