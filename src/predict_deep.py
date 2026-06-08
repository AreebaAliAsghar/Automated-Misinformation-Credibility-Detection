import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from linguistic_report import generate_linguistic_report, calculate_linguistic_risk
from credibility_score import get_credibility_score, get_risk_score, get_final_category

MODEL_PATH = "models/deep_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_deep_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

def predict_news_deep(text):
    tokenizer, model = load_deep_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs.logits, dim=1)[0]

    id2label = model.config.id2label
    probability_dict = {}

    for i, prob in enumerate(probabilities):
        label = str(id2label[i]).lower()
        probability_dict[label] = float(prob.item())

    predicted_id = int(torch.argmax(probabilities).item())
    predicted_label = str(id2label[predicted_id]).lower()
    model_confidence = float(probabilities[predicted_id].item())

    credibility_score = get_credibility_score(predicted_label)
    model_risk = get_risk_score(predicted_label)
    linguistic_risk = calculate_linguistic_risk(text)

    final_risk = (model_risk * 0.85) + (linguistic_risk * 0.15)
    final_category = get_final_category(final_risk)

    report = generate_linguistic_report(text)

    return {
        "model_used": "DistilBERT 6-Class LIAR Credibility Model",
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
    result = predict_news_deep(sample_text)
    print(result)