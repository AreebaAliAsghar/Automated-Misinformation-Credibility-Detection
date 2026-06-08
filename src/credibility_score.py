CREDIBILITY_MAP = {
    "true": 1.00,
    "mostly-true": 0.80,
    "half-true": 0.60,
    "barely-true": 0.40,
    "false": 0.20,
    "pants-fire": 0.00
}

def get_credibility_score(predicted_label):
    return CREDIBILITY_MAP.get(str(predicted_label).lower(), 0.50)

def get_risk_score(predicted_label):
    return 1 - get_credibility_score(predicted_label)

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