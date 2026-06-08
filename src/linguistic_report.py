EMOTIONAL_WORDS = [
    "shocking", "unbelievable", "breaking", "exposed", "secret", "urgent",
    "miracle", "danger", "warning", "terrifying", "amazing", "hate",
    "destroy", "scandal"
]

CLICKBAIT_PHRASES = [
    "you won't believe", "watch now", "share now", "must see",
    "doctors hate", "secret trick", "what happened next", "before it's too late"
]

def check_capital_words(text):
    words = text.split()
    return len([word for word in words if word.isupper() and len(word) > 2])

def check_exclamation_marks(text):
    return text.count("!")

def check_emotional_words(text):
    text_lower = text.lower()
    return [word for word in EMOTIONAL_WORDS if word in text_lower]

def check_clickbait_phrases(text):
    text_lower = text.lower()
    return [phrase for phrase in CLICKBAIT_PHRASES if phrase in text_lower]

def generate_linguistic_report(text):
    reasons = []

    capital_count = check_capital_words(text)
    exclamation_count = check_exclamation_marks(text)
    emotional_words = check_emotional_words(text)
    clickbait_phrases = check_clickbait_phrases(text)

    if capital_count >= 2:
        reasons.append("The text contains multiple fully capitalized words.")

    if exclamation_count >= 2:
        reasons.append("The text uses too many exclamation marks.")

    if len(emotional_words) > 0:
        reasons.append("Emotional or dramatic words detected: " + ", ".join(emotional_words))

    if len(clickbait_phrases) > 0:
        reasons.append("Clickbait-style phrases detected: " + ", ".join(clickbait_phrases))

    if len(reasons) == 0:
        reasons.append("No major suspicious linguistic pattern detected.")

    return reasons

def calculate_linguistic_risk(text):
    score = 0

    if check_capital_words(text) >= 2:
        score += 0.25

    if check_exclamation_marks(text) >= 2:
        score += 0.25

    if len(check_emotional_words(text)) > 0:
        score += 0.25

    if len(check_clickbait_phrases(text)) > 0:
        score += 0.25

    return min(score, 1.0)