import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

LIAR_COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "speaker_job",
    "state", "party", "barely_true_counts", "false_counts",
    "half_true_counts", "mostly_true_counts", "pants_fire_counts", "context"
]

LIAR_LABELS = [
    "pants-fire",
    "false",
    "barely-true",
    "half-true",
    "mostly-true",
    "true"
]

def load_liar_dataset(file_path):
    return pd.read_csv(file_path, sep="\t", names=LIAR_COLUMNS)

def convert_label(label):
    label = str(label).strip().lower()
    if label in LIAR_LABELS:
        return label
    return "unknown"

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [word for word in words if word not in STOP_WORDS]
    return " ".join(words)

def preprocess_and_save(input_path, output_path):
    df = load_liar_dataset(input_path)
    df = df[["label", "statement"]].copy()
    df["target"] = df["label"].apply(convert_label)
    df = df[df["target"] != "unknown"]
    df = df.dropna(subset=["statement"])
    df["clean_text"] = df["statement"].apply(clean_text)
    df = df[df["clean_text"].str.strip() != ""]
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    preprocess_and_save("data/raw/train.tsv", "data/processed/train_clean.csv")
    preprocess_and_save("data/raw/valid.tsv", "data/processed/valid_clean.csv")
    preprocess_and_save("data/raw/test.tsv", "data/processed/test_clean.csv")