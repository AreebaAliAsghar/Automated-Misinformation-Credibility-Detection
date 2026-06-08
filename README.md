# 📰 Automated Misinformation Credibility Detection

## Project Overview

This project is an NLP-based misinformation credibility detection system. It analyzes a news headline or claim and predicts a **LIAR truthfulness label**. The system then converts that label into a **Credibility Score**, **Risk Score**, and **Final Credibility Category**.

Instead of only classifying news as `Real` or `Fake`, this project uses a more realistic 6-class credibility approach:

- `true`
- `mostly-true`
- `half-true`
- `barely-true`
- `false`
- `pants-fire`

This allows the system to handle in-between cases such as partially true, uncertain, or suspicious claims.

---

## Project Title

**A Hybrid Deep-Learning Approach to Automated Misinformation Detection**

---

## Main Features

- 6-class LIAR truthfulness classification
- Credibility score generation
- Risk meter
- Soft risk scoring using all class probabilities
- Linguistic report for suspicious writing patterns
- Semantic ML model using Sentence-BERT embeddings
- Deep learning model using DistilBERT
- Streamlit-based user interface
- Google Colab GPU support for training

---

## Dataset

The project uses the **LIAR Dataset**, which contains short political statements labeled with six truthfulness categories.

Dataset files used:

```text
train.tsv
valid.tsv
test.tsv
```

The dataset is preprocessed and saved into:

```text
data/processed/train_clean.csv
data/processed/valid_clean.csv
data/processed/test_clean.csv
```

---

## Truthfulness Labels

| LIAR Label | Credibility Score | Risk Score |
|---|---:|---:|
| true | 100% | 0% |
| mostly-true | 80% | 20% |
| half-true | 60% | 40% |
| barely-true | 40% | 60% |
| false | 20% | 80% |
| pants-fire | 0% | 100% |

---

## Final Output Categories

| Final Risk Score | Category |
|---|---|
| 0% - 25% | Likely Real |
| 26% - 45% | Mostly Credible |
| 46% - 60% | Uncertain / Needs Verification |
| 61% - 80% | Suspicious |
| 81% - 100% | Likely Fake |

---

## Models Used

### 1. Semantic Machine Learning Model

The traditional TF-IDF approach was replaced with a more context-aware technique:

```text
Sentence-BERT Embeddings + Softmax Logistic Regression
```

Sentence-BERT converts each news statement into a semantic vector, which captures meaning better than simple word frequency.

### 2. Deep Learning Model

The main deep learning model is:

```text
DistilBERT 6-Class Sequence Classification
```

DistilBERT is used to classify the input statement into one of the six LIAR truthfulness labels.

---

## Why Not Only TF-IDF?

TF-IDF mainly captures word frequency and word importance. It does not deeply understand context.

Example:

```text
"The minister confirmed the claim."
"The minister denied the claim."
```

Both sentences may share similar words, but their meanings are different. Sentence-BERT and DistilBERT are better suited for capturing such contextual differences.

---

## Soft Risk Scoring

Instead of depending only on the highest predicted label, the system calculates risk using the full probability distribution.

Example:

```text
false: 30%
barely-true: 25%
half-true: 20%
mostly-true: 15%
true: 10%
```

The system combines these probabilities to calculate a more stable risk score.

This is useful because LIAR labels are close to each other, especially:

```text
half-true
barely-true
mostly-true
```

---

## Linguistic Report

The system also checks suspicious writing patterns such as:

- Excessive capital letters
- Too many exclamation marks
- Emotional or dramatic words
- Clickbait-style phrases
- Urgent commands like “WATCH NOW” or “SHARE NOW”

The linguistic report is used as a supporting signal, not the main decision-maker.

---

## Final Risk Formula

The final risk score is calculated as:

```text
Final Risk = 90% Model Risk + 10% Linguistic Risk
```

This keeps the model prediction as the main factor while still considering suspicious language patterns.

---

## Project Workflow

```text
User enters news headline or claim
        ↓
Text is processed
        ↓
Semantic ML model or DistilBERT predicts LIAR label
        ↓
Class probabilities are generated
        ↓
Soft risk score is calculated
        ↓
Linguistic risk is added
        ↓
Final credibility category is displayed
        ↓
Streamlit app shows result, risk meter, and explanation
```

---

## Folder Structure

```text
NLP_Project/
│
├── app/
│   ├── streamlit_app.py
│   └── utils.py
│
├── data/
│   ├── raw/
│   │   ├── train.tsv
│   │   ├── valid.tsv
│   │   └── test.tsv
│   │
│   └── processed/
│       ├── train_clean.csv
│       ├── valid_clean.csv
│       └── test_clean.csv
│
├── models/
│   ├── semantic_ml_model.pkl
│   ├── embedding_model_name.txt
│   └── deep_model/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── vocab.txt
│
├── results/
│   ├── semantic_ml_report.txt
│   ├── semantic_ml_confusion_matrix.png
│   └── deep_learning_report.txt
│
├── src/
│   ├── linguistic_report.py
│   ├── predict_semantic_ml.py
│   └── predict_deep.py
│
└── README.md
```

---

## Technologies Used

| Layer | Technology |
|---|---|
| Programming Language | Python |
| Data Handling | Pandas, NumPy |
| NLP Preprocessing | NLTK |
| Semantic Embeddings | Sentence-BERT |
| Machine Learning | Scikit-learn |
| ML Classifier | Softmax Logistic Regression |
| Deep Learning | PyTorch |
| Transformer Model | DistilBERT |
| UI | Streamlit |
| Training Platform | Google Colab GPU |
| Model Saving | Joblib, Hugging Face save_pretrained |

---

## How to Run in Google Colab

### Step 1: Install Dependencies

```python
!pip install -q transformers torch pandas scikit-learn nltk joblib matplotlib sentence-transformers streamlit
```

### Step 2: Upload Dataset

Upload:

```text
train.tsv
valid.tsv
test.tsv
```

### Step 3: Preprocess Dataset

Run the preprocessing cells to generate:

```text
data/processed/train_clean.csv
data/processed/valid_clean.csv
data/processed/test_clean.csv
```

### Step 4: Train Semantic ML Model

Run the Sentence-BERT embedding cell and train:

```text
Sentence-BERT Embeddings + Softmax Logistic Regression
```

### Step 5: Train DistilBERT Model

Run the DistilBERT 6-class training cell using Colab GPU.

### Step 6: Run Streamlit App

Start the Streamlit app and expose it using Cloudflare Tunnel.

---

## How to Run Streamlit in Colab

Use the Streamlit + Cloudflare tunnel cell. It will generate a public URL like:

```text
https://something.trycloudflare.com
```

Open the URL to access the app.

---

## Sample Input

```text
Morocco wants tourists to visit Western Sahara. Some say it's tightening its control
```

Possible output:

```text
Predicted LIAR Label: half-true
Credibility Score: 60%
Final Risk: 40%
Final Category: Mostly Credible
```

---

## Important Note

This system does **not** perform live fact-checking from the internet. It estimates the credibility of a news statement based on learned patterns from the LIAR dataset, semantic embeddings, transformer-based classification, and linguistic cues.

For a more advanced version, the system can be extended with:

- RAG-based evidence retrieval
- Trusted news source verification
- Claim-evidence matching
- Google Fact Check API
- Source credibility analysis

---

## Future Enhancements

- Add RAG-based fact verification
- Compare claims with trusted sources
- Add source credibility checking
- Add article URL analysis
- Add multilingual misinformation detection
- Improve model performance using larger transformer models
- Add explainable AI visualizations

---

## Project Summary

This project improves basic fake news detection by moving from simple binary classification to a credibility-based misinformation detection system. It uses six LIAR truthfulness labels, semantic embeddings, DistilBERT, soft risk scoring, and a Streamlit interface to provide a more realistic and explainable credibility analysis.
