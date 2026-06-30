# 📧 AI Spam Email Detection System

A feature-rich spam/phishing email classifier built with Python, scikit-learn, and Streamlit — designed to go beyond a basic "spam vs. ham" classifier with explainability, phishing detection, URL risk analysis, and a prediction history dashboard.

## 🚀 Overview

Spam emails waste time, spread malware, and are often used for phishing. This project classifies emails as **Spam** or **Not Spam**, explains *why*, and surfaces phishing/URL red flags with a 0–100 risk score — making it feel like a real-world security tool rather than a toy classifier.

## ✨ Features

**Core**
- Upload a `.txt` email or paste text directly
- Spam / Not Spam classification with confidence score
- Highlights spam-triggering words
- Email statistics: word count, character count, links, special characters, exclamation marks, ALL-CAPS words

**Advanced**
- Explainable AI — bar chart of words that pushed the prediction toward spam (Naive Bayes / Logistic Regression / Linear SVM)
- Phishing keyword detection (bank, OTP, verify, password, urgent, etc.)
- Suspicious URL analysis — flags shortened links, raw IPs, suspicious TLDs, missing HTTPS, spoofed-looking domains
- Composite 0–100 risk score blending model confidence with rule-based signals
- Prediction history saved to SQLite, with a dashboard (spam vs. ham pie chart, risk score distribution, recent predictions table)
- Model comparison tab (Accuracy / Precision / Recall / F1 across all trained models)

## 🧰 Tech Stack

| Layer | Tools |
|---|---|
| Frontend | Streamlit |
| Backend / ML | Python, scikit-learn, XGBoost |
| NLP | NLTK (with offline fallback), TF-IDF |
| Database | SQLite |
| Visualization | Plotly |

## 📊 Models Compared

The training script trains and evaluates all of the following, then automatically selects the best by F1-score:
- Naive Bayes
- Logistic Regression
- Linear SVM (calibrated for probability output)
- Random Forest
- XGBoost *(if installed)*

## 📁 Folder Structure

```text
spam-email-detector/
│
├── data/                # place your dataset CSV here
├── models/              # saved model, vectorizer, metrics, history.db (generated)
├── app.py               # Streamlit app
├── train_model.py       # training + model comparison script
├── utils.py             # preprocessing, phishing/URL detection, risk scoring
├── requirements.txt
└── README.md
```

## 📦 Dataset

The repo ships with a small **bundled sample dataset** (in `train_model.py`) so the project runs out of the box with no downloads.

For a stronger, resume-worthy project, swap in a real dataset:
- [SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (good for learning)
- An email spam dataset from Kaggle (e.g. search "Email Spam Classification Dataset") for real email content

```bash
python train_model.py --data data/spam.csv --text-col text --label-col label
```

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/spam-email-detector.git
cd spam-email-detector
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## ▶️ Usage

1. Train the model (creates `models/spam_model.pkl`, `models/vectorizer.pkl`, `models/metrics.json`):
   ```bash
   python train_model.py
   ```
2. Launch the app:
   ```bash
   streamlit run app.py
   ```
3. Open the local URL Streamlit prints (usually `http://localhost:8501`), paste or upload an email, and click **Analyze Email**.

## 📸 Screenshots

*(Add screenshots/GIFs of the Detect tab, Dashboard tab, and Model Info tab here once you run the app locally.)*

## ☁️ Deployment

- **Streamlit Community Cloud**: push this repo to GitHub, connect it at [share.streamlit.io](https://share.streamlit.io), set `app.py` as the entry point.
- **Render / Railway**: deploy as a web service with the start command `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.

## 🔮 Future Improvements

- Fine-tune a transformer (e.g. DistilBERT) for higher accuracy on longer emails
- Add an "AI safer-rewrite" suggestion for suspicious emails using an LLM
- Multi-class categorization (Spam / Promotions / Social / Personal / Work)
- Email thread summarization for long emails
- User accounts + per-user history instead of a single shared SQLite file
- Browser extension / Gmail API integration for live inbox scanning

## 📝 Resume Description

**AI Spam Email Detection System**
- Built a machine learning application to classify emails as spam or legitimate using NLP techniques.
- Performed text preprocessing and feature extraction using TF-IDF.
- Compared multiple classification algorithms (Naive Bayes, Logistic Regression, SVM, Random Forest, XGBoost) and evaluated them using accuracy, precision, recall, and F1-score.
- Implemented explainable AI, phishing keyword detection, suspicious URL analysis, and a composite risk-scoring system.
- Developed an interactive Streamlit interface with a prediction history dashboard for real-time email analysis.

---

**Difficulty:** ⭐⭐⭐☆☆ (Intermediate) · **Time:** 2–3 weeks if learning as you build.
