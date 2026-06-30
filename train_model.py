"""
train_model.py
Trains and compares multiple spam-classification models, then saves
the best one (by F1-score) to models/spam_model.pkl along with the
fitted TF-IDF vectorizer (models/vectorizer.pkl) and a metrics report
(models/metrics.json) used by the Streamlit dashboard.

Usage:
    python train_model.py                     # uses bundled sample data
    python train_model.py --data data/spam.csv --text-col text --label-col label
"""

import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from utils import clean_text

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Bundled sample dataset (small, for demo / offline use).
# For a real project, replace with the Kaggle "Email Spam Classification"
# dataset or the SMS Spam Collection Dataset via --data.
# ---------------------------------------------------------------------
def load_sample_data() -> pd.DataFrame:
    spam_samples = [
        "Congratulations! You have WON a $1000 Walmart gift card. Click here to claim now!!!",
        "URGENT: Your bank account has been suspended. Verify your account immediately at http://secure-bank-verify.xyz",
        "You are pre-approved for a $5000 loan. No credit check needed. Apply now bit.ly/loan123",
        "FREE entry in our weekly lottery! Text WIN to 80085 to claim your prize now.",
        "Your PayPal account has unusual activity. Reset your password here: http://paypal-secure.tk/reset",
        "CONGRATULATIONS!!! You've been selected for a free iPhone 15. Click the link to claim your gift now!",
        "Hot singles in your area want to meet you tonight! Click here.",
        "Limited time offer: Buy one get one free on all crypto wallet deposits. Act now!",
        "Your OTP for the transaction is required. Verify your account at http://192.168.1.1/verify",
        "IRS Notice: You have an unclaimed tax refund of $987. Confirm your information now.",
        "WINNER! As a valued network customer you have been selected to receive a prize.",
        "Get rich quick! Invest in bitcoin today and double your money in 24 hours.",
        "Your account will be suspended unless you verify your password immediately!!!",
        "Claim your inheritance of $4.5 million USD. Reply with your bank account details.",
        "Cheap viagra and pills, no prescription needed, order now at lowest price guaranteed.",
        "Act now! Your debit card has been locked due to unusual activity. Click here to unlock.",
        "Free gift card worth $500 waiting for you, click below to claim before it expires.",
        "Dear customer, your Amazon order could not be delivered, confirm your address urgently.",
        "You have a security alert from your bank. Login attempt detected, verify now.",
        "Make money fast working from home, no experience needed, click here to start earning today.",
    ]

    ham_samples = [
        "Hi John, can we move our meeting to 3pm tomorrow? Let me know if that works.",
        "Thanks for sending over the report. I'll review it and get back to you by Friday.",
        "Don't forget to pick up milk and eggs on your way home tonight.",
        "The quarterly numbers look good, great work team on hitting the targets this quarter.",
        "Happy birthday! Hope you have a wonderful day with family and friends.",
        "Attached is the agenda for next week's project sync. Let me know if anything's missing.",
        "Can you send me the slides from yesterday's presentation when you get a chance?",
        "Reminder: dentist appointment is scheduled for next Tuesday at 10am.",
        "Great catching up with you at the conference, let's grab coffee next time you're in town.",
        "The flight has been delayed by two hours, new departure time is 6:45pm.",
        "I've reviewed the contract and left a few comments, please take a look when free.",
        "Thanks for the recommendation, I just finished the book and really enjoyed it.",
        "Let's plan the team offsite for next month, I'll send a poll for dates.",
        "Your package has been shipped and is expected to arrive on Thursday.",
        "Just checking in to see how the project is going, let me know if you need help.",
        "The recipe you shared was amazing, I'm going to make it again this weekend.",
        "Please find attached the invoice for last month's services, let me know if questions.",
        "Looking forward to seeing you at the wedding next weekend!",
        "Can we reschedule our call to Thursday instead of Wednesday this week?",
        "The new office layout looks great, thanks for organizing the move.",
    ]

    df = pd.DataFrame({
        "text": spam_samples + ham_samples,
        "label": [1] * len(spam_samples) + [0] * len(ham_samples),
    })
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def load_data(path: str, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df = df.dropna(subset=["text", "label"])

    # Normalize label values like "spam"/"ham" or "1"/"0"
    if df["label"].dtype == object:
        df["label"] = df["label"].str.strip().str.lower().map(
            {"spam": 1, "ham": 0, "1": 1, "0": 0}
        )
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def build_models():
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(), cv=3),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, use_label_encoder=False, eval_metric="logloss", random_state=42
        )
    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to CSV dataset")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    args = parser.parse_args()

    if args.data and os.path.exists(args.data):
        print(f"Loading dataset from {args.data} ...")
        df = load_data(args.data, args.text_col, args.label_col)
    else:
        print("No --data provided (or file not found). Using bundled sample dataset.")
        print("For a real project, download an email/SMS spam dataset from Kaggle and pass --data.")
        df = load_sample_data()

    print(f"Dataset size: {len(df)} rows | Spam: {df['label'].sum()} | Ham: {(df['label']==0).sum()}")

    print("Cleaning text...")
    df["clean_text"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )

    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = build_models()
    results = {}
    best_model_name, best_model, best_f1 = None, None, -1

    print("\nTraining and comparing models...\n")
    for name, model in models.items():
        start = time.time()
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        elapsed = time.time() - start

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "train_time_sec": round(elapsed, 3),
        }

        print(f"{name:22s} | Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model

    print(f"\nBest model: {best_model_name} (F1 = {best_f1:.3f})")

    joblib.dump(best_model, os.path.join(MODELS_DIR, "spam_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "vectorizer.pkl"))

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump({"results": results, "best_model": best_model_name}, f, indent=2)

    print(f"\nSaved model -> {MODELS_DIR}/spam_model.pkl")
    print(f"Saved vectorizer -> {MODELS_DIR}/vectorizer.pkl")
    print(f"Saved metrics -> {MODELS_DIR}/metrics.json")


if __name__ == "__main__":
    main()
