"""
app.py
Streamlit front-end for the AI Spam Email Detection System.

Run with:
    streamlit run app.py
"""

import json
import os
import sqlite3
from datetime import datetime

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (
    analyze_urls,
    calculate_risk_score,
    clean_text,
    detect_phishing_keywords,
    explain_prediction,
    get_email_stats,
    risk_level,
)

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(page_title="AI Spam Email Detector", page_icon="📧", layout="wide")

MODELS_DIR = "models"
DB_PATH = "models/history.db"


# ----------------------------------------------------------------------
# Database (SQLite) for prediction history
# ----------------------------------------------------------------------
def init_db():
    os.makedirs(MODELS_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            preview TEXT,
            prediction TEXT,
            confidence REAL,
            risk_score INTEGER,
            word_count INTEGER,
            link_count INTEGER
        )
        """
    )
    conn.commit()
    conn.close()


def save_to_history(text, prediction, confidence, risk_score, stats):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO history (timestamp, preview, prediction, confidence, risk_score, word_count, link_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now().isoformat(timespec="seconds"),
            text[:120].replace("\n", " "),
            prediction,
            confidence,
            risk_score,
            stats["word_count"],
            stats["link_count"],
        ),
    )
    conn.commit()
    conn.close()


def load_history() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM history ORDER BY id DESC", conn)
    conn.close()
    return df


def clear_history():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM history")
    conn.commit()
    conn.close()


# ----------------------------------------------------------------------
# Model loading (cached)
# ----------------------------------------------------------------------
@st.cache_resource
def load_model_and_vectorizer():
    model_path = os.path.join(MODELS_DIR, "spam_model.pkl")
    vec_path = os.path.join(MODELS_DIR, "vectorizer.pkl")
    if not (os.path.exists(model_path) and os.path.exists(vec_path)):
        return None, None
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer


@st.cache_data
def load_metrics():
    path = os.path.join(MODELS_DIR, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Prediction pipeline
# ----------------------------------------------------------------------
def predict_email(text, model, vectorizer):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])

    pred = model.predict(vec)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        spam_prob = proba[1]
    else:
        spam_prob = float(pred)  # fallback

    label = "Spam" if pred == 1 else "Not Spam"
    confidence = spam_prob if pred == 1 else 1 - spam_prob
    return label, confidence, spam_prob


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
def main():
    init_db()
    model, vectorizer = load_model_and_vectorizer()

    st.title("📧 AI Spam Email Detection System")
    st.caption("Classify emails as Spam or Not Spam with explainability, phishing detection, and risk scoring.")

    if model is None:
        st.error(
            "No trained model found. Run `python train_model.py` first to train and save a model "
            "to the `models/` folder, then refresh this page."
        )
        st.stop()

    tab1, tab2, tab3 = st.tabs(["🔍 Detect", "📊 Dashboard", "ℹ️ Model Info"])

    # ------------------------------------------------------------
    # TAB 1: Detect
    # ------------------------------------------------------------
    with tab1:
        col_input, col_settings = st.columns([3, 1])

        with col_input:
            uploaded_file = st.file_uploader("Upload an email (.txt)", type=["txt"])
            default_text = ""
            if uploaded_file is not None:
                default_text = uploaded_file.read().decode("utf-8", errors="ignore")

            email_text = st.text_area(
                "Or paste email text here",
                value=default_text,
                height=220,
                placeholder="Paste the full email content (subject + body) here...",
            )

        with col_settings:
            st.markdown("#### Options")
            show_explain = st.checkbox("Explain prediction", value=True)
            show_phishing = st.checkbox("Detect phishing keywords", value=True)
            show_urls = st.checkbox("Check suspicious URLs", value=True)
            save_history_chk = st.checkbox("Save to history", value=True)

        analyze_clicked = st.button("🔎 Analyze Email", type="primary", use_container_width=True)

        if analyze_clicked:
            if not email_text.strip():
                st.warning("Please paste some email text or upload a .txt file first.")
            else:
                label, confidence, spam_prob = predict_email(email_text, model, vectorizer)
                stats = get_email_stats(email_text)
                phishing_words = detect_phishing_keywords(email_text) if show_phishing else []
                url_analysis = analyze_urls(email_text) if show_urls else []
                risk_score = calculate_risk_score(spam_prob, stats, phishing_words, url_analysis)

                if save_history_chk:
                    save_to_history(email_text, label, confidence, risk_score, stats)

                st.divider()

                # ---- Result header ----
                res_col1, res_col2, res_col3 = st.columns(3)
                with res_col1:
                    if label == "Spam":
                        st.error(f"### 🚫 {label}")
                    else:
                        st.success(f"### ✅ {label}")
                with res_col2:
                    st.metric("Confidence", f"{confidence*100:.1f}%")
                with res_col3:
                    st.metric("Risk Score", f"{risk_score}/100", risk_level(risk_score))

                st.progress(int(spam_prob * 100), text=f"Spam probability: {spam_prob*100:.1f}%")

                # ---- Email statistics ----
                st.markdown("#### 📈 Email Statistics")
                s1, s2, s3, s4, s5, s6 = st.columns(6)
                s1.metric("Words", stats["word_count"])
                s2.metric("Characters", stats["char_count"])
                s3.metric("Links", stats["link_count"])
                s4.metric("Special chars", stats["special_char_count"])
                s5.metric("Exclamations", stats["exclamation_count"])
                s6.metric("ALL-CAPS words", stats["uppercase_word_count"])

                # ---- Explainability ----
                if show_explain:
                    st.markdown("#### 🧠 Why this prediction? (Top contributing words)")
                    contributions = explain_prediction(email_text, model, vectorizer, top_n=10)
                    if contributions:
                        words = [c[0] for c in contributions]
                        weights = [c[1] for c in contributions]
                        fig = px.bar(
                            x=weights, y=words, orientation="h",
                            labels={"x": "Contribution toward Spam", "y": "Word"},
                            color=weights, color_continuous_scale="RdYlGn_r",
                        )
                        fig.update_layout(yaxis=dict(autorange="reversed"), height=380, coloraxis_showscale=False)
                        st.plotly_chart(fig, use_container_width=True)

                        highlight = ", ".join(f"**{w}**" for w in words[:8])
                        st.caption(f"Spam-triggering words found in this email: {highlight}")
                    else:
                        st.info("This model type doesn't support word-level explanations.")

                # ---- Phishing keywords ----
                if show_phishing:
                    st.markdown("#### 🎣 Phishing Keyword Detection")
                    if phishing_words:
                        st.warning("Detected phishing-style phrases: " + ", ".join(f"`{w}`" for w in phishing_words))
                    else:
                        st.info("No common phishing keywords detected.")

                # ---- URL analysis ----
                if show_urls:
                    st.markdown("#### 🔗 URL / Link Analysis")
                    if url_analysis:
                        for u in url_analysis:
                            if u["suspicious"]:
                                st.error(f"⚠️ `{u['url']}` — " + "; ".join(u["reasons"]))
                            else:
                                st.success(f"✅ `{u['url']}` — no obvious red flags")
                    else:
                        st.info("No links found in this email.")

                st.divider()
                with st.expander("📋 Full risk explanation"):
                    st.write(
                        f"- **Model spam probability:** {spam_prob*100:.1f}%\n"
                        f"- **Phishing keywords found:** {len(phishing_words)}\n"
                        f"- **Suspicious links:** {sum(1 for u in url_analysis if u['suspicious'])} / {len(url_analysis)}\n"
                        f"- **Final risk score:** {risk_score}/100 ({risk_level(risk_score)})"
                    )

    # ------------------------------------------------------------
    # TAB 2: Dashboard
    # ------------------------------------------------------------
    with tab2:
        st.subheader("📊 Prediction History Dashboard")
        df = load_history()

        if df.empty:
            st.info("No predictions saved yet. Analyze an email in the Detect tab to populate this dashboard.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Emails Analyzed", len(df))
            c2.metric("Spam Detected", int((df["prediction"] == "Spam").sum()))
            c3.metric("Not Spam", int((df["prediction"] == "Not Spam").sum()))

            colA, colB = st.columns(2)
            with colA:
                pie_df = df["prediction"].value_counts().reset_index()
                pie_df.columns = ["prediction", "count"]
                fig = px.pie(pie_df, names="prediction", values="count", title="Spam vs Not Spam",
                             color="prediction", color_discrete_map={"Spam": "#e74c3c", "Not Spam": "#2ecc71"})
                st.plotly_chart(fig, use_container_width=True)

            with colB:
                fig2 = px.histogram(df, x="risk_score", nbins=20, title="Risk Score Distribution")
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("#### Recent Predictions")
            st.dataframe(
                df[["timestamp", "preview", "prediction", "confidence", "risk_score"]],
                use_container_width=True,
                hide_index=True,
            )

            if st.button("🗑️ Clear History"):
                clear_history()
                st.rerun()

    # ------------------------------------------------------------
    # TAB 3: Model Info
    # ------------------------------------------------------------
    with tab3:
        st.subheader("ℹ️ Model Comparison & Info")
        metrics = load_metrics()
        if metrics is None:
            st.info("No metrics file found. Run train_model.py to generate model comparison metrics.")
        else:
            st.success(f"Best model in use: **{metrics['best_model']}**")
            results_df = pd.DataFrame(metrics["results"]).T.reset_index().rename(columns={"index": "model"})
            st.dataframe(results_df, use_container_width=True, hide_index=True)

            fig = px.bar(
                results_df.melt(id_vars="model", value_vars=["accuracy", "precision", "recall", "f1_score"]),
                x="model", y="value", color="variable", barmode="group",
                title="Model Comparison: Accuracy / Precision / Recall / F1",
            )
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
