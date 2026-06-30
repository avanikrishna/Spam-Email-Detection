"""
utils.py
Shared utilities for the Spam Email Detector:
- Text cleaning / preprocessing
- Email statistics (word count, links, special chars)
- Phishing keyword detection
- Suspicious URL detection
- Risk score calculation
- Explainability helper (top contributing words)
"""

import re
import string
from urllib.parse import urlparse

# ---------------------------------------------------------------------
# Optional NLTK setup (falls back gracefully if NLTK data isn't found)
# ---------------------------------------------------------------------
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer

    try:
        STOPWORDS = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        STOPWORDS = set(stopwords.words("english"))

    STEMMER = PorterStemmer()
    NLTK_AVAILABLE = True
except Exception:
    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
        "on", "for", "and", "or", "it", "this", "that", "with", "as",
        "be", "at", "by", "from", "you", "your", "i", "we", "they",
    }
    STEMMER = None
    NLTK_AVAILABLE = False

URL_REGEX = re.compile(
    r"(https?://\S+|www\.\S+|\b\S+\.(com|net|org|info|biz|xyz|ru|cn|top|click|link)\b\S*)",
    re.IGNORECASE,
)

PHISHING_KEYWORDS = [
    "verify your account", "verify account", "bank account", "social security",
    "password", "username", "click here", "click below", "act now",
    "urgent", "immediately", "suspended", "limited time", "confirm your",
    "update your information", "otp", "one time password", "pin number",
    "credit card", "debit card", "wire transfer", "tax refund",
    "claim your prize", "winner", "congratulations you", "free gift",
    "lottery", "inheritance", "bitcoin", "crypto wallet", "gift card",
    "login attempt", "unusual activity", "reset your password",
    "security alert", "account locked", "irs", "paypal", "amazon order",
]

SUSPICIOUS_TLDS = {".ru", ".cn", ".tk", ".xyz", ".top", ".click", ".link",
                    ".info", ".biz", ".gq", ".cf", ".ml"}

URL_SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly",
                   "is.gd", "buff.ly", "rebrand.ly"}


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/punctuation/numbers, remove stopwords, stem."""
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = URL_REGEX.sub(" url ", text)
    text = re.sub(r"\S+@\S+", " email ", text)
    text = re.sub(r"\d+", " number ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    if STEMMER is not None:
        tokens = [STEMMER.stem(t) for t in tokens]

    return " ".join(tokens)


def get_email_stats(text: str) -> dict:
    words = text.split()
    urls = extract_urls(text)
    special_chars = sum(1 for c in text if c in "!$%^&*#@~`")
    exclamations = text.count("!")
    uppercase_words = sum(1 for w in words if w.isupper() and len(w) > 1)

    return {
        "word_count": len(words),
        "char_count": len(text),
        "link_count": len(urls),
        "special_char_count": special_chars,
        "exclamation_count": exclamations,
        "uppercase_word_count": uppercase_words,
    }


def extract_urls(text: str) -> list:
    return [m.group(0) for m in URL_REGEX.finditer(text)]


def detect_phishing_keywords(text: str) -> list:
    text_lower = text.lower()
    return [kw for kw in PHISHING_KEYWORDS if kw in text_lower]


def analyze_urls(text: str) -> list:
    """Return a list of dicts describing each URL and why it might be risky."""
    urls = extract_urls(text)
    results = []
    for raw_url in urls:
        url = raw_url if raw_url.startswith(("http://", "https://")) else "http://" + raw_url
        reasons = []
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
        except Exception:
            domain = raw_url.lower()

        if any(domain.endswith(tld) for tld in SUSPICIOUS_TLDS):
            reasons.append("Suspicious top-level domain")
        if any(short in domain for short in URL_SHORTENERS):
            reasons.append("Shortened URL (hides real destination)")
        if re.search(r"\d{1,3}(\.\d{1,3}){3}", domain):
            reasons.append("Raw IP address used instead of domain name")
        if domain.count("-") >= 2:
            reasons.append("Multiple hyphens in domain (common spoofing trick)")
        if domain.count(".") >= 3:
            reasons.append("Excessive subdomains")
        if not url.startswith("https://"):
            reasons.append("Not using HTTPS")

        results.append({
            "url": raw_url,
            "domain": domain,
            "suspicious": len(reasons) > 0,
            "reasons": reasons,
        })
    return results


def calculate_risk_score(spam_probability: float, stats: dict,
                          phishing_words: list, url_analysis: list) -> int:
    """Combine model probability with rule-based signals into a 0-100 score."""
    score = spam_probability * 60

    score += min(len(phishing_words) * 6, 18)
    suspicious_urls = sum(1 for u in url_analysis if u["suspicious"])
    score += min(suspicious_urls * 8, 16)

    if stats["exclamation_count"] >= 3:
        score += 3
    if stats["uppercase_word_count"] >= 3:
        score += 3

    return int(min(round(score), 100))


def risk_level(score: int) -> str:
    if score >= 70:
        return "High Risk"
    elif score >= 40:
        return "Medium Risk"
    else:
        return "Low Risk"


def explain_prediction(raw_text: str, model, vectorizer, top_n: int = 10) -> list:
    """
    Returns a list of (word, weight) tuples showing which words in THIS
    email pushed the prediction toward spam. Works for linear models
    (Naive Bayes, Logistic Regression, Linear SVM) that expose either
    feature_log_prob_ or coef_.
    """
    cleaned = clean_text(raw_text)
    vec = vectorizer.transform([cleaned])
    feature_names = vectorizer.get_feature_names_out()
    indices = vec.nonzero()[1]

    if hasattr(model, "coef_"):
        weights = model.coef_[0]
    elif hasattr(model, "feature_log_prob_"):
        weights = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    else:
        return []

    contributions = [(feature_names[i], float(weights[i])) for i in indices]
    contributions.sort(key=lambda x: x[1], reverse=True)
    return contributions[:top_n]
