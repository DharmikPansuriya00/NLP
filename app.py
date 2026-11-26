"""Airline Tweet Sentiment Analyzer — Streamlit Web Application (Task 8).

Uses the shared preprocessing and heuristic lexicon from nlp_utils.py and the
trained TF-IDF + Logistic Regression pipeline saved in sentiment_model.joblib.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import joblib
import nltk
import pandas as pd
import sklearn
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nlp_utils import (
    NEGATIVE_LEXICON,
    POSITIVE_LEXICON,
    clean_text,
    ensure_nltk_data,
    heuristic_sentiment,
)

# ---------------------------------------------------------------------------
# Page configuration (must be the very first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Airline Tweet Sentiment Analyzer",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Safely ensure required NLTK data is available on startup (e.g., Streamlit Cloud)
try:
    ensure_nltk_data()
except Exception:
    pass

# ---------------------------------------------------------------------------
# Custom CSS for clean, modern aviation-inspired UI
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .hero-banner {
        background: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        color: #FFFFFF;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px rgba(15, 32, 39, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(42, 157, 143, 0.25);
        color: #80FFDB;
        border: 1px solid rgba(128, 255, 219, 0.45);
        padding: 0.22rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        margin-bottom: 0.6rem;
        text-transform: uppercase;
    }
    .hero-title {
        font-size: 2.05rem;
        font-weight: 800;
        margin: 0.15rem 0 0.4rem 0;
        line-height: 1.2;
        color: #FFFFFF;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        color: #DCEEF2;
        margin: 0;
        max-width: 780px;
    }
    .clean-box {
        background: rgba(44, 83, 100, 0.08);
        border-left: 4px solid #2A9D8F;
        padding: 0.85rem 1rem;
        border-radius: 8px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.92rem;
        margin-bottom: 1rem;
    }
    .metric-pill {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    .metric-pill-label {
        font-size: 0.78rem;
        opacity: 0.78;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .metric-pill-val {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }
    div.stButton > button[kind="secondary"] {
        border-radius: 10px;
        font-size: 0.86rem;
        padding: 0.45rem 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentiment_model.joblib")


@st.cache_resource
def load_ml_model():
    """Load the trained TF-IDF + Logistic Regression pipeline once."""
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_vader() -> SentimentIntensityAnalyzer:
    """Load NLTK VADER SentimentIntensityAnalyzer once."""
    ensure_nltk_data()
    return SentimentIntensityAnalyzer()


POS_EXPLANATIONS: Dict[str, str] = {
    "NN": "Noun (singular)",
    "NNS": "Noun (plural)",
    "NNP": "Proper noun (singular)",
    "NNPS": "Proper noun (plural)",
    "VB": "Verb (base form)",
    "VBD": "Verb (past tense)",
    "VBG": "Verb (gerund / -ing)",
    "VBN": "Verb (past participle)",
    "VBP": "Verb (present, non-3rd)",
    "VBZ": "Verb (present, 3rd person)",
    "JJ": "Adjective",
    "JJR": "Adjective (comparative)",
    "JJS": "Adjective (superlative)",
    "RB": "Adverb",
    "RBR": "Adverb (comparative)",
    "RBS": "Adverb (superlative)",
    "PRP": "Personal pronoun",
    "PRP$": "Possessive pronoun",
    "IN": "Preposition / conjunction",
    "DT": "Determiner",
    "CC": "Coordinating conjunction",
    "MD": "Modal auxiliary",
    "UH": "Interjection",
}


def get_pos_table(text: str) -> pd.DataFrame:
    """Tokenise and POS-tag text safely using NLTK."""
    ensure_nltk_data()
    tokens = nltk.word_tokenize(text)
    tagged: List[Tuple[str, str]] = nltk.pos_tag(tokens)
    rows = [
        {
            "Token": word,
            "POS Tag": tag,
            "Grammar Role": POS_EXPLANATIONS.get(tag, "Other / Punctuation"),
        }
        for word, tag in tagged
    ]
    return pd.DataFrame(rows)


def render_sentiment_banner(label: str, subtitle: str) -> None:
    """Render a coloured Streamlit alert matching sentiment polarity."""
    if label == "positive":
        st.success(f"**Predicted Sentiment: POSITIVE** — {subtitle}", icon="✅")
    elif label == "neutral":
        st.warning(f"**Predicted Sentiment: NEUTRAL** — {subtitle}", icon="⚖️")
    else:
        st.error(f"**Predicted Sentiment: NEGATIVE** — {subtitle}", icon="🚨")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ✈️ About This Project")
    st.write(
        "This app analyses passenger tweets about US airlines using **NLP Fundamentals** "
        "(RegEx cleaning, domain lexicons, NLTK VADER, TF-IDF + N-grams, and POS tagging)."
    )

    st.markdown("### 📊 Dataset Snapshot")
    st.markdown(
        """
        - **Dataset:** [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
        - **Size:** 14,640 raw tweets (14,427 unique)
        - **Class Balance:** ~63% Negative · ~21% Neutral · ~16% Positive
        - **Primary Metric:** Macro-F1 (handles class imbalance)
        """
    )

    st.markdown("### 🧠 Approaches Available")
    st.markdown(
        """
        1. **ML Model (Deployed Best):** `TF-IDF (1,2) + LogisticRegression(class_weight='balanced')`
        2. **API-Style (VADER):** Pre-built social-media lexicon (`compound >= 0.05` / `<= -0.05`)
        3. **Heuristic:** 38 positive + 38 negative airline words with 2-word negation flipping
        """
    )

    st.divider()
    st.caption(f"scikit-learn `{sklearn.__version__}` · Red & White Skill Education PR 1")

# ---------------------------------------------------------------------------
# Header Hero Banner
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-badge">NLP Practical Report 1 · Live Classifier</div>
        <div class="hero-title">✈️ Airline Tweet Sentiment Analyzer</div>
        <p class="hero-subtitle">
            Test real airline tweets across three NLP paradigms — hand-crafted Heuristic rules,
            API-style VADER scoring, and a trained TF-IDF N-gram Machine Learning pipeline —
            plus inspect live RegEx cleaning and NLTK Part-of-Speech tags.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state for example tweet buttons
# ---------------------------------------------------------------------------
EXAMPLES = {
    "positive": "@VirginAmerica thanks for the super smooth landing and friendly crew today! #BestAirline",
    "negative": "@United flight cancelled again and I'm NOT happy with the 3-hour delay on hold https://t.co/demo",
    "neutral": "@JetBlue what is the carry-on bag size limit for Boston to Chicago tomorrow?",
}

if "tweet_input" not in st.session_state:
    st.session_state["tweet_input"] = ""

st.markdown("#### 1. Try an Example Tweet or Write Your Own")
ex_col1, ex_col2, ex_col3 = st.columns(3)
with ex_col1:
    if st.button("😊 Positive Praise", width="stretch"):
        st.session_state["tweet_input"] = EXAMPLES["positive"]
with ex_col2:
    if st.button("😠 Negative + Negation", width="stretch"):
        st.session_state["tweet_input"] = EXAMPLES["negative"]
with ex_col3:
    if st.button("❓ Neutral Inquiry", width="stretch"):
        st.session_state["tweet_input"] = EXAMPLES["neutral"]

tweet_text = st.text_area(
    "Enter an airline tweet to analyse (max 500 characters):",
    key="tweet_input",
    height=120,
    placeholder="e.g., @AmericanAir not happy with the delay and lost luggage at gate B4...",
)

col_ctrl1, col_ctrl2 = st.columns([2.2, 1.0])
with col_ctrl1:
    approach = st.radio(
        "2. Select NLP Approach:",
        options=[
            "ML Model (TF-IDF + Logistic Regression)",
            "API-Based (NLTK VADER)",
            "Heuristic (Domain Lexicon + Negation)",
        ],
        horizontal=True,
    )
with col_ctrl2:
    st.write("")
    st.write("")
    predict_clicked = st.button("🔍 Predict Sentiment", type="primary", width="stretch")

# ---------------------------------------------------------------------------
# Prediction & Results Panel
# ---------------------------------------------------------------------------
if predict_clicked:
    raw_input = tweet_text.strip()

    # Edge case 1: Empty input
    if not raw_input:
        st.warning("Please enter a tweet or click one of the example buttons above before predicting.")
        st.stop()

    # Edge case 2: Very long input (> 500 chars)
    if len(raw_input) > 500:
        st.info(
            f"Input was {len(raw_input)} characters long and has been truncated to the first **500 characters**."
        )
        raw_input = raw_input[:500]

    cleaned = clean_text(raw_input)

    st.markdown("---")
    st.markdown("### 📋 Analysis Results")

    st.markdown("**Cleaned Text (`clean_text()`):**")
    if cleaned:
        st.markdown(f'<div class="clean-box">{cleaned}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="clean-box"><em>(empty after removing URLs, @mentions, numbers, and symbols)</em></div>',
            unsafe_allow_html=True,
        )

    # Edge case 3: Input becomes empty after cleaning (e.g., emoji-only or URL-only)
    if not cleaned:
        if approach == "API-Based (NLTK VADER)":
            st.warning(
                "Cleaning removed all alphabetic words (for example, URL-only or emoji-only input). "
                "VADER is evaluating the raw text directly."
            )
            vader = load_vader()
            vs = vader.polarity_scores(raw_input)
            comp = vs["compound"]
            label = "positive" if comp >= 0.05 else ("negative" if comp <= -0.05 else "neutral")
            render_sentiment_banner(label, f"VADER compound score = {comp:+.3f}")
        else:
            st.warning(
                "No alphabetic words remained after text cleaning (e.g., only URLs, @mentions, numbers, or emojis were entered). "
                "Please include some descriptive words or switch to **API-Based (NLTK VADER)** to score raw symbols/emojis."
            )
            st.stop()
    else:
        if approach == "ML Model (TF-IDF + Logistic Regression)":
            model = load_ml_model()
            pred_label = str(model.predict([cleaned])[0])
            probs = model.predict_proba([cleaned])[0]
            class_names = [str(c) for c in model.classes_]
            prob_map = {cls: float(p) for cls, p in zip(class_names, probs)}
            top_conf = prob_map[pred_label]

            render_sentiment_banner(pred_label, f"Confidence: {top_conf:.1%}")

            # KPI Pills
            m1, m2, m3 = st.columns(3)
            for col_w, cls_name, emoji in [
                (m1, "negative", "🔴"),
                (m2, "neutral", "🟡"),
                (m3, "positive", "🟢"),
            ]:
                with col_w:
                    st.markdown(
                        f"""
                        <div class="metric-pill">
                            <div class="metric-pill-label">{emoji} {cls_name}</div>
                            <div class="metric-pill-val">{prob_map.get(cls_name, 0.0):.1%}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("#### Class Probabilities")
            ordered_classes = ["negative", "neutral", "positive"]
            prob_df = pd.DataFrame(
                {"Probability": [prob_map.get(c, 0.0) for c in ordered_classes]},
                index=[c.capitalize() for c in ordered_classes],
            )
            st.bar_chart(prob_df, height=250)

        elif approach == "API-Based (NLTK VADER)":
            vader = load_vader()
            vs = vader.polarity_scores(raw_input)
            comp = vs["compound"]
            label = "positive" if comp >= 0.05 else ("negative" if comp <= -0.05 else "neutral")
            render_sentiment_banner(label, f"VADER compound score = {comp:+.3f} (evaluated on raw tweet)")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Compound", f"{vs['compound']:+.3f}")
            c2.metric("Positive", f"{vs['pos']:.3f}")
            c3.metric("Neutral", f"{vs['neu']:.3f}")
            c4.metric("Negative", f"{vs['neg']:.3f}")

        else:
            label, details = heuristic_sentiment(raw_input, return_details=True)
            render_sentiment_banner(
                label,
                f"Net Lexicon Score = {details['score']} "
                f"({len(POSITIVE_LEXICON)} pos / {len(NEGATIVE_LEXICON)} neg words in lexicon)",
            )
            hc1, hc2, hc3 = st.columns(3)
            hc1.metric("Net Score", int(details["score"]))
            hc2.write(f"**Positive matches:** {', '.join(details['positive_hits']) or 'None'}")
            hc3.write(f"**Negative matches:** {', '.join(details['negative_hits']) or 'None'}")
            if details["negated_flips"]:
                st.info(f"Negation flips triggered: {', '.join(details['negated_flips'])}")

    # Collapsible POS tagging section using NLTK
    with st.expander("🏷️ Inspect Part-of-Speech (POS) Tags (NLTK averaged perceptron)"):
        pos_source = cleaned if cleaned else raw_input
        pos_df = get_pos_table(pos_source)
        st.dataframe(pos_df, width="stretch", hide_index=True)
        st.caption(
            "Tags use the Penn Treebank tagset on cleaned full-sentence tokens (stop-words kept for grammatical context)."
        )
