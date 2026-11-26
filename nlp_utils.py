"""Shared NLP utilities for Airline Tweet Sentiment Analyzer (PR 1).

Both NLP_PR1.ipynb and app.py import from this module so text cleaning,
stop-word handling, heuristic lexicon scoring, and NLTK setup stay identical
across experimentation and deployment.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple, Union

import nltk
from nltk.corpus import stopwords


NEGATION_WORDS: Set[str] = {"no", "not", "nor", "never"}

# 38 airline-specific positive words informed by Task 2.5 vocabulary statistics
POSITIVE_LEXICON: Set[str] = {
    "thanks",
    "thank",
    "thankful",
    "great",
    "love",
    "loved",
    "loving",
    "good",
    "best",
    "awesome",
    "amazing",
    "excellent",
    "wonderful",
    "fantastic",
    "friendly",
    "helpful",
    "kind",
    "nice",
    "smooth",
    "easy",
    "fast",
    "quick",
    "early",
    "ontime",
    "happy",
    "pleased",
    "impressed",
    "appreciate",
    "appreciated",
    "enjoy",
    "enjoyed",
    "comfortable",
    "clean",
    "upgrade",
    "upgraded",
    "kudos",
    "rock",
    "brilliant",
}

# 38 airline-specific negative words informed by Task 2.5 vocabulary statistics
NEGATIVE_LEXICON: Set[str] = {
    "cancelled",
    "canceled",
    "flightled",
    "delayed",
    "delay",
    "delays",
    "late",
    "rude",
    "worst",
    "worse",
    "bad",
    "terrible",
    "horrible",
    "awful",
    "poor",
    "sucks",
    "hate",
    "angry",
    "frustrated",
    "frustrating",
    "disappointed",
    "disappointing",
    "lost",
    "missing",
    "broken",
    "stuck",
    "stranded",
    "hold",
    "waiting",
    "wait",
    "unacceptable",
    "useless",
    "pathetic",
    "nightmare",
    "fail",
    "failed",
    "mess",
    "ridiculous",
}

HEURISTIC_NEGATIONS: Set[str] = {"not", "no", "never"}


def ensure_nltk_data() -> None:
    """Safely download required NLTK corpora and taggers if missing."""
    resources = [
        (["corpora/stopwords", "corpora/stopwords.zip"], "stopwords"),
        (["tokenizers/punkt", "tokenizers/punkt.zip"], "punkt"),
        (["tokenizers/punkt_tab", "tokenizers/punkt_tab.zip"], "punkt_tab"),
        (["sentiment/vader_lexicon", "sentiment/vader_lexicon.zip"], "vader_lexicon"),
        (
            [
                "taggers/averaged_perceptron_tagger",
                "taggers/averaged_perceptron_tagger.zip",
            ],
            "averaged_perceptron_tagger",
        ),
        (
            [
                "taggers/averaged_perceptron_tagger_eng",
                "taggers/averaged_perceptron_tagger_eng.zip",
            ],
            "averaged_perceptron_tagger_eng",
        ),
    ]
    for paths, name in resources:
        found = False
        for path in paths:
            try:
                nltk.data.find(path)
                found = True
                break
            except LookupError:
                continue
        if not found:
            try:
                nltk.download(name, quiet=True)
            except Exception:
                pass


def clean_text(text: str) -> str:
    """Clean a raw tweet using RegEx rules in safe order.

    Steps:
    1. Lowercase text.
    2. Remove URLs (http/https) before punctuation removal.
    3. Remove @mentions before punctuation removal.
    4. Keep hashtag word and drop the '#' symbol.
    5. Remove numbers and non-alphabetic punctuation.
    6. Collapse repeated whitespace and strip edges.
    """
    if text is None:
        return ""
    s = str(text).lower()
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"@\w+", " ", s)
    s = re.sub(r"#(\w+)", r"\1", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def get_stopwords(keep_negations: bool = True) -> Set[str]:
    """Return English stop-words, optionally preserving negation words."""
    ensure_nltk_data()
    sw = set(stopwords.words("english"))
    if keep_negations:
        sw = sw - NEGATION_WORDS
    return sw


def tokenize_and_remove_stopwords(text: str, keep_negations: bool = True) -> List[str]:
    """Tokenise cleaned text with NLTK and filter stop-words."""
    ensure_nltk_data()
    sw = get_stopwords(keep_negations=keep_negations)
    tokens = nltk.word_tokenize(clean_text(text))
    return [tok for tok in tokens if tok not in sw]


def heuristic_sentiment(
    text: str, return_details: bool = False
) -> Union[str, Tuple[str, Dict[str, object]]]:
    """Classify tweet sentiment using domain lexicons and 2-word negation flipping.

    For each matched positive (+1) or negative (-1) word, if any of the previous
    two tokens is in {'not', 'no', 'never'}, the polarity is flipped.
    Returns 'positive', 'negative', or 'neutral' (tie or zero matches).
    """
    cleaned = clean_text(text)
    tokens = cleaned.split()

    score = 0
    matched_pos: List[str] = []
    matched_neg: List[str] = []
    negated_matches: List[str] = []

    for idx, tok in enumerate(tokens):
        polarity = 0
        if tok in POSITIVE_LEXICON:
            polarity = 1
        elif tok in NEGATIVE_LEXICON:
            polarity = -1

        if polarity != 0:
            prev_window = tokens[max(0, idx - 2) : idx]
            is_negated = any(w in HEURISTIC_NEGATIONS for w in prev_window)
            if is_negated:
                polarity *= -1
                negated_matches.append(f"{prev_window[-1]} ... {tok}" if prev_window else tok)

            score += polarity
            if polarity > 0:
                matched_pos.append(tok)
            else:
                matched_neg.append(tok)

    if score > 0:
        label = "positive"
    elif score < 0:
        label = "negative"
    else:
        label = "neutral"

    if return_details:
        details: Dict[str, object] = {
            "score": score,
            "positive_hits": matched_pos,
            "negative_hits": matched_neg,
            "negated_flips": negated_matches,
        }
        return label, details
    return label
