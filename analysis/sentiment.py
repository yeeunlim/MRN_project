"""
감성 분석 함수 (TextBlob / FinBERT)
"""
import numpy as np
from textblob import TextBlob
from nltk.tokenize import sent_tokenize
from transformers import pipeline

from mrn_project.config import FINBERT_MODEL_NAME


# TextBlob

def calculate_sentiment_textblob(text: str) -> float:
    """TextBlob 기반 감성 점수 (최대 절댓값)."""
    sentences = sent_tokenize(text)
    scores = [TextBlob(sentence).sentiment.polarity for sentence in sentences]
    return max(scores, key=abs) if scores else 0.0


# FinBERT

_finbert_pipeline = None


def _get_finbert_pipeline():
    global _finbert_pipeline
    if _finbert_pipeline is None:
        _finbert_pipeline = pipeline("sentiment-analysis", model=FINBERT_MODEL_NAME)
    return _finbert_pipeline


def calculate_sentiment_finbert(text: str) -> float:
    """FinBERT 기반 감성 점수 (평균)."""
    sentiment_pipe = _get_finbert_pipeline()
    sentences = sent_tokenize(text)
    sentences = [s[:512] for s in sentences]

    results = sentiment_pipe(sentences)
    scores = []
    for result in results:
        if result["label"] == "Positive":
            scores.append(result["score"])
        elif result["label"] == "Negative":
            scores.append(-result["score"])
        else:
            scores.append(0)

    return float(np.mean(scores)) if scores else 0.0


# 감성 점수 집계

def compute_mean_sentiment_per_security(df):
    """종목별 평균 감성 점수를 계산한다."""
    df = df[df["secName"].apply(lambda x: x != [])]
    exploded = df.explode("secName")
    return exploded.groupby("secName", as_index=False)["sentiment_score"].mean()


def get_top_sentiment_secNames(df, k: int = 5):
    """감성 점수 상위 k 종목."""
    mean_df = compute_mean_sentiment_per_security(df)
    return mean_df.nlargest(k, "sentiment_score")[["secName", "sentiment_score"]].reset_index(drop=True)


def get_bottom_sentiment_secNames(df, k: int = 5):
    """감성 점수 하위 k 종목."""
    mean_df = compute_mean_sentiment_per_security(df)
    return mean_df.nsmallest(k, "sentiment_score")[["secName", "sentiment_score"]].reset_index(drop=True)


def calculate_daily_mean_sentiment_per_subject(df, selected_subjects: list, subjects_column: str = "subjects"):
    """날짜 · subject 별 평균 감성 점수를 계산한다."""
    filtered = df[df[subjects_column].apply(lambda x: any(s in selected_subjects for s in x))]
    exploded = filtered.explode(subjects_column)
    exploded = exploded[exploded[subjects_column].isin(selected_subjects)]
    daily = exploded.groupby(["date(versionCreated)", subjects_column])["sentiment_score"].mean().reset_index()
    return daily
