"""
키워드 추출 · 트렌드 분석 · 키워드별 감성 분석
"""
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import zscore
from sklearn.feature_extraction.text import TfidfVectorizer


# TF-IDF 키워드 추출

def extract_keywords_tfidf(texts, top_n: int = 5) -> list:
    """TF-IDF 기반으로 문서별 상위 키워드를 추출한다."""
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    keywords = []
    for doc_idx in range(tfidf_matrix.shape[0]):
        sorted_indices = np.argsort(-tfidf_matrix[doc_idx].toarray()).flatten()
        top_keywords = [feature_names[idx] for idx in sorted_indices[:top_n]]
        keywords.append(top_keywords)

    return keywords


# LLM 키워드 추출

def extract_keyword_llm(texts, keyword_chain, batch_size: int = 100) -> list:
    """LLM 기반으로 문서별 키워드를 추출한다."""
    num_batches = (len(texts) + batch_size - 1) // batch_size
    results = []
    for i in range(num_batches):
        batch_texts = texts[i * batch_size : (i + 1) * batch_size].tolist()
        batch_inputs = [{"combined_text": text} for text in batch_texts]
        batch_outputs = keyword_chain.batch(batch_inputs)
        results.extend(batch_outputs)
    return results


# 키워드 트렌드 분석

def analyze_keyword_trends(df, keyword_col: str) -> pd.DataFrame:
    """키워드 빈도 추이 · z-score · 증가율을 계산한다."""
    df = df.copy()
    df["date(versionCreated)"] = pd.to_datetime(df["date(versionCreated)"])

    keyword_counts = (
        df.explode(keyword_col)
        .groupby(["date(versionCreated)", keyword_col])
        .size()
        .unstack(fill_value=0)
    )

    recent_avg = keyword_counts.iloc[-7:].mean()
    past_avg = keyword_counts.iloc[:-7].mean()

    increase_rate_avg = ((recent_avg - past_avg) / (past_avg + 1)) * 100
    z_scores = keyword_counts.apply(zscore, axis=0)

    trend_df = pd.DataFrame({
        "latest_count": keyword_counts.iloc[-1],
        "recent_avg": recent_avg,
        "past_avg": past_avg,
        "increase_rate_avg(%)": increase_rate_avg,
        "z_score": z_scores.iloc[-1],
    }).fillna(0)

    trend_df = trend_df[trend_df["latest_count"] > 1]
    trend_df = trend_df.sort_values(
        by=["increase_rate_avg(%)", "recent_avg", "latest_count", "z_score"],
        ascending=[False, False, False, False],
    )
    return trend_df


# 키워드별 감성 분석

def calculate_keyword_sentiment(df, list_col: str) -> dict:
    """키워드별 평균 감성 점수를 계산한다."""
    keyword_sentiment = defaultdict(list)
    for _, row in df.iterrows():
        for keyword in row[list_col]:
            keyword_sentiment[keyword].append(row["sentiment_score"])

    return {k: sum(v) / len(v) for k, v in keyword_sentiment.items()}
