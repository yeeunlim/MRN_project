"""
토픽 모델링 (BERTopic)
"""
from bertopic import BERTopic


def calculate_top_topics(df, top_k: int = 5, articles_per_topic: int = 1) -> list:
    """BERTopic 으로 상위 k 개 토픽과 대표 기사를 반환한다."""
    topic_model = BERTopic(language="english")
    documents = df["headline"].tolist()
    topics, _ = topic_model.fit_transform(documents)
    df = df.copy()
    df["topic"] = topics

    top_topics = df["topic"].value_counts().head(top_k).index
    result = []
    for topic_id in top_topics:
        related = df[df["topic"] == topic_id].head(articles_per_topic)
        result.append(related[["id", "headline"]].to_dict(orient="records"))

    return result
