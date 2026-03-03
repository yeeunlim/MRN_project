"""
RAG 챗봇 및 뉴스 검색
"""
import re
from elasticsearch import Elasticsearch
from langchain_core.output_parsers import StrOutputParser

from mrn_project.llm.prompts import RAG_PROMPT
from mrn_project.elasticsearch_utils.search import search_similarity


def build_rag_chain(llm):
    """RAG 체인을 생성한다."""
    return RAG_PROMPT | llm | StrOutputParser()


def rag_chatbot(question: str, es: Elasticsearch, embedding_model, llm,
                index_name: str = "mrn_data"):
    """질문에 대해 벡터 검색 → RAG 체인으로 답변을 생성한다."""
    query_vector = embedding_model.embed_query(question)
    results = search_similarity(es, index_name, query_vector, "vector_combined_text")

    context_texts = []
    sources = []
    confidence_scores = []

    for hit in results:
        doc = hit["_source"]
        score = hit["_score"]

        context_texts.append(
            f"Ticker: {doc['secName']}\n"
            f"Sentiment Score: {doc['sentiment_score']}\n"
            f"Date: {doc['date(versionCreated)']}\n"
            f"News Content: {doc['headline']} {doc['body']}\n"
        )
        sources.append(doc)
        confidence_scores.append(score)

    combined_context = "\n\n".join(context_texts)

    rag_chain = build_rag_chain(llm)
    response = rag_chain.invoke({
        "question": question,
        "context": combined_context,
    })

    return response, sources, confidence_scores


def clean_response(response: str) -> str:
    """<think> 태그를 제거하여 깨끗한 응답을 반환한다."""
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()


def search_news(documents, query: str, es: Elasticsearch, embedding_model,
                index_name: str = "mrn_data") -> list:
    """벡터 유사도 기반으로 뉴스를 검색한다."""
    query_vector = embedding_model.embed_query(query)
    results = search_similarity(es, index_name, query_vector, "vector_headline")

    sources = []
    for hit in results:
        doc = hit["_source"]
        doc["confidence"] = hit["_score"]
        sources.append(doc)

    return sources
