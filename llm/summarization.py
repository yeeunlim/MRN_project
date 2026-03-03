"""
뉴스 요약 함수
"""
from langchain_core.output_parsers import StrOutputParser

from mrn_project.llm.prompts import SUMMARY_PROMPT


def build_summarization_chain(llm):
    """요약 체인을 생성한다."""
    return SUMMARY_PROMPT | llm | StrOutputParser()


def summarize_single_document(text: str, summarization_chain, max_length: int = 1000) -> str:
    """단일 문서를 요약한다."""
    return summarization_chain.invoke({"text": text[:max_length]})


def summarize_multiple_documents(df, summarization_chain, max_length: int = 10000) -> str:
    """여러 문서의 헤드라인을 합쳐서 요약한다."""
    combined_headlines = "\n".join(df["headline"])[:max_length]
    return summarization_chain.invoke({"text": combined_headlines})
