"""
MRN 프로젝트 - 메인 실행 파이프라인

전체 흐름:
  1. 데이터 로딩 및 전처리
  2. Elasticsearch 저장 / 조회
  3. 감성 분석
  4. 기업 분석 (언급량 · 감성 스파이크)
  5. 토픽 모델링
  6. 키워드 추출 · 트렌드 분석
  7. RAG 챗봇 · 뉴스 검색 · 요약
  8. 네트워크 그래프 시각화
"""
import os
import re
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# 설정
from mrn_project.config import (
    MRN_DATA_FILE_PATH,
    RIC_MAPPING_FILE_PATH,
    NEWSCODE_MAPPING_FILE_PATH,
    START_DATE, END_DATE,
    ES_INDEX_NAME, ES_SAMPLED_INDEX_NAME,
    OLLAMA_MODEL_NAME, OLLAMA_URL,
    GRAPH_OUTPUT_DIR,
)

os.makedirs(GRAPH_OUTPUT_DIR, exist_ok=True)

# 1. 데이터 로딩
from mrn_project.data.loader import load_zipfile_to_df
from mrn_project.data.mapping import (
    load_mapping_dict,
    map_ric_to_sec_nm, map_ric_list_to_sec_nm,
    map_topic_code_to_description, map_topic_code_list_to_description,
)
from mrn_project.data.processing import filter_data, process_data

data_df = load_zipfile_to_df(MRN_DATA_FILE_PATH, START_DATE, END_DATE)

# 매핑 딕셔너리 생성
topic_code_mapping_dict = load_mapping_dict(
    NEWSCODE_MAPPING_FILE_PATH,
    key_columns=["Topic_RCS_Code", "Topic_Primary_N2000_Code", "Topic_Secondary_N2000_Code"],
    value_column="Topic_Description",
)
ric_mapping_dict = load_mapping_dict(
    RIC_MAPPING_FILE_PATH,
    key_columns=["RIC_ID"],
    value_column="SEC_NM",
)

# 필터 · 전처리
filtered_df = filter_data(data_df, START_DATE, END_DATE)
processed_df = process_data(filtered_df, ric_mapping_dict)
print(processed_df.head())

# 2. Elasticsearch
from mrn_project.elasticsearch_utils.client import initialize_elasticsearch
from mrn_project.elasticsearch_utils.crud import fetch_all_documents, store_data, update_data, query_by_id

es = initialize_elasticsearch()
index_name = ES_INDEX_NAME

mrn_df = fetch_all_documents(index_name, es, start_date=START_DATE, end_date=END_DATE)
print(mrn_df.head())

# 3. 감성 분석
from mrn_project.analysis.sentiment import (
    calculate_sentiment_textblob, calculate_sentiment_finbert,
    get_top_sentiment_secNames, get_bottom_sentiment_secNames,
    calculate_daily_mean_sentiment_per_subject,
)

processed_df["combined_text"] = processed_df["headline"] + processed_df["body"]
processed_df["sentiment_score_finbert"] = processed_df["combined_text"].apply(calculate_sentiment_finbert)
processed_df["sentiment_score_textblob"] = processed_df["combined_text"].apply(calculate_sentiment_textblob)
processed_df["sentiment_score"] = processed_df["sentiment_score_textblob"]

# 감성 점수 Top / Bottom
top_sent = get_top_sentiment_secNames(mrn_df, k=5)
bottom_sent = get_bottom_sentiment_secNames(mrn_df, k=5)
print(top_sent)
print("-" * 30)
print(bottom_sent)

# 4. 시각화
from mrn_project.visualization.sentiment_plots import (
    plot_sentiment_distribution,
    visualize_sentiment_trends,
)

plot_sentiment_distribution(mrn_df, "sentiment_score_textblob",
                           "sentiment score distribution (textblob)",
                           "sentiment_score_distribution_textblob.png")

plot_sentiment_distribution(mrn_df, "sentiment_score_finbert",
                           "sentiment score distribution (finbert)",
                           "sentiment_score_distribution_finbert.png")

# 5. 기업 분석
from mrn_project.analysis.company import (
    calculate_company_mentions, detect_company_mention_spikes,
    calculate_company_sentiment, detect_company_sentiment_spikes,
)

mentions_df = calculate_company_mentions(mrn_df)
mention_spikes = detect_company_mention_spikes(mentions_df)
print("=== 언급량 스파이크 ===")
print(mention_spikes)

company_sentiment_df = calculate_company_sentiment(mrn_df)
rising, falling = detect_company_sentiment_spikes(company_sentiment_df)
print("=== 감성 상승 스파이크 ===")
print(rising)
print("=== 감성 하락 스파이크 ===")
print(falling)

# 6. 토픽 모델링
from mrn_project.analysis.topics import calculate_top_topics

# mrn_1day 등 특정 날짜 데이터가 필요하면 별도로 필터링
# top_topics = calculate_top_topics(mrn_1day)
# for topic_articles in top_topics:
#     for article in topic_articles:
#         print(article)

# 7. 키워드 분석
from mrn_project.analysis.keywords import (
    extract_keywords_tfidf, analyze_keyword_trends, calculate_keyword_sentiment,
)
from mrn_project.visualization.network import build_network_graph, visualize_single_network

mrn_df["combined_text"] = mrn_df["headline"] + mrn_df["body"]
mrn_df["keywords_tfidf"] = extract_keywords_tfidf(mrn_df["combined_text"])

trend_df = analyze_keyword_trends(mrn_df, "keywords_tfidf")
print(trend_df.head(10))

keyword_avg_sentiment = calculate_keyword_sentiment(mrn_df, "keywords_tfidf")

G = build_network_graph(mrn_df, "keywords_tfidf", "secName")
visualize_single_network(G, "deepseek", keyword_avg_sentiment)

# 8. RAG 챗봇
from langchain_ollama import OllamaLLM
from mrn_project.llm.chatbot import rag_chatbot, clean_response, search_news
from mrn_project.llm.summarization import build_summarization_chain, summarize_single_document

ollama_llm = OllamaLLM(model=OLLAMA_MODEL_NAME, base_url=OLLAMA_URL)

question = "Which emerging companies in AI and semiconductor industries are gaining attention?"
# embedding_model 은 별도로 초기화 필요
# response, sources, confidence_scores = rag_chatbot(question, es, embedding_model, ollama_llm)
# print(clean_response(response))

# 뉴스 검색
# query = "Trump tariff"
# sources = search_news(mrn_df, query, es, embedding_model, index_name)
# print(pd.DataFrame(sources).head())

# 요약
# summarization_chain = build_summarization_chain(ollama_llm)
# summary = summarize_single_document("news text here...", summarization_chain)
# print(summary)

print("\n✅ Pipeline complete.")
