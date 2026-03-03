# MRN Project - 금융 뉴스 감성 분석 및 키워드 분석 파이프라인

MRN(Machine Readable News) 데이터를 활용한 감성 분석, 키워드 추출, RAG 챗봇 프로젝트입니다.

## 프로젝트 구조

```
mrn_project/
├── __init__.py
├── config.py                      # 설정값 (경로, 날짜, 모델명 등)
├── main.py                        # 메인 실행 파이프라인
├── README.md
│
├── data/                          # 데이터 로딩 · 매핑 · 전처리
│   ├── __init__.py
│   ├── loader.py                  # gzip 파일 → DataFrame 로딩
│   ├── mapping.py                 # 뉴스코드/RIC → 설명/종목명 매핑
│   └── processing.py              # 필터링 · 열 생성 · 전처리
│
├── elasticsearch_utils/           # Elasticsearch 연동
│   ├── __init__.py
│   ├── client.py                  # ES 클라이언트 초기화
│   ├── index.py                   # 인덱스 생성/수정/삭제
│   ├── crud.py                    # 저장/업데이트/삭제/조회
│   └── search.py                  # 벡터 유사도 검색
│
├── analysis/                      # 분석 기능
│   ├── __init__.py
│   ├── sentiment.py               # 감성 분석 (TextBlob, FinBERT)
│   ├── company.py                 # 기업 언급량 · 감성 스파이크 탐지
│   ├── events.py                  # 주요 이벤트 탐지
│   ├── topics.py                  # BERTopic 토픽 모델링
│   └── keywords.py                # 키워드 추출 · 트렌드 · 감성 분석
│
├── llm/                           # LLM 관련 기능
│   ├── __init__.py
│   ├── prompts.py                 # 프롬프트 템플릿 (RAG, 요약, 키워드)
│   ├── chatbot.py                 # RAG 챗봇 · 뉴스 검색
│   └── summarization.py           # 뉴스 요약
│
├── visualization/                 # 시각화
│   ├── __init__.py
│   ├── sentiment_plots.py         # 감성 분포 · 일별 트렌드 시각화
│   └── network.py                 # 키워드-종목 네트워크 그래프
│
└── graphs/                        # 그래프 출력 디렉토리
```

## 사용법

```python
# 전체 파이프라인 실행
python -m mrn_project.main

# 또는 모듈 개별 임포트
from mrn_project.data.loader import load_zipfile_to_df
from mrn_project.analysis.sentiment import calculate_sentiment_textblob
from mrn_project.llm.chatbot import rag_chatbot
```

## 주요 기능

| 모듈 | 설명 |
|------|------|
| `data` | 데이터 로딩, 뉴스코드/RIC 매핑, 필터링 및 전처리 |
| `elasticsearch_utils` | ES 클라이언트, 인덱스 관리, CRUD, 벡터 유사도 검색 |
| `analysis` | 감성 분석, 기업 분석, 이벤트 탐지, 토픽 모델링, 키워드 분석 |
| `llm` | RAG 챗봇, 뉴스 검색, 요약, 프롬프트 관리 |
| `visualization` | 감성 분포/트렌드 시각화, 네트워크 그래프 |
