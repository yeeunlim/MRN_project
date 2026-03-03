"""
프로젝트 설정값 (경로, 날짜, 모델명 등)
"""
import os

# 데이터 경로
MRN_DATA_FILE_PATH = "/data/users/GuestQuant/YeeunLim/data"
RIC_MAPPING_FILE_PATH = "mapping_info/RIC_ID_NAME.xlsx"
NEWSCODE_MAPPING_FILE_PATH = "mapping_info/NewsCodes.xlsx"
NEWSCODE_BEFORE_FILE_PATH = "mapping_info/NewsCodes_before.xlsx"

# 날짜 범위
START_DATE = "2025-01-01"
END_DATE = "2025-01-31"

# Elasticsearch
ES_INDEX_NAME = "mrn_data"
ES_SAMPLED_INDEX_NAME = "mrn_sampled"
ES_TIMEOUT = 60

# 모델
FINBERT_MODEL_NAME = "yiyanghkust/finbert-tone"
OLLAMA_MODEL_NAME = "llama3.2:latest"

# 감성 분석
MENTION_SPIKE_STD_FACTOR = 2.0
SENTIMENT_SPIKE_STD_FACTOR = 1.7

# 환경 변수 (Elasticsearch / Ollama)
ELASTIC_URL = os.getenv("ELASTIC_URL")
ELASTIC_USERNAME = os.getenv("ELASTIC_USERNAME")
ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")
OLLAMA_URL = os.getenv("OLLAMA_URL")

# 그래프 저장 경로
GRAPH_OUTPUT_DIR = "graphs"
