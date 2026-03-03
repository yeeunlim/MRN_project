"""
Elasticsearch 클라이언트 초기화
"""
import os
from elasticsearch import Elasticsearch

from mrn_project.config import ELASTIC_URL, ELASTIC_USERNAME, ELASTIC_PASSWORD, ES_TIMEOUT


def initialize_elasticsearch() -> Elasticsearch:
    """환경 변수에서 접속 정보를 읽어 Elasticsearch 클라이언트를 반환한다."""
    elastic_url = ELASTIC_URL or os.getenv("ELASTIC_URL")
    elastic_username = ELASTIC_USERNAME or os.getenv("ELASTIC_USERNAME")
    elastic_password = ELASTIC_PASSWORD or os.getenv("ELASTIC_PASSWORD")

    return Elasticsearch(
        hosts=[elastic_url],
        basic_auth=(elastic_username, elastic_password),
        timeout=ES_TIMEOUT,
    )
