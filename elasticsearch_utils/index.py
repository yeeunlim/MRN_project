"""
Elasticsearch 인덱스 생성 / 수정 / 삭제
"""
from elasticsearch import Elasticsearch


def initialize_index(es: Elasticsearch, index_name: str, properties: dict):
    """인덱스가 없으면 생성한다."""
    mapping = {"mappings": {"properties": properties}}
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=mapping)
        print(f"Index '{index_name}' created successfully.")
    else:
        print(f"Index '{index_name}' already exists.")


def update_index(es: Elasticsearch, index_name: str, updated_properties: dict):
    """기존 인덱스의 매핑을 업데이트한다."""
    if es.indices.exists(index=index_name):
        mapping = {"mappings": {"properties": updated_properties}}
        es.indices.put_mapping(index=index_name, body=mapping)
        print(f"Index '{index_name}' updated successfully.")
    else:
        print(f"Index '{index_name}' does not exist.")


def delete_index(es: Elasticsearch, index_name: str):
    """인덱스를 삭제한다."""
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print(f"Index '{index_name}' deleted successfully.")
    else:
        print(f"Index '{index_name}' does not exist.")
