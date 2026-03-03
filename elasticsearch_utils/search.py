"""
Elasticsearch 벡터 유사도 검색
"""
from elasticsearch import Elasticsearch


def search_similarity(
    es: Elasticsearch,
    index_name: str,
    query_vector: list,
    vector_column: str,
    size: int = 10,
) -> list:
    """cosine similarity 기반으로 유사 문서를 검색한다."""
    search_query = {
        "size": size,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, params.document_vector) + 1.0",
                    "params": {
                        "query_vector": query_vector,
                        "document_vector": vector_column,
                    },
                },
            }
        },
    }

    response = es.search(index=index_name, body=search_query)
    return response["hits"]["hits"]
