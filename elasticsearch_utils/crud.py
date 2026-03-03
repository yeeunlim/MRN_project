"""
Elasticsearch CRUD 함수 (저장 / 업데이트 / 삭제 / 조회)
"""
import logging
import pandas as pd
from elasticsearch import Elasticsearch, helpers

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# 저장 (store)

def store_data(
    es: Elasticsearch,
    index_name: str,
    data_df: pd.DataFrame,
    id_column: str,
    embedding_model=None,
    columns_to_vectorize: list = None,
    exclude_columns: list = None,
    batch_size: int = 100,
    retry_batch_size: int = 10,
):
    """DataFrame 을 Elasticsearch 에 배치 저장한다."""
    batches = [data_df.iloc[i : i + batch_size] for i in range(0, len(data_df), batch_size)]
    failed_batches = []

    def process_batch(batch):
        try:
            if columns_to_vectorize:
                for col in columns_to_vectorize:
                    texts = batch[col].tolist()
                    vectors = embedding_model.embed_documents(texts) if embedding_model else None
                    batch = batch.copy()
                    batch[f"vector_{col}"] = vectors

            original_batch = batch.copy()

            if exclude_columns:
                batch.drop(columns=exclude_columns, inplace=True, errors="ignore")

            actions = batch.apply(
                lambda row: {
                    "_index": index_name,
                    "_id": row[id_column],
                    "_source": row.to_dict(),
                },
                axis=1,
            ).to_list()

            helpers.bulk(es, actions)
            print(f"{len(actions)} saved to Elasticsearch.")
        except Exception as e:
            logging.error(f"Error processing batch: {e}", exc_info=True)
            return original_batch

    for batch in batches:
        result = process_batch(batch)
        if result is not None:
            failed_batches.append(result)

    # 실패 배치 재시도
    if failed_batches:
        logging.warning(f"{len(failed_batches)} batches failed. Retrying...")
        smaller = [
            batch.iloc[i : i + retry_batch_size]
            for batch in failed_batches
            for i in range(0, len(batch), retry_batch_size)
        ]
        for fb in smaller:
            try:
                process_batch(fb)
            except Exception as e:
                logging.error(f"Retry failed for batch: {e}", exc_info=True)

    logging.info("All batches processed")


# 업데이트 (update)

def update_data(
    es: Elasticsearch,
    index_name: str,
    data_df: pd.DataFrame,
    id_column: str,
    update_columns: list,
    embedding_model=None,
    columns_to_vectorize: list = None,
    exclude_columns: list = None,
    batch_size: int = 100,
    retry_batch_size: int = 10,
):
    """DataFrame 의 특정 열만 Elasticsearch 에 업데이트한다."""
    batches = [data_df.iloc[i : i + batch_size] for i in range(0, len(data_df), batch_size)]
    failed_batches = []

    def process_batch(batch):
        try:
            if columns_to_vectorize:
                for col in columns_to_vectorize:
                    texts = batch[col].tolist()
                    vectors = embedding_model.embed_documents(texts) if embedding_model else None
                    batch = batch.copy()
                    batch[f"vector_{col}"] = vectors

            original_batch = batch.copy()

            if exclude_columns:
                batch.drop(columns=exclude_columns, inplace=True, errors="ignore")

            actions = batch.apply(
                lambda row: {
                    "_op_type": "update",
                    "_index": index_name,
                    "_id": row[id_column],
                    "doc": {col: row[col] for col in update_columns if col in row},
                },
                axis=1,
            ).to_list()

            helpers.bulk(es, actions)
            print(f"{len(actions)} saved to Elasticsearch.")
        except Exception as e:
            logging.error(f"Error processing batch: {e}", exc_info=True)
            return original_batch

    for batch in batches:
        result = process_batch(batch)
        if result is not None:
            failed_batches.append(result)

    if failed_batches:
        logging.warning(f"{len(failed_batches)} batches failed. Retrying...")
        smaller = [
            batch.iloc[i : i + retry_batch_size]
            for batch in failed_batches
            for i in range(0, len(batch), retry_batch_size)
        ]
        for fb in smaller:
            try:
                process_batch(fb)
            except Exception as e:
                logging.error(f"Retry failed for batch: {e}", exc_info=True)

    logging.info("All batches processed")


# 삭제 (delete)

def delete_data(
    es: Elasticsearch,
    index_name: str,
    df: pd.DataFrame = None,
    id_column: str = "id",
    id_list: list = None,
    batch_size: int = 500,
):
    """DataFrame 또는 id_list 로 데이터를 삭제한다."""
    if df is not None:
        if id_column not in df.columns:
            raise ValueError(f"DataFrame에 {id_column} 열이 없습니다.")
        id_list = df[id_column].dropna().astype(str).tolist()

    if not id_list:
        print("삭제할 ID 리스트가 비어있습니다.")
        return

    batches = [id_list[i : i + batch_size] for i in range(0, len(id_list), batch_size)]
    failed_batches = []

    def process_batch(batch):
        try:
            actions = [
                {"_op_type": "delete", "_index": index_name, "_id": doc_id}
                for doc_id in batch
            ]
            helpers.bulk(es, actions)
            print(f"{len(actions)} deleted from Elasticsearch Index '{index_name}'.")
        except Exception as e:
            logging.error(f"Error processing batch: {e}", exc_info=True)
            failed_batches.append(batch)

    for batch in batches:
        process_batch(batch)

    if failed_batches:
        logging.warning(f"{len(failed_batches)} batches failed. Retrying...")
        for fb in failed_batches:
            try:
                process_batch(fb)
            except Exception as e:
                logging.error(f"Retry failed for batch: {e}", exc_info=True)

    logging.info("All batches processed")


# 조회 (fetch)

def fetch_all_documents(
    index_name: str,
    es: Elasticsearch,
    query: str = None,
    start_date: str = None,
    end_date: str = None,
    subjects: list = None,
    batch_size: int = 1000,
) -> pd.DataFrame:
    """Scroll API 로 문서를 모두 조회하여 DataFrame 으로 반환한다."""
    search_body = {
        "query": {"bool": {"must": [], "filter": []}},
        "size": batch_size,
        "_source": {"excludes": ["vector", "vector_headline"]},
        "sort": [{"_doc": "asc"}],
    }

    if query:
        search_body["query"]["bool"]["must"].append({"match": {"body": query}})

    if start_date or end_date:
        date_range = {}
        if start_date:
            date_range["gte"] = start_date
        if end_date:
            date_range["lte"] = end_date
        search_body["query"]["bool"]["filter"].append({"range": {"versionCreated": date_range}})

    if subjects:
        search_body["query"]["bool"]["filter"].append({"terms": {"subjects.keyword": subjects}})

    response = es.search(index=index_name, body=search_body, scroll="5m")
    scroll_id = response["_scroll_id"]
    hits = response["hits"]["hits"]

    all_data = [{**hit["_source"], "id": hit["_id"]} for hit in hits]

    while len(hits) > 0:
        response = es.scroll(scroll_id=scroll_id, scroll="5m")
        scroll_id = response["_scroll_id"]
        hits = response["hits"]["hits"]
        all_data.extend([{**hit["_source"], "id": hit["_id"]} for hit in hits])

    return pd.DataFrame(all_data)


def query_by_id(df: pd.DataFrame, doc_id: str):
    """DataFrame 에서 id 로 단일 문서를 조회한다."""
    result = df[df["id"] == doc_id]
    return result if not result.empty else None
