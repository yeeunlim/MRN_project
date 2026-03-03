"""
데이터 필터링 및 전처리 함수
"""
import pandas as pd

from mrn_project.data.mapping import map_ric_list_to_sec_nm


def filter_data(data_df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """날짜·언어·본문 기준으로 필터링한다."""
    df = data_df.copy()
    df["firstCreated"] = pd.to_datetime(df["firstCreated"], errors="coerce")
    df["versionCreated"] = pd.to_datetime(df["versionCreated"], errors="coerce")
    df["date"] = df["versionCreated"].dt.date

    # 날짜 필터
    filtered = df[
        (df["date"] >= pd.to_datetime(start_date).date())
        & (df["date"] <= pd.to_datetime(end_date).date())
    ]

    # 영어 필터
    filtered = filtered[filtered["language"] == "en"]

    # body 비어있지 않은 데이터
    filtered = filtered[filtered["body"].str.strip() != ""]

    return filtered


def process_data(data_df: pd.DataFrame, ric_mapping_dict: dict) -> pd.DataFrame:
    """분석에 필요한 열을 생성하고 선택한다."""
    df = data_df.copy()

    # date(versionCreated) 열 생성
    df["date(versionCreated)"] = df["versionCreated"].dt.date
    df["date(versionCreated)"] = pd.to_datetime(df["date(versionCreated)"], errors="coerce")

    # secName 열 생성
    df["secName"] = df["subjects"].apply(
        lambda subjects: map_ric_list_to_sec_nm(subjects, ric_mapping_dict)
    )

    # 필요한 열만 선택
    processed = df[
        [
            "id", "date(versionCreated)", "headline", "body", "subjects", "secName",
            "language", "provider", "firstCreated", "versionCreated",
        ]
    ]

    return processed
