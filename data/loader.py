"""
데이터 로딩 함수 (gzip 파일 → DataFrame)
"""
import os
import gzip
import json
import pandas as pd


def load_zipfile_to_df(path: str, start_date: str = "2025-01-01", end_date: str = "2025-01-31") -> pd.DataFrame:
    """gzip 뉴스 데이터 파일을 읽어 DataFrame 으로 반환한다."""
    files_df = pd.Series(os.listdir(path), name="filename").to_frame()

    # 파일명에서 날짜 추출
    files_df.loc[:, "date"] = files_df.filename.str.extract(r"(\d{4}-\d{2}-\d{2})\d{6}")[0]
    files_df.loc[:, "history_date"] = files_df.filename.str.extract(r"\.(\d{6})\.")[0]
    files_df["date"] = pd.to_datetime(files_df["date"], format=None, errors="coerce")

    # 날짜 범위 필터
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    files_df = files_df[(files_df["date"] >= start) & (files_df["date"] <= end)]

    # 파일을 읽어서 DataFrame 으로 합치기
    data_frames = []
    for filename in files_df["filename"]:
        file_path = os.path.join(path, filename)
        with gzip.open(file_path, "rt") as f:
            content = f.read()
        content_json = json.loads(content)
        data_detail = [item["data"] for item in content_json["Items"]]
        data_frames.append(pd.DataFrame(data_detail))

    if data_frames:
        return pd.concat(data_frames, ignore_index=True)
    return pd.DataFrame()
