"""
코드 매핑 함수 (뉴스 코드 ↔ 설명, RIC ↔ 종목명)
"""
import pandas as pd


# 매핑 딕셔너리 로드

def load_mapping_dict(file_path: str, key_columns: list, value_column: str) -> dict:
    """엑셀 매핑 파일에서 key → value 딕셔너리를 생성한다."""
    mapping_df = pd.read_excel(file_path)
    mapping_dict = {}
    for key_column in key_columns:
        mapping_dict.update(mapping_df.set_index(key_column)[value_column].to_dict())
    return mapping_dict


def prepare_newscodes_file(before_path: str, output_path: str):
    """NewsCodes_before.xlsx 의 코드에 'N:' 접두사를 붙여 저장한다."""
    df = pd.read_excel(before_path)
    df["Topic_Primary_N2000_Code"] = "N:" + df["Topic_Primary_N2000_Code"]
    df["Topic_Secondary_N2000_Code"] = "N:" + df["Topic_Secondary_N2000_Code"]
    df.to_excel(output_path, index=False)


# Topic Code 매핑

def map_topic_code_to_description(topic_code: str, mapping_dict: dict) -> str:
    """단일 topic code → 설명 매핑"""
    return mapping_dict.get(topic_code, topic_code)


def map_topic_code_list_to_description(subject_list, mapping_dict: dict) -> list:
    """topic code 리스트 → 설명 리스트 매핑"""
    if isinstance(subject_list, str):
        subject_list = [subject_list]
    return [map_topic_code_to_description(s, mapping_dict) for s in subject_list]


# RIC → 종목명 매핑

def map_ric_to_sec_nm(ric: str, mapping_dict: dict):
    """단일 RIC 코드(예: R:MSFT.O) → 종목명 매핑"""
    ticker = ric.split(":")[1].split(".")[0]
    for key, value in mapping_dict.items():
        if key.split(":")[1] == ticker:
            return value
    return None


def map_ric_list_to_sec_nm(subject_list, mapping_dict: dict) -> list:
    """subject 리스트에서 R: 로 시작하는 항목만 종목명으로 매핑"""
    if isinstance(subject_list, str):
        subject_list = [subject_list]
    sec_name_list = []
    for subject in subject_list:
        if subject.startswith("R:"):
            sec_name = map_ric_to_sec_nm(subject, mapping_dict)
            if sec_name is not None:
                sec_name_list.append(sec_name)
    return sec_name_list
