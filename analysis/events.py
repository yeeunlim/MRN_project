"""
주요 이벤트 탐지
"""
import pandas as pd


def detect_important_events(df, important_events: list) -> pd.DataFrame:
    """subject 에 주요 이벤트가 포함된 뉴스를 탐지한다."""
    alerts = []
    for _, row in df.iterrows():
        for subject in row["subjects"]:
            if subject in important_events:
                alerts.append({
                    "event": subject,
                    "secName": row["secName"],
                    "headline": row["headline"],
                })

    alert_df = pd.DataFrame(alerts)
    alert_df = alert_df.explode("secName")

    # 이벤트·종목 동일 → 중복 기사 제거
    nonan = alert_df.dropna(subset=["secName"]).drop_duplicates(subset=["event", "secName"])
    nan_rows = alert_df[alert_df["secName"].isna()]
    alert_df = pd.concat([nonan, nan_rows]).reset_index(drop=True)

    alert_df.groupby(["headline", "event"], as_index=False).agg({"secName": list})
    return alert_df
