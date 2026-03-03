"""
기업 언급량 / 감성 스파이크 탐지
"""
import pandas as pd


def calculate_company_mentions(df) -> pd.DataFrame:
    """날짜 · 종목별 언급량을 계산한다."""
    mention_counts = []
    for _, row in df.iterrows():
        for ticker in row["secName"]:
            mention_counts.append({"date": row["date(versionCreated)"], "ticker": ticker, "count": 1})

    mentions_df = pd.DataFrame(mention_counts)
    mentions_df = mentions_df.groupby(["date", "ticker"]).sum().reset_index()
    return mentions_df


def detect_company_mention_spikes(mentions_df: pd.DataFrame, std_factor: float = 2.0) -> pd.DataFrame:
    """언급량이 볼린저 밴드(평균 ± std_factor×표준편차)를 넘는 종목을 찾는다."""
    latest_date = mentions_df["date"].max()
    today = mentions_df[mentions_df["date"] == latest_date]

    rolling_stats = mentions_df.groupby("ticker")["count"].agg(["mean", "std"]).reset_index()
    rolling_stats.rename(columns={"mean": "moving_avg", "std": "std_dev"}, inplace=True)
    rolling_stats = rolling_stats[rolling_stats["ticker"].isin(today["ticker"])]

    today = today.merge(rolling_stats, on="ticker", how="left")
    today["upper_band"] = today["moving_avg"] + (std_factor * today["std_dev"])
    spikes = today[today["count"] > today["upper_band"]].sort_values(by="count", ascending=False)
    return spikes


def calculate_company_sentiment(df) -> pd.DataFrame:
    """날짜 · 종목별 평균 감성 점수를 계산한다."""
    exploded = df.explode("secName").reset_index(drop=True)
    result = exploded.groupby(["date(versionCreated)", "secName"])["sentiment_score"].agg(["mean"]).reset_index()
    result.rename(columns={"mean": "sentiment_score"}, inplace=True)
    return result[["date(versionCreated)", "secName", "sentiment_score"]]


def detect_company_sentiment_spikes(company_sentiment_df: pd.DataFrame, std_factor: float = 1.7):
    """감성 점수가 볼린저 밴드를 벗어난 종목(상승/하락)을 반환한다."""
    latest_date = company_sentiment_df["date(versionCreated)"].max()
    today = company_sentiment_df[company_sentiment_df["date(versionCreated)"] == latest_date]

    stats = company_sentiment_df.groupby("secName")["sentiment_score"].agg(["mean", "std"]).reset_index()
    stats.rename(columns={"mean": "moving_avg", "std": "std_dev"}, inplace=True)
    stats = stats[stats["secName"].isin(today["secName"])]

    merged = today.merge(stats, on="secName", how="left")
    merged["upper_band"] = merged["moving_avg"] + (std_factor * merged["std_dev"])
    merged["lower_band"] = merged["moving_avg"] - (std_factor * merged["std_dev"])

    rising = merged[merged["sentiment_score"] > merged["upper_band"]][
        ["secName", "sentiment_score", "moving_avg", "std_dev", "upper_band"]
    ]
    falling = merged[merged["sentiment_score"] < merged["lower_band"]][
        ["secName", "sentiment_score", "moving_avg", "std_dev", "lower_band"]
    ]

    return rising, falling
