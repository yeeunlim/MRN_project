"""
감성 분석 시각화 (분포 히스토그램, 일별 트렌드)
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from mrn_project.data.mapping import map_ric_to_sec_nm, map_topic_code_to_description
from mrn_project.config import GRAPH_OUTPUT_DIR


# 감성 점수 분포 히스토그램

def plot_sentiment_distribution(df, column: str, title: str, filename: str):
    """감성 점수 분포 히스토그램을 그린다."""
    plt.figure(figsize=(10, 6))
    plt.hist(df[column], bins=30, edgecolor="black")
    plt.title(title)
    plt.xlabel("sentiment score")
    plt.ylabel("frequency")
    plt.grid(True)
    plt.savefig(f"{GRAPH_OUTPUT_DIR}/{filename}")
    plt.show()


# 일별 감성 트렌드 시각화

def visualize_sentiment_trends(daily_sentiment, label_column: str,
                                ric_mapping_dict: dict, topic_code_mapping_dict: dict):
    """subject 별 일별 감성 점수 추이를 시각화한다."""
    plt.figure(figsize=(20, 8))

    daily_sentiment = daily_sentiment.copy()
    daily_sentiment["date(versionCreated)"] = pd.to_datetime(daily_sentiment["date(versionCreated)"])
    daily_sentiment.sort_values("date(versionCreated)", inplace=True)

    full_date_range = pd.DataFrame({
        "date(versionCreated)": pd.date_range(
            start=daily_sentiment["date(versionCreated)"].min(),
            end=daily_sentiment["date(versionCreated)"].max(),
        )
    })
    merged = full_date_range.merge(daily_sentiment, on="date(versionCreated)", how="left")

    subjects = daily_sentiment["subjects"].unique()
    for subject in subjects:
        subject_data = merged[merged["subjects"] == subject]

        if subject.startswith("R:"):
            mapped_name = map_ric_to_sec_nm(subject, ric_mapping_dict)
        elif subject in topic_code_mapping_dict:
            mapped_name = map_topic_code_to_description(subject, topic_code_mapping_dict)
        else:
            mapped_name = subject

        plt.plot(
            subject_data["date(versionCreated)"],
            subject_data["sentiment_score"],
            marker="o",
            label=mapped_name,
        )

    plt.title("sentiment score trends", fontsize=16)
    plt.xlabel("date", fontsize=12)
    plt.ylabel("ave sentiment score", fontsize=12)
    plt.ylim(-1, 1)
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.legend(title=label_column, fontsize=10, loc="upper left", bbox_to_anchor=(1, 1), borderaxespad=0)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{GRAPH_OUTPUT_DIR}/daily_sentiment_trends_{label_column}.png")
    plt.show()
