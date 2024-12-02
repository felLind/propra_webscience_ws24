from typing import Iterator
from transformers import pipeline
import pandas as pd


def roberta_baseline_sentiment_predictions(df: pd.Dataframe) -> Iterator[int]:
    sentiment_analysis = pipeline(
        "sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    return df["text"].apply(sentiment_analysis).toiter()
