"""Module containing to use local deepseek models for inference."""

import asyncio
import itertools
import json
from datetime import datetime
from typing import Iterator, Literal, get_args

import pandas as pd
from loguru import logger
from ollama import AsyncClient
from pydantic import BaseModel
from sklearn.metrics import accuracy_score

from propra_webscience_ws24.constants import TEST_DATASET_FILE_PATH

N_PARALLEL_PROMPTS = 10

MODEL_NAME = "deepseek-r1:32b"
# MODEL_NAME = 'deepseek-r1:1.5b'

SENTIMENT_LITERAL = Literal["positive", "negative"]

PROMPT_TEMPLATE = """
Given the following tweet, what is the sentiment of the tweet?
Answer either with positive or negative. Return the reasoning and the sentiment in JSON format.

Tweet: '{tweet}'
"""

PROMPT_OPTIONS = {
    "seed": 42,
    "temperature": 0.0,  # Deterministic output
    "max_tokens": 10,  # Limit the response length
    "top_p": 0.9,  # Focused sampling
    "frequency_penalty": 0.5,  # Reduce repetition
    "presence_penalty": 0.0,  # Neutral on introducing new tokens
}

SENTIMENT_MAPPING = {
    0: get_args(SENTIMENT_LITERAL)[1],
    4: get_args(SENTIMENT_LITERAL)[0],
}


def batched(iterable, n) -> Iterator:
    """
    Yield successive n-sized chunks from iterable.

    Args:
        iterable: The iterable to be batched.
        n: The size of each batch.

    Yields:
        Iterator: An iterator over the batches.
    """
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        yield batch


def stream_raw_tweets_in_batches():
    """
    Stream raw tweets in batches from the dataset.

    Yields:
        Iterator: An iterator over batches of tweets.
    """
    df = pd.read_parquet(TEST_DATASET_FILE_PATH)
    df = df.loc[df.sentiment.isin([0, 4]), :]
    # df = df.sample(2*N_PARALLEL_PROMPTS)
    for batch in batched(df.iterrows(), N_PARALLEL_PROMPTS):
        yield batch


class ModelOutput(BaseModel):
    reasoning: str
    sentiment: SENTIMENT_LITERAL


class EnrichedModelOutput(ModelOutput):
    tweet: str
    ground_truth: SENTIMENT_LITERAL


async def process_tweet_w_model(client: AsyncClient, text: str) -> str:
    """
    Process a single tweet with the model.

    Args:
        client (AsyncClient): The async client to use for the model.
        text (str): The tweet text to be processed.

    Returns:
        str: The model's response.
    """
    response = await client.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(tweet=text)}],
        options=PROMPT_OPTIONS,
        format=ModelOutput.model_json_schema(),
    )
    return response["message"]["content"]


def create_enriched_model_output(
    tweet: str, ground_truth: int, model_output: ModelOutput
) -> EnrichedModelOutput:
    """
    Create an enriched model output.

    Args:
        tweet (str): The original tweet text.
        ground_truth (int): The ground truth sentiment value.
        model_output (ModelOutput): The model output.

    Returns:
        EnrichedModelOutput: The enriched model output.
    """
    return EnrichedModelOutput(
        tweet=tweet,
        ground_truth=SENTIMENT_MAPPING[ground_truth],
        **model_output.model_dump(),
    )


async def process_tweets() -> list[EnrichedModelOutput]:
    """
    Process tweets in batches and get model outputs.

    Returns:
        list[EnrichedModelOutput]: A list of enriched model outputs.
    """
    client = AsyncClient()

    results = []

    idx = 1
    for tweet_batch in stream_raw_tweets_in_batches():
        tweet_data = list(tweet_batch)
        tweets = [tweet[1]["text"] for tweet in tweet_data]
        sentiments = [tweet[1]["sentiment"] for tweet in tweet_data]

        logger.info(f"Batch {idx=}")
        idx += 1

        prompt_results = await asyncio.gather(
            *[process_tweet_w_model(client, tweet) for tweet in tweets]
        )
        prompt_result_models = [
            ModelOutput.model_validate_json(result.strip()) for result in prompt_results
        ]

        results += [
            create_enriched_model_output(tweet, ground_truth, prompt_result_model)
            for tweet, ground_truth, prompt_result_model in zip(
                tweets, sentiments, prompt_result_models
            )
        ]

    if len(results) <= 10:
        logger.info(f"{results=}")
    return results


def save_results(results: list[EnrichedModelOutput]):
    """
    Save the results to a JSON file.

    Args:
        results (list[EnrichedModelOutput]): The results to be saved.
    """
    with open(
        f"{MODEL_NAME.replace(':', '-').lower()}-prompts-results-"
        f'{datetime.now().strftime("%Y%m%d-%H%M%S")}.json',
        "w",
    ) as f:
        json.dump([result.model_dump() for result in results], f, indent=4)


async def calculate_accuracy_for_model_output(
    model_output: list[EnrichedModelOutput],
) -> float:
    """
    Calculate the accuracy of the model output.

    Args:
        model_output (list[EnrichedModelOutput]): The model output to be evaluated.

    Returns:
        float: The accuracy score.
    """
    inverse_sentiment_mapping = {v: k for k, v in SENTIMENT_MAPPING.items()}

    model_sentiment_classes = [
        inverse_sentiment_mapping[o.sentiment] for o in model_output
    ]
    ground_truth_sentiment_classes = [
        inverse_sentiment_mapping[o.ground_truth] for o in model_output
    ]

    return accuracy_score(model_sentiment_classes, ground_truth_sentiment_classes)


async def main():
    logger.info(f"Start sending prompts to model {MODEL_NAME=}")
    model_output = await process_tweets()

    accuracy = await calculate_accuracy_for_model_output(model_output)

    logger.info(f"Accuracy: {accuracy}")

    save_results(model_output)

    logger.info("Finished")


if __name__ == "__main__":
    asyncio.run(main())
