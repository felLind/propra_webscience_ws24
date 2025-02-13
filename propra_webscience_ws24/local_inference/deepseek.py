"""Module containing to use local deepseek models for inference."""

import asyncio
import itertools
from datetime import datetime
from typing import Iterator, Literal, get_args

import pandas as pd
from loguru import logger
from ollama import AsyncClient
from pydantic import BaseModel
from sklearn.metrics import accuracy_score

from propra_webscience_ws24.constants import TEST_DATASET_FILE_PATH

N_PARALLEL_PROMPTS = 10

# MODEL_NAME = 'deepseek-r1:1.5b'
MODEL_NAME = "deepseek-r1:8b"
# MODEL_NAME = "deepseek-r1:32b"
# MODEL_NAME = "deepseek-r1:70b"

SENTIMENT_LITERAL = Literal["positive", "negative"]

QUERY_TERM_TEMPLATE = "Sentiment topic: '{query}'\n"
PROMPT_TEMPLATE = (
    "Tweet sentiment?\n{query_term_prompt}Answer with positive or "
    "negative. Provide reasoning in JSON.\nTweet: '{tweet}'\n"
)

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
    query_term: str | None


class InferenceResults(BaseModel):
    accuracy: float
    with_query_term: bool
    results: list[EnrichedModelOutput]


async def process_tweet_w_model(
    client: AsyncClient, text: str, query_term: str | None = None
) -> str:
    """
    Process a single tweet with the model.

    Args:
        client (AsyncClient): The async client to use for the model.
        text (str): The tweet text to be processed.
        query_term (str): The query term to be used if not None.

    Returns:
        str: The model's response.
    """
    query_term_part = (
        QUERY_TERM_TEMPLATE.format(query=query_term) if query_term is not None else ""
    )
    prompt = PROMPT_TEMPLATE.format(tweet=text, query_term_prompt=query_term_part)
    response = await client.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options=PROMPT_OPTIONS,
        format=ModelOutput.model_json_schema(),
    )
    return response["message"]["content"]


def create_enriched_model_output(
    tweet: str, ground_truth: int, query_term: str | None, model_output: ModelOutput
) -> EnrichedModelOutput:
    """
    Create an enriched model output.

    Args:
        tweet (str): The original tweet text.
        ground_truth (int): The ground truth sentiment value.
        query_term (str): The query term or None
        model_output (ModelOutput): The model output.

    Returns:
        EnrichedModelOutput: The enriched model output.
    """
    return EnrichedModelOutput(
        tweet=tweet,
        ground_truth=SENTIMENT_MAPPING[ground_truth],
        query_term=query_term,
        **model_output.model_dump(),
    )


async def process_tweets(with_query_term: bool = False) -> list[EnrichedModelOutput]:
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

        if with_query_term:
            query_terms = [
                tweet[1]["query"] if with_query_term else None for tweet in tweet_data
            ]
        else:
            query_terms = [None] * len(tweets)

        logger.info(f"Batch {idx=}")
        idx += 1

        prompt_results = await asyncio.gather(
            *[
                process_tweet_w_model(client, tweet, query_terms[idx])
                for idx, tweet in enumerate(tweets)
            ]
        )
        prompt_result_models = [
            ModelOutput.model_validate_json(result.strip()) for result in prompt_results
        ]

        results += [
            create_enriched_model_output(
                tweet, ground_truth, query_term, prompt_result_model
            )
            for tweet, ground_truth, query_term, prompt_result_model in zip(
                tweets, sentiments, query_terms, prompt_result_models
            )
        ]

    if len(results) <= 10:
        logger.info(f"{results=}")
    return results


def save_results(
    results: list[EnrichedModelOutput], accuracy: float, with_query_term: bool
):
    """
    Save the results to a JSON file.

    Args:
        results (list[EnrichedModelOutput]): The results to be saved.
        accuracy (float): The accuracy of the model output.
        with_query_term (bool): Whether the query term was used.
    """
    with open(
        f"{MODEL_NAME.replace(':', '-').lower()}-prompts-results-"
        f"{'with' if with_query_term else 'without'}-query-term-"
        f'{datetime.now().strftime("%Y%m%d-%H%M%S")}.json',
        "w",
    ) as f:
        f.write(
            InferenceResults(
                accuracy=accuracy, with_query_term=with_query_term, results=results
            ).model_dump_json(indent=4)
        )


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
    for with_query_term in [False, True]:
        logger.info(f"Processing tweets with query term: {with_query_term=}")
        model_output = await process_tweets(with_query_term)

        accuracy = await calculate_accuracy_for_model_output(model_output)

        logger.info(f"Accuracy: {accuracy}")

        save_results(model_output, accuracy, with_query_term)

    logger.info("Finished")


if __name__ == "__main__":
    asyncio.run(main())
