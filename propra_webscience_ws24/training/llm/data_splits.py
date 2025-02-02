"""
Module for creating train and test datasets for the LLM training.
"""

from datasets import Dataset, load_dataset, concatenate_datasets  # type: ignore[attr-defined]

SEED = 42


def get_train_and_test_datasets(
    sentiment_mapping: dict[int, int],
) -> tuple[Dataset, Dataset]:
    """
    Load the sentiment140 dataset and map the sentiment labels according to the provided mapping.

    The test dataset will not contain any neutral examples.

    Args:
        sentiment_mapping: the mapping to use for the sentiment labels

    Returns:
        A tuple containing the training and test datasets.
    """
    ds_train, ds_test = load_dataset(
        "sentiment140", trust_remote_code=True, split=["train", "test"]
    )

    columns_to_remove = ["date", "user", "query"]

    ds_train = (
        ds_train.rename_column("sentiment", "label")
        .remove_columns(columns_to_remove)
        .map(lambda example: map_sentiment(example, sentiment_mapping))
    )
    ds_test = (
        ds_test.rename_column("sentiment", "label")
        .remove_columns(columns_to_remove)
        .filter(lambda example: example["label"] != 2)
        .map(lambda example: map_sentiment(example, sentiment_mapping))
    )

    return ds_train, ds_test


def create_train_and_eval_split(
    ds: Dataset, positive_class_id: int, dataset_size: int = 100, test_size: float = 0.1
) -> tuple[Dataset, Dataset]:
    """
    Create a train and evaluation split from the provided dataset.

    Args:
        ds: The dataset to split.
        positive_class_id: The id of the positive class.
        dataset_size: The size of the dataset to create.
        test_size: The fraction of the dataset to use for evaluation.

    Returns:
        A tuple containing the training and evaluation datasets
    """
    ds_negative = (
        ds.filter(lambda example: example["label"] == 0)
        .shuffle(seed=SEED)
        .select(range(dataset_size // 2))
    )
    ds_positive = (
        ds.filter(lambda example: example["label"] == positive_class_id)
        .shuffle(seed=SEED)
        .select(range(dataset_size // 2))
    )

    if len(ds_negative) + len(ds_positive) != dataset_size:
        raise RuntimeError("Dataset sizes do not match")

    ds_negative_train_test_split = ds_negative.train_test_split(test_size=test_size)
    ds_positive_train_test_split = ds_positive.train_test_split(test_size=test_size)

    ds_negative_train, ds_negative_eval = (
        ds_negative_train_test_split["train"],
        ds_negative_train_test_split["test"],
    )
    ds_positive_train, ds_positive_eval = (
        ds_positive_train_test_split["train"],
        ds_positive_train_test_split["test"],
    )

    ds_train = concatenate_datasets([ds_negative_train, ds_positive_train]).shuffle(
        seed=SEED
    )
    ds_eval = concatenate_datasets([ds_negative_eval, ds_positive_eval]).shuffle(
        seed=SEED
    )
    return ds_train, ds_eval


def map_sentiment(example: dict, mapper: dict[int, int]) -> dict:
    """
    Map the sentiment label of the example using the provided mapper.

    Args:
        example: The example for which to map the sentiment label.
        mapper: A dictionary mapping the old sentiment labels to the new ones.

    Returns:
        The modified example with the mapped sentiment label.
    """
    example["label"] = mapper[example["label"]]
    return example
