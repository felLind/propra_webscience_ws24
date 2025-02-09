"""
Module for fine-tuning BERT-based models on the sentiment140 dataset.
"""

import dataclasses
import statistics
import time

import pandas as pd
import torch
from datasets import Dataset  # type: ignore[attr-defined]
from loguru import logger
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

from propra_webscience_ws24 import constants
from propra_webscience_ws24.training.llm.data_splits import (
    get_train_and_test_datasets,
    create_train_and_eval_split,
)
from propra_webscience_ws24.training.llm.training_results import EvalResult
from propra_webscience_ws24.training.llm.utils import compute_metrics, tokenize_text

ROBERTA_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
DISTILBERT_MODEL_NAME = "distilbert-base-uncased"

MODEL_NAMES = [DISTILBERT_MODEL_NAME, ROBERTA_MODEL_NAME]

SENTIMENT_MAPS = {
    ROBERTA_MODEL_NAME: {
        0: 0,
        4: 2,
    },
    DISTILBERT_MODEL_NAME: {
        0: 0,
        4: 1,
    },
}

LEARNING_RATES = [1e-4, 5 * 1e-5, 1e-5, 5 * 1e-6, 1e-6]

DATASET_SIZES = [2_500, 5_000, 7_500, 10_000, 15_000, 20_000]

logger.info(f"cuda enabled: {torch.cuda.is_available()}")


def evaluate_default_model(model_name: str, ds_test: Dataset) -> EvalResult:
    """
    Evaluate the default model on the test dataset.

    Args:
        model_name: The name of the model to evaluate.
        ds_test: The test dataset.

    Returns:
        The evaluation results.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized_test = ds_test.map(
        lambda example: tokenize_text(example, tokenizer), batched=True
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="output/",
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_test,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    result = trainer.evaluate()
    eval_result = EvalResult(
        model_name=model_name,
        data_size=None,
        learning_rate=None,
        accuracy=result["eval_accuracy"]["accuracy"],
        f1=statistics.mean(result["eval_f1"]["f1"]),
    )
    logger.info(f"Default Model: {eval_result}")
    return eval_result


def fine_tune_and_evaluate(
    model_name: str,
    ds_train: Dataset,
    ds_eval: Dataset,
    ds_test: Dataset,
    learning_rate: float,
) -> EvalResult:
    """
    Fine-tune the model on the training and evaluation dataset and finally evaluate it on the test dataset.

    Args:
        model_name: The name of the model to fine-tune.
        ds_train: The training dataset.
        ds_eval: The evaluation dataset.
        ds_test: The test dataset.
        learning_rate: The initial learning rate.

    Returns:
        The evaluation results.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized_train = ds_train.map(
        lambda example: tokenize_text(example, tokenizer), batched=True
    )
    tokenized_eval = ds_eval.map(
        lambda example: tokenize_text(example, tokenizer), batched=True
    )
    tokenized_test = ds_test.map(
        lambda example: tokenize_text(example, tokenizer), batched=True
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="output/",
        learning_rate=learning_rate,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info(f"Fine Tuning Model: {model_name=}, {dataset_size=}, {learning_rate=}")
    trainer.train()

    result = trainer.evaluate(tokenized_test)

    eval_result = EvalResult(
        model_name=model_name,
        data_size=len(ds_train) + len(ds_eval),
        learning_rate=learning_rate,
        accuracy=result["eval_accuracy"]["accuracy"],
        f1=statistics.mean(result["eval_f1"]["f1"]),
    )
    logger.info(f"Fine Tuned Model ({model_name}):\n{eval_result}")
    return eval_result


if __name__ == "__main__":
    logger.info("fine tuning started")

    eval_results = []

    for dataset_size in DATASET_SIZES:
        for model_name in MODEL_NAMES:
            ds_train_, ds_test = get_train_and_test_datasets(SENTIMENT_MAPS[model_name])

            if dataset_size == DATASET_SIZES[0]:
                result = evaluate_default_model(model_name, ds_test)
                eval_results.append(result)

            ds_train, ds_eval = create_train_and_eval_split(
                ds_train_, SENTIMENT_MAPS[model_name][4], dataset_size=dataset_size
            )
            for learning_rate in LEARNING_RATES:
                result = fine_tune_and_evaluate(
                    model_name,
                    ds_train,
                    ds_eval,
                    ds_test,
                    learning_rate,
                )
                eval_results.append(result)

    logger.info("fine tuning done")

    for eval_result in sorted(eval_results, key=lambda x: (x.model_name, x.accuracy)):
        logger.info(eval_result)

    df = pd.DataFrame(dataclasses.asdict(eval_result) for eval_result in eval_results)
    timestr = time.strftime("%Y%m%d-%H%M%S")

    df.to_csv(constants.RESULTS_PATH / f"fine_tuning_{timestr}.csv", index=False)
