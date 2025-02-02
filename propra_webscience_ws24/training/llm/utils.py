"""
Module containing utility functions for the LLM training.
"""

from typing import Callable

import numpy as np
from evaluate import load


def compute_metrics(eval_pred: tuple[list[float], list[int]]) -> dict[str, float]:
    """
    Compute the evaluation metrics for the given predictions

    Args:
        eval_pred: The predictions and labels to evaluate.

    Returns:
        The evaluation metrics (accuracy and F1) in a dictionary.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load("accuracy").compute(predictions=predictions, references=labels)
    f1 = load("f1").compute(predictions=predictions, references=labels, average=None)
    return {"accuracy": accuracy, "f1": f1}


def tokenize_text(example: dict, tokenizer: Callable) -> dict:
    """
    Preprocess the examples using the provided tokenizer.

    Args:
        example: The example with the text to preprocess.
        tokenizer: The tokenizer to use.

    Returns:
        The preprocessed example.
    """
    return tokenizer(example["text"], truncation=True)
