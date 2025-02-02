"""
Module containing the data objects for the training results for LLMs.
"""

import dataclasses


@dataclasses.dataclass
class EvalResult:
    """
    A dataclass to store the results of a LLM fine-tuning run.
    """

    model_name: str
    data_size: int | None
    learning_rate: float | None
    accuracy: float
    f1: float
