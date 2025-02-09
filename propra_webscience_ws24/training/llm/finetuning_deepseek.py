import dataclasses
import statistics
import time

import pandas as pd
import torch
from loguru import logger
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from transformers import AutoTokenizer

from propra_webscience_ws24 import constants
from propra_webscience_ws24.training.llm.data_splits import (
    get_train_and_test_datasets,
    create_train_and_eval_split,
)
from propra_webscience_ws24.training.llm.utils import tokenize_text, compute_metrics

SEED = 42
DEFAULT_BATCH_SIZE = 16
DEEPSEEK_MODEL_NAME_8B = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DEEPSEEK_MODEL_NAME_1_5B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# MODEL_NAMES = [DEEPSEEK_MODEL_NAME_1_5B, DEEPSEEK_MODEL_NAME_8B]
MODEL_NAMES = [DEEPSEEK_MODEL_NAME_1_5B]

SENTIMENT_MAPPING = {
    0: 0,
    4: 1,
}

ORIGINAL_MODEL_OUTPUT_DIM = {
    DEEPSEEK_MODEL_NAME_1_5B: 1536,
    DEEPSEEK_MODEL_NAME_8B: 4096,
}

LEARNING_RATES = [1e-4, 5 * 1e-5, 1e-5, 5 * 1e-6, 1e-6]

DATASET_SIZES = [2_500, 5_000, 7_500, 10_000, 15_000, 20_000]

logger.info(f"cuda enabled: {torch.cuda.is_available()}")


def map_sentiment(example, mapper):
    example["label"] = mapper[example["label"]]
    return example


@dataclasses.dataclass
class EvalResult:
    model_name: str
    data_size: int | None
    learning_rate: float | None
    accuracy: float
    f1: float


def create_deepseek_classifier_model(model_name, output_dim=1536, n_labels=2):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    class CustomHead(nn.Module):
        def __init__(self, output_dim, n_labels):
            super().__init__()
            self.dense = nn.Linear(
                in_features=output_dim, out_features=output_dim, bias=True
            )
            self.dropout = nn.Dropout(p=0.1)
            self.out_proj = nn.Linear(
                in_features=output_dim, out_features=n_labels, bias=False
            )

        def forward(self, x):
            x = self.dense(x)
            x = torch.relu(x)  # Todo: Modify?
            x = self.dropout(x)
            x = self.out_proj(x)
            return x

    # Replace the model's classifier (last layer)
    model.score = CustomHead(output_dim, n_labels)

    logger.debug(f"{model=}")

    return model


def get_original_model_output_dim(model_name) -> int:
    try:
        return ORIGINAL_MODEL_OUTPUT_DIM[model_name]
    except KeyError:
        raise ValueError(
            f"Model {model_name} not defined in the 'ORIGINAL_MODEL_OUTPUT_DIM' map. "
            f"New models must be specified in the map."
        )


def train_classifier_and_evaluate(
    model_name, dataset_size, ds_train, ds_eval, ds_test, learning_rate
):
    model = create_deepseek_classifier_model(
        model_name, output_dim=get_original_model_output_dim(model_name)
    )
    model.config.pad_token_id = model.config.eos_token_id
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

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

    batch_size = DEFAULT_BATCH_SIZE
    if dataset_size > 5_000:
        # For larger datasets, the batch size must be reduced to avoid memory errors
        batch_size = DEFAULT_BATCH_SIZE // 2

    training_args = TrainingArguments(
        output_dir="output/",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=2,
        save_strategy="no",
        fp16=True,
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

    logger.info(f"Fine Tuning Model: {dataset_size=}, {learning_rate=}")
    trainer.train()

    logger.info("Evaluating Model with test data...")
    result = trainer.evaluate(tokenized_test)

    eval_result = EvalResult(
        model_name=model_name,
        data_size=len(ds_train) + len(ds_eval),
        learning_rate=learning_rate,
        accuracy=result["eval_accuracy"]["accuracy"],
        f1=statistics.mean(result["eval_f1"]["f1"]),
    )
    logger.info(f"Fine Tuned Model:\n{eval_result}")
    return eval_result


if __name__ == "__main__":
    logger.info("fine tuning started")

    eval_results = []

    for model_name in MODEL_NAMES:
        logger.info(f"Fine tuning model: {model_name}")
        for dataset_size in DATASET_SIZES:
            ds_train_, ds_test = get_train_and_test_datasets(SENTIMENT_MAPPING)

            ds_train, ds_eval = create_train_and_eval_split(
                ds_train_, SENTIMENT_MAPPING[4], dataset_size=dataset_size
            )
            for learning_rate in LEARNING_RATES:
                result = train_classifier_and_evaluate(
                    model_name,
                    dataset_size,
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

    df.to_csv(
        constants.RESULTS_PATH / f"fine_tuning_deepseek_{timestr}.csv", index=False
    )
