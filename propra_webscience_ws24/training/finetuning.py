import torch
import statistics
import numpy as np
import pandas as pd
import time
from datasets import combine, load_dataset  # type: ignore[attr-defined]
from evaluate import load
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

untrained_eval = "untrained_eval"


def compute_metrics(eval_pred):
    load_accuracy = load("accuracy")
    load_f1 = load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)
    f1 = load_f1.compute(predictions=predictions, references=labels, average=None)
    return {"accuracy": accuracy, "f1": f1}


def preprocess_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True)


def map_sentiment(example, mapper):
    example["label"] = mapper[example["label"]]
    return example


def add_data_row(rows, model_name, data_size, learning_rate, result):
    row = {
        "model_name": model_name,
        "data_size": data_size,
        "learning_rate": learning_rate,
        "accuracy": result["eval_accuracy"]["accuracy"],
        "f1": statistics.mean(result["eval_f1"]["f1"]),
    }
    rows.append(row)


def train_params(
    train_dataset,
    test_dataset,
    model_name,
    dataset_size,
    learning_rate,
    tokenizer,
    mapper,
    data_rows,
):
    positive = (
        train_dataset.select([i for i in list(range(0, 799999))])
        .shuffle(seed=42)
        .select([i for i in list(range(int(dataset_size / 2)))])
    )
    negative = (
        train_dataset.select([i for i in list(range(800000, 1599999))])
        .shuffle(seed=42)
        .select([i for i in list(range(int(dataset_size / 2)))])
    )
    small_train_dataset = combine.concatenate_datasets([positive, negative])

    small_train_dataset = small_train_dataset.map(
        lambda example: map_sentiment(example, mapper)
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    tokenized_train = small_train_dataset.map(
        lambda example: preprocess_function(example, tokenizer), batched=True
    )

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
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    result = trainer.evaluate()
    print(result)
    add_data_row(data_rows, model_name, dataset_size, learning_rate, result)


def train_model(
    model_name, model_args, dataset, mapper, dataset_sizes, learning_rates, data_rows
):
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    train_dataset = train_dataset.rename_column("sentiment", "label")
    test_dataset = test_dataset.rename_column("sentiment", "label")

    small_train_dataset = train_dataset.shuffle(seed=42).select(
        [i for i in list(range(1000))]
    )

    test_dataset = test_dataset.filter(lambda example: example["label"] != 2)

    test_dataset = test_dataset.map(lambda example: map_sentiment(example, mapper))

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, **model_args[model_name]
    )

    tokenized_train = small_train_dataset.map(
        lambda example: preprocess_function(example, tokenizer), batched=True
    )
    tokenized_test = test_dataset.map(
        lambda example: preprocess_function(example, tokenizer), batched=True
    )

    training_args = TrainingArguments(
        output_dir="output/",
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    result = trainer.evaluate()

    print(result)

    add_data_row(data_rows, model_name, untrained_eval, untrained_eval, result)

    for dataset_size in dataset_sizes:
        for learning_rate in learning_rates:
            print(
                f"model: {model_name}, learning rate: {learning_rate}, dataset size: {dataset_size}"
            )
            train_params(
                train_dataset,
                tokenized_test,
                model_name,
                dataset_size,
                learning_rate,
                tokenizer,
                mapper,
                data_rows,
            )


print(f"cuda enabled: {torch.cuda.is_available()}")

roberta = "cardiffnlp/twitter-roberta-base-sentiment"
distilbert = "distilbert-base-uncased"

model_names = [distilbert, roberta]

sentiment140 = load_dataset("sentiment140", trust_remote_code=True)

sentiment_map_roberta = {
    0: 0,
    4: 2,
}

sentiment_map_distilbert = {
    0: 0,
    4: 1,
}

sentiment_maps = {
    roberta: sentiment_map_roberta,
    distilbert: sentiment_map_distilbert,
}

model_args_distilbert = {
    "num_labels": 2,
}

model_args_roberta = {}  # type: ignore[var-annotated]

model_args = {
    roberta: model_args_roberta,
    distilbert: model_args_distilbert,
}

learning_rates = [1e-3, 1e-4, 1e-5, 1e-6]

dataset_sizes = [2500, 5000, 7500, 10000]

rows = []  # type: ignore[var-annotated]

for model_name in model_names:
    train_model(
        model_name,
        model_args,
        sentiment140,
        sentiment_maps[model_name],
        dataset_sizes,
        learning_rates,
        rows,
    )

df = pd.DataFrame(rows)
timestr = time.strftime("%Y%m%d-%H%M%S")
df.to_csv(f"results/fine_tuning_{timestr}.csv", index=False)
