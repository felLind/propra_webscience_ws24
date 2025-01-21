import torch
import numpy as np
from datasets import combine, load_dataset  # type: ignore[attr-defined]
from evaluate import load
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

model_name = "cardiffnlp/twitter-roberta-base-sentiment"

print(f"cuda enabled: {torch.cuda.is_available()}")

sentiment140 = load_dataset("sentiment140", trust_remote_code=True)

dataset_size = 10000

positive = (
    sentiment140["train"]
    .select([i for i in list(range(0, 799999))])
    .shuffle(seed=42)
    .select([i for i in list(range(int(dataset_size / 2)))])
)
negative = (
    sentiment140["train"]
    .select([i for i in list(range(800000, 1599999))])
    .shuffle(seed=42)
    .select([i for i in list(range(int(dataset_size / 2)))])
)

small_train_dataset = combine.concatenate_datasets([positive, negative])


#
# small_train_dataset = (
#     sentiment140["train"].shuffle(seed=42).select([i for i in list(range(10000))])
# )
small_test_dataset = sentiment140["test"]

small_train_dataset = small_train_dataset.rename_column("sentiment", "label")
small_test_dataset = small_test_dataset.rename_column("sentiment", "label")

sentiment_map = {
    0: 0,
    2: 1,
    4: 2,
}


def map_sentiment(example):
    example["label"] = sentiment_map[example["label"]]
    return example


small_train_dataset = small_train_dataset.map(map_sentiment)
small_test_dataset = small_test_dataset.map(map_sentiment)

small_test_dataset = small_test_dataset.filter(lambda example: example["label"] != 1)

tokenizer = AutoTokenizer.from_pretrained(model_name)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
tokenized_test = small_test_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(model_name)


def compute_metrics(eval_pred):
    load_accuracy = load("accuracy")
    load_f1 = load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)
    f1 = load_f1.compute(predictions=predictions, references=labels, average=None)
    return {"accuracy": accuracy, "f1": f1}


training_args = TrainingArguments(
    output_dir="output/",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    save_strategy="epoch",
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


print(trainer.evaluate())

trainer.train()

print(trainer.evaluate())
