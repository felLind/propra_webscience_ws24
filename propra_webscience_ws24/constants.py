import pathlib

# Datasets
DATASETS_ROOT_PATH = pathlib.Path(".").absolute().parent / "datasets" / "sentiment140"
TRAIN_DATASET_FILE_NAME = "train-data.parquet"
TEST_DATASET_FILE_NAME = "test-data.parquet"

TRAIN_DATASET_FILE_PATH = DATASETS_ROOT_PATH / TRAIN_DATASET_FILE_NAME
TEST_DATASET_FILE_PATH = DATASETS_ROOT_PATH / TEST_DATASET_FILE_NAME

PREPARED_TRAIN_DATASET_FILE_PATH = TRAIN_DATASET_FILE_PATH.with_stem(
    f"{TRAIN_DATASET_FILE_PATH.stem}-prepared"
)
PREPARED_TEST_DATASET_FILE_PATH = TEST_DATASET_FILE_PATH.with_stem(
    f"{TEST_DATASET_FILE_PATH.stem}-prepared"
)

# Datasets with Embedding


# Word Embeddings
GLOVE_EMBEDDING_27B_25D_FILE_PATH = (
    DATASETS_ROOT_PATH / "word_embeddings" / "glove.twitter.27B.25d.txt"
)
GLOVE_EMBEDDING_27B_50D_FILE_PATH = (
    DATASETS_ROOT_PATH / "word_embeddings" / "glove.twitter.27B.50d.txt"
)
GLOVE_EMBEDDING_27B_100D_FILE_PATH = (
    DATASETS_ROOT_PATH / "word_embeddings" / "glove.twitter.27B.100d.txt"
)
GLOVE_EMBEDDING_27B_200D_FILE_PATH = (
    DATASETS_ROOT_PATH / "word_embeddings" / "glove.twitter.27B.200d.txt"
)


# Results
CLASSIFICATION_RESULTS_OUTPUT_FILE_PATH = (
    DATASETS_ROOT_PATH / "classification-results.json"
)
