import requests
import zipfile

from propra_webscience_ws24.constants import (
    DATASETS_ROOT_PATH,
)


def download_glove_word_embeddings():
    glove_twitter_url = (
        "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.twitter.27B.zip"
    )

    response = requests.get(glove_twitter_url, timeout=10, stream=True)
    response.raise_for_status()

    zip_file_path = DATASETS_ROOT_PATH / "word_embeddings" / "original_file.zip"
    zip_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(zip_file_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(zip_file_path.parent)

    zip_file_path.unlink()
