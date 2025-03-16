# Projektpraktikum Webscience WS24-25 - Stimmungsanalyse mit Twitter

Data-Source: [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)

## Projekt-Einrichtung

Wir verwenden _Poetry_ zur Verwaltung von verwendeten _Python_-Paketen und [_pre-commit_](https://pre-commit.com/) hooks um insbesondere eine einheitliche Formatierung unserer Dateien zu erreichen.

Um das Projekt lokal aufzusetzen, müssen folgende Schritte ausgeführt werden:

1. Installation von _Poetry_ (s. [_Poetry_ Dokumentation](https://python-poetry.org/docs/))

2. Installation der verwendeten _Python_ Pakete: `poetry install`

3. Installation der _pre-commit_ hooks: `poetry run pre-commit install`

## Daten-Vorverarbeitung und Training von ML-Modellen

### Erstellung der vorverarbeiteten Datensätze

> [!NOTE]
> Dieser Schritt ist notwendig für alle weiteren Schritte.

Um den ursprünglichen Datensatz zu bereinigen und die Daten vorzuverarbeiten kann der folgende Befehl ausgeführt werden:

```bash
python -m propra_webscience_ws24.data.data_preprocessing
```

Durch Aufruf dieses Befehls werden die folgenden Schritte (nach Download des Trainings- und Testdatensatzes) ausgeführt:

- Bereinigung der Tweets

  - _URLs_ werden entfernt
  - Hashtags werden entfernt
  - Ziffernfolgen werden entfernt
  - Erwähnungen von Nutzern (bspw. `@NASA`) werden entfernt
  - Sonderzeichen (aber nicht einfache Anführungszeichen, weil diese in Kontraktionen wie `don't` verwendet werden) werden entfernt

- Vorverarbeitung der Tweets (s. `data_preprocessing.py`)

  Es werden Kombinationen der folgenden Vorverarbeitungsschritte ausgeführt, wobei für jede Kombination ein eigener Datensatz erstellt wird:

  - Text-Normalisierungen (s. Parameter `--normalization-strategy` im [folgenden Abschnitt](#training-von-ml-modellen))

    - Ohne Normalisierung, d.h. alle Token werden ohne Anpassung übernommen.
    - Lemmatization: `WordNetLemmatizer`
    - Stemmer: `PorterStemmer`

  - Umgang mit Stopp-Wörtern (s. Parameter `--stopword-removal-strategy` im [folgenden Abschnitt](#training-von-ml-modellen))

    - Beibehaltung aller Stopp-Wörter
    - Entfernung der Stopp-Wörter gemäß  `nltk.corpus.stopwords.words("english")` 
    - Entfernung der Stopp-Wörter mit angepasster Menge an Stopp-Wörtern (s. `_get_custom_stopwords` für Details)

### Training von klassischen ML Modellen

#### Auswahl Vectorizer Varianten

Für die Umwandlung der Textdaten in numerische Vektoren können folgende Vectorizer verwendet werden:

- `TfidfVectorizer`
- `HashingVectorizer`

Bei den Vectorizer-Varianten können die Parameter für die Anzahl der Merkmale und der berücksichtigten Wort-Gruppen verändert werden:

- Anzahl der berücksichtigten Merkmale: `max_features`
  Zulässige Werte: 

  - `10_000`: Es werden maximal 10.000 Merkmale berücksichtigt
  - `50_000`: Es werden maximal 50.000 Merkmale berücksichtigt
  - `250_000`: Es werden maximal 250.000 Merkmale berücksichtigt
  - `ALL_FEATURES`: Alle Merkmale werden berücksichtigt

- Anzahl der berücksichtigten Wort-Gruppen: `ngram_range`

  - `UNIGRAMS`: Es werden nur einzelne Wörter berücksichtigt
  - `UNI_AND_BIGRAMS`: Es werden einzelne Wörter und Wort-Paare berücksichtigt
  - `UNI_AND_BI_AND_TRIGRAMS`: Es werden einzelne Wörter, Wort-Paare und Wort-Tripel berücksichtigt

#### Auswahl ML Methoden

Es können die folgenden ML Methoden trainiert werden:

- `LINEAR_SVC`: _Support-Vector-Machines_
- `DECISION_TREE`: Entscheidungsbaum
- `KNN`: k-nächste Nachbarn
- `LOGISTIC_REGRESSION`: logistische Regression
- `NAIVE_BAYES`: naiver Bayes(Bernoulli)
- `RANDOM_FOREST`: Entscheidungswald

#### Ausführung des Trainings

> [!WARNING]
> Das Training aller Kombination kann je nach Hardware lange dauern, da für jede Kombination von Vorverarbeitungsschritten ein Modell trainiert wird.
> Deshalb wird empfohlen, die Anzahl der zu trainierenden Modelle durch Verwendung der nachfolgend genannten Parameter zu reduzieren.

- Um eine ML Methode auf allen erstellten Datensätzen mit den Vectorizer-Varianten zu trainieren, kann der folgende Befehl ausgeführt werden:

  ```bash
  python -m propra_webscience_ws24.training.classical.main --model-type=LINEAR_SVC
  ```

- Um weniger Modelle zu trainieren, können die folgenden Parameter angepasst werden:

  | Parameter                     | Beschreibung                                    | Mögliche Argumente                                                                          |
  |-------------------------------|-------------------------------------------------|---------------------------------------------------------------------------------------------|
  | `--model-type`                | Typ des zu trainierenden Modells                | `LINEAR_SVC`, `LOGISTIC_REGRESSION`, `NAIVE_BAYES`, `KNN`, `DECISION_TREE`, `RANDOM_FOREST` |
  | `--normalization-strategy`    | Text-Normalisierungsstrategie                   | `NONE`, `LEMMATIZER`, `PORTER`                                                              |
  | `--stopword-removal-strategy` | Strategie zur Entfernung von Stoppwörtern       | `KEEP`, `REMOVE_DEFAULT_NLTK`, `REMOVE_CUSTOM`                                              |
  | `--vectorizer`                | Vectorizer zur Umwandlung von Textdaten         | `TFIDF`, `HASHING`                                                                          |
  | `--max-features`              | Maximale Anzahl der Merkmale für den Vectorizer | `ALL_FEATURES`, `10000`, `50000`, `250000`                                                  |
  | `--ngram-range`               | N-Gramm-Bereich für den Vectorizer              | `UNIGRAMS`, `UNI_AND_BIGRAMS`, `UNI_AND_BI_AND_TRIGRAMS`                                    |
  | `--max-workers`               | Anzahl der zu verwendenden Worker               | Positive ganze Zahl (default `1`, d.h. sequentiell)                                         |
  | `--model-args`                | spezifische Argumente für die Modelle           | "Key1:value,Key2:value,..."                                                                 |

  Beispiel um alle Kombinationen für ein lineares _Support-Vector-Machine_ Modell zu trainieren, bei dem die Token nicht normalisiert wurden, alle Stoppwörter enthalten sind und ein `TfidfVectorizer` verwendet wird:
  ```bash
  python -m propra_webscience_ws24.training.classical.main.py --model-type LINEAR_SVC --normalization-strategy=NONE --stopword-removal-strategy=KEEP --vectorizer=TFIDF --max-workers 4
  ```

### Fine-Tuning von BERT-basierten Modellen

Für das Fine-Tuning von BERT-basierten Modellen wird die Bibliothek `transformers` von _Hugging Face_ verwendet, um auf bereits vortrainierte Modell zuzugreifen.
Es werden die vortrainierten Modelle `distilbert-base-uncased` und `roberta-base` verwendet.

- [`cardiffnlp/twitter-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)

- [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased)

#### Parameter für das Fine-Tuning

Um die Auswirkung unterschiedlicher Parameter auf das Fine-Tuning zu untersuchen, werden die folgenden Parameter verwendet:

- Initiale Lernrate: `[1e-4, 5 * 1e-5, 1e-5, 5 * 1e-6, 1e-6]`
- Daten-Größe (Anzahl aus Beispielen aus dem Gesamt-Datensatz): `[2_500, 5_000, 7_500, 10_000, 15_000, 20_000]`

#### Ausführung des Fine-Tunings für die beiden Modelle

Um das Fine-Tuning für die beiden Modelle durchzuführen, kann der folgende Befehl ausgeführt werden:

```bash
python -m propra_webscience_ws24.training.llm.finetuning_bert_based 
```

### Fine-Tuning von Deepseek-basierten Modellen

Für das Fine-Tuning von Deepseek-basierten Modellen, können die Modelle analog zum Vorgehen für BERT-basierte Modelle geladen werden (s. [Fine-Tuning von BERT-basierten Modellen](#fine-tuning-von-bert-basierten-modellen)).
Es wird aufgrund der Speicheranforderung lediglich das kleinste distilled Modell verwendet (alle anderen Modelle können nicht ohne Anpassung auf ein GPU mit 40GB RAM geladen werden):

- [`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)

#### Parameter für das Fine-Tuning

Um die Auswirkung unterschiedlicher Parameter auf das Fine-Tuning zu untersuchen, werden die gleichen Parameter verwendet, wie für das Fine-Tuning von BERT-basierten Modellen.

Zusätzlich wird die Option verwendet, Gleitkommazahlen mit 16-bit, anstelle der standardmäßigen 32-bit, zu verwenden.
Dadurch wird der Speicherbedarf reduziert und das Training beschleunigt.

#### Ausführung des Fine-Tunings für die beiden Modelle

Um das Fine-Tuning für die beiden Modelle durchzuführen, kann der folgende Befehl ausgeführt werden:

```bash
python -m propra_webscience_ws24.training.llm.finetuning_deepseek 
```

## Direkte Verwendung von DeepSeek R1 über lokale Installation

Um DeepSeek R1 lokal zu verwenden, muss zunächst Ollama installiert werden:
> [!WARNING]
> Der nachfolgende Befehl installiert Ollama ohne Prüfung der Quelle. Es wird empfohlen, die Quelle zu überprüfen, bevor der Befehl ausgeführt wird.

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

> [!NOTE]
> [Ollama](https://github.com/ollama/ollama) empfiehlt mindestens 16 GB RAM für die Verwendung von Modellen mit 13 Milliarden Parametern und 32 GB für Modelle mit 33 Milliarden Parametern.

Anschließend kann das DeepSeek R1 mit 32 Milliarden Parametern wie folgt geladen werden:
```bash
ollama pull deepseek-r1:32b
```

_ollama_ kann mit dem Befehl `ollama serve` gestartet werden.
Mit dem nachfolgenden Aufruf können die Tweets des Testdatensatzes über die exponierte (lokale) REST-API and das ausgewählte Deepseek R1 Modell übergeben werden:
```bash
python -m propra_webscience_ws24.local_inference.deepseek
```
