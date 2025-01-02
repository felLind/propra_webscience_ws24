# Projektpraktikum Webscience WS24-25 - Stimmungsanalyse mit Twitter

Data-Source: [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)

## Projekt-Einrichtung

Wir verwenden _Poetry_ zur Verwaltung von verwendeten _Python_ Paketen und [_pre-commit_](https://pre-commit.com/) hooks um insbesondere eine einheitliche Formatierung unserer Dateien zu erreichen.

Um das Projekt lokal aufzusetzen müssen folgende Schritte ausgeführt werden:

1. Installation von _Poetry_ (s. [_Poetry_ Dokumentation](https://python-poetry.org/docs/))

2. Installation der verwendeten _Python_ Pakete: `poetry install`

3. Installation der _pre-commit_ hooks: `poetry run pre-commit install`

## Daten-Vorverarbeitung und Training von ML-Modellen

### Erstellung der vorverarbeiteten Datensätze

> [!NOTE]
> Dieser Schritt ist notwendig für alle weiteren Schritte.

Um den ursprünglichen Datensatz zu bereinigen und die Daten vorzuverarbeiten kann der folgende Befehl ausgeführt werden:

```bash
# In einer Poetry shell und nachdem die restlichen Python Pakete installiert wurden
python -m propra_webscience_ws24.data.data_preprocessing
```

Durch Aufruf dieses Befehls werden die folgenden Schritte (nach Download des Trainings- und Testdatensatzes) ausgeführt:

- Bereinigung der Tweets

  - _URLs_ werden entfernt
  - Hashtags werden entfernt
  - Ziffernfolgen werden entfernt
  - Erwähnungen von Nutzern (bspw. `@NASA`) werden entfernt
  - Sonderzeichen (aber nicht einfache Anführungszeichen weil diese in Kontraktionen wie `don't` verwendet werden) werden entfernt

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

### Training von ML Modellen

#### Auswahl Vectorizer Varianten

Für die Umwandlung der Textdaten in numerische Vektoren können folgende Vectorizer verwendet werden:

- `TfidfVectorizer`
- `HashingVectorizer`

Bei den Vectorizer-Varianten können die Parameter für die Anzahl der Merkmale und der berücksichtigten Wort-Gruppen verändert werden:

- Anzahl der berückschtigten Merkmale: `max_features`
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

- _Support-Vector-Machines_
- ...

#### Ausführung des Trainings

> [!WARNING]
> Das Training aller Kombination kann je nach Hardware lange dauern, da für jede Kombination von Vorverarbeitungsschritten ein Modell trainiert wird.
> Deshalb wird empfohlen, die Anzahl der zu trainierenden Modelle durch Verwendung der nachfolgend genannten Parameter zu reduzieren.

- Um eine ML Methode auf allen erstellten Datensätzen mit den Vectorizer-Varianten zu trainieren kann der folgende Befehl ausgeführt werden:

  ```bash
  # In einer Poetry shell und nachdem die restlichen Python Pakete installiert wurden
  python -m propra_webscience_ws24.training.main --model-type=LINEAR_SVC
  ```

- Um weniger Modelle zu trainieren können die folgenden Parameter angepasst werden:

  | Parameter                     | Beschreibung                                    | Mögliche Argumente                                                                 |
  |-------------------------------|-------------------------------------------------|------------------------------------------------------------------------------------|
  | `--model-type`                | Typ des zu trainierenden Modells                | `LINEAR_SVC`                                                                       |
  | `--normalization-strategy`    | Text-Normalisierungsstrategie                   | `NONE`, `LEMMATIZER`, `PORTER`                                                     |
  | `--stopword-removal-strategy` | Strategie zur Entfernung von Stoppwörtern       | `KEEP`, `REMOVE_DEFAULT_NLTK`, `REMOVE_CUSTOM`                                     |
  | `--vectorizer`                | Vectorizer zur Umwandlung von Textdaten         | `TFIDF`, `HASHING`                                                                 |
  | `--max-features`              | Maximale Anzahl der Merkmale für den Vectorizer | `ALL_FEATURES`, `10000`, `50000`, `250000`                                         |
  | `--ngram-range`               | N-Gramm-Bereich für den Vectorizer              | `UNIGRAMS`, `UNI_AND_BIGRAMS`, `UNI_AND_BI_AND_TRIGRAMS`                           |
  | `--max-workers`               | Anzahl der zu verwendenden Worker               | Positive ganze Zahl (default `1`, d.h. sequentiell)                                |

  Beispiel um alle Kombinationen für ein lineares _SUport-Vector-Machine_ Modell zu trainieren, bei dem die Token nicht normlisiert wurden, alle Stoppwörter enthalten sind und ein `TfidfVectorizer` verwendet wird:
  ```bash
  python main.py --model-type LINEAR_SVC --normalization-strategy=NONE --stopword-removal-strategy=KEEP --vectorizer=TFIDF --max-workers 4
  ```
