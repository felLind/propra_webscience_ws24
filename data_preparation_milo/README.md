
# Sentiment Analysis with Sentiment140 Dataset

Dieses Projekt analysiert die Stimmung von Tweets, um positive und negative Inhalte zu klassifizieren. Dazu verwenden wir das **Sentiment140**-Datenset und bereiten die Daten umfassend auf, um ein Machine-Learning-Modell zu trainieren.

## Inhaltsverzeichnis
1. [Importieren der Bibliotheken](#1-importieren-der-bibliotheken)
2. [Daten Laden und Überprüfen](#2-daten-laden-und-überprüfen)
3. [Explorative Datenanalyse (EDA)](#3-explorative-datenanalyse-eda)
4. [Datenbereinigung (Data Cleaning)](#4-datenbereinigung-data-cleaning)
5. [Text-Vorverarbeitung](#5-text-vorverarbeitung)
6. [Speichern der Vorverarbeiteten Daten](#6-speichern-der-vorverarbeiteten-daten)

---

### 1. Importieren der Bibliotheken

Wir importieren die notwendigen Bibliotheken für Datenverarbeitung, Visualisierung und Textverarbeitung:

- `pandas`: Zum Laden und Verarbeiten der Daten.
- `matplotlib` & `seaborn`: Zur Visualisierung der Datenverteilung und der Tweet-Längen.
- `nltk`: Für die Textverarbeitung, insbesondere die Bereinigung und Vorverarbeitung.

Zusätzlich laden wir einige **NLTK-Ressourcen** wie `stopwords`, `punkt` und `wordnet`, um die Textdaten effizient zu verarbeiten.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

---

### 2. Daten Laden und Überprüfen

Wir lesen die CSV-Datei ein und passen die Spaltenbezeichnungen an, da die Originaldaten keine Kopfzeile hatten. Die wichtigsten Spalten sind:

- **Sentiment**: 0 für negative und 4 für positive Tweets.
- **Text**: Der eigentliche Inhalt des Tweets.
- **ID**, **Date**, **Query**, **Username**: Metadaten, die für die Sentiment-Analyse nicht relevant sind.

Zur Überprüfung wird die Datenstruktur zusammen mit den ersten Zeilen der Daten angezeigt.

```python
df = pd.read_csv('sentiment140.csv', header=None)
df.columns = ['Sentiment', 'ID', 'Date', 'Query', 'Username', 'Text']
df.head()
```

---

### 3. Explorative Datenanalyse (EDA)

#### Sentiment-Verteilung
Wir visualisieren die Anzahl der positiven und negativen Tweets mit `sns.countplot`, um zu prüfen, ob die Daten gut verteilt sind.

```python
sns.countplot(x='Sentiment', data=df)
plt.title("Sentiment-Verteilung")
plt.xlabel("Sentiment")
plt.ylabel("Anzahl der Tweets")
plt.show()
```

#### Textlängenverteilung
Um die Länge der Tweets zu untersuchen, berechnen wir die Länge jedes Tweets und visualisieren die Verteilung.

```python
df['Text Length'] = df['Text'].apply(len)
df['Text Length'].plot(kind='hist', bins=50)
plt.title("Verteilung der Textlängen")
plt.xlabel("Textlänge")
plt.ylabel("Anzahl der Tweets")
plt.show()
```

---

### 4. Datenbereinigung (Data Cleaning)

Wir erstellen eine Funktion zur Bereinigung des Textes, die Folgendes umsetzt:

- Entfernen von `@-Erwähnungen`, um Benutzerreferenzen zu vermeiden.
- Entfernen von URLs, um irrelevante Links zu beseitigen.
- Entfernen von Sonderzeichen und Ziffern.
- Umwandeln des gesamten Texts in Kleinbuchstaben.

Die bereinigten Texte werden in einer neuen Spalte `clean_text` gespeichert.

```python
import re

def clean_text(text):
    text = re.sub(r'@\w+', '', text)  # Entfernen von @-Erwähnungen
    text = re.sub(r'http\S+', '', text)  # Entfernen von URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Entfernen von Sonderzeichen und Ziffern
    text = text.lower()  # Umwandeln in Kleinbuchstaben
    return text

df['clean_text'] = df['Text'].apply(clean_text)
```

---

### 5. Text-Vorverarbeitung

#### Stopwords und Lemmatizer einrichten
Wir initialisieren Stopwords und einen Lemmatizer für die Textverarbeitung:

- **Stopwords** sind häufige Wörter ohne großen Informationsgehalt, die entfernt werden sollen.
- **Lemmatizer** reduziert Wörter auf ihre Grundform, um Varianten eines Wortes zu vereinheitlichen.

```python
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
```

#### Text-Tokenisierung und -Lemmatisierung
In diesem Schritt wird der bereinigte Text in einzelne Wörter zerlegt, Stopwords entfernt und die verbleibenden Wörter lemmatisiert. Der vorverarbeitete Text wird in einer neuen Spalte `processed_text` gespeichert.

```python
def preprocess_text(text):
    tokens = nltk.word_tokenize(text)  # Tokenisierung
    tokens = [word for word in tokens if word not in stop_words]  # Entfernen von Stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatisierung
    return ' '.join(tokens)

df['processed_text'] = df['clean_text'].apply(preprocess_text)
```

---

### 6. Speichern der Vorverarbeiteten Daten

Die endgültig bereinigten und vorverarbeiteten Daten werden in einer CSV-Datei namens `processed_sentiment140.csv` gespeichert. Diese Datei enthält nur die `Sentiment`- und `processed_text`-Spalten und ist bereit für das Training von Machine-Learning-Modellen.

```python
df[['Sentiment', 'processed_text']].to_csv('processed_sentiment140.csv', index=False)
```

---

## Abschluss

Die Daten sind nun vollständig vorverarbeitet und bereit für die Verwendung in einem Machine-Learning-Modell zur Sentiment-Analyse. Das Ziel dieser Schritte war es, die Texte zu bereinigen, irrelevante Informationen zu entfernen und eine strukturierte Textrepräsentation zu schaffen, die für Modellierungszwecke geeignet ist.


