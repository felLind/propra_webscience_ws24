
==============================================

### ProPra Web Science Protokoll 05.11.24

Datum: 05.11.24 19:00 - 20:15  
Ort: Discord  
Teilnehmer: Andreas, Anne, Burak, Felix, Milo  
Fehlende Personen: niemand

==============================================


### Inhalt:
- Alle: Gegenseitige Vorstellung der Preprocessing Schritte
	- das allgemeine Vorgehen war ähnlich:
		- clean tweets (Username, Links, Sonderzeichen)
		- Tokenizing
		- stopwords entfernen
		- Stemming / Lemmatizing
	- Unterschiede /Auffälligkeiten:
		- Felix: Tweets nach den unterschiedlichen Preprocessing Schritten in separaten Spalten, unterschiedliches Tokenizing bei Benutzung unterschiedlicher nltk packages
		- Burak: Vergleich der gegebenen Klassifikation mit roBERTa
		- Anne: entfernte Stopwords enthalten Wörter wie "not", es gibt doppelte Tweets mit unterschiedlichem Sentiment im Datensatz
		- Milo: alle irrelevanten Spalten gelöscht, leere Tweets nach dem Lemmatizing
		- Andreas: Beschreibung, wie der Datensatz hergestellt wurde, Preprocessing des Testdatensatzes
	- => Preprocessed-Datensatz ist nicht final, sondern muss evtl. noch für die Klassifikationsmethoden angepasst werden.
- Alle: Diskussion / Auswahl der klassischen Methoden
	- Welche Methoden können wir anwenden? 
	  ->SVM, Naiver Bayes Klassifikator, K-Nächster Nachbar, Entscheidungsbäume, Logistische Regression
	- Felix: Vorschlag: alle 5 Methoden ausprobieren und 2 gut funktionierende und 1 schlecht funktionierendes Modell am Ende auswählen
	- alle sind damit einverstanden, dass wir 5 Modell machen
	- Aufteilung der Methoden:
		- SVM: Andreas
		- Naiver Bayes Klassifikator: Burak
		- K-Nächster Nachbar: Anne
		- Entscheidungsbäume: Felix
		- Logistische Regression: Milo
- Diskussion über den Deep Learning Ansatz und die eigene Implementierung
	- Andreas: als existierenden Ansatz BERT und als eigenen Ansatz Feintuning eines Pretrained-Model
	- wird später noch genauer geklärt, bzw. kann nachgefragt werden
- Milo: Sollen wie einen gemeinsamen Datensatz nutzen, um die klassischen Modelle zu trainieren? -> Im Moment nicht, da der Datensatz für die einzelnen Modell noch modifiziert wird
- Zeitplan bis zur Zwischenpräsentation (17.12.)
	- 3 Wochen klassische Modell trainieren
	- 3 Wochen Ergebnisse vergleichen, Präsentation vorbereiten
- Nächster Termin: 26.11.24 19:00


---------------------------------------------


### Links:
- [Sentiment140 dataset with 1.6 million tweets (kaggle)](https://www.kaggle.com/datasets/kazanova/sentiment140/code?datasetId=2477&sortBy=commentCount) oder [Sentiment140 Datensatz (zip-file)](https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip)
- [Sentiment140 paper: TwitterDistantSupervision09.pdf](https://www-cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf)
- [Data Science Life Cycle](Data_Science_Life_Cycle.png)
- [felLind/propra_webscience_ws24 (GitHub Repo)](https://github.com/felLind/propra_webscience_ws24/tree/main)

### Ergebnisse:
- die grundsätzlichen Schritte der Data Preparation waren bei allen ähnlich oder sogar gleich
- klassische Methoden Aufteilung:
	- SVM: Andreas
	- Entscheidungsbäume: Felix
	- KNN: Anne
	- Naiver Bayes Klassifikator: Burak
	- Logistische Regression: Milo
- für die Präsentationen / Abschlussbericht werden 3 Methoden ausgewählt

### Aufgaben:
- Alle: 
	- die jeweiligen Methoden am Sentiment140 Datensatz anwenden und das Modell möglichst gut trainieren + evtl. für das Modelltraining nötige Data Preparation Schritte 

### Nächster Termin: 
- Di, 26.11.24 19:00, Discord Voicy
