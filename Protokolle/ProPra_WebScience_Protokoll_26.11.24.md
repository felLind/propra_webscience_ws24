
==============================================

### ProPra Web Science Protokoll 26.11.24

Datum: 26.11.24 19:00 - 20:10  
Ort: Discord  
Teilnehmer: Andreas, Anne, Burak, Felix, Milo  
Fehlende Personen: niemand

==============================================


### Inhalt:
- Info zum Kontakt mit Prof. Thimm
- Vorstellung der Vorgehensweise und Ergebnisse bei der Anwendung der jeweiligen klassischen überwachten Lernmethoden:
	- Milo (Logistische Regression): 
		- LinReg und LogReg gleiche Ergebnisraten
		- Train/Test split
		- tfidf, Bigramme
		- Accuracy bis zu 76%
	- Burak (Bayes-Klassifikator): 
		- Train/Test split, unterschiedliche Werte machen keinen großen Unterschied
		- tfidf
		- Accuracy 78%, ohne Preprocessing bessere Werte (80%) (liegt vermutlich an den stopwords, die nicht entfernt werden)
	- Felix (Entscheidungsbäume): 
		- mehrfache Buchstaben verkürzen auf 2
		- lange Trainingszeiten, tfidf
		- Test auf Testdatensatz (Lemmatized) 74% (ohne Neutral), bei Stemming < 70%
		- mit Word2Vec 50%
	- Anne (KNN): 
		- Datensätze mit unterschiedlichem Preprocessing
		- Train/Test split, Vorhersage der Testdaten dauert lange
		- GridSearch 
		- auf Teildatensatz: Bag-of-Words 68%, tfidf 62%
	- Andreas (SVM): 
		- Datensätze mit unterschiedlichem Preprocessing
		- tfidf, GLOVE 
		- Testdatensatz, Uni+Bigramme 84%
		- Vergleich mit BERT basiertem Modell
		- Visualisierung der Ergebnisse
- Alle: Besprechung der Inhalte der Zwischenpräsentation
	- Grobe Struktur: (noch an die Vorgaben anzupassen)
		- Gruppenorganisation
		- Datensatzauswahl
			- Warum nicht der gegebene Datensatz?
			- Warum sentiment140?
		- Preprocessing
			- mögliche Schritte
		- Präsentation der Modellergebnisse
			- Auflistung der Ergebnisse (Betonung: nicht vergleichbar)
			- schon vorab mit baseline Modell/BERT getestet
		- Ausblick
- Festlegung der Person, die die Zwischenpräsentation hält: Anne
- Besprechung der formalen Voraussetzungen und Vorgaben
	- Guidelines
	- Sprache: Deutsch
- Die Präsentation wird beim nächsten Treffen (in einer Woche, 03.12.) gemeinsam erstellt
- in der darauffolgenden Woche (10.12.) wird die Präsentation verbessert (falls nötig) und geübt 
- Nächster Termin: 03.12.24 19:00


---------------------------------------------


### Links:
- [Sentiment140 dataset with 1.6 million tweets (kaggle)](https://www.kaggle.com/datasets/kazanova/sentiment140/code?datasetId=2477&sortBy=commentCount) oder [Sentiment140 Datensatz (zip-file)](https://www.google.com/url?q=https%3A%2F%2Fcs.stanford.edu%2Fpeople%2Falecmgo%2Ftrainingandtestdata.zip)
- [Sentiment140 paper: TwitterDistantSupervision09.pdf](https://www-cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf)
- [Data Science Life Cycle](Data_Science_Life_Cycle.png)
- [felLind/propra_webscience_ws24 (GitHub Repo)](https://github.com/felLind/propra_webscience_ws24/tree/main)

### Ergebnisse:
- Vorläufige Ergebnisse der klassischen Modelle
- Grundstruktur der Präsentation
- Vortragende der Zwischenpräsentation: Anne

### Aufgaben:
- Alle: 
	- evtl. jeweiligen Modelle verbessern
	- guidelines für die Präsentation lesen
	- Notizen für die Folien

### Nächster Termin: 
- Di, 03.12.24 19:00, Discord Voicy
