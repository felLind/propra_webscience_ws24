
==============================================

**ProPra Web Science Protokoll 15.10.24**

Datum: 15.10.24 20:00 - 20:40
Ort: Discord
Teilnehmer: Andreas, Anne, Burak, Felix, Milo
Fehlende Personen: niemand

==============================================


### Inhalt:
- Diskussion über den gegebenen Datensatz
	- Allgemeine Meinung: gegebener Kaggle Datensatz ist nicht geeignet (twitter-entity-sentiment-analyse)
- Burak: hat einen alternativen Datensatz gefunden (Sentiment140)
- Diskussion über den Datensatz Sentment140
	- balancierter Datensatz
	- wurde im Zusammenhang mit einem Paper erstellt (Twitter Sentiment Classification using Distant Supervision)
	- hat Datum des Tweets als weiteres Merkmal
- Diskussion über die Problemstellung
	- Milo: Klassifikation
	- Burak: Sentiment von anderen Datensätzen mit unseren Modellen überprüfen
	- Andreas: Mehrdimensionale Analyse anhand des Entity Merkmals
	- Andreas: zunächst Modell trainieren, um das vorhandene Sentiment möglichst genau vorherzusagen
- Diskussion über das Preprocessing des Sentiment140 Datensatzes
	- Andreas: Tokenisation, Lemmatisation, Word embeddings
- Anne: Sollen wir den Datensatz Sentiment140 verwenden oder nicht?
- Felix: jeder sucht Datensätze die wir verwenden könnten
- Anforderungen an einen passenden Datensatz:
	- Tweet
	- Sentiment
	- Datum, wenn möglich
	- weitere, wenn möglich
- Felix: Sollen wir uns auf eine Programmiersprache, bzw. Pytorch /TensorFlow / Scikit-learn
	- Für den Deep-Learning Ansatz wichtig, wird aber später entschieden
- Nächster Termin: Di, 22.10.24 20:00
- Anne: Zusammenfassung der Aufgaben bis nächste Woche
	- Nach geeigneten Datensätzen schauen
	- Auf Discord teilen 
	- Die geteilten Datensätze anschauen 


---------------------------------------------

### Links:
- [twitter-entity-sentiment-analyse](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
- [Sentiment140 dataset with 1.6 million tweets](https://www.kaggle.com/datasets/kazanova/sentiment140/code?datasetId=2477&sortBy=commentCount)
- [TwitterDistantSupervision09.pdf](https://www-cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf)

### Ergebnisse:
- twitter-entity-sentiment-analyse Datensatz wird nicht verwendet
- Sentiment140 Datensatz könnten wir benutzen
- vorerst Training von Klassifikationsmodellen zur Vorhersage des Sentiment 
- noch keine Festlegung auf einen Datensatz, sondern Suche bis zum nächsten Mal
	- Merkmale: Tweet, Sentiment, (Datum und weitere, wenn möglich)

### Aufgaben:
- Alle: 
	- Nach geeigneten Datensätzen suchen
	- Auf Discord teilen 
	- Die geteilten Datensätze anschauen 

### Nächster Termin: 
- Di, 22.10.24 20:00, Discord Voicy


