
==============================================

### ProPra Web Science Protokoll 21.01.25

Datum: 21.01.25 19:00 - 20:50  
Ort: Discord  
Teilnehmer: Andreas, Anne, Burak, Felix, Milo  
Fehlende Personen: niemand

==============================================


### Inhalt:
- Felix: berichtet von seinen Experimenten
	- RoBERTa base sentiment, 10.000 Datensätzen zum Finetuning, 2 epochs
	- Accuracy vorher 83% auf 90.5% mit Finetuning, in 2 Min
- Felix lässt das Modell mit 10 Epochs trainieren
- Burak: in der transformers-Bibliothek wird beim Finetuning die letzte Schicht/Ausgabeschicht eines voll-vernetzten neuronalen Netzes (nach dem Attention Mechanismus) trainiert
- Anne: Kurze Erklärung von LoRA, eine Möglichkeit für einen eigenen Ansatz
- Burak: Vergleich von RoBERTa Ergebnissen mit DistilBERT auch möglich
- Diskussion: Anwendung mindestens eines Deep Learning Modells:
	- Burak: Vorschlag: ein Modell, dass auf Twitter Daten vortrainiert wurde + ein Modell, dass nicht auf Twitter Daten vortrainiert wurde + Vergleich 
	  -> DistilBERT und Twitter RoBERTa base sentiment vergleichen
	- Diskussion darüber welche Parameter und welche Werte variiert werden sollen:
		- Anzahl an Trainingsdaten?
		- Learning rate?
		- Anzahl von Epochs?
		-> Anzahl an Trainingsdaten (2500, 5000, 7500, 10000), Learning rate auf Basis des [TweetEval Paper](https://arxiv.org/pdf/2010.12421) variieren
- Felix lässt das Modell mit 5 Epochs trainieren -> Accuracy 88%
- Felix lässt das Modell mit learning rate 1e^-5 trainieren -> Accuracy >90%
- Felix übernimmt das Trainieren der Modelle mit den entsprechenden Parametern
- Weiteres Vorgehen besprechen:
	- bis zum nächsten Mal Ideen für den eigenen Ansatz sammeln
	- bis zum nächsten Mal überlegen welche 3 klassischen Ansätze in den Abschlussbericht übernommen werden sollen
- Nächster Termin: 28.01.25, 19:00



---------------------------------------------


### Links:
- [Sentiment140 dataset with 1.6 million tweets (kaggle)](https://www.kaggle.com/datasets/kazanova/sentiment140/code?datasetId=2477&sortBy=commentCount) oder [Sentiment140 Datensatz (zip-file)](https://www.google.com/url?q=https%3A%2F%2Fcs.stanford.edu%2Fpeople%2Falecmgo%2Ftrainingandtestdata.zip)
- [Sentiment140 paper: TwitterDistantSupervision09.pdf](https://www-cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf)
- [Data Science Life Cycle](Data_Science_Life_Cycle.png)
- [felLind/propra_webscience_ws24 (GitHub Repo)](https://github.com/felLind/propra_webscience_ws24/tree/main)

### Ergebnisse:
- Vorgehensweise für die Anwendung von mind. einem Deep Learning Modell wurde festgelegt

### Aufgaben:
- Felix: DistilBERT und RoBERTa base sentiment trainieren
- Alle: Ideen für den eigenen Ansatz
- Alle: klassische Ansätze für den Abschlussbericht

### Nächster Termin: 
- Di, 28.01.25 19:00, Discord Voicy

