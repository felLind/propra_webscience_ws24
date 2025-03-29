
==============================================

### ProPra Web Science Protokoll 14.01.25

Datum: 14.01.25 19:00 - 20:10  
Ort: Discord  
Teilnehmer: Andreas, Anne, Burak, Felix, Milo  
Fehlende Personen: niemand

==============================================


### Inhalt:
- Deep Learning Ansatz:
	- Abstimmung, ob wir Finetuning als Deep Learning Ansatz verwenden sollen -> Ja, alle stimmen zu
	- Diskussion über verschiedene Vorgehensweisen und Transformer Modelle (Alle):
		- Welche BERT Modell sollen verwendet werden? -> Twitter-RoBERTa, DistilBERT
		- Mehrere Modelle vergleichen? 
		- Wie viele Daten sollen zum Finetuning verwendet werden? -> einige tausend, nicht alle 1.6 Mio.
	- Burak erklärt das Vorgehen bei einer ähnlichen Aufgabenstellung
	- Felix: macht Experimente mit den Infos/Parametern von Burak, um einschätzen zu können, wie schnell das Finetuning mit dem Sentiment140 Datensatz durchgeführt werden kann
- Klassische Modelle:
	- Felix hat das Refactoring gemacht
	- Burak hat die Naiver Bayes Modell durchlaufen lassen 
		- Genauigkeit bis über 90%
		- Unterschied zwischen Train und Testaccuracy ist groß
		-> nochmal überprüfen
	- Felix hat Entscheidungswälder hinzugefügt
	- Unterschiedliche bisherige Ergebnisse besprechen und hinterfragen
- Weitere Vorgehen:
	- Felix testet das Finetuning
	- Alle überlegen sich mögliche nächste Schritte/eigene Ansätze
- Nächster Termin: 21.01.25, 19:00



---------------------------------------------


### Links:
- [Sentiment140 dataset with 1.6 million tweets (kaggle)](https://www.kaggle.com/datasets/kazanova/sentiment140/code?datasetId=2477&sortBy=commentCount) oder [Sentiment140 Datensatz (zip-file)](https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip)
- [Sentiment140 paper: TwitterDistantSupervision09.pdf](https://www-cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf)
- [Data Science Life Cycle](Data_Science_Life_Cycle.png)
- [felLind/propra_webscience_ws24 (GitHub Repo)](https://github.com/felLind/propra_webscience_ws24/tree/main)

### Ergebnisse:
- Finetuning eines Transformers wird als Deep Learning Ansatz gewählt

### Aufgaben:
- Felix: Finetuning testen
- Alle: mögliche nächste Schritte/eigene Ansätze 

### Nächster Termin: 
- Di, 21.01.25 19:00, Discord Voicy

