
==============================================

### ProPra Web Science Protokoll 22.10.24

Datum: 22.10.24 20:00 - 21:00  
Ort: Discord  
Teilnehmer: Andreas, Anne, Burak, Felix, Milo  
Fehlende Personen: niemand

==============================================


### Inhalt:
- Alle: Diskussion über die Datensätze 
	- Airline (Milo)
		- muss vorverarbeitet werden
		- gut für Klassifizierung
		- hat viele Merkmale, von denen nur die nützlichen verwendet werden
	- Financal Tweets (Felix)
		- 2 Files
		- in File 2 sind nur bereinigte Tweets und Sentiment, d.h. die Zuordnung von Sentiment zu Tweet in File 1 ist schwer (wenn überhaupt möglich)
	- Sentiment140 (Burak)
		- sehr clean
		- es sind nur die Merkmale Sentiment und Date (Tweet + ID) vorhanden, keine weiteren
		- gleiche Anzahl an positiv und negativ Sentiment Tweets
		- es gibt einen vermutlich manuell gelabelten Testdatensatz dazu
		- wurde für ein Paper erstellt und wurde schon öfter verwendet
- Abstimmung über den Datensatz
	- alle sind mit Sentiment140 einverstanden
	- bzw. Airline Datensatz als Backup
- Alle: Besprechung über das weitere Vorgehen / nächsten Schritte nach dem Data Science Life Cycle 
	- Data Exploration wurde von allen gemacht
	- Hypothesis Generation: wir wollen ein/mehrere Modell zur Klassifizierung der Tweets nach Sentiment trainieren
	- Data Cleaning/Organization/Merging muss nicht mehr viel durchgeführt werden
	- Data Preparation/Feature Selection: 
		- Feature Selection ist in unserem Fall nicht wirklich relevant, da der Datensatz nur 4 Merkmale besitzt
		- Data Preparation ist der nächste Schritt
- Grober Zeitplan bis zur Zwischenpräsentation (17.12.) aufstellen
	- 2 Wochen Data Preparation
	- 4 Wochen klassische Modell trainieren und Ergebnisse vergleichen
- Nächster Termin: 05.11.24, 19:00
- Anne: Zusammenfassung der Aufgaben bis nächstes Mal
	- Data Preparation von Sentiment140
	- Ideen, welche Klassifikationsmodell wir dann trainieren wollen
- Felix: Jeder macht sich Notizen zu seinem Vorgehen, die dann für den Abschlussbericht zusammengefasst werden
- Andreas: Zum Teilen der Ergebnisse kann jeder einen Branch erstellen


---------------------------------------------


### Links:
- [Sentiment140 dataset with 1.6 million tweets](https://www.kaggle.com/datasets/kazanova/sentiment140/code?datasetId=2477&sortBy=commentCount)
- [TwitterDistantSupervision09.pdf](https://www-cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf)
- [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment/data)
- [Sentiment Analysis on Financial Tweets](https://www.kaggle.com/datasets/vivekrathi055/sentiment-analysis-on-financial-tweets?select=stockerbot-export1.csv)
- [Data Science Life Cycle](Data_Science_Life_Cycle.png)

### Ergebnisse:
- Datensatz Sentiment140 wird verwendet
- Data Preparation ist der nächste Schritt, um danach den Datensatz klassifizieren zu können
- Zeitplan bis zur Zwischenpräsentation (17.12.)
	- 2 Wochen Data Preparation
	- 4 Wochen klassische Modell trainieren und Ergebnisse vergleichen

### Aufgaben:
- Alle: 
	- Data Preparation von Sentiment140
	- Ideen, welche Klassifikationsmodell wir dann trainieren wollen

### Nächster Termin: 
- Di, 05.11.24 19:00, Discord Voicy
