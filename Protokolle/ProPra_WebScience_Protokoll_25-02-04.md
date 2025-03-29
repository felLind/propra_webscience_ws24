
==============================================

### ProPra Web Science Protokoll 04.02.25

Datum: 04.02.25 19:00 - 20:00  
Ort: Discord  
Teilnehmer: Andreas, Anne, Burak, Felix, Milo  
Fehlende Personen: niemand

==============================================


### Inhalt:
- Bericht der Experimente mit DeepSeek (Andreas, Felix):
	- 1.5 Modell funktioniert, hat aber nur eine Genauigkeit von ca. 85%
	- Genauigkeit wird geringer bei größerem Datensatz
	- 7B, 8B Modelle sind zu groß
- Alternativen:
	- Zero-Shot Ansatz auf 32B Modell (evtl. auch 70B Modell mit 4-bit Quantisierung)
	- Einigung darauf den Zero-Shot Ansatz zu verwenden
- Zum Zero-Shot Ansatz:
	- Zugriff über API?
	- Burak: Wie soll der Prompt aussehen?
	- Recherche bis zum nächsten Mal
- Abschlussbericht:
	- Abschnitt zu klassischen Modellen ist soweit fertig
	- Andreas: An wen richtet sich der Bericht? (Wie detailliert müssen bspw. die klassischen Modelle beschrieben werden)
	- -> je nach Länge des Berichts kann bei der Erklärung der klassischen Modellen gekürzt werden
	- Anne: insb. die Abschlusspräsentation geht auch an die Mitstudenten
	- Burak: Laut Angabe ist keine Erklärung zu den klassischen Modellen gefragt
	- Andreas: Beweggründe und Zusammenhänge sollen vermutlich beschrieben werden
	- Was genau gehört zu den geforderten 12 Seiten?
	- -> Anne: Referenzen gehörten nicht dazu
	- Weitere Aufteilung des Abschlussberichts:
		- Anne. Aufgabenverteilung (Problemstellung)
		- Felix: Deep Learning
		- Felix + Andreas: eigener Ansatz
		- Andreas: Experimente
	- Erwähnung von den klassischen Verfahren (kNN, Entscheidungsbäume) bei der Aufgabenverteilung und Begründung, warum sie nicht weiter verfolgt wurden, bei den klassischen Ansätzen
- Andreas: Soll Accuracy als Evaluationsmaß beibehalten werden oder nicht? -> Ausprobieren ob es einen Unterschied gibt, Entscheidung danach
- Nächster Termin: 11.02.25, 20:15



---------------------------------------------


### Links:
- [Sentiment140 dataset with 1.6 million tweets (kaggle)](https://www.kaggle.com/datasets/kazanova/sentiment140/code?datasetId=2477&sortBy=commentCount) oder [Sentiment140 Datensatz (zip-file)](https://www.google.com/url?q=https%3A%2F%2Fcs.stanford.edu%2Fpeople%2Falecmgo%2Ftrainingandtestdata.zip)
- [Sentiment140 paper: TwitterDistantSupervision09.pdf](https://www-cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf)
- [Data Science Life Cycle](Data_Science_Life_Cycle.png)
- [felLind/propra_webscience_ws24 (GitHub Repo)](https://github.com/felLind/propra_webscience_ws24/tree/main)
- [Lizenz (Google Group)](https://groups.google.com/g/sentiment140/c/IZUgbwH99L8)

### Ergebnisse:
- Finetuning von DeepSeek Modellen klappt nur beim 1.5B Modell
- Daher: Ausprobieren/Vergleich mit Zero-Shot Ansatz 

### Aufgaben:
- Alle: Zero-Shot Ansatz auf einem DeepSeek Modell Recherche, bzw. ausprobieren
- Alle: jeweilige Abschnitte des Abschlussberichts schreiben

### Nächster Termin: 
- Di, 11.02.25 20:15, Discord Voicy

