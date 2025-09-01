# Bachelorarbeit – Aktienkursvorhersage mit Hybridmodell (Zeitreihen + Sentiment)

Dieses Projekt wurde im Rahmen der Bachelorarbeit entwickelt und umfasst eine komplette Pipeline zur **Aktienkursvorhersage**, die historische Kursdaten und KI-basierte Sentiment-Analysen kombiniert.  
Das Projekt ist modular aufgebaut und gliedert sich in folgende Hauptbereiche:

---

## 📂 Projektstruktur

```
.
├── Applikation/               # Zentrale Scripts für Prediction & Datenaufbereitung
│   └── Prediction erstellen/  # Einzelne Modelle + Pipeline-Skripte
│
├── Ausgabe_GUI/               # Streamlit-Oberfläche zur Ausgabe
│
├── Auswertung/                # Skripte zur statistischen Auswertung der Modelle
│
├── Datenbanken/               # SQLite-Datenbanken für Artikel, Kurse und Predictions
│
├── Modelle/                   # Trainierte Modellparameter (.pt)
│
├── README.md                  # Hauptübersicht (dieses Dokument)
└── requirements.txt           # Python-Abhängigkeiten
```

---



## 📊 Datenbanken

Alle Daten liegen in SQLite-Datenbanken im Ordner `Datenbanken/`:
- **news_cache.db** → Zwischenspeicher für gescrapte Artikel
- **market_cache.db** → Kurswerte
- **gpt_predictions.db** → Ergebnisse GPT-Modell
- **historisch_predictions.db** → Ergebnisse historisches Modell
- **hybrid_predictions.db** → Ergebnisse Hybrid-Modell
- **random_predictions.db** → Ergebnisse Random-Modell
- **kummulierte_artikel.db** → Zusammengefasste & gruppierte Artikel
- **trainingsdaten.db** → Datensatz für Modelltraining

---

## 🧠 Modelle

Im Ordner `Modelle/` liegen die trainierten Modelle:
- **final_best_gpt_hparams.pt**
- **final_best_historisch_hparams.pt**
- **final_best_hybrid_hparams.pt**

---

## 📂 Ordnerdokumentationen

Jeder Unterordner enthält eine eigene `README.md` mit mehr Details:

- [Applikation/Prediction erstellen](Applikation/Prediction%20erstellen/README.md)  
- [Ausgabe_GUI](Ausgabe_GUI/README.md)  
- [Auswertung](Auswertung/README.md)  
- [Datenbanken](Datenbanken/README.md)  
- [Modelle](Modelle/README.md)
