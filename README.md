Bachelorarbeit – Aktienkursvorhersage mit Hybridmodell (Zeitreihen + Sentiment)

Dieses Projekt wurde im Rahmen der Bachelorarbeit entwickelt und umfasst eine komplette Pipeline zur Aktienkursvorhersage, die historische Kursdaten und KI-basierte Sentiment-Analysen kombiniert.
Das Projekt ist modular aufgebaut und gliedert sich in folgende Hauptbereiche:

📂 Projektstruktur
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
🚀 Quickstart

Virtuelle Umgebung einrichten

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

Abhängigkeiten installieren

pip install -r requirements.txt

Pipeline starten

cd Applikation/Prediction\ erstellen
python run_pipeline.py

GUI starten

cd Ausgabe_GUI
streamlit run Streamlit_GUI.py
📊 Datenbanken

Alle Daten liegen in SQLite-Datenbanken im Ordner Datenbanken/:

news_cache.db → Zwischenspeicher für gescrapte Artikel

market_cache.db → Kurswerte

*_predictions.db → Ergebnisse der Modelle

kummulierte_artikel.db → Zusammengefasste & gruppierte Artikel

trainingsdaten.db → Datensatz für Modelltraining

🧠 Modelle

Im Ordner Modelle/ liegen die trainierten Modelle:

final_best_gpt_hparams.pt

final_best_historisch_hparams.pt

final_best_hybrid_hparams.pt

📂 Ordnerdokumentationen

Jeder Unterordner enthält eine eigene README.md mit mehr Details:

Applikation/Prediction erstellen

Ausgabe_GUI

Auswertung

Datenbanken

Modelle
