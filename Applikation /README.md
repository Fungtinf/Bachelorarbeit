## Applikation/Prediction erstellen/README.md


```markdown
# Prediction erstellen


In diesem Ordner liegen die zentralen Skripte, um Vorhersagen zu erstellen, Artikel vorzubereiten und Datenflüsse zu steuern.


## Dateien


- **gpt_echtzeit_prediction_und_speichern.py**
Führt Echtzeit-Vorhersagen mit GPT-basiertem Modell durch und speichert die Ergebnisse.


- **historisch_echtzeit_prediction_und_speichern.py**
Prognosen nur auf Basis historischer Kursdaten.


- **hybrid_echtzeit_prediction_und_speichern.py**
Kombiniert historische Daten und Sentiment-Bewertungen (zentrales Modell der Arbeit).


- **randome_echtzeit_prediction_und_speichern.py**
Zufallsmodell als Vergleichsbaseline.


- **gpt_impact_bewertung.py**
Analysiert Artikel mit LLM und weist Impact-Werte (+/- %) für 1/3/7 Tage zu.


- **group_articles_by_similarity.py**
Clustert Artikel nach inhaltlicher Ähnlichkeit und bildet Gruppen.


- **impact_kummulieren.py**
Aggregiert Impact-Werte pro Aktie und Zeitraum.


- **precheck_with_llm.py**
Prüft Artikel vorab mit GPT (Relevanzfilter).


- **scrapping_script_news_cash.py**
Web-Scraper für Artikel von cash.ch.


- **update_actual_prices.py**
Holt die aktuellen Kurswerte (z. B. von Yahoo Finance).


- **run_pipeline.py**
Steuert den gesamten Workflow: Scraping → Bewertung → Gruppierung → Impact → Prediction.
```


---
