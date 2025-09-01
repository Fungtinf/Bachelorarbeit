import sqlite3
import requests
from bs4 import BeautifulSoup
import time
import json
import re
import os

DB_PATH = r"C:\Bachelorarbeit\Datenbank\news_cache.db"

def get_article_text(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        if "finanzen.ch" in url:
            lead = soup.select_one("div.asset-intro") or soup.select_one("p.intro")
            body = soup.select_one("div.asset-content")
        else:
            lead = soup.find("div", class_="article-lead") or soup.select_one("p.lead") or soup.select_one("div.lead")
            body = soup.find("div", class_="article-body") or soup.select_one("article") or soup.select_one("div.content")

        lead_text = lead.get_text(strip=True) if lead else ""
        body_text = body.get_text(" ", strip=True) if body else ""
        return (lead_text + "\n\n" + body_text).strip()
    except Exception as e:
        print(f"[Fehler] Artikeltext konnte nicht geladen werden: {e}")
        return ""

# JSON aus LLM-Antwort extrahieren
def extract_json(text):
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    match = re.search(r'\{.*\}', s, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            return None
    return None

# Anfrage an LLM (lokal via Ollama)
def run_llm_check(text):
    prompt = f"""
Du bist ein Analyst für den Schweizer Aktienmarkt und überwachst nur folgende Unternehmen:
UBS, Nestlé, Novartis, Roche und Zurich Insurance.

Bitte bewerte den folgenden Artikeltext und bewerte, ob dieser Artikel inhaltlich relevant ist für eine, der von dir überwachten Unternehmen.

Gib ein JSON-Objekt in folgendem Format zurück:
{{
  "relevant": true/false,
  "stocks": []
}}

Hier ist der Artikeltext:
{text}
""".strip()

    try:
        model_name = os.environ.get("OLLAMA_MODEL", "llama3:latest")
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.0}
            },
            timeout=30
        )
        if response.status_code != 200:
            print(f"[Fehler] LLM HTTP {response.status_code}: {response.text[:200]}")
            return None

        data = response.json()
        raw = data.get("response", "")
        obj = extract_json(raw)
        if not obj:
            print("[Fehler] LLM lieferte kein parsebares JSON. Auszug:", repr(raw[:200]))
            return None
        return obj

    except Exception as e:
        print(f"[Fehler] LLM lokal fehlgeschlagen: {e}")
        return None

# Hauptlogik
def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT id, url FROM articles WHERE pre_checked IS NULL")
    rows = cursor.fetchall()

    print(f"[INFO] {len(rows)} Artikel zur Analyse geladen")

    for article_id, url in rows:
        print(f"\n[ID] {article_id}")
        print(f"[Link] {url}")

        text = get_article_text(url)
        if not text:
            print("[WARNUNG] Kein Artikeltext – übersprungen")
            continue

        result = run_llm_check(text)
        if not result:
            print("[PRECHECK] Nicht relevant (API/JSON-Fehler)")
            continue

        is_relevant = result.get("relevant", False)

        # Stocks extrahieren
        raw_stocks = result.get("stocks", [])
        relevant_stocks = []
        for entry in raw_stocks:
            if isinstance(entry, dict):
                relevant_stocks.append((entry.get("name") or "").strip())
            elif isinstance(entry, str):
                relevant_stocks.append(entry.strip())
        relevant_stocks = [s for s in relevant_stocks if s]

        print(f"[PRECHECK] {'Relevant' if is_relevant else 'Nicht relevant'}")
        if relevant_stocks:
            print(f"[STOCKS] {', '.join(relevant_stocks)}")

        if is_relevant and relevant_stocks:
            for stock in relevant_stocks:
                cursor.execute("""
                    INSERT OR IGNORE INTO articles (
                        title, url, date, article_text, stock, pre_checked, relevant, source
                    )
                    SELECT title, url || '?stock=' || ?, date, ?, ?, 1, 1, source
                    FROM articles WHERE id = ?
                """, (stock, text, stock, article_id))
                print(f"➕ Duplikat erstellt für {stock}")

            cursor.execute("UPDATE articles SET pre_checked = 1, relevant = 0 WHERE id = ?", (article_id,))
        else:
            cursor.execute("""
                UPDATE articles SET pre_checked = 1, relevant = 0, article_text = ?
                WHERE id = ?
            """, (text, article_id))

        conn.commit()
        time.sleep(1)

    conn.close()

if __name__ == "__main__":
    main()
