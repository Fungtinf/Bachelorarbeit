import sqlite3
import openai
import time
import json
from dotenv import load_dotenv
import os
import re

load_dotenv()
openai.api_key = "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"

DB_PATH = r"C:\Bachelorarbeit\Datenbank\news_cache.db"

PROMPT_TEMPLATE = """
Analysiere die folgende wirtschaftsrelevante Nachricht für das Unternehmen {stock}:

{text}

Ziel: Schätze, wie stark sich der Aktienkurs von {stock} in den nächsten
- 1 Tag,
- 3 Tagen und
- 7 Tagen
verändern wird, dies kann positiv sein also +% Angabe oder negative Auswirkungen haben mit -% Angaben, wenn du denkst dass es keinen spürbaren effekt auf den Kurs haben wird setze 0%.

**Vorgaben**
1. Berücksichtige direkte Auswirkungen, Marktstimmung und historische Parallelen.
2. Gib für **jeden Zeitraum separat** eine kurze Begründung (1 – 2 Sätze).
3. Kannst du für einen Zeitraum keine Veränderung ableiten, setze **"0%"** und erkläre kurz warum (z. B. „keine neue Information“).
4. Wenn du insgesamt keine fundierte Schätzung abgeben kannst, antworte exakt mit **"Nicht genügend Daten"** (ohne JSON).

**Antwortformat (ausschließlich dieses JSON-Objekt)**
{{
  "impact": {{
    "1d": {{ "value": "+/- ..%", "explanation": "..." }},
    "3d": {{ "value": "+/- ..%", "explanation": "..." }},
    "7d": {{ "value": "+/- ..%",    "explanation": "..." }}
  }}
}}
Nur dieses JSON, keine weiteren Texte oder Zeilen davor/dahinter.
"""

def get_unchecked_articles():
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, stock, article_text
            FROM articles
            WHERE relevant = 1 AND gpt_checked = 0
        """)
        return cur.fetchall()

def extract_json_block(raw: str):
    """liefert dict oder None; behandelt 'Nicht genügend Daten'."""
    if "Nicht genügend Daten" in raw:
        return {
            "impact": {
                "1d": {"value": "0%", "explanation": "Nicht genügend Daten"},
                "3d": {"value": "0%", "explanation": "Nicht genügend Daten"},
                "7d": {"value": "0%", "explanation": "Nicht genügend Daten"}
            },
            "forced_irrelevant": True
        }
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    d = json.loads(m.group())
    d["forced_irrelevant"] = False
    return d

def to_float(percent_str: str) -> float:
    if percent_str is None:
        return 0.0
    s = str(percent_str).strip()
    if not s:
        return 0.0
    s = s.replace("−", "-").replace("–", "-")
    s = s.replace("%", "").replace("+", "")
    s = s.replace(",", ".")
    return float(s)

def update_article(article_id: int, block: dict):
    data = block["impact"]
    vals = {
        "impact_1d":       to_float(data["1d"]["value"]),
        "explanation_1d":  data["1d"]["explanation"],
        "impact_3d":       to_float(data["3d"]["value"]),
        "explanation_3d":  data["3d"]["explanation"],
        "impact_7d":       to_float(data["7d"]["value"]),
        "explanation_7d":  data["7d"]["explanation"],
        "relevant":        0 if block["forced_irrelevant"] else 1
    }

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE articles SET
              gpt_checked    = 1,
              relevant       = :relevant,
              impact_1d      = :impact_1d,
              explanation_1d = :explanation_1d,
              impact_3d      = :impact_3d,
              explanation_3d = :explanation_3d,
              impact_7d      = :impact_7d,
              explanation_7d = :explanation_7d
            WHERE id = :aid
        """, {**vals, "aid": article_id})
        conn.commit()

def query_gpt(prompt: str) -> str | None:
    try:
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Fehler] GPT-Request: {e}")
        return None

# ────────────────────────────────────────────────────────────
# Hauptablauf
# ────────────────────────────────────────────────────────────
def main():
    rows = get_unchecked_articles()
    print(f"🧠 {len(rows)} Artikel benötigen Impact-Prediction")

    for aid, stock, text in rows:
        stock = stock.strip()
        print(f"\n[ID {aid}] {stock}")

        if not stock:
            print("⚠️ Kein Stock – übersprungen")
            continue

        prompt = PROMPT_TEMPLATE.format(stock=stock, text=text[:3500])
        ans = query_gpt(prompt)
        if not ans:
            print("⚠️ Keine Antwort – übersprungen"); continue

        block = extract_json_block(ans)
        if not block:
            print(f"⚠️ JSON nicht gefunden – Antwort: {ans[:120]}…"); continue

        update_article(aid, block)
        print(f"✅ Gespeichert: 1d {block['impact']['1d']['value']} | "
              f"3d {block['impact']['3d']['value']} | "
              f"7d {block['impact']['7d']['value']}")

        time.sleep(1.5)

if __name__ == "__main__":
    main()
