import os
import json
import time
import subprocess
import sqlite3
from datetime import datetime, date
import sys, os
if os.name == "nt":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ================== CONFIG ==================
PYTHON_EXE = r"C:\Users\pasca\AppData\Local\Microsoft\WindowsApps\python3.11.exe"

# Skripte
SCRAPER               = r"C:\Bachelorarbeit\scrapping_script_news_cash.py"
PRECHECK              = r"C:\Bachelorarbeit\precheck_with_llm.py"
GROUP_SIM             = r"C:\Bachelorarbeit\group_articels_by_similarity.py"
GPT_IMPACT            = r"C:\Bachelorarbeit\gpt_impact_bewertung.py"
IMPACT_AGG            = r"C:\Bachelorarbeit\impact_kummulieren.py"
UPDATE_PRICES         = r"C:\Bachelorarbeit\update_actual_prices.py"  # optional

GPT_PREDICT_STORE         = r"C:\Bachelorarbeit\LSTM_Modell_prediction\gpt_echtzeit_prediction_und_speichern.py"
HYBRID_PREDICT_STORE      = r"C:\Bachelorarbeit\LSTM_Modell_prediction\hybrid_echtzeit_prediction_und_speichern.py"
HISTORISCH_PREDICT_STORE  = r"C:\Bachelorarbeit\LSTM_Modell_prediction\historisch_echtzeit_prediction_und_speichern.py"
RANDOME_PREDICT_STORE     = r"C:\Bachelorarbeit\LSTM_Modell_prediction\randome_echtzeit_prediction_und_speichern.py"  # NEU


# Datenbanken
NEWS_DB   = r"C:\Bachelorarbeit\Datenbank\news_cache.db"
AGG_DB    = r"C:\Bachelorarbeit\Datenbank\kummulierte_artikel.db"
PRED_DB   = r"C:\Bachelorarbeit\Datenbank\gpt_predictions.db"
HYBRID_DB = r"C:\Bachelorarbeit\Datenbank\hybrid_predictions.db"
HIST_DB   = r"C:\Bachelorarbeit\Datenbank\historisch_predictions.db"
RAND_DB   = r"C:\Bachelorarbeit\Datenbank\randome_predictions.db"
MKT_DB    = r"C:\Bachelorarbeit\Datenbank\market_cache.db"

# Modelle / Market
HYBRID_CKPT = r"C:\Bachelorarbeit\LSTM_hybrid_trained\hybrid_from_trainingsdaten_h7_symbol_emb\final_on_all\final_best_hybrid_hparams.pt"
HIST_CKPT   = r"C:\Bachelorarbeit\LSTM_historisch_trained\historisch_only_from_trainingsdaten_h7_symbol_emb\final_on_all\final_best_historisch_hparams.pt"

MKT_TICKER   = "^SSMI"

# Intervall-Steuerung
RUN_MODE = "interval"
INTERVAL_MINUTES = 15

SYMBOL_FILTER = None


# ================== HELPERS ==================
def get_as_of_date() -> date:
    env = os.getenv("AS_OF_DATE")
    if env:
        try:
            return date.fromisoformat(env.strip())
        except Exception:
            pass
    return date.today()

def run_cmd(args, cwd=None, env=None, name=""):
    import os, subprocess, sys
    print(f"\n🚀 Starte: {name or args}")
    env_full = os.environ.copy()
    env_full["PYTHONIOENCODING"] = "utf-8"
    env_full["PYTHONUTF8"]       = "1"
    env_full["PYTHONUNBUFFERED"] = "1"
    if env:
        env_full.update(env)

    proc = subprocess.Popen(
        [PYTHON_EXE, "-u", *args],
        cwd=cwd,
        env=env_full,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )

    try:
        for chunk in iter(lambda: proc.stdout.readline(), b""):
            if not chunk:
                break
            try:
                line = chunk.decode("utf-8", errors="replace")
            except Exception:
                line = chunk.decode("cp1252", errors="replace")
            print(line, end="")
    finally:
        if proc.stdout:
            proc.stdout.close()

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"❌ Fehler in {name or args}: rc={proc.returncode}")
    return proc


def get_symbols_from_agg_db(as_of: date):
    if not os.path.exists(AGG_DB):
        return []
    con = sqlite3.connect(AGG_DB); con.row_factory = sqlite3.Row
    rows = con.execute("""
        SELECT DISTINCT stock
        FROM aggregated_impacts
        WHERE as_of_date = ?
    """, (as_of.isoformat(),)).fetchall()
    con.close()
    syms = [r["stock"] for r in rows]
    if SYMBOL_FILTER:
        syms = [s for s in syms if s in SYMBOL_FILTER]
    return syms

def get_agg_impacts_for_symbol(as_of: date, symbol: str):
    con = sqlite3.connect(AGG_DB); con.row_factory = sqlite3.Row
    r = con.execute("""
        SELECT impact_1d, impact_3d, impact_7d
        FROM aggregated_impacts
        WHERE as_of_date = ? AND stock = ?
        LIMIT 1
    """, (as_of.isoformat(), symbol)).fetchone()
    con.close()
    if not r: return None
    return float(r["impact_1d"]), float(r["impact_3d"]), float(r["impact_7d"])

def get_base_price(symbol: str, as_of: date):
    """
    Holt den letzten Preis aus predictions.db -> actual_prices
    mit Cutoff <= as_of (ENV), Fallback: neustes close_t aus forecast_header.
    """
    price = None
    if os.path.exists(PRED_DB):
        con = sqlite3.connect(PRED_DB); con.row_factory = sqlite3.Row
        row = con.execute("""
            SELECT price FROM actual_prices
            WHERE symbol = ? AND date <= ?
            ORDER BY date DESC
            LIMIT 1
        """, (symbol, as_of.isoformat())).fetchone()
        if row:
            price = float(row["price"])
        else:
            row2 = con.execute("""
                SELECT close_t FROM forecast_header
                WHERE symbol = ?
                ORDER BY datetime(created_at) DESC
                LIMIT 1
            """, (symbol,)).fetchone()
            if row2 and row2["close_t"] is not None:
                price = float(row2["close_t"])
        con.close()
    return price


def call_update_prices(symbols):
    if not os.path.exists(UPDATE_PRICES):
        print("ℹ️  update_actual_prices.py nicht gefunden – überspringe Preis-Update.")
        return
    args = [UPDATE_PRICES, "--db", PRED_DB, "--symbols", *symbols]
    print(f"[PRICE-UPDATE] rufe auf mit: {args}")
    run_cmd(args, name="update_actual_prices.py")

def call_gpt_predict_store(items, source_label="pipeline"):
    tmp_json = r"C:\Bachelorarbeit\temp_batch_prediction.json"
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    args = [GPT_PREDICT_STORE,
            "--db", PRED_DB,
            "--json", tmp_json,
            "--source", source_label]
    run_cmd(args, name="gpt_echtzeit_prediction_und_speichern.py")
    os.remove(tmp_json)

def call_randome_predict_store(items, source_label="pipeline"):
    tmp_json = r"C:\Bachelorarbeit\temp_batch_prediction_randome.json"
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    args = [RANDOME_PREDICT_STORE,
            "--db", RAND_DB,
            "--json", tmp_json,
            "--source", source_label,
            "--ckpt",  HYBRID_CKPT,
            "--prices_db", PRED_DB,
            "--market_ticker", MKT_TICKER,
            "--market_cache_db", MKT_DB]
    run_cmd(args, name="randome_echtzeit_prediction_und_speichern.py")
    os.remove(tmp_json)

def call_hybrid_predict_store(items, source_label="pipeline"):
    tmp_json = r"C:\Bachelorarbeit\temp_batch_prediction_hybrid.json"
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    args = [HYBRID_PREDICT_STORE,
            "--db", HYBRID_DB,
            "--json", tmp_json,
            "--source", source_label,
            "--ckpt",  HYBRID_CKPT,
            "--prices_db", PRED_DB,
            "--market_ticker", MKT_TICKER,
            "--market_cache_db", MKT_DB]
    run_cmd(args, name="hybrid_echtzeit_prediction_und_speichern.py")
    os.remove(tmp_json)

def call_historisch_predict_store(items, source_label="pipeline"):
    tmp_json = r"C:\Bachelorarbeit\temp_batch_prediction_historisch.json"
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    args = [HISTORISCH_PREDICT_STORE,
            "--db", HIST_DB,
            "--json", tmp_json,
            "--source", source_label,
            "--ckpt",  HIST_CKPT,
            "--prices_db", PRED_DB,
            "--market_ticker", MKT_TICKER,
            "--market_cache_db", MKT_DB]
    run_cmd(args, name="historisch_echtzeit_prediction_und_speichern.py")
    os.remove(tmp_json)

# ================== PIPELINE ==================
def run_once():
    as_of = get_as_of_date()
    as_of_env = os.getenv("AS_OF_DATE")
    if as_of_env:
        today_d = date.fromisoformat(as_of_env)
    else:
        today_d = date.today()
    print(f"\n================ PIPELINE START für {today_d.isoformat()} ({datetime.now().isoformat(timespec='seconds')}) ================\n")

    # 1) Scraper
    if os.path.exists(SCRAPER):
        run_cmd([SCRAPER], name="scrapping_script_news_cash.py")
    else:
        print("ℹ️  Scraper-Script nicht gefunden – überspringe Schritt 1.")

    # 2) precheck_with_llm
    if os.path.exists(PRECHECK):
        run_cmd([PRECHECK], name="precheck_with_llm.py")
    else:
        print("ℹ️  precheck_with_llm.py nicht gefunden – überspringe Schritt 3.")

    # 3) group_articels_by_similarity
    if os.path.exists(GROUP_SIM):
        run_cmd([GROUP_SIM], name="group_articels_by_similarity.py")
    else:
        print("ℹ️  group_articels_by_similarity.py nicht gefunden – überspringe Schritt 4.")

    # 4) gpt_impact_bewertung
    if os.path.exists(GPT_IMPACT):
        run_cmd([GPT_IMPACT], name="gpt_impact_bewertung.py")
    else:
        print("ℹ️  gpt_impact_bewertung.py nicht gefunden – überspringe Schritt 5.")

    # 5) impact_kummulieren (neu)
    run_cmd([IMPACT_AGG], name="impact_kummulieren.py")

    # 6) Symbole mit kumulierten Impacts laden
    symbols = get_symbols_from_agg_db(as_of)
    if not symbols:
        print("⚠️  Keine kumulierten Impacts gefunden – breche vor Prediction ab.")
        return

    # 7) Preise aktualisieren (optional) + Predict & Store
    call_update_prices(symbols)

    # Items bauen
    items = []
    for sym in symbols:
        impacts = get_agg_impacts_for_symbol(as_of, sym)
        if not impacts:
            print(f"⚠️  Keine Impacts für {sym} – skip.")
            continue
        base = get_base_price(sym,as_of)
        if base is None:
            print(f"⚠️  Kein Base-Preis für {sym} – skip.")
            continue
        i1d, i3d, i7d = impacts
        items.append({
            "symbol": sym,
            "impact_1d": float(i1d),
            "impact_3d": float(i3d),
            "impact_7d": float(i7d),
            "close_t": float(base),
            "article_id": f"agg-{as_of.isoformat()}-{sym}"
        })

    import random

    items_rand = []
    for sym in symbols:
        base = get_base_price(sym, as_of)
        if base is None:
            print(f"⚠️  Kein Base-Preis für {sym} (Randome) – skip.")
            continue
        r1 = round(random.uniform(-2.0,  2.0), 2)
        r3 = round(random.uniform(-3.0,  3.0), 2)
        r7 = round(random.uniform(-5.0,  5.0), 2)
        items_rand.append({
            "symbol":   sym,
            "impact_1d": r1,
            "impact_3d": r3,
            "impact_7d": r7,
            "close_t":   float(base),
            "article_id": f"rand-{today_d.isoformat()}-{sym}"
        })


    if items:
        call_gpt_predict_store(items, source_label="pipeline")
        call_hybrid_predict_store(items, source_label="pipeline")
        call_historisch_predict_store(items, source_label="pipeline")
        print(f"✓ {len(items)} Vorhersagen gespeichert (GPT + Hybrid + Historisch).")
    else:
        print("⚠️  Keine Items für Prediction zusammengestellt.")

    # Randome immer separat prüfen
    if items_rand:
        call_randome_predict_store(items_rand, source_label="pipeline")
        print(f"✓ {len(items_rand)} Vorhersagen gespeichert (Randome).")
    else:
        print("⚠️  Keine Randome-Items erstellt.")

    print("\n================ PIPELINE ENDE ================================\n")
    print("Hinweis: Streamlit liest je nach Toggle aus den getrennten DBs.\n"
          "Nach Live-Run im GUI 'Daten aktualisieren' klicken oder Auto-Refresh nutzen.")


def main():
    run_once()

     --- Normalbetrieb (wieder einkommentieren wenn nötig) ---
     if RUN_MODE == "once":
         run_once()
     else:
         interval_sec = max(60, int(INTERVAL_MINUTES * 60))
         while True:
             try:
                 run_once()
             except Exception as e:
                 print(f"❌ Pipeline-Fehler: {e}")
             print(f"⏲️  Schlafe {INTERVAL_MINUTES} Minuten … (Strg+C zum Abbrechen)")
             try:
                 time.sleep(interval_sec)
             except KeyboardInterrupt:
                 print("⏹️  Abgebrochen.")
                 break

if __name__ == "__main__":
    main()
