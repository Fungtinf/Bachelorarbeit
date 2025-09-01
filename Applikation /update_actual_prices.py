# C:\Bachelorarbeit\update_actual_prices.py
import os, sqlite3, argparse
from datetime import datetime, timezone, timedelta
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Bitte zuerst installieren:  pip install yfinance pandas")

PRED_DB = r"C:\Bachelorarbeit\Datenbank\predictions.db"

TICKERS = {
    "UBS": "UBSG.SW",
    "Nestlé": "NESN.SW",
    "Novartis": "NOVN.SW",
    "Roche": "ROG.SW",
    "Zurich Insurance": "ZURN.SW",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS actual_prices(
  symbol TEXT,
  date   TEXT,   -- ISO yyyy-mm-dd
  price  REAL,
  PRIMARY KEY(symbol, date)
);
"""

def ensure_db(db):
    con = sqlite3.connect(db)
    con.execute(SCHEMA)
    con.commit()
    con.close()

def save_prices(db, symbol, rows):
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.executemany(
        "INSERT OR REPLACE INTO actual_prices(symbol, date, price) VALUES (?,?,?)",
        rows
    )
    con.commit()
    con.close()

def fetch_last_n_closes(ticker: str, n_days: int = 800):
    """
    Holt die letzten n_days gültigen Schlusskurse.
    Robust gegen fehlende 'Close'-Spalte und MultiIndex-Columns.
    """
    try:
        df = yf.Ticker(ticker).history(period="800d", interval="1d", auto_adjust=False)
        if df is None or df.empty:
            return []

        # Spalte für Schlusskurs finden
        def pick_close_series(frame: pd.DataFrame):
            if isinstance(frame.columns, pd.MultiIndex):
                lvl0 = frame.columns.get_level_values(0)
                if "Close" in set(lvl0):
                    s = frame["Close"]
                elif "Adj Close" in set(lvl0):
                    s = frame["Adj Close"]
                else:
                    return None
                if isinstance(s, pd.DataFrame):
                    if s.shape[1] == 1:
                        s = s.iloc[:, 0]
                    else:
                        s = s.iloc[:, 0]
                return s.dropna()
            else:
                for col in ("Close", "Adj Close", "close", "adjclose"):
                    if col in frame.columns:
                        return frame[col].dropna()
            return None

        s = pick_close_series(df)
        if s is None or s.empty:
            return []

        # Letzten n_days Handelstage
        s = s.tail(n_days)
        out = []
        for idx, price in s.items():
            try:
                d = idx.date().isoformat()
            except Exception:
                d = str(idx)[:10]
            out.append((d, float(price)))
        return out
    except Exception:
        return []

def main(symbols=None, db=PRED_DB):
    ensure_db(db)
    symbols = symbols or list(TICKERS.keys())
    for sym in symbols:
        ticker = TICKERS.get(sym)
        if not ticker:
            print(f"⚠️  Kein Ticker-Mapping für {sym} – übersprungen.")
            continue
        rows = fetch_last_n_closes(ticker, n_days=800)
        if not rows:
            print(f"⚠️  Keine Kurse für {sym} ({ticker}) gefunden.")
            continue
        payload = [(sym, d, p) for d, p in rows]
        save_prices(db, sym, payload)
        print(f"✓ {sym}: gespeichert -> {rows[-1][0]}:{rows[-1][1]:.2f} (insgesamt {len(rows)} Tage)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=PRED_DB)
    ap.add_argument("--symbols", nargs="*", default=None, help="z.B. UBS Roche")
    args = ap.parse_args()
    main(args.symbols, db=args.db)
