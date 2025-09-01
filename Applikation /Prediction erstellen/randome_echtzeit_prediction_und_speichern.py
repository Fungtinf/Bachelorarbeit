# C:\Bachelorarbeit\LSTM_Modell_prediction\randome_echtzeit_prediction_und_speichern.py
# Randome-Modell: identische Pipeline wie Hybrid (ENV-Cutoff, Lookback=14, Market-Features),
# aber die Impact-Werte (1d/3d/7d) werden IMMER zufällig erzeugt (JSON-Impacts werden ignoriert).
# FIX: Nur Horizont 1..6 verwenden; Tag 7 = Tag 6 (für Streamlit-Kompatibilität, um Ausreißer zu vermeiden).

import argparse, os, json, uuid, sqlite3, numpy as np, torch, random
from datetime import datetime, date, timedelta
import torch.nn as nn

# ================== ENV / Zeit ==================
def env_as_of_date() -> date | None:
    env = os.getenv("AS_OF_DATE")
    if not env:
        return None
    try:
        return date.fromisoformat(env.strip())
    except Exception:
        return None

def sim_created_at() -> str:
    """
    created_at = AS_OF_DATE + (AS_OF_TIME oder aktuelle UTC-Uhrzeit)
    Ausgabe im ISO-Format mit 'Z', damit Frontend nicht irritiert.
    """
    d = env_as_of_date()
    if d:
        hh, mm = None, None
        t = os.getenv("AS_OF_TIME")
        if t:
            try:
                hh_str, mm_str = t.strip()[:5].split(":")
                hh, mm = int(hh_str), int(mm_str)
            except Exception:
                hh, mm = None, None
        if hh is None or mm is None:
            now = datetime.utcnow()
            hh, mm = now.hour, now.minute
        return datetime(d.year, d.month, d.day, hh, mm, 0).isoformat() + "Z"
    return datetime.utcnow().isoformat() + "Z"

if os.getenv("RAND_SEED"):
    try:
        seed = int(os.getenv("RAND_SEED"))
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    except Exception:
        pass

# ================== CONFIG ==================
DEFAULT_CKPT      = r"C:\Bachelorarbeit\LSTM_hybrid_trained\hybrid_from_trainingsdaten_h7_symbol_emb\final_on_all\final_best_hybrid_hparams.pt"
DEFAULT_OUT_DB    = r"C:\Bachelorarbeit\Datenbank\randome_predictions.db"
DEFAULT_PRICES_DB = r"C:\Bachelorarbeit\Datenbank\predictions.db"
DEFAULT_MKT_DB    = r"C:\Bachelorarbeit\Datenbank\market_cache.db"
DEFAULT_MKT_TKR   = "^SSMI"

LOOKBACK = 14

# Zufalls-Impact-Bereiche in Prozentpunkten (≈ %-Angaben)
RAND_RANGES = {"1d": (-2.0, 2.0), "3d": (-3.0, 3.0), "7d": (-5.0, 5.0)}

MODEL_TYPE = "randome"
MODEL_NAME = "Randome_LSTM_live"

# ================== Symbol-Alias (für Embedding wie im Hybrid) ==================
SYMBOL_ALIASES = {
    "Nestlé": "Nestle",
}
def map_symbol_for_vocab(sym: str) -> str:
    return SYMBOL_ALIASES.get(sym, sym)

# ================== Modell (Hybrid-Architektur mit Symbol-Embedding) ==================
class LSTMSeqMultiHeadWithSym(nn.Module):
    """
    Wie im Hybrid-Training:
    - Input: Sequenz (T,F) + sym_id (Embedding)
    - Output: Pfad (HORIZON) relativer Returns
    """
    def __init__(self, in_dim_seq, n_symbols, emb_dim=8, hidden=256, layers=3, lstm_dropout=0.2, horizon=7):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=n_symbols, embedding_dim=emb_dim)
        do = float(lstm_dropout) if layers > 1 else 0.0
        self.lstm = nn.LSTM(in_dim_seq + emb_dim, hidden, num_layers=layers, batch_first=True, dropout=do)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, horizon))
        self.horizon = horizon

    def forward(self, x, sym_id):
        e = self.emb(sym_id).unsqueeze(1).expand(-1, x.size(1), -1)
        x_cat = torch.cat([x, e], dim=-1)
        out, _ = self.lstm(x_cat)
        last = out[:, -1, :]
        return self.head(last)

def load_ckpt(path):
    ck = torch.load(path, map_location="cpu")
    vocab = ck.get("sym_vocab", ["<UNK>"])
    stoi  = {s:i for i,s in enumerate(vocab)}
    model = LSTMSeqMultiHeadWithSym(
        in_dim_seq=ck["in_dim"],
        n_symbols=len(vocab),
        emb_dim=ck.get("emb_dim", 8),
        hidden=ck["hidden"],
        layers=ck["layers"],
        lstm_dropout=ck.get("cfg", {}).get("lstm_dropout", 0.2),
        horizon=int(ck["horizon"])
    )
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return {
        "model": model,
        "in_dim": ck["in_dim"],
        "sym_vocab": vocab,
        "stoi": stoi,
        "mean": np.array(ck["scaler_mean"], dtype=np.float32),
        "scale": np.array(ck["scaler_scale"], dtype=np.float32),
        "horizon": int(ck["horizon"]),
        "cfg": ck.get("cfg", {})
    }

# ================== DB Schema ==================
SCHEMA_HEADER = """
CREATE TABLE IF NOT EXISTS forecast_header (
  group_id       TEXT PRIMARY KEY,
  created_at     TEXT,
  source         TEXT,
  article_id     TEXT,
  symbol         TEXT,
  close_t        REAL,
  impact_1d      REAL,
  impact_3d      REAL,
  impact_7d      REAL,
  horizon        INTEGER,
  model_name     TEXT,
  model_cfg_json TEXT
);
"""
SCHEMA_LINES = """
CREATE TABLE IF NOT EXISTS forecast_lines (
  group_id   TEXT,
  model_type TEXT,       -- 'randome'
  step       INTEGER,    -- 1..horizon
  rel_return REAL,
  price      REAL,
  PRIMARY KEY (group_id, model_type, step)
);
"""

def ensure_db(db_path: str):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(SCHEMA_HEADER)
    cur.execute(SCHEMA_LINES)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_forecast_lines_group ON forecast_lines(group_id)")
    con.commit()
    con.close()

def insert_forecast(db_path: str, header: dict, lines: list):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO forecast_header
        (group_id, created_at, source, article_id, symbol, close_t,
         impact_1d, impact_3d, impact_7d, horizon, model_name, model_cfg_json)
        VALUES (:group_id, :created_at, :source, :article_id, :symbol, :close_t,
                :impact_1d, :impact_3d, :impact_7d, :horizon, :model_name, :model_cfg_json)
    """, header)
    cur.executemany("""
        INSERT OR REPLACE INTO forecast_lines
        (group_id, model_type, step, rel_return, price)
        VALUES (:group_id, :model_type, :step, :rel_return, :price)
    """, lines)
    con.commit()
    con.close()

# ================== Feature-Bau (wie Hybrid-Training) ==================
def _ffill_bfill(arr):
    arr = arr.copy()
    last = None
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            arr[i] = last if last is not None else np.nan
        else:
            last = arr[i]
    nxt = None
    for i in range(len(arr)-1, -1, -1):
        if np.isnan(arr[i]):
            arr[i] = nxt if nxt is not None else np.nan
        else:
            nxt = arr[i]
    return arr

def _log_returns(closes):
    c = np.asarray(closes, dtype=np.float32)
    if np.isnan(c).any():
        c = _ffill_bfill(c)
    if not np.all(np.isfinite(c)) or np.any(c <= 0) or len(c) < 2:
        return None
    r = np.diff(np.log(c))
    return r if np.all(np.isfinite(r)) else None

def _ewma(x, alpha):
    y = np.zeros_like(x, dtype=np.float32)
    if len(x) == 0: return y
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha*x[i] + (1-alpha)*y[i-1]
    return y

def normalize_to_iso_date(s: str) -> str:
    s = str(s).strip()
    s_part = s.split("T")[0].split(" ")[0]
    if len(s_part) == 10 and s_part[4] == "-" and s_part[7] == "-":
        return s_part
    try:
        dt = datetime.strptime(s_part, "%d.%m.%Y")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return s_part

def get_last_closes(prices_db, symbol, lookback, as_of=None):
    """
    Holt die letzten 'lookback' Schlusskurse <= as_of (ISO yyyy-mm-dd).
    Wenn as_of=None, wird 'heute' verwendet.
    """
    if as_of is None:
        from datetime import date as _date
        as_of = _date.today().isoformat()

    con = sqlite3.connect(prices_db); con.row_factory = sqlite3.Row
    rows = con.execute("""
        SELECT date, price FROM actual_prices
        WHERE symbol = ? AND date <= ?
        ORDER BY date DESC
        LIMIT ?
    """, (symbol, as_of, int(lookback))).fetchall()
    con.close()

    if not rows or len(rows) < lookback:
        return None, None

    rows = list(reversed(rows))
    dates  = [normalize_to_iso_date(r["date"]) for r in rows]
    prices = [float(r["price"]) for r in rows]
    if any(d is None for d in dates):
        return None, None
    return dates, prices

def _ensure_market_table(db_path: str):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = sqlite3.connect(db_path); cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS market_prices(
            symbol TEXT, date TEXT, price REAL,
            PRIMARY KEY(symbol, date)
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_market_symbol_date ON market_prices(symbol,date)")
    con.commit(); con.close()

def _load_market_map(db_path, ticker):
    if not os.path.exists(db_path): return {}
    con = sqlite3.connect(db_path); con.row_factory = sqlite3.Row
    rows = con.execute("SELECT date, price FROM market_prices WHERE symbol = ?", (ticker,)).fetchall()
    con.close()
    return {str(r["date"]): float(r["price"]) for r in rows}

def _fetch_yf_and_cache(db_path, ticker, start_iso, end_iso):
    try:
        import yfinance as yf
    except ImportError:
        return
    _ensure_market_table(db_path)
    end_dt = datetime.strptime(end_iso, "%Y-%m-%d") + timedelta(days=1)
    df = yf.download(ticker, start=start_iso, end=end_dt.strftime("%Y-%m-%d"),
                     progress=False, auto_adjust=False)
    if df is None or df.empty:
        return
    to_ins = []
    for idx, row in df.iterrows():
        iso = idx.strftime("%Y-%m-%d")
        val = row.get("Close", None)
        try:
            pr = float(val.item()) if hasattr(val, "item") else float(val)
        except Exception:
            continue
        if not np.isfinite(pr) or pr <= 0:
            continue
        to_ins.append((ticker, iso, pr))
    if to_ins:
        con = sqlite3.connect(db_path); cur = con.cursor()
        cur.executemany("INSERT OR REPLACE INTO market_prices(symbol,date,price) VALUES(?,?,?)", to_ins)
        con.commit(); con.close()

def get_market_returns_for_dates(cache_db, ticker, dates_iso):
    _ensure_market_table(cache_db)
    m = _load_market_map(cache_db, ticker)
    if any(d not in m for d in dates_iso):
        start = (datetime.strptime(dates_iso[0], "%Y-%m-%d") - timedelta(days=120)).strftime("%Y-%m-%d")
        _fetch_yf_and_cache(cache_db, ticker, start, dates_iso[-1])
        m = _load_market_map(cache_db, ticker)
    closes = [m.get(d, np.nan) for d in dates_iso]
    arr = _ffill_bfill(np.array(closes, dtype=np.float32))
    rets = _log_returns(arr)
    if rets is None or len(rets) != (LOOKBACK-1):
        return np.zeros((LOOKBACK-1,), dtype=np.float32), np.zeros((LOOKBACK-1,), dtype=np.float32)
    ema = _ewma(rets, alpha=0.5)
    return rets.astype(np.float32), ema.astype(np.float32)

def build_feature_sequence(symbol_name, ck, dates_iso, closes, market_db, market_tkr,
                           imp1: float, imp3: float, imp7: float):
    """
    Feature-Layout (T,F) wie im HYBRID-Training:
      [ret, mkt_ret, mkt_ema, vol_std, vol_ewma, imp1, imp3, imp7, weekday_onehot(7)]
      F = 5 + 3 + 7 = 15  (Symbol kommt über Embedding)
    Standardisierung mit ck['mean'] / ck['scale'] (shape = F).
    Gibt (Xn: Tensor[1,T,F], sym_id: Tensor[1]) zurück.
    """
    T = LOOKBACK - 1
    F = ck["in_dim"]  # sollte 15 sein

    rets = _log_returns(closes)
    if rets is None or len(rets) != T:
        raise RuntimeError(f"Zu wenige/ungültige Kurse für {symbol_name} (LOOKBACK={LOOKBACK}).")

    vol_std  = float(np.std(rets, ddof=1)) if T > 1 else 0.0
    vol_ewma = float(np.mean(_ewma(np.abs(rets), alpha=0.3))) if T > 0 else 0.0
    if not np.isfinite(vol_std):  vol_std = 0.0
    if not np.isfinite(vol_ewma): vol_ewma = 0.0

    mret, mema = get_market_returns_for_dates(market_db, market_tkr, dates_iso)

    wd = datetime.strptime(dates_iso[-1], "%Y-%m-%d").weekday()
    wd_oh = np.zeros((7,), dtype=np.float32); wd_oh[wd] = 1.0

    X = np.zeros((T, F), dtype=np.float32)
    for t in range(T):
        X[t, 0] = rets[t]
        X[t, 1] = mret[t]
        X[t, 2] = mema[t]
        X[t, 3] = vol_std
        X[t, 4] = vol_ewma
        X[t, 5] = float(imp1)
        X[t, 6] = float(imp3)
        X[t, 7] = float(imp7)
        X[t, 8:] = wd_oh

    Xn = (X - ck["mean"].reshape(1, -1)) / ck["scale"].reshape(1, -1)
    Xn = torch.from_numpy(Xn).unsqueeze(0).float()

    mapped = map_symbol_for_vocab(symbol_name)
    sym_id_val = ck["stoi"].get(mapped, 0)
    if sym_id_val == 0 and mapped != symbol_name:
        print(f"[INFO] Symbol-Alias verwendet: '{symbol_name}' -> '{mapped}', aber im Vokab nicht gefunden → <UNK>.")
    elif sym_id_val == 0 and symbol_name == mapped:
        print(f"[INFO] Symbol nicht im Vokab: '{symbol_name}' → <UNK>.")
    sym_id = torch.tensor([sym_id_val], dtype=torch.long)

    return Xn, sym_id

# ================== Pipeline ==================
def _rand_impacts() -> tuple[float, float, float]:
    r1 = round(random.uniform(*RAND_RANGES["1d"]), 2)
    r3 = round(random.uniform(*RAND_RANGES["3d"]), 2)
    r7 = round(random.uniform(*RAND_RANGES["7d"]), 2)
    return r1, r3, r7

def predict_one(ck, item, prices_db, market_db, market_tkr, source):
    db_symbol = item["symbol"]
    article_id = item.get("article_id")

    as_of = env_as_of_date()
    con = sqlite3.connect(prices_db); con.row_factory = sqlite3.Row
    if as_of:
        row = con.execute("""
            SELECT price FROM actual_prices
            WHERE symbol = ? AND date <= ?
            ORDER BY date DESC
            LIMIT 1
        """, (db_symbol, as_of.isoformat())).fetchone()
    else:
        row = con.execute("""
            SELECT price FROM actual_prices
            WHERE symbol = ?
            ORDER BY date DESC
            LIMIT 1
        """, (db_symbol,)).fetchone()
    con.close()
    if not row:
        raise RuntimeError(f"Kein Base-Preis in actual_prices für {db_symbol} gefunden.")
    base = float(row["price"])

    # Lookback-Kurse bis inkl. ENV
    dates_iso, closes = get_last_closes(prices_db, db_symbol, LOOKBACK, as_of)
    if not dates_iso:
        raise RuntimeError(f"Nicht genug Preise für {db_symbol} (LOOKBACK={LOOKBACK}).")

    imp1, imp3, imp7 = _rand_impacts()

    # Feature-Bau
    Xn, sym_id = build_feature_sequence(db_symbol, ck, dates_iso, closes, market_db, market_tkr,
                                        imp1=imp1, imp3=imp3, imp7=imp7)

    with torch.no_grad():
        yp = ck["model"](Xn, sym_id)
        rel = yp.squeeze(0).numpy().astype(np.float32)

    if len(rel) >= 6:
        rel_used = np.concatenate([rel[:6], rel[5:6]])
    else:
        last = rel[-1] if len(rel) > 0 else 0.0
        rel_used = np.array([*(rel.tolist()), *([last] * (7 - len(rel)))], dtype=np.float32)

    prices = (base * (1.0 + rel_used)).astype(np.float32)

    gid = str(uuid.uuid4())
    header = {
        "group_id": gid,
        "created_at": sim_created_at(),
        "source": source,
        "article_id": article_id,
        "symbol": db_symbol,
        "close_t": base,
        "impact_1d": imp1, "impact_3d": imp3, "impact_7d": imp7,
        "horizon": 7,
        "model_name": MODEL_NAME,
        "model_cfg_json": json.dumps({
            "features": "hybrid: ret+mkt+ema+vol+imp(RANDOM)+weekday, sym_emb",
            "rand_ranges_pct": RAND_RANGES,
            "store_h7_as_h6": True
        }, ensure_ascii=False)
    }
    lines = [
        {
            "group_id": gid, "model_type": MODEL_TYPE, "step": i + 1,
            "rel_return": float(rel_used[i]), "price": float(prices[i])
        }
        for i in range(7)
    ]
    return header, lines

# ================== CLI ==================
def parse_args():
    ap = argparse.ArgumentParser("Randome (Impacts zufällig) Echtzeit-Predictions -> randome_predictions.db")
    ap.add_argument("--ckpt",      default=DEFAULT_CKPT)
    ap.add_argument("--db",        default=DEFAULT_OUT_DB)
    ap.add_argument("--json",      default=None, help="Optional: Items-JSON (symbol[, article_id]). Impacts werden IGNORIERT.")
    ap.add_argument("--source",    default="randome-live")
    ap.add_argument("--prices_db", default=DEFAULT_PRICES_DB)
    ap.add_argument("--market_cache_db", default=DEFAULT_MKT_DB)
    ap.add_argument("--market_ticker",   default=DEFAULT_MKT_TKR)
    return ap.parse_args()

def main():
    args = parse_args()
    ensure_db(args.db)
    ck = load_ckpt(args.ckpt)

    if not os.path.exists(args.prices_db):
        raise SystemExit(f"❌ Prices-DB nicht gefunden: {args.prices_db}")
    con = sqlite3.connect(args.prices_db)
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='actual_prices'")
    if not cur.fetchone():
        raise SystemExit(f"❌ Tabelle 'actual_prices' fehlt in {args.prices_db}")

    items = []
    if args.json and os.path.exists(args.json):
        try:
            with open(args.json, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except json.JSONDecodeError:
            with open(args.json, "r", encoding="utf-8-sig") as f:
                raw = json.load(f)
        if isinstance(raw, dict):
            raw = [raw]
        for r in raw:
            if "symbol" in r:
                items.append({"symbol": r["symbol"], "article_id": r.get("article_id")})
    else:
        cur.execute("SELECT DISTINCT symbol FROM actual_prices")
        symbols = [r[0] for r in cur.fetchall()]
        for sym in symbols:
            items.append({"symbol": sym})

    saved = 0
    total = len(items)
    for it in items:
        try:
            header, lines = predict_one(ck, it, args.prices_db, args.market_cache_db, args.market_ticker, args.source)
            insert_forecast(args.db, header, lines)
            print(f"✓ gespeichert — symbol={it['symbol']}, group_id={header['group_id']}")
            saved += 1
        except Exception as e:
            print(f"❌ {it.get('symbol','?')}: {e}")

    print(f"Fertig. {saved}/{total} Vorhersagen geschrieben nach {args.db}")

if __name__ == "__main__":
    main()
