import argparse, json, os, sqlite3, uuid
from datetime import datetime, date
import numpy as np
import torch
import torch.nn as nn

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
    Ausgabe im ISO-Format mit 'Z', damit Streamlit lokal korrekt formatiert.
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

# ================== Defaults ==================
DEFAULT_CKPT   = r"C:\Bachelorarbeit\LSTM_modell_tuning_best_run_3layers\final_on_all\final_best_hparams.pt"
DEFAULT_DB     = r"C:\Bachelorarbeit\Datenbank\gpt_predictions.db"

# ================== Modell ==================
class LSTMSeqRegressor(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=1, lstm_dropout=0.0):
        super().__init__()
        do = float(lstm_dropout) if layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=do
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out).squeeze(-1)

# ================== Laden & Preprocessing ==================
def load_model(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = LSTMSeqRegressor(
        in_dim=ckpt["in_dim"],
        hidden=ckpt["hidden"],
        layers=ckpt["layers"],
        lstm_dropout=ckpt.get("cfg", {}).get("lstm_dropout", 0.0),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    sym_classes = ckpt["sym_classes"]
    mean  = np.array(ckpt["scaler_mean"],  dtype=np.float32)
    scale = np.array(ckpt["scaler_scale"], dtype=np.float32)
    horizon = int(ckpt.get("horizon", 6))
    cfg = ckpt.get("cfg", {})
    return model, sym_classes, mean, scale, horizon, cfg

def build_input(symbol: str, impact1: float, impact3: float, impact7: float,
                sym_classes, mean, scale, horizon: int):
    xstat = (np.array([impact1, impact3, impact7], dtype=np.float32) - mean) / scale
    oh = np.zeros((len(sym_classes),), dtype=np.float32)
    if symbol in sym_classes:
        oh[sym_classes.index(symbol)] = 1.0
    steps = np.arange(1, horizon + 1, dtype=np.float32) / float(horizon)
    F = 3 + len(sym_classes) + 1
    seq = np.zeros((horizon, F), dtype=np.float32)
    for k in range(horizon):
        seq[k, :3] = xstat
        seq[k, 3:3+len(sym_classes)] = oh
        seq[k, 3+len(sym_classes)] = steps[k]
    return torch.from_numpy(seq).unsqueeze(0).float()

# ================== DB Schema & Insert ==================
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
  model_type TEXT,       -- 'gpt'
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
        (group_id, created_at, source, article_id, symbol, close_t, impact_1d, impact_3d, impact_7d, horizon, model_name, model_cfg_json)
        VALUES (:group_id, :created_at, :source, :article_id, :symbol, :close_t, :impact_1d, :impact_3d, :impact_7d, :horizon, :model_name, :model_cfg_json)
    """, header)
    cur.executemany("""
        INSERT OR REPLACE INTO forecast_lines 
        (group_id, model_type, step, rel_return, price)
        VALUES (:group_id, :model_type, :step, :rel_return, :price)
    """, lines)
    con.commit()
    con.close()

# ================== Pipeline ==================
def predict_item(model, sym_classes, mean, scale, horizon, cfg, item: dict, source: str, db_path: str):
    symbol = item["symbol"]
    imp1 = float(item["impact_1d"])
    imp3 = float(item["impact_3d"])
    imp7 = float(item["impact_7d"])
    base = float(item["close_t"])
    article_id = item.get("article_id")

    x = build_input(symbol, imp1, imp3, imp7, sym_classes, mean, scale, horizon)
    with torch.no_grad():
        rel = model(x).squeeze(0).numpy().astype(np.float32)
    prices_gpt = (base * (1.0 + rel)).astype(np.float32)

    group_id = str(uuid.uuid4())
    created_at = sim_created_at()
    header = {
        "group_id": group_id,
        "created_at": created_at,
        "source": source,
        "article_id": article_id,
        "symbol": symbol,
        "close_t": base,
        "impact_1d": imp1,
        "impact_3d": imp3,
        "impact_7d": imp7,
        "horizon": int(horizon),
        "model_name": "GPT_only_h6",
        "model_cfg_json": json.dumps({
            "hidden": cfg.get("hidden"),
            "layers": cfg.get("layers"),
            "lr": cfg.get("lr"),
            "batch": cfg.get("batch"),
            "epochs": cfg.get("epochs"),
            "weight_decay": cfg.get("weight_decay"),
            "lstm_dropout": cfg.get("lstm_dropout", 0.0)
        }, ensure_ascii=False),
    }

    lines = [{
        "group_id": group_id, "model_type": "gpt", "step": i+1,
        "rel_return": float(rel[i]), "price": float(prices_gpt[i]),
    } for i in range(horizon)]

    insert_forecast(db_path, header, lines)
    return group_id

def run(items, ckpt_path: str, db_path: str, source: str):
    ensure_db(db_path)
    model, sym_classes, mean, scale, horizon, cfg = load_model(ckpt_path)
    out_groups = []
    for it in items:
        gid = predict_item(model, sym_classes, mean, scale, horizon, cfg, it, source, db_path)
        out_groups.append(gid)
        print(f"✓ gespeichert — group_id={gid}, symbol={it['symbol']}, horizon={horizon}")
    return out_groups

# ================== CLI ==================
def parse_args():
    ap = argparse.ArgumentParser(description="GPT-only LSTM predictions -> SQLite DB.")
    ap.add_argument("--ckpt",    default=DEFAULT_CKPT)
    ap.add_argument("--db",      default=DEFAULT_DB)
    ap.add_argument("--json",    default=None)
    ap.add_argument("--source",  default="gpt-realtime")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.json and os.path.exists(args.json):
        try:
            with open(args.json, "r", encoding="utf-8") as f:
                items = json.load(f)
        except json.JSONDecodeError:
            with open(args.json, "r", encoding="utf-8-sig") as f:
                items = json.load(f)
        if isinstance(items, dict):
            items = [items]
    else:
        items = [{
            "symbol": "UBS", "impact_1d": 0.3, "impact_3d": 0.8, "impact_7d": 1.4,
            "close_t": 27.90, "article_id": "UBS-TEST-DEMOINPUT"
        }]
        print("⚠️  Keine --json übergeben. Nutze Demo-Item …")

    run(items, args.ckpt, args.db, args.source)
