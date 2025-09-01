import os, json, sqlite3
from datetime import datetime, timedelta, date
import math

# ================== CONFIG ==================
NEWS_DB = r"C:\Bachelorarbeit\Datenbank\news_cache.db"
OUT_DB  = r"C:\Bachelorarbeit\Datenbank\kummulierte_artikel.db"

ALLOWED = {
    "UBS": {"ubs", "ubs group"},
    "Nestlé": {"nestlé", "nestle", "nesn"},
    "Novartis": {"novartis", "novn"},
    "Roche": {"roche", "rog"},
    "Zurich Insurance": {"zurich insurance", "zurich", "zurich insurance group", "zurn"},
}
CANONICAL = list(ALLOWED.keys())

AGG_MODE = "mean"

SOFT_CAPS = {"1d": 8.0, "3d": 12.0, "7d": 20.0}

# ================== DB HELPERS ==================
def _connect(db_path):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con

def ensure_out_db(path=OUT_DB):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _connect(path) as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS aggregated_impacts (
          as_of_date   TEXT,
          stock        TEXT,
          impact_1d    REAL,
          impact_3d    REAL,
          impact_7d    REAL,
          computed_at  TEXT,
          details_json TEXT,
          PRIMARY KEY (as_of_date, stock)
        )
        """)
        con.commit()

# ================== UTIL ==================
def normalize_stock(name: str):
    if not name: return None
    s = name.strip().lower()
    for canon, variants in ALLOWED.items():
        if s == canon.lower() or s in variants:
            return canon
    return None

def parse_article_date(date_str: str):
    if not date_str: return None
    s = date_str.strip()
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S", "%d.%m.%Y %H:%M"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    if s.lower() in ("unbekannt", "unknown"):
        return None
    return None

def soft_cap(x: float, cap: float | None) -> float:
    """Sanfte Sättigung mit tanh; cap in %-Punkten. None -> keine Sättigung."""
    if cap is None or cap <= 0:
        return x
    return math.copysign(cap, x) * math.tanh(abs(x) / cap)

# ================== FETCH ==================
def get_articles_window(canon: str, today: date, window_days: int = 7):
    """
    Holt Artikel der letzten `window_days` (inkl. heute) für das Canonical inkl. Synonyme;
    dedupliziert per group_id (beste source_quality).
    """
    variants = {canon.lower()} | set(ALLOWED[canon])
    qmarks   = ",".join(["?"] * len(variants))
    params   = tuple(sorted(variants))

    start_date = today - timedelta(days=window_days - 1)

    with _connect(NEWS_DB) as con:
        rows = con.execute(f"""
            SELECT id, group_id, date, impact_1d, impact_3d, impact_7d, source_quality, stock
            FROM articles
            WHERE gpt_checked=1 AND relevant=1
              AND stock IS NOT NULL AND TRIM(stock) <> ''
              AND LOWER(stock) IN ({qmarks})
        """, params).fetchall()

    best_by_group = {}
    for r in rows:
        art_date = parse_article_date(r["date"])
        if not art_date:
            continue
        if art_date < start_date or art_date > today:
            continue
        gid = r["group_id"]
        sq  = r["source_quality"] if r["source_quality"] is not None else -1e9
        keep = best_by_group.get(gid)
        if keep is None or sq >= (keep["source_quality"] if keep and keep["source_quality"] is not None else -1e9):
            rr = dict(r)
            rr["_art_date"] = art_date
            best_by_group[gid] = rr

    return list(best_by_group.values())

# ================== CORE ==================
def residual_weight(h: int, age: int) -> float:
    """Linearer Restanteil einer h-Tage-Wirkung nach 'age' Tagen; age=0..6, h in {1,3,7}."""
    if age < 0: age = 0
    if age >= h: return 0.0
    return (h - age) / float(h)

def aggregate_for_today(canon: str, today_d: date):
    """
    Aggregation exakt wie in deiner Doku (Cross-Mapping nach Alter):
      Ziel 1T (t+1):  age=0 → Impact_1T*1.0;  age=1–2 → Impact_3T*w3;  age=3–6 → Impact_7T*w7
      Ziel 3T (t+3):  age=0 → Impact_3T*1.0;  age=1–4 → Impact_7T*w7
      Ziel 7T (t+7):  age=0 → Impact_7T*1.0
    Danach optional sanfte Sättigung (tanh) pro Horizont.
    """
    arts = get_articles_window(canon, today_d, window_days=7)

    # Zähler/Denominator je Ziel-Horizont (1d, 3d, 7d)
    num = {"1d": 0.0, "3d": 0.0, "7d": 0.0}
    den = {"1d": 0.0, "3d": 0.0, "7d": 0.0}

    details = []

    for a in arts:
        art_date = a["_art_date"]
        age = (today_d - art_date).days  # 0..6

        i1 = float(a["impact_1d"]) if a["impact_1d"] is not None else None
        i3 = float(a["impact_3d"]) if a["impact_3d"] is not None else None
        i7 = float(a["impact_7d"]) if a["impact_7d"] is not None else None

        # Residualgewichte
        w3 = residual_weight(3, age)
        w7 = residual_weight(7, age)

        # ---- Ziel 1T ----
        if age == 0:
            if i1 is not None:
                num["1d"] += i1 * 1.0
                den["1d"] += 1.0
                details.append({
                    "target": "1d", "article_id": a["id"], "group_id": a["group_id"],
                    "article_date": art_date.isoformat(), "age_days": age,
                    "source_impact": "impact_1d", "impact_value": i1,
                    "weight": 1.0, "contribution": i1 * 1.0
                })
        elif age in (1, 2):
            # Impact_3T trägt mit w3
            if i3 is not None and w3 > 0.0:
                num["1d"] += i3 * w3
                den["1d"] += w3
                details.append({
                    "target": "1d", "article_id": a["id"], "group_id": a["group_id"],
                    "article_date": art_date.isoformat(), "age_days": age,
                    "source_impact": "impact_3d", "impact_value": i3,
                    "weight": w3, "contribution": i3 * w3
                })
        elif age in (3, 4, 5, 6):
            # Impact_7T trägt mit w7
            if i7 is not None and w7 > 0.0:
                num["1d"] += i7 * w7
                den["1d"] += w7
                details.append({
                    "target": "1d", "article_id": a["id"], "group_id": a["group_id"],
                    "article_date": art_date.isoformat(), "age_days": age,
                    "source_impact": "impact_7d", "impact_value": i7,
                    "weight": w7, "contribution": i7 * w7
                })

        # ---- Ziel 3T ----
        if age == 0:
            # Impact_3T zählt voll
            if i3 is not None:
                num["3d"] += i3 * 1.0
                den["3d"] += 1.0
                details.append({
                    "target": "3d", "article_id": a["id"], "group_id": a["group_id"],
                    "article_date": art_date.isoformat(), "age_days": age,
                    "source_impact": "impact_3d", "impact_value": i3,
                    "weight": 1.0, "contribution": i3 * 1.0
                })
        elif age in (1, 2, 3, 4):
            # Impact_7T trägt mit w7
            if i7 is not None and w7 > 0.0:
                num["3d"] += i7 * w7
                den["3d"] += w7
                details.append({
                    "target": "3d", "article_id": a["id"], "group_id": a["group_id"],
                    "article_date": art_date.isoformat(), "age_days": age,
                    "source_impact": "impact_7d", "impact_value": i7,
                    "weight": w7, "contribution": i7 * w7
                })

        # ---- Ziel 7T ----
        if age == 0:
            # nur Impact_7T zählt
            if i7 is not None:
                num["7d"] += i7 * 1.0
                den["7d"] += 1.0
                details.append({
                    "target": "7d", "article_id": a["id"], "group_id": a["group_id"],
                    "article_date": art_date.isoformat(), "age_days": age,
                    "source_impact": "impact_7d", "impact_value": i7,
                    "weight": 1.0, "contribution": i7 * 1.0
                })

    # Aggregierte Werte berechnen (gewichteter Mittelwert oder Summe)
    if AGG_MODE == "sum":
        totals = {
            "1d": num["1d"],
            "3d": num["3d"],
            "7d": num["7d"],
        }
    else:  # "mean" (default; entspricht Doku)
        totals = {
            "1d": (num["1d"] / den["1d"]) if den["1d"] > 0 else 0.0,
            "3d": (num["3d"] / den["3d"]) if den["3d"] > 0 else 0.0,
            "7d": (num["7d"] / den["7d"]) if den["7d"] > 0 else 0.0,
        }

    # Sanfte Caps anwenden (gegen Ausreißer)
    totals["1d"] = soft_cap(totals["1d"], SOFT_CAPS.get("1d"))
    totals["3d"] = soft_cap(totals["3d"], SOFT_CAPS.get("3d"))
    totals["7d"] = soft_cap(totals["7d"], SOFT_CAPS.get("7d"))

    # Debug-Hinweis bei sehr großen Werten
    extreme = []
    for k in ("1d", "3d", "7d"):
        th = SOFT_CAPS.get(k)
        if th and abs(totals[k]) >= th * 0.9:
            extreme.append(k)
    if extreme:
        print(f"[WARN] Starke aggregierte Impacts für {canon} @ {today_d}: {', '.join(extreme)} -> {totals}")

    return totals, {
        "agg_mode": AGG_MODE,
        "soft_caps": SOFT_CAPS,
        "items": details,
        "num": num,
        "den": den,
    }

def write_row(as_of_date: date, canon: str, totals: dict, details_obj: dict):
    ensure_out_db(OUT_DB)
    payload = {
        "as_of_date": as_of_date.isoformat(),
        "stock": canon,
        "impact_1d": float(totals.get("1d", 0.0)),
        "impact_3d": float(totals.get("3d", 0.0)),
        "impact_7d": float(totals.get("7d", 0.0)),
        "computed_at": datetime.utcnow().isoformat() + "Z",
        "details_json": json.dumps(details_obj, ensure_ascii=False),
    }
    with _connect(OUT_DB) as con:
        con.execute("""
            INSERT OR REPLACE INTO aggregated_impacts
            (as_of_date, stock, impact_1d, impact_3d, impact_7d, computed_at, details_json)
            VALUES (:as_of_date, :stock, :impact_1d, :impact_3d, :impact_7d, :computed_at, :details_json)
        """, payload)
        con.commit()

def run_all(as_of: date = None):
    # Simulationsdatum aus ENV übernehmen
    env_as_of = os.getenv("AS_OF_DATE")
    if env_as_of:
        try:
            today_d = date.fromisoformat(env_as_of)
        except ValueError:
            print(f"[WARN] Ungültiges AS_OF_DATE='{env_as_of}', fallback auf heutiges Datum.")
            today_d = as_of or date.today()
    else:
        today_d = as_of or date.today()

    print(f"[INFO] Kumulierte Impacts ({AGG_MODE}, soft_caps={SOFT_CAPS}) für {today_d.isoformat()}")

    present = []
    for canon in CANONICAL:
        arts = get_articles_window(canon, today_d, window_days=7)
        if arts:
            present.append(canon)

    if not present:
        print("[WARN] Keine relevanten Artikel in den letzten 7 Tagen für unsere fünf Aktien.")
        return

    for canon in present:
        totals, details_obj = aggregate_for_today(canon, today_d)
        write_row(today_d, canon, totals, details_obj)
        print(f"OK {canon}: 1d={totals['1d']:+.3f}  3d={totals['3d']:+.3f}  7d={totals['7d']:+.3f}")

if __name__ == "__main__":
    run_all()
