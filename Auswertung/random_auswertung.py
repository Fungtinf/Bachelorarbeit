# evaluate_by_groupid_rules_weekend_last_of_day.py
import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import timedelta

# ================== Pfade ==================
BASE_DIR = r"C:\Bachelorarbeit\Datenbank"

DB_FORECAST = os.path.join(BASE_DIR, "randome_predictions.db")
DB_ACTUAL   = os.path.join(BASE_DIR, "predictions.db")

DATE_FROM = "2025-06-15"
DATE_TO   = "2025-08-15"

# Nur diese Symbole werten (None = alle)
SYMBOLS_WHITELIST = {"UBS", "Nestlé", "Novartis", "Roche", "Zurich Insurance"}

EVAL_STEPS = {1, 3, 7}

COMPARE_MODE = "return"

# CSV-Export
WRITE_CSV = True
CSV_DETAILS = os.path.join(BASE_DIR, "forecast_eval_details_rules.csv")
CSV_SUMMARY = os.path.join(BASE_DIR, "forecast_eval_summary_rules.csv")


# ================== Helpers ==================
def hitrate(df: pd.DataFrame) -> float:
    if df.empty:
        return np.nan
    p = df["predicted"].to_numpy(float)
    a = df["actual"].to_numpy(float)
    mask = ~np.isnan(p) & ~np.isnan(a)
    if not np.any(mask):
        return np.nan
    hits = np.sum((p[mask] >= 0) == (a[mask] >= 0))
    return 100.0 * hits / np.sum(mask)

def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-12
    mask = np.abs(y_true) > eps
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100.0)

def calc_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "MSE": np.nan, "MAPE": np.nan, "N": 0}
    err = df["predicted"] - df["actual"]
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mape = safe_mape(df["actual"].to_numpy(float), df["predicted"].to_numpy(float))
    return {"MAE": mae, "RMSE": rmse, "MSE": mse, "MAPE": mape, "N": int(len(df))}

def tidy_metrics_row(m: dict) -> dict:
    out = {k: (None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), 6))
           for k, v in m.items() if k in ("MAE", "RMSE", "MSE", "MAPE")}
    out["N"] = m.get("N", 0)
    return out

def is_saturday(ts: pd.Timestamp) -> bool:
    return ts.weekday() == 5

def is_sunday(ts: pd.Timestamp) -> bool:
    return ts.weekday() == 6


# ================== Loaders ==================
def load_headers_raw() -> pd.DataFrame:
    con = sqlite3.connect(DB_FORECAST)
    df = pd.read_sql("""
        SELECT group_id, created_at, symbol, close_t
        FROM forecast_header
        WHERE date(created_at) BETWEEN date(?) AND date(?)
    """, con, params=[DATE_FROM, DATE_TO])
    con.close()

    created = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["created_at_raw"] = created.dt.tz_convert(None)
    df["created_at_day"] = created.dt.tz_convert(None).dt.normalize()
    df["created_date"]   = df["created_at_day"].dt.date

    if SYMBOLS_WHITELIST:
        df = df[df["symbol"].isin(SYMBOLS_WHITELIST)]

    # Regel 1: Prognosen, die am Samstag erstellt wurden -> ignorieren
    df = df[~df["created_at_raw"].dt.dayofweek.eq(5)]

    df = df.dropna(subset=["group_id", "created_at_raw", "created_at_day", "symbol"]).reset_index(drop=True)
    return df[["group_id", "symbol", "close_t", "created_at_raw", "created_at_day", "created_date"]]

def keep_last_per_symbol_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regel: pro (symbol, created_date) nur die zuletzt erstellte Prognose (max(created_at_raw)).
    """
    if df.empty:
        return df.copy()
    df_sorted = df.sort_values(["symbol", "created_date", "created_at_raw"])
    last_rows = df_sorted.groupby(["symbol", "created_date"], as_index=False).tail(1)
    return last_rows.reset_index(drop=True)

def load_headers() -> pd.DataFrame:
    df = load_headers_raw()
    return keep_last_per_symbol_day(df)

def load_lines_for_group(group_id: str) -> pd.DataFrame:
    con = sqlite3.connect(DB_FORECAST)
    df = pd.read_sql("""
        SELECT group_id, model_type, step, rel_return, price
        FROM forecast_lines
        WHERE group_id = ?
    """, con, params=[group_id])
    con.close()
    df = df[df["step"].isin(EVAL_STEPS)]
    return df

def load_prices() -> pd.DataFrame:
    con = sqlite3.connect(DB_ACTUAL)
    px = pd.read_sql("SELECT symbol, date, price FROM actual_prices", con)
    con.close()
    px["date"] = pd.to_datetime(px["date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    if SYMBOLS_WHITELIST:
        px = px[px["symbol"].isin(SYMBOLS_WHITELIST)]
    px = px.dropna(subset=["symbol", "date", "price"]).sort_values(["symbol", "date"]).reset_index(drop=True)
    return px

def build_price_index(px: pd.DataFrame) -> dict[str, pd.DataFrame]:
    idx = {}
    for sym, df_sym in px.groupby("symbol"):
        df_sym = df_sym.drop_duplicates(subset=["date"], keep="last").set_index("date")[["price"]]
        idx[sym] = df_sym
    return idx

def exact_price_on(price_df: pd.DataFrame, target_date: pd.Timestamp):
    td = pd.Timestamp(target_date).tz_localize(None).normalize()
    if td in price_df.index:
        return td, float(price_df.loc[td, "price"])
    return None, None

def next_trading_day_on_or_after(price_df: pd.DataFrame, target_date: pd.Timestamp):
    """
    Gibt den ersten vorhandenen Handelstag >= target_date zurück (z. B. Sonntag -> Montag).
    """
    td = pd.Timestamp(target_date).tz_localize(None).normalize()
    idx = price_df.index
    pos = idx.get_indexer([td], method="backfill")[0]
    if pos == -1:
        return None, None
    d = idx[pos]
    return d, float(price_df.loc[d, "price"])


# ================== Evaluation (mit deinen Regeln) ==================
def evaluate(headers: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    price_idx = build_price_index(prices)
    rows = []

    for _, h in headers.iterrows():
        gid     = h["group_id"]
        sym     = h["symbol"]
        created_day = h["created_at_day"]
        close_t = h.get("close_t")

        if sym not in price_idx:
            continue
        df_sym = price_idx[sym]

        # Regel 5: Startkurs MUSS der Schlusskurs des Prognosetages sein
        start_d, start_px = exact_price_on(df_sym, created_day)
        if start_d is None:
            # kein offizieller Schlusskurs am Prognosetag -> Prognose nicht bewerten
            continue

        lines = load_lines_for_group(gid)
        if lines.empty:
            continue

        for _, l in lines.iterrows():
            model = l["model_type"]
            step  = int(l["step"])
            if step not in EVAL_STEPS:
                continue

            # Vorhersagewert
            if COMPARE_MODE == "return":
                if pd.isna(l["rel_return"]):
                    continue
                predicted = float(l["rel_return"])
            else:
                if pd.notnull(l["price"]) and float(l["price"]) > 0:
                    predicted = float(l["price"])
                elif pd.notnull(l["rel_return"]):
                    predicted = float(start_px) * (1.0 + float(l["rel_return"]))
                else:
                    continue

            # Zieltermin nach Kalendertagen
            target_cal = (created_day + timedelta(days=step)).normalize()

            # Regel 2 & 3:
            # - Ziel = Samstag -> IGNORIEREN (keine Bewertung)
            # - Ziel = Sonntag -> auf nächsten Handelstag verschieben
            if is_saturday(target_cal):
                continue  # ignorieren
            elif is_sunday(target_cal):
                end_d, end_px = next_trading_day_on_or_after(df_sym, target_cal + timedelta(days=1))
            else:
                # Regulärer Werktag: es muss ein Schlusskurs am Ziel-Kalendertag existieren
                end_d, end_px = exact_price_on(df_sym, target_cal)

            if end_d is None:
                # kein Kurs verfügbar -> nicht werten
                continue

            # Ist-Vergleich
            if COMPARE_MODE == "return":
                actual = (end_px - start_px) / start_px
            else:
                actual = end_px

            rows.append({
                "model": model,
                "symbol": sym,
                "created_at": created_day,
                "horizon": step,
                "predicted": float(predicted),
                "actual": float(actual),
            })

    return pd.DataFrame(rows)


# ================== Ausgabe ==================
def print_summary(df_eval: pd.DataFrame):
    if df_eval.empty:
        print("❌ Keine auswertbaren Zeilen. Prüfe Zeitraum, Symbole oder Kursabdeckung.")
        return

    df_eval = df_eval.sort_values(["symbol", "model", "horizon", "created_at"]).reset_index(drop=True)

    if WRITE_CSV:
        df_eval.to_csv(CSV_DETAILS, index=False, encoding="utf-8")
        print(f"✅ Details-CSV: {CSV_DETAILS}")

    rows_sum = []
    for (model, sym, h), sub in df_eval.groupby(["model", "symbol", "horizon"]):
        m = calc_metrics(sub)
        hit = hitrate(sub)
        rows_sum.append({
            "model": model,
            "symbol": sym,
            "horizon": h,
            **tidy_metrics_row(m),
            "HitRate_%": None if np.isnan(hit) else round(hit, 2)
        })

    df_sum = pd.DataFrame(rows_sum).sort_values(["symbol","model","horizon"]).reset_index(drop=True)

    if WRITE_CSV:
        csv_cols = ["model","symbol","horizon","MAE","RMSE","MSE","MAPE","HitRate_%","N"]
        df_sum.to_csv(CSV_SUMMARY, index=False, encoding="utf-8", columns=csv_cols)
        print(f"✅ Summary-CSV: {CSV_SUMMARY}")

    for sym in df_sum["symbol"].unique():
        print(f"\n================  {sym}  ================")
        sub = df_sum[df_sum["symbol"] == sym]
        cols = ["model","horizon","MAE","RMSE","MSE","MAPE","HitRate_%","N"]
        print(sub[cols].to_string(index=False))


# ================== Main ==================
def main():
    headers = load_headers()
    prices  = load_prices()

    print(f"Headers (nach Regeln & Last-of-Day): {len(headers)} | Prices: {len(prices)}")
    if headers.empty or prices.empty:
        print("⚠️ Datenbasis leer – Abbruch.")
        return

    df_eval = evaluate(headers, prices)
    print(f"Eval-Zeilen: {len(df_eval)}  (Mode: {COMPARE_MODE})")

    print_summary(df_eval)

if __name__ == "__main__":
    main()
