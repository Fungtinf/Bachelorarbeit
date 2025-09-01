# t_test_mae_rmse.py  (aktualisiert)
import os, sqlite3, numpy as np, pandas as pd
from datetime import timedelta
from scipy.stats import ttest_rel

BASE_DIR = r"C:\Bachelorarbeit\Datenbank"
DB_ACTUAL = os.path.join(BASE_DIR, "predictions.db")
DB_HYBRID = os.path.join(BASE_DIR, "hybrid_predictions.db")
DB_HIST   = os.path.join(BASE_DIR, "historisch_predictions.db")

DATE_FROM = "2025-06-15"
DATE_TO   = "2025-08-15"

SYMBOLS_WHITELIST = {"UBS", "Nestlé", "Novartis", "Roche", "Zurich Insurance"}

# TEST 1 (unverändert): alle Horizonte
EVAL_STEPS_ALL   = {1, 3, 7}
# TEST 2 (neu): pro Aktie UND alle Horizonte 1,3,7 separat
EVAL_STEPS_TEST2 = {1, 3, 7}

COMPARE_MODE = "return"
ALPHA = 0.05

def is_saturday(ts): return ts.weekday() == 5
def is_sunday(ts):   return ts.weekday() == 6

def load_prices():
    con = sqlite3.connect(DB_ACTUAL)
    px = pd.read_sql("SELECT symbol, date, price FROM actual_prices", con)
    con.close()
    px["date"] = pd.to_datetime(px["date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    px = px[px["symbol"].isin(SYMBOLS_WHITELIST)]
    return px.dropna(subset=["symbol","date","price"]).sort_values(["symbol","date"]).reset_index(drop=True)

def build_price_index(px):
    idx = {}
    for sym, df_sym in px.groupby("symbol"):
        df_sym = df_sym.drop_duplicates(subset=["date"], keep="last").set_index("date")[["price"]]
        idx[sym] = df_sym
    return idx

def exact_price_on(df, d):
    td = pd.Timestamp(d).tz_localize(None).normalize()
    if td in df.index:
        return td, float(df.loc[td, "price"])
    return None, None

def next_trading_day_on_or_after(df, d):
    td = pd.Timestamp(d).tz_localize(None).normalize()
    pos = df.index.get_indexer([td], method="backfill")[0]
    if pos == -1: return None, None
    dd = df.index[pos]
    return dd, float(df.loc[dd, "price"])

def load_headers_last_of_day(db):
    con = sqlite3.connect(db)
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
    df = df[df["symbol"].isin(SYMBOLS_WHITELIST)]
    # Samstags-Erstellungen ignorieren
    df = df[~df["created_at_raw"].dt.dayofweek.eq(5)]
    df = df.dropna(subset=["group_id","created_at_raw","created_at_day","symbol"]).reset_index(drop=True)
    # pro (symbol, Tag) nur letzte Prognose des Tages
    df = df.sort_values(["symbol","created_date","created_at_raw"]).groupby(["symbol","created_date"], as_index=False).tail(1)
    return df[["group_id","symbol","close_t","created_at_day"]].reset_index(drop=True)

def load_lines_for_group(db, gid, steps):
    con = sqlite3.connect(db)
    df = pd.read_sql("""
        SELECT group_id, model_type, step, rel_return, price
        FROM forecast_lines
        WHERE group_id = ?
    """, con, params=[gid])
    con.close()
    return df[df["step"].isin(steps)]

def evaluate_db(db, eval_steps):
    headers = load_headers_last_of_day(db)
    prices = load_prices()
    price_idx = build_price_index(prices)
    rows = []
    for _, h in headers.iterrows():
        gid, sym, created_day = h["group_id"], h["symbol"], h["created_at_day"]
        if sym not in price_idx: continue
        df_sym = price_idx[sym]
        # Start = Schlusskurs am Prognosetag (harte Regel)
        _, start_px = exact_price_on(df_sym, created_day)
        if start_px is None: continue
        lines = load_lines_for_group(db, gid, eval_steps)
        if lines.empty: continue
        for _, l in lines.iterrows():
            step = int(l["step"])
            target = (created_day + pd.Timedelta(days=step)).normalize()
            if is_saturday(target):
                continue
            if is_sunday(target):
                _, end_px = next_trading_day_on_or_after(df_sym, target + pd.Timedelta(days=1))
            else:
                _, end_px = exact_price_on(df_sym, target)
            if end_px is None: continue
            if COMPARE_MODE == "return":
                if pd.isna(l["rel_return"]): continue
                predicted = float(l["rel_return"])
                actual = (end_px - start_px) / start_px
            else:
                if pd.notnull(l["price"]) and float(l["price"]) > 0:
                    predicted = float(l["price"])
                elif pd.notnull(l["rel_return"]):
                    predicted = float(start_px) * (1.0 + float(l["rel_return"]))
                else:
                    continue
                actual = end_px
            err = predicted - actual
            rows.append({"symbol": sym, "horizon": step, "created_at": created_day,
                         "abs_err": abs(err), "sq_err": err*err})
    return pd.DataFrame(rows)

from math import sqrt
def paired_t_mae_rmse_with_means(df_hyb, df_hist, subset=None):
    if subset is not None:
        A = subset(df_hyb)
        B = subset(df_hist)
    else:
        A, B = df_hyb.copy(), df_hist.copy()
    key = ["symbol","horizon","created_at"]
    A_ = A[key+["abs_err","sq_err"]].rename(columns={"abs_err":"abs_hyb","sq_err":"sq_hyb"})
    B_ = B[key+["abs_err","sq_err"]].rename(columns={"abs_err":"abs_hist","sq_err":"sq_hist"})
    M  = pd.merge(A_, B_, on=key, how="inner")
    if len(M)==0:
        return {"N":0, "mean_mae_hyb":np.nan, "mean_mae_hist":np.nan, "delta_mae":np.nan, "t_mae":np.nan, "p_mae":np.nan,
                "mean_rmse_hyb":np.nan, "mean_rmse_hist":np.nan, "delta_rmse":np.nan, "t_rmse":np.nan, "p_rmse":np.nan}
    # MAE (auf abs_err)
    mean_mae_hyb  = float(M["abs_hyb"].mean())
    mean_mae_hist = float(M["abs_hist"].mean())
    delta_mae     = mean_mae_hyb - mean_mae_hist
    t_mae, p_mae  = ttest_rel(M["abs_hyb"], M["abs_hist"], alternative="less")
    mean_rmse_hyb  = float(np.sqrt(M["sq_hyb"].mean()))
    mean_rmse_hist = float(np.sqrt(M["sq_hist"].mean()))
    delta_rmse     = mean_rmse_hyb - mean_rmse_hist
    t_rmse, p_rmse = ttest_rel(M["sq_hyb"], M["sq_hist"], alternative="less")
    return {"N":len(M),
            "mean_mae_hyb":mean_mae_hyb, "mean_mae_hist":mean_mae_hist, "delta_mae":delta_mae, "t_mae":t_mae, "p_mae":p_mae,
            "mean_rmse_hyb":mean_rmse_hyb, "mean_rmse_hist":mean_rmse_hist, "delta_rmse":delta_rmse, "t_rmse":t_rmse, "p_rmse":p_rmse}

def fmt(x, nd=6):
    if x is None or (isinstance(x,float) and (np.isnan(x) or np.isinf(x))): return "-"
    return f"{x:.{nd}f}"

def main():
    # TEST 1: alle Aktien & Horizonte (1,3,7)
    df_hyb_all  = evaluate_db(DB_HYBRID, EVAL_STEPS_ALL)
    df_hist_all = evaluate_db(DB_HIST,   EVAL_STEPS_ALL)
    res1 = paired_t_mae_rmse_with_means(df_hyb_all, df_hist_all)

    print("=== TEST 1: Alle Aktien & Horizonte (α=0.05, H1: Hybrid < Historisch) ===")
    print("Metrik\tMittelwert Hybrid\tMittelwert Historisch\tΔ (Hybrid−Historisch)\tt-Wert\tp-Wert\tSignifikant")
    print(f"MAE\t{fmt(res1['mean_mae_hyb'])}\t{fmt(res1['mean_mae_hist'])}\t{fmt(res1['delta_mae'])}\t{fmt(res1['t_mae'])}\t{fmt(res1['p_mae'])}\t{'JA' if res1['p_mae']<ALPHA else 'NEIN'}")
    print(f"RMSE\t{fmt(res1['mean_rmse_hyb'])}\t{fmt(res1['mean_rmse_hist'])}\t{fmt(res1['delta_rmse'])}\t{fmt(res1['t_rmse'])}\t{fmt(res1['p_rmse'])}\t{'JA' if res1['p_rmse']<ALPHA else 'NEIN'}")
    print(f"N (gepairte Prognosen): {res1['N']}\n")

    # TEST 2: pro Aktie & pro Horizont (1,3,7 separat)
    df_hyb_sep  = evaluate_db(DB_HYBRID, EVAL_STEPS_TEST2)
    df_hist_sep = evaluate_db(DB_HIST,   EVAL_STEPS_TEST2)

    print("=== TEST 2: Pro Aktie & pro Horizont (α=0.05, H1: Hybrid < Historisch) ===")
    print("Aktie\tHorizont\tMetrik\tMittelwert Hybrid\tMittelwert Historisch\tΔ (Hybrid−Historisch)\tt-Wert\tp-Wert\tSignifikant")
    for sym in sorted(SYMBOLS_WHITELIST):
        for h in (1,3,7):
            subset = (lambda df, s=sym, hh=h: df[(df["symbol"]==s) & (df["horizon"]==hh)])
            r = paired_t_mae_rmse_with_means(df_hyb_sep, df_hist_sep, subset=subset)
            # MAE
            print(f"{sym}\tImpact_{h}T\tMAE\t{fmt(r['mean_mae_hyb'])}\t{fmt(r['mean_mae_hist'])}\t"
                  f"{fmt(r['delta_mae'])}\t{fmt(r['t_mae'])}\t{fmt(r['p_mae'])}\t"
                  f"{'JA' if r['p_mae']<ALPHA else 'NEIN'}")
            # RMSE
            print(f"{sym}\tImpact_{h}T\tRMSE\t{fmt(r['mean_rmse_hyb'])}\t{fmt(r['mean_rmse_hist'])}\t"
                  f"{fmt(r['delta_rmse'])}\t{fmt(r['t_rmse'])}\t{fmt(r['p_rmse'])}\t"
                  f"{'JA' if r['p_rmse']<ALPHA else 'NEIN'}")

if __name__ == "__main__":
    main()
