# Streamlit_GUI.py
# Viewer für Hybrid (Basis) + optionale Overlays GPT, Historisch und Randome
# - Basis: Hybrid (hybrid_predictions.db)
# - Overlay: GPT (gpt_predictions.db, model_type='gpt')
# - Overlay: Historisch (historisch_predictions.db, model_type='historisch')
# - Overlay: Randome (randome_predictions.db, model_type='randome')
# - Echte Kurse: t-4..t0 aus predictions.db -> actual_prices, wobei t0 = Datum aus created_at der gewählten Vorhersage
# - Echte Kurse nach t0 (t+1..), wenn eine ältere Prediction-Version gewählt ist (Overlay)
# - Prozent-Labels (+1,+3,+7) je sichtbarem Modell in Linienfarbe
# - Legende unten in drei Spalten
# - X-Achse zeigt echte Kalenderdaten (Tag 0 = created_at-Datum)

import os
import sqlite3
from datetime import datetime, timedelta, date
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.dates as mdates

# Optionales Auto-Refresh Add-on
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

# ================== CONFIG ==================
HYBRID_DB  = os.environ.get("HYBRID_DB",  r"C:\Bachelorarbeit\Datenbank\hybrid_predictions.db")
GPT_DB     = os.environ.get("PRED_DB",    r"C:\Bachelorarbeit\Datenbank\gpt_predictions.db")
HIST_DB    = os.environ.get("HIST_DB",    r"C:\Bachelorarbeit\Datenbank\historisch_predictions.db")
RAND_DB    = os.environ.get("RAND_DB",    r"C:\Bachelorarbeit\Datenbank\randome_predictions.db")
PRICES_DB  = r"C:\Bachelorarbeit\Datenbank\predictions.db"

HORIZON_DEFAULT = 7
PAGE_TITLE = "Aktienkurs-Vorhersage"

ACTUAL_LOOKBACK = 5

# ================== DB-HELPERS ==================
def _connect(db_path: str):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con

def _mtime_or_zero(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0.0

def table_exists(db_path: str, table_name: str) -> bool:
    try:
        with _connect(db_path) as con:
            row = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            ).fetchone()
        return row is not None
    except Exception:
        return False

@st.cache_data(show_spinner=False)
def get_symbols_hybrid(db_path: str, _mtime: float) -> List[str]:
    if not table_exists(db_path, "forecast_header"):
        return []
    with _connect(db_path) as con:
        rows = con.execute("SELECT DISTINCT symbol FROM forecast_header ORDER BY symbol").fetchall()
    return [r[0] for r in rows]

@st.cache_data(show_spinner=False)
def get_header_history(db_path: str, symbol: str, limit: int, _mtime: float) -> list[dict]:
    if not table_exists(db_path, "forecast_header"):
        return []
    with _connect(db_path) as con:
        rows = con.execute(
            """
            SELECT group_id, created_at, symbol, close_t, horizon, model_name, source
            FROM forecast_header
            WHERE symbol=?
            ORDER BY datetime(created_at) DESC
            LIMIT ?
            """,
            (symbol, limit),
        ).fetchall()
    return [dict(r) for r in rows]

@st.cache_data(show_spinner=False)
def get_latest_header(db_path: str, symbol: str, _mtime: float) -> Optional[dict]:
    hist = get_header_history(db_path, symbol, limit=1, _mtime=_mtime)
    return hist[0] if hist else None

@st.cache_data(show_spinner=False)
def get_forecast_lines(db_path: str, group_id: str, model_type: str, _mtime: float) -> pd.DataFrame:
    if not table_exists(db_path, "forecast_lines"):
        return pd.DataFrame(columns=["step","rel_return","price"])
    with _connect(db_path) as con:
        df = pd.read_sql_query(
            """
            SELECT step, rel_return, price
            FROM forecast_lines
            WHERE group_id=? AND model_type=?
            ORDER BY step
            """,
            con,
            params=(group_id, model_type),
        )
    return df

@st.cache_data(show_spinner=False)
def get_actual_lastN_upto(db_path: str, symbol: str, n: int, cutoff_date: str, _mtime: float) -> pd.DataFrame:
    """
    Holt die letzten n Schlusskurse (aufsteigend sortiert) mit date <= cutoff_date (YYYY-MM-DD).
    Erwartet: Tabelle actual_prices(date, price, symbol) in PRICES_DB.
    """
    if not table_exists(db_path, "actual_prices"):
        return pd.DataFrame(columns=["date","price"])
    with _connect(db_path) as con:
        df = pd.read_sql_query(
            """
            SELECT date, price
            FROM actual_prices
            WHERE symbol=? AND date <= ?
            ORDER BY date DESC
            LIMIT ?
            """,
            con,
            params=(symbol, cutoff_date, n),
        )
    return df.sort_values("date").reset_index(drop=True)

@st.cache_data(show_spinner=False)
def get_overlay_actuals_after(db_path: str, symbol: str, after_date: str, _mtime: float) -> pd.DataFrame:
    """
    Echte Kurse NACH after_date (YYYY-MM-DD) als Overlay (t+1..), aufsteigend sortiert.
    """
    if not table_exists(db_path, "actual_prices"):
        return pd.DataFrame(columns=["date","price"])
    with _connect(db_path) as con:
        df = pd.read_sql_query(
            """
            SELECT date, price
            FROM actual_prices
            WHERE symbol=? AND date > ?
            ORDER BY date ASC
            """,
            con,
            params=(symbol, after_date),
        )
    return df

def _fmt_local(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.replace("Z","+00:00")).astimezone()
        return dt.strftime("%d.%m.%y um %H:%M")
    except Exception:
        return ts

def _created_at_date_str(created_at_iso: str) -> str:
    """
    Extrahiert das YYYY-MM-DD (lokale Trading-Referenz) aus created_at (ISO, typ. mit 'Z').
    """
    try:
        return created_at_iso.split("T", 1)[0]
    except Exception:
        return created_at_iso[:10]

def _date_list_from_offsets(t0_date: date, offsets: list[int]) -> list[datetime]:
    return [datetime(t0_date.year, t0_date.month, t0_date.day) + timedelta(days=o) for o in offsets]

# ================== PLOT ==================
def _build_path_from_lines_dates(df: pd.DataFrame, t0_dt: datetime, base_price: float, horizon: int) -> Tuple[list, list, dict]:
    steps = df["step"].astype(int).tolist() if not df.empty else []
    prices = df["price"].astype(float).tolist() if not df.empty else []
    xy = sorted(zip(steps, prices))
    x_model = [t0_dt]           # Tag 0
    y_model = [base_price]
    pts_map = {}                # {step: price}
    for s, p in xy:
        if 1 <= s <= horizon:
            x_model.append(t0_dt + timedelta(days=int(s)))
            y_model.append(float(p))
            pts_map[int(s)] = float(p)
    # Pfad bis Tag +7 strecken, falls Horizon < 7
    last_val = y_model[-1] if y_model else base_price
    if horizon < 7:
        x_model.append(t0_dt + timedelta(days=7))
        y_model.append(last_val)
    return x_model, y_model, pts_map

def _annotate_pct(ax, x_dt, y, color, text):
    ax.annotate(text, (x_dt, y),
                xytext=(0, 10), textcoords="offset points",
                ha="center", va="bottom", color=color, fontsize=10)

def _compute_delta(prices: dict, step: int, base_price: float, delta_mode: str) -> Optional[float]:
    if step not in prices:
        return None
    price_val = float(prices[step])
    if delta_mode == "Tag 0":
        return (price_val - base_price) / max(base_price, 1e-9) * 100.0
    else:
        prev_step = step - 1
        if prev_step in prices:
            prev_val = float(prices[prev_step])
            return (price_val - prev_val) / max(prev_val, 1e-9) * 100.0
        else:
            return (price_val - base_price) / max(base_price, 1e-9) * 100.0

def plot_all(symbol: str,
             created_at: str,
             base_price: float,
             actual_df: pd.DataFrame,
             hybrid_df: pd.DataFrame,
             gpt_df: Optional[pd.DataFrame],
             hist_df: Optional[pd.DataFrame],
             rand_df: Optional[pd.DataFrame],
             show_gpt: bool,
             show_hist: bool,
             show_rand: bool,
             horizon: int,
             delta_mode: str,
             overlay_actuals: Optional[pd.DataFrame] = None):

    # Tag 0 = created_at-Datum
    t0_str = _created_at_date_str(created_at)
    t0_date = date.fromisoformat(t0_str)
    t0_dt = datetime(t0_date.year, t0_date.month, t0_date.day)

    if not actual_df.empty:
        dates_past = [date.fromisoformat(str(d)[:10]) for d in actual_df["date"].tolist()]
        xa_past = [datetime(d.year, d.month, d.day) for d in dates_past]  # bis t0
        ya_past = actual_df["price"].astype(float).tolist()
    else:
        xa_past = [t0_dt]
        ya_past = [float(base_price)]

    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()

    # Echte Kurse (bis Tag 0)
    h_actual, = ax.plot(xa_past, ya_past, marker="o", linewidth=2, label=f"Echter Kurs (bis Tag 0)")

    # Overlay echte Kurse (t+ nach t0)
    h_actual_future = None
    if overlay_actuals is not None and not overlay_actuals.empty:
        x_overlay = [t0_dt] + [
            datetime.fromisoformat(str(d)[:10]) for d in overlay_actuals["date"].tolist()
        ]
        y_overlay = [ya_past[-1]] + overlay_actuals["price"].astype(float).tolist()
        h_actual_future, = ax.plot(
            x_overlay, y_overlay,
            marker="o", linewidth=2, linestyle="--",
            color=h_actual.get_color(),
            label="Echter Kurs (t+)"
        )

    # Hybrid
    x_h, y_h, pts_h = _build_path_from_lines_dates(hybrid_df, t0_dt, ya_past[-1], horizon)
    h_hybrid, = ax.plot(x_h, y_h, linewidth=2, label="Prediction (Hybrid)")

    # GPT
    pts_g = {}
    h_gpt = None
    if show_gpt and gpt_df is not None and not gpt_df.empty:
        x_g, y_g, pts_g = _build_path_from_lines_dates(gpt_df, t0_dt, ya_past[-1], horizon)
        h_gpt, = ax.plot(x_g, y_g, linewidth=2, linestyle="--", label="Prediction (GPT)")

    # Historisch
    pts_r = {}
    h_hist = None
    if show_hist and hist_df is not None and not hist_df.empty:
        x_r, y_r, pts_r = _build_path_from_lines_dates(hist_df, t0_dt, ya_past[-1], horizon)
        h_hist, = ax.plot(x_r, y_r, linewidth=2, linestyle=":", label="Prediction (Historisch)")

    # Randome
    pts_rand = {}
    h_rand = None
    if show_rand and rand_df is not None and not rand_df.empty:
        x_rand, y_rand, pts_rand = _build_path_from_lines_dates(rand_df, t0_dt, ya_past[-1], horizon)
        h_rand, = ax.plot(x_rand, y_rand, linewidth=2, linestyle="-.", label="Prediction (Randome)")

    # Punkte + Prozent-Annotation (+1, +3, +7 bzw. letzter verfügbarer Schritt)
    steps_to_annotate = [1, 3]
    if horizon >= 7:
        steps_to_annotate.append(7)
    elif horizon >= 6:
        steps_to_annotate.append(6)
    else:
        steps_to_annotate.append(horizon)

    for step in steps_to_annotate:
        x_dt = t0_dt + timedelta(days=step)
        if step in pts_h:
            y_val = pts_h[step]
            ax.scatter([x_dt], [y_val], s=60, marker="o", color=h_hybrid.get_color())
            delta = _compute_delta(pts_h, step, ya_past[-1], delta_mode)
            if delta is not None:
                _annotate_pct(ax, x_dt, y_val, h_hybrid.get_color(), f"{delta:+.2f}%")
        if h_gpt is not None and step in pts_g:
            y_val = pts_g[step]
            ax.scatter([x_dt], [y_val], s=60, marker="o", color=h_gpt.get_color())
            delta = _compute_delta(pts_g, step, ya_past[-1], delta_mode)
            if delta is not None:
                _annotate_pct(ax, x_dt, y_val, h_gpt.get_color(), f"{delta:+.2f}%")
        if h_hist is not None and step in pts_r:
            y_val = pts_r[step]
            ax.scatter([x_dt], [y_val], s=60, marker="o", color=h_hist.get_color())
            delta = _compute_delta(pts_r, step, ya_past[-1], delta_mode)
            if delta is not None:
                _annotate_pct(ax, x_dt, y_val, h_hist.get_color(), f"{delta:+.2f}%")
        if h_rand is not None and step in pts_rand:
            y_val = pts_rand[step]
            ax.scatter([x_dt], [y_val], s=60, marker="o", color=h_rand.get_color())
            delta = _compute_delta(pts_rand, step, ya_past[-1], delta_mode)
            if delta is not None:
                _annotate_pct(ax, x_dt, y_val, h_rand.get_color(), f"{delta:+.2f}%")

    # Achsen/Deko (echte Kalenderdaten auf X-Achse)
    ax.set_title(f"{symbol} — erstellt {_fmt_local(created_at)}")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Preis")
    ax.grid(True, alpha=0.25)

    # Datumsformatierung
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate(rotation=30)

    # Legende
    handles, labels = [], []
    handles.append(Line2D([0],[0], color=h_actual.get_color(), marker="o", linewidth=2))
    labels.append("Echter Kurs (bis Tag 0)")
    if h_actual_future is not None:
        handles.append(Line2D([0],[0], color=h_actual.get_color(), linestyle="--", marker="o", linewidth=2))
        labels.append("Echter Kurs (t+)")
    handles.append(Line2D([0],[0], color=h_hybrid.get_color(), linewidth=2))
    labels.append("Hybrid")
    if h_gpt is not None:
        handles.append(Line2D([0],[0], color=h_gpt.get_color(), linestyle="--", linewidth=2))
        labels.append("GPT")
    if h_hist is not None:
        handles.append(Line2D([0],[0], color=h_hist.get_color(), linestyle=":", linewidth=2))
        labels.append("Historisch")
    if h_rand is not None:
        handles.append(Line2D([0],[0], color=h_rand.get_color(), linestyle="-.", linewidth=2))
        labels.append("Randome")
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.15))

    plt.subplots_adjust(bottom=0.25)
    st.pyplot(fig)

# ================== UI ==================
st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(PAGE_TITLE)

col1, col2, col3, col4, col5 = st.columns([3,2,2,2,2])

# Aktuelle mtimes (steuern Cache-Invalidierung)
mtime_hybrid = _mtime_or_zero(HYBRID_DB)
mtime_gpt    = _mtime_or_zero(GPT_DB)
mtime_hist   = _mtime_or_zero(HIST_DB)
mtime_rand   = _mtime_or_zero(RAND_DB)
mtime_prices = _mtime_or_zero(PRICES_DB)

symbols = get_symbols_hybrid(HYBRID_DB, mtime_hybrid)
if not symbols:
    st.warning("Keine Symbole in der Hybrid-DB gefunden. Prüfe Pfad/DB:\n" + HYBRID_DB)
    st.stop()

with col1:
    symbol = st.selectbox("Symbol auswählen", symbols, index=0)
with col2:
    auto = st.toggle("Auto-Refresh (alle 10s)", value=False)
with col3:
    show_gpt = st.toggle("GPT-Modell einblenden", value=False)
with col4:
    show_hist = st.toggle("Historisches Modell einblenden", value=False)
with col5:
    show_rand = st.toggle("Randome-Modell einblenden", value=False)

if st.button("Daten aktualisieren", use_container_width=True):
    get_symbols_hybrid.clear()
    get_latest_header.clear()
    get_header_history.clear()
    get_forecast_lines.clear()
    get_actual_lastN_upto.clear()
    get_overlay_actuals_after.clear()
    st.rerun()

# Auto-Refresh
if auto:
    if st_autorefresh:
        st_autorefresh(interval=10_000, key="db-poll")
    else:
        now_ts = datetime.utcnow().timestamp()
        last = st.session_state.get("_last_auto_refresh", 0.0)
        if now_ts - last >= 10:
            st.session_state["_last_auto_refresh"] = now_ts
            st.rerun()

# Hybrid Auswahl
hist = get_header_history(HYBRID_DB, symbol, limit=30, _mtime=mtime_hybrid)
if not hist:
    st.info("Für dieses Symbol existiert (noch) keine Hybrid-Vorhersage.")
    st.stop()

cA, cB = st.columns([2, 3])
with cA:
    use_latest = st.toggle("Immer neueste Hybrid-Vorhersage verwenden", value=True)
with cB:
    active_hdr = hist[0]
    if not use_latest:
        options = [f"{_fmt_local(h['created_at'])} — {h.get('model_name') or 'Hybrid'} — group={h['group_id'][:8]}" for h in hist]
        sel = st.selectbox("Hybrid-Vorhersage-Version wählen", options, index=0)
        active_hdr = hist[options.index(sel)]

st.success(f"Hybrid-Modell wurde am {_fmt_local(active_hdr['created_at'])} aktualisiert.", icon="🕒")

group_id   = active_hdr["group_id"]
created_at = active_hdr["created_at"]
base_price = float(active_hdr["close_t"]) if active_hdr["close_t"] is not None else np.nan
horizon    = int(active_hdr.get("horizon") or HORIZON_DEFAULT)

# Tag 0 Datum (YYYY-MM-DD)
t0_str = _created_at_date_str(created_at)

# Daten laden
hybrid_df = get_forecast_lines(HYBRID_DB, group_id, "hybrid", _mtime=mtime_hybrid)
actual_df = get_actual_lastN_upto(PRICES_DB, symbol, n=ACTUAL_LOOKBACK, cutoff_date=t0_str, _mtime=mtime_prices)

gpt_df = None
if show_gpt:
    gpt_hdr = get_latest_header(GPT_DB, symbol, _mtime=mtime_gpt)
    if gpt_hdr:
        gpt_df = get_forecast_lines(GPT_DB, gpt_hdr["group_id"], "gpt", _mtime=mtime_gpt)

hist_df = None
if show_hist:
    rh_hdr = get_latest_header(HIST_DB, symbol, _mtime=mtime_hist)
    if rh_hdr:
        hist_df = get_forecast_lines(HIST_DB, rh_hdr["group_id"], "historisch", _mtime=mtime_hist)

rand_df = None
if show_rand:
    rand_hdr = get_latest_header(RAND_DB, symbol, _mtime=mtime_rand)
    if rand_hdr:
        rand_df = get_forecast_lines(RAND_DB, rand_hdr["group_id"], "randome", _mtime=mtime_rand)

overlay_actuals = None
if not use_latest and not actual_df.empty:
    last_actual_date = str(actual_df.iloc[-1]["date"])
    overlay_actuals = get_overlay_actuals_after(PRICES_DB, symbol, last_actual_date, _mtime=mtime_prices)
    if overlay_actuals.empty:
        overlay_actuals = None

plot_all(symbol, created_at, base_price,
         actual_df, hybrid_df, gpt_df, hist_df, rand_df,
         show_gpt, show_hist, show_rand,
         horizon, delta_mode="Tag 0",
         overlay_actuals=overlay_actuals)

with st.expander("Debug: Datenstatus"):
    st.write(f"HYBRID_DB mtime: {mtime_hybrid}")
    st.write(f"PRICES_DB mtime: {mtime_prices}")
    if table_exists(PRICES_DB, "actual_prices"):
        with _connect(PRICES_DB) as con:
            cnt_total = con.execute("SELECT COUNT(*) FROM actual_prices WHERE symbol=?", (symbol,)).fetchone()[0]
        st.write(f"Rows in actual_prices für {symbol}: {cnt_total}")
        st.write("Letzte Kurse bis Tag 0 (gefiltert auf created_at-Datum):")
        st.dataframe(actual_df)
        if overlay_actuals is not None:
            st.write("Overlay-Kurse (t+):")
            st.dataframe(overlay_actuals)
        else:
            st.write("Keine Overlay-Kurse gefunden (entweder neueste Prediction gewählt oder keine neuen Schlusskurse).")
    else:
        st.write("Tabelle actual_prices nicht gefunden.")
