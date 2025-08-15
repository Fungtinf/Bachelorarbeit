# Streamlit_GUI.py
# Viewer für Hybrid (Basis) + optionale Overlays GPT & Historisch
# - Basis: Hybrid (hybrid_predictions.db)
# - Overlay: GPT (predictions.db, model_type='gpt')
# - Overlay: Historisch (historisch_predictions.db, model_type='historisch')
# - Echte Kurse (Tag-1, Tag0): predictions.db -> actual_prices
# - Prozent-Labels (+1,+3,+7) je sichtbarem Modell in Linienfarbe
# - Legende unten in drei Spalten (wie zuvor)

import os
import sqlite3
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # /workspaces/Bachelorarbeit/Ausgabe_GUI
DB_DIR = os.path.join(os.path.dirname(BASE_DIR), "Datenbank")  # /workspaces/Bachelorarbeit/Datenbank

# ================== CONFIG ==================
HYBRID_DB = os.environ.get("HYBRID_DB", os.path.join(DB_DIR, "hybrid_predictions.db"))
GPT_DB    = os.environ.get("PRED_DB",   os.path.join(DB_DIR, "gpt_predictions.db"))
HIST_DB   = os.environ.get("HIST_DB",   os.path.join(DB_DIR, "historisch_predictions.db"))
PRICES_DB = os.path.join(DB_DIR, "predictions.db")

HORIZON_DEFAULT = 6
PAGE_TITLE = "Aktienprognose Modell"

ACTUAL_LOOKBACK = 5  # t-4..t0 anzeigen
X_AXIS = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
X_AXIS_LABELS = ["Tag -4","Tag -3","Tag -2","Tag -1","Tag 0","Tag +1","Tag +2","Tag +3","Tag +4","Tag +5","Tag +6","Tag +7"]

# ================== DB-HELPERS ==================
def _connect(db_path: str):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con

@st.cache_data(show_spinner=False)
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
def get_symbols_hybrid(db_path: str) -> List[str]:
    if not table_exists(db_path, "forecast_header"):
        return []
    with _connect(db_path) as con:
        rows = con.execute("SELECT DISTINCT symbol FROM forecast_header ORDER BY symbol").fetchall()
    return [r[0] for r in rows]

@st.cache_data(show_spinner=False)
def get_latest_header(db_path: str, symbol: str) -> Optional[dict]:
    if not table_exists(db_path, "forecast_header"):
        return None
    with _connect(db_path) as con:
        row = con.execute(
            """
            SELECT group_id, created_at, symbol, close_t, horizon, model_name, source
            FROM forecast_header
            WHERE symbol=?
            ORDER BY datetime(created_at) DESC
            LIMIT 1
            """,
            (symbol,),
        ).fetchone()
    return dict(row) if row else None

@st.cache_data(show_spinner=False)
def get_header_history(db_path: str, symbol: str, limit: int = 30) -> list[dict]:
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
def get_forecast_lines(db_path: str, group_id: str, model_type: str) -> pd.DataFrame:
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
def get_actual_lastN(db_path: str, symbol: str, n: int = 5) -> pd.DataFrame:
    """
    Holt die letzten n Schlusskurse (aufsteigend sortiert).
    Erwartet: Tabelle actual_prices(date, price) in PRICES_DB.
    """
    if not table_exists(db_path, "actual_prices"):
        return pd.DataFrame(columns=["date","price"])
    with _connect(db_path) as con:
        df = pd.read_sql_query(
            """
            SELECT date, price
            FROM actual_prices
            WHERE symbol=?
            ORDER BY date DESC
            LIMIT ?
            """,
            con,
            params=(symbol, n),
        )
    return df.sort_values("date").reset_index(drop=True)


def _fmt_local(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.replace("Z","+00:00")).astimezone()
        return dt.strftime("%d.%m.%y um %H:%M")
    except Exception:
        return ts

# ================== PLOT ==================
def _build_path_from_lines(df: pd.DataFrame, base_price: float, horizon: int) -> Tuple[list, list, dict]:
    steps = df["step"].astype(int).tolist() if not df.empty else []
    prices = df["price"].astype(float).tolist() if not df.empty else []
    xy = sorted(zip(steps, prices))
    x_model = [0]
    y_model = [base_price]
    for s, p in xy:
        if 1 <= s <= horizon:
            x_model.append(int(s))
            y_model.append(float(p))
    last_val = y_model[-1] if y_model else base_price
    x_model.append(7)
    y_model.append(last_val)
    # Werte für 1/3/7 (7 == step 6)
    points = {1: None, 3: None, 6: None}
    for s in (1, 3, 6):
        if s in steps:
            val = float(df.loc[df["step"] == s, "price"].iloc[0])
        elif s == 6 and 6 in steps:
            val = float(df.loc[df["step"] == 6, "price"].iloc[0])
        else:
            val = last_val
        points[s] = val
    return x_model, y_model, points

def _annotate_pct(ax, x, y, color, text):
    ax.annotate(text, (x, y),
                xytext=(0, 10), textcoords="offset points",
                ha="center", va="bottom", color=color, fontsize=10)

def _pct_from_rel(df: pd.DataFrame, step: int) -> Optional[float]:
    if df.empty: return None
    if step in df["step"].values:
        return float(df.loc[df["step"] == step, "rel_return"].iloc[0]) * 100.0
    if step == 6 and 6 in df["step"].values:
        return float(df.loc[df["step"] == 6, "rel_return"].iloc[0]) * 100.0
    return None

def plot_all(symbol: str,
             created_at: str,
             base_price: float,
             actual_df: pd.DataFrame,
             hybrid_df: pd.DataFrame,
             gpt_df: Optional[pd.DataFrame],
             hist_df: Optional[pd.DataFrame],
             show_gpt: bool,
             show_hist: bool,
             horizon: int):
    # ---- echte Kurse: t-(k-1) .. t0
    xa, ya = [], []
    k = len(actual_df)
    if k >= 1:
        # actual_df ist aufsteigend sortiert (ältestes zuerst)
        ya = actual_df["price"].astype(float).tolist()
        # mappe auf x: -k+1 .. 0  (z.B. k=5 -> -4..0)
        xa = list(range(-k+1, 1))
        ya_t0 = ya[-1]
    else:
        ya_t0 = float(base_price)
        xa = [0]; ya = [ya_t0]


    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()

    # Echte Kurse
    h_actual, = ax.plot(xa, ya, marker="o", linewidth=2, label=f"Echte Kurse (Tag {xa[0]} bis Tag 0)")


    # Hybrid (Basis)
    x_h, y_h, pts_h = _build_path_from_lines(hybrid_df, ya[-1], horizon)
    h_hybrid, = ax.plot(x_h, y_h, linewidth=2, label="Prediction (Hybrid)")

    # GPT Overlay (optional)
    if show_gpt and gpt_df is not None and not gpt_df.empty:
        x_g, y_g, pts_g = _build_path_from_lines(gpt_df, ya[-1], horizon)
        h_gpt, = ax.plot(x_g, y_g, linewidth=2, linestyle="--", label="Prediction (GPT)")
    else:
        x_g = y_g = []; pts_g = {}
        h_gpt = None

    # Historisch Overlay (optional)
    if show_hist and hist_df is not None and not hist_df.empty:
        x_r, y_r, pts_r = _build_path_from_lines(hist_df, ya[-1], horizon)
        h_hist, = ax.plot(x_r, y_r, linewidth=2, linestyle=":", label="Prediction (Historisch)")
    else:
        x_r = y_r = []; pts_r = {}
        h_hist = None

    # Punkte markieren & Prozente annotieren (+1, +3, +7)
    # Hybrid
    color_h = h_hybrid.get_color()
    for step, x_plot in [(1,1), (3,3), (6,7)]:
        if step in hybrid_df["step"].values:
            y_val = float(hybrid_df.loc[hybrid_df["step"]==step, "price"].iloc[0])
        else:
            y_val = y_h[-1]
        ax.scatter([x_plot], [y_val], s=60, marker="o", color=color_h)
        pct = _pct_from_rel(hybrid_df, step)
        if pct is not None:
            _annotate_pct(ax, x_plot, y_val, color_h, f"{pct:+.2f}%")

    # GPT (wenn sichtbar)
    if h_gpt is not None:
        color_g = h_gpt.get_color()
        for step, x_plot in [(1,1), (3,3), (6,7)]:
            if step in gpt_df["step"].values:
                y_val = float(gpt_df.loc[gpt_df["step"]==step, "price"].iloc[0])
            else:
                y_val = y_g[-1] if y_g else ya[-1]
            ax.scatter([x_plot], [y_val], s=60, marker="o", color=color_g)
            pct = _pct_from_rel(gpt_df, step)
            if pct is not None:
                _annotate_pct(ax, x_plot, y_val, color_g, f"{pct:+.2f}%")

    # Historisch (wenn sichtbar)
    if h_hist is not None:
        color_r = h_hist.get_color()
        for step, x_plot in [(1,1), (3,3), (6,7)]:
            if step in hist_df["step"].values:
                y_val = float(hist_df.loc[hist_df["step"]==step, "price"].iloc[0])
            else:
                y_val = y_r[-1] if y_r else ya[-1]
            ax.scatter([x_plot], [y_val], s=60, marker="o", color=color_r)
            pct = _pct_from_rel(hist_df, step)
            if pct is not None:
                _annotate_pct(ax, x_plot, y_val, color_r, f"{pct:+.2f}%")

    # Achsen/Deko
    ax.set_title(f"{symbol} — erstellt {_fmt_local(created_at)}")
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Preis")
    ax.set_xticks(X_AXIS)
    ax.set_xticklabels(X_AXIS_LABELS)
    ax.grid(True, alpha=0.25)

    # ---------- Legende unten in 3 Spalten ----------
    col1_handles = [
        Line2D([0],[0], color=h_actual.get_color(), marker="o", linewidth=2),
        Line2D([0],[0], color=h_hybrid.get_color(), linewidth=2),
    ]
    col1_labels = [
        "Echte Kurse (Tag -1 bis Tag 0)",
        "Prediction (Hybrid)",
    ]
    if h_gpt is not None:
        col1_handles.append(Line2D([0],[0], color=h_gpt.get_color(), linestyle="--", linewidth=2))
        col1_labels.append("Prediction (GPT)")
    if h_hist is not None:
        col1_handles.append(Line2D([0],[0], color=h_hist.get_color(), linestyle=":", linewidth=2))
        col1_labels.append("Prediction (Historisch)")

    # Spalte 2: Punkt-Erklärungen (Hybrid)
    col2_handles = [
        Line2D([0],[0], marker="o", linestyle="None", markersize=8, color=h_hybrid.get_color()),
    ]
    col2_labels = ["Punkte +1 / +3 / +7 (Hybrid)"]

    # Spalte 3: Punkt-Erklärungen (Overlays, wenn aktiv)
    col3_handles, col3_labels = [], []
    if h_gpt is not None:
        col3_handles.append(Line2D([0],[0], marker="o", linestyle="None", markersize=8, color=h_gpt.get_color()))
        col3_labels.append("Punkte (GPT)")
    if h_hist is not None:
        col3_handles.append(Line2D([0],[0], marker="o", linestyle="None", markersize=8, color=h_hist.get_color()))
        col3_labels.append("Punkte (Historisch)")

    leg1 = fig.legend(col1_handles, col1_labels, loc="lower left",
                      bbox_to_anchor=(0.02, -0.04), frameon=False, ncol=1, borderaxespad=0.0)
    fig.add_artist(leg1)

    leg2 = fig.legend(col2_handles, col2_labels, loc="lower center",
                      bbox_to_anchor=(0.50, -0.04), frameon=False, ncol=1, borderaxespad=0.0)
    fig.add_artist(leg2)

    if col3_handles:
        leg3 = fig.legend(col3_handles, col3_labels, loc="lower right",
                          bbox_to_anchor=(0.98, -0.04), frameon=False, ncol=1, borderaxespad=0.0)
        fig.add_artist(leg3)

    plt.subplots_adjust(bottom=0.30)
    st.pyplot(fig)

# ================== UI ==================
st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(PAGE_TITLE)

col1, col2, col3, col4 = st.columns([3,2,2,2])

symbols = get_symbols_hybrid(HYBRID_DB)
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

if st.button("Daten aktualisieren", use_container_width=True):
    # Cache invalidieren
    get_symbols_hybrid.clear()
    get_latest_header.clear()
    get_header_history.clear()
    get_forecast_lines.clear()
    get_actual_lastN.clear()
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

# Hybrid-Basis — Auswahl neueste oder manuell
hist = get_header_history(HYBRID_DB, symbol, limit=30)
if not hist:
    st.info("Für dieses Symbol existiert (noch) keine Hybrid-Vorhersage.")
    st.stop()

cA, cB = st.columns([2, 3])
with cA:
    use_latest = st.toggle("Immer neueste Hybrid-Vorhersage verwenden", value=True,
                           help="Wenn aus: manuelle Auswahl unten.")
with cB:
    active_hdr = hist[0]
    if not use_latest:
        options = [
            f"{_fmt_local(h['created_at'])} — {h.get('model_name') or 'Hybrid'} — group={h['group_id'][:8]}"
            for h in hist
        ]
        sel = st.selectbox("Hybrid-Vorhersage-Version wählen", options, index=0)
        active_hdr = hist[options.index(sel)]

st.success(f"Hybrid-Modell wurde am {_fmt_local(active_hdr['created_at'])} aktualisiert.", icon="🕒")

group_id   = active_hdr["group_id"]
created_at = active_hdr["created_at"]
base_price = float(active_hdr["close_t"]) if active_hdr["close_t"] is not None else np.nan
horizon    = int(active_hdr.get("horizon") or HORIZON_DEFAULT)
model_name = active_hdr.get("model_name") or "Hybrid"
st.caption(f"Aktives Hybrid: {model_name} — group_id: {group_id} — close_t: {base_price}")

# Daten laden: Hybrid immer, Overlays optional (jeweils "neuester" Header dieses Symbols aus der jeweiligen DB)
hybrid_df = get_forecast_lines(HYBRID_DB, group_id, "hybrid")
actual_df = get_actual_lastN(PRICES_DB, symbol, n=ACTUAL_LOOKBACK)


gpt_df = None
if show_gpt:
    gpt_hdr = get_latest_header(GPT_DB, symbol)
    if gpt_hdr:
        gpt_df = get_forecast_lines(GPT_DB, gpt_hdr["group_id"], "gpt")

hist_df = None
if show_hist:
    rh_hdr = get_latest_header(HIST_DB, symbol)
    if rh_hdr:
        hist_df = get_forecast_lines(HIST_DB, rh_hdr["group_id"], "historisch")

if actual_df.empty:
    st.info("Hinweis: 'actual_prices' nicht vorhanden oder leer. Es wird nur Tag 0 (close_t) als Startpunkt genutzt.")
else:
    st.caption(f"Echte Kurse geladen: {len(actual_df)} Punkte (zeigt bis Tag {-(len(actual_df)-1)}).")


# Plot
plot_all(symbol, created_at, base_price, actual_df, hybrid_df, gpt_df, hist_df, show_gpt, show_hist, horizon)

st.markdown("---")

# Details
with st.expander("Details / Tabellenansicht"):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.subheader("Actual (Tag -4 & Tag 0)")
        st.dataframe(actual_df if not actual_df.empty else pd.DataFrame())
    with c2:
        st.subheader("Hybrid (forecast_lines)")
        st.dataframe(hybrid_df)
    with c3:
        st.subheader("GPT (forecast_lines)")
        st.dataframe(gpt_df if gpt_df is not None else pd.DataFrame())
    with c4:
        st.subheader("Historisch (forecast_lines)")
        st.dataframe(hist_df if hist_df is not None else pd.DataFrame())

# Historie (Hybrid)
with st.expander("Hybrid: ältere Vorhersagen (Historie)"):
    df_hist = pd.DataFrame([{
        "Zeit": _fmt_local(h["created_at"]),
        "Modell": h.get("model_name") or "Hybrid",
        "group_id": h["group_id"],
        "Quelle": h.get("source") or "",
        "close_t": h.get("close_t"),
        "Horizont": h.get("horizon")
    } for h in hist])
    st.dataframe(df_hist, use_container_width=True)

st.markdown("\n")
st.caption(f"DBs — Hybrid: {HYBRID_DB} | GPT: {GPT_DB} | Historisch: {HIST_DB} | Preise: {PRICES_DB}")
