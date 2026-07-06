import os
import time
import json
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# ============================================================
# PINEHURST MAIN KALMAN — STREAMLIT FULL 150 VERIFY v17
# ============================================================
# This worker mirrors the uploaded Streamlit Main Kalman visible-tab production path:
#   • live 15m Main Kalman uses 30 calendar days of data
#   • yfinance auto_adjust=False
#   • same CT timezone conversion and dropna cleaning
#   • same adaptive Kalman + institutional trend rail
#   • same per-ticker optimized params
#   • same risk firewall when enabled
#   • same non-repaint signal lock
#   • same BacktestEngine trade accounting
#   • same institutional append-only trade ledger
#
# IMPORTANT:
# For existing Streamlit positions/history to match, upload the read-only
# Streamlit state bundle as a Render Secret File:
#   /etc/secrets/streamlit_kalman_render_bundle.json

BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
CHAT_ID = os.environ.get("CHAT_ID", "").strip()

INTERVAL = os.environ.get("KALMAN_INTERVAL", "15m").strip()
LOOKBACK_DAYS = int(os.environ.get("KALMAN_LOOKBACK_DAYS", "30"))
SCAN_EVERY_MINUTES = int(os.environ.get("SCAN_EVERY_MINUTES", "15"))
SCAN_DELAY_SECONDS = int(os.environ.get("SCAN_DELAY_SECONDS", "30"))
SLEEP_BETWEEN_TICKERS_SEC = float(os.environ.get("SLEEP_BETWEEN_TICKERS_SEC", "0.35"))

# Streamlit visible-tab defaults.
PARAMS = {
    "fast_gain": float(os.environ.get("KALMAN_FAST_GAIN", "0.34")),
    "slow_gain": float(os.environ.get("KALMAN_SLOW_GAIN", "0.055")),
    "polish_span": int(os.environ.get("KALMAN_POLISH_SPAN", "3")),
    "atr_window": int(os.environ.get("KALMAN_ATR_WINDOW", "14")),
    "rail_mult": float(os.environ.get("KALMAN_RAIL_MULT", "1.35")),
    "buffer_pct": float(os.environ.get("KALMAN_BUFFER_PCT", "0.0125")),
    "confirm_bars": int(os.environ.get("KALMAN_CONFIRM_BARS", "3")),
    "min_hold_bars": int(os.environ.get("KALMAN_MIN_HOLD_BARS", "5")),
    "cooldown_bars": int(os.environ.get("KALMAN_COOLDOWN_BARS", "3")),
    "slope_confirm": os.environ.get("KALMAN_SLOPE_CONFIRM", "true").lower() == "true",
    "atr_safety": os.environ.get("KALMAN_ATR_SAFETY", "true").lower() == "true",
}

USE_RISK_FIREWALL = os.environ.get("KALMAN_USE_RISK_FIREWALL", "false").lower() == "true"
TRADE_STOP_PCT = float(os.environ.get("KALMAN_TRADE_STOP_PCT", "16.0"))
TRAIL_STOP_PCT = float(os.environ.get("KALMAN_TRAIL_STOP_PCT", "22.0"))
EQUITY_DD_STOP_PCT = float(os.environ.get("KALMAN_EQUITY_DD_STOP_PCT", "28.0"))
FIREWALL_COOLDOWN = int(os.environ.get("KALMAN_FIREWALL_COOLDOWN", "8"))
USE_INSTITUTIONAL_LEDGER = os.environ.get("KALMAN_INSTITUTIONAL_LIVE_LEDGER", "true").lower() == "true"

BUNDLE_FILE = os.environ.get("STREAMLIT_KALMAN_BUNDLE_FILE", "/etc/secrets/streamlit_kalman_render_bundle.json")
PARAMS_SECRET_FILE = os.environ.get("KALMAN_PARAMS_SECRET_FILE", "/etc/secrets/kalman_params.json")

STATE_FILE = os.environ.get("STATE_FILE", "kalman_render_state_v17.json")
SIGNAL_LOCK_FILE = os.environ.get("SIGNAL_LOCK_FILE", "kalman_render_signal_lock_v17.json")
INSTITUTIONAL_LEDGER_FILE = os.environ.get("INSTITUTIONAL_LEDGER_FILE", "kalman_render_institutional_ledger_v17.json")
UPDATE_OFFSET_FILE = os.environ.get("UPDATE_OFFSET_FILE", "kalman_render_update_offset_v17.json")

WATCHLIST = [
    "AAPL", "ACN", "ADI", "AEVA", "AFRM", "AI", "ALAB", "AMAT", "AMD", "AMLX", "AMPX", "AMR",
    "AMZN", "APEI", "APLD", "APP", "APPF", "APPS", "ARQQ", "ASTS", "AVGO", "AXON", "AXP", "AZZ",
    "BABA", "BBAI", "BE", "BR", "BROS", "BTBT", "BULL", "CCL", "CDE", "CEG", "CELC", "CGNX",
    "CIFR", "CLSK", "CMG", "COIN", "CORT", "CPB", "CRCL", "CRM", "CRML", "CRWD", "CRWV", "CSGP",
    "DAL", "DELL", "EFX", "ELF", "ETN", "EXK", "FSLR", "FVRR", "GLXY", "GOOGL", "GTES", "HCC",
    "HIMS", "HOOD", "HPE", "HTZ", "HUT", "IHS", "INGR", "INTC", "INTU", "IONQ", "IREN", "IRON",
    "JKHY", "KKR", "LULU", "LUNR", "MARA", "META", "MOS", "MRK", "MRVL", "MSFT", "MSTR", "MTZ",
    "MU", "NBIS", "NEE", "NEGG", "NFLX", "NIO", "NNE", "NVAX", "NVDA", "NVTS", "ONDS", "OPEN",
    "ORCL", "OUST", "PGY", "PINS", "PLTR", "PNRG", "PRCH", "QBTS", "QCOM", "QS", "QUBT", "RBLX",
    "RDDT", "RDW", "RELX", "RELY", "RGTI", "RIOT", "RIVN", "RKLB", "ROK", "S", "SAP", "SBUX",
    "SCHW", "SEDG", "SG", "SHAK", "SHOP", "SMR", "SNDK", "SNOW", "SOFI", "SOUN", "SPCX", "SYM",
    "T", "TOST", "TPR", "TRI", "TSLA", "UA", "UAL", "UBER", "UFPT", "ULTA", "UNH", "UPST",
    "V", "VST", "WING", "WMT", "WULF", "XYZ"
]

def _load_json_file(path, default=None):
    default = {} if default is None else default
    try:
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
    except Exception as e:
        print(f"JSON load error {path}: {e}")
    return default

def _save_json_file(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"JSON save error {path}: {e}")
        return False

def load_streamlit_bundle():
    data = _load_json_file(BUNDLE_FILE, {})
    if not isinstance(data, dict):
        data = {}
    params = data.get("per_ticker_params", data.get("params", {}))
    signal_lock = data.get("signal_lock", {})
    institutional_ledger = data.get("institutional_ledger", {})
    if not isinstance(params, dict):
        params = {}
    if not isinstance(signal_lock, dict):
        signal_lock = {}
    if not isinstance(institutional_ledger, dict):
        institutional_ledger = {}
    open_tickers = data.get("streamlit_open_tickers", [])
    if not isinstance(open_tickers, list):
        open_tickers = []
    sync_summary = data.get("sync_summary", {})
    if not isinstance(sync_summary, dict):
        sync_summary = {}
    data_path = data.get("data_path", {})
    if not isinstance(data_path, dict):
        data_path = {}
    return {
        "per_ticker_params": {str(k).upper(): v for k, v in params.items() if isinstance(v, dict)},
        "signal_lock": signal_lock,
        "institutional_ledger": institutional_ledger,
        "streamlit_open_tickers": sorted({str(x).upper() for x in open_tickers if str(x).strip()}),
        "sync_summary": sync_summary,
        "data_path": data_path,
        "bundle_version": data.get("bundle_version", 0),
        "exported_ct": data.get("exported_ct", ""),
    }

STREAMLIT_BUNDLE = load_streamlit_bundle()

def _load_per_ticker_params():
    # Priority 1: full Streamlit bundle.
    from_bundle = STREAMLIT_BUNDLE.get("per_ticker_params", {})
    if from_bundle:
        return dict(from_bundle)

    # Priority 2: params-only Render secret file.
    data = _load_json_file(PARAMS_SECRET_FILE, {})
    if isinstance(data, dict) and data:
        return {str(k).upper(): v for k, v in data.items() if isinstance(v, dict)}

    # Priority 3: old env JSON.
    raw = os.environ.get("PER_TICKER_PARAMS_JSON", "").strip()
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return {str(k).upper(): v for k, v in data.items() if isinstance(v, dict)}
        except Exception as e:
            print(f"PER_TICKER_PARAMS_JSON parse error: {e}")
    return {}

PER_TICKER_PARAMS = _load_per_ticker_params()

# Full 150-ticker coverage verification. The worker still has a 150-name watchlist,
# but now it explicitly reports whether every watchlist ticker has a parameter record.
PARAM_TICKERS = set(PER_TICKER_PARAMS.keys())
MISSING_PARAM_TICKERS = sorted(set(WATCHLIST) - PARAM_TICKERS)
EXTRA_PARAM_TICKERS = sorted(PARAM_TICKERS - set(WATCHLIST))
PARAM_COVERAGE_OK = (len(MISSING_PARAM_TICKERS) == 0 and len(WATCHLIST) == 150)

def param_source_for_ticker(ticker):
    rec = PER_TICKER_PARAMS.get(str(ticker).upper(), {})
    if isinstance(rec, dict):
        return str(rec.get("_sync_source", rec.get("source", "BUNDLE_OR_SAVED_PARAMS")))
    return "DEFAULTS"

def params_for_ticker(ticker):
    p = dict(PARAMS)
    override = PER_TICKER_PARAMS.get(str(ticker).upper(), {})
    for k, v in override.items():
        if k not in p:
            continue
        try:
            if isinstance(p[k], bool):
                p[k] = bool(v)
            elif isinstance(p[k], int):
                p[k] = int(v)
            else:
                p[k] = float(v)
        except Exception:
            pass
    return p

positions = {ticker: "UNKNOWN" for ticker in WATCHLIST}
last_alert_bar = {}
last_checked = {}
last_error = {}
scan_started_at = None
scan_finished_at = None
scan_in_progress = False
full_scans_completed = 0
rescan_requested = False
last_update_id = None
next_scheduled_scan_ct = None

def ct_now():
    return datetime.now(ZoneInfo("America/Chicago"))

def fmt_ct_now():
    return ct_now().strftime("%Y-%m-%d %I:%M %p CT")

def fmt_ct_dt(dt):
    if dt is None:
        return "N/A"
    return dt.strftime("%Y-%m-%d %I:%M:%S %p CT")

def send_telegram(text):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram disabled: BOT_TOKEN or CHAT_ID missing")
        return False
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        txt = str(text or "")
        chunks = []
        while len(txt) > 3500:
            cut = txt.rfind("\n", 0, 3500)
            if cut < 1000:
                cut = 3500
            chunks.append(txt[:cut])
            txt = txt[cut:].lstrip()
        chunks.append(txt)
        ok = True
        for ch in chunks:
            r = requests.post(url, json={"chat_id": CHAT_ID, "text": ch, "parse_mode": "HTML"}, timeout=8)
            if not r.ok:
                r = requests.post(url, json={"chat_id": CHAT_ID, "text": ch.replace("<b>", "").replace("</b>", "")}, timeout=8)
            ok = ok and r.ok
        return ok
    except Exception as e:
        print(f"Telegram send error: {e}")
        return False

def load_state():
    global positions, last_alert_bar, last_checked, last_error, full_scans_completed
    data = _load_json_file(STATE_FILE, {})
    if not isinstance(data, dict):
        return
    for t in WATCHLIST:
        if data.get("positions", {}).get(t) in ("LONG", "CASH", "UNKNOWN"):
            positions[t] = data["positions"][t]
    if isinstance(data.get("last_alert_bar"), dict):
        last_alert_bar.update(data["last_alert_bar"])
    if isinstance(data.get("last_checked"), dict):
        last_checked.update(data["last_checked"])
    if isinstance(data.get("last_error"), dict):
        last_error.update(data["last_error"])
    try:
        full_scans_completed = int(data.get("full_scans_completed", 0))
    except Exception:
        full_scans_completed = 0

def save_state():
    _save_json_file(STATE_FILE, {
        "positions": positions,
        "last_alert_bar": last_alert_bar,
        "last_checked": last_checked,
        "last_error": last_error,
        "full_scans_completed": full_scans_completed,
        "scan_started_at": scan_started_at,
        "scan_finished_at": scan_finished_at,
    })

def seed_streamlit_state_once():
    """Seed Render's local lock/ledger from the read-only Streamlit bundle once."""
    if not os.path.exists(SIGNAL_LOCK_FILE):
        seed = STREAMLIT_BUNDLE.get("signal_lock", {})
        if seed:
            _save_json_file(SIGNAL_LOCK_FILE, seed)
            print(f"Seeded Render signal lock from Streamlit bundle: {len(seed)} keys")
    if not os.path.exists(INSTITUTIONAL_LEDGER_FILE):
        seed = STREAMLIT_BUNDLE.get("institutional_ledger", {})
        if seed:
            _save_json_file(INSTITUTIONAL_LEDGER_FILE, seed)
            print(f"Seeded Render institutional ledger from Streamlit bundle: {len(seed)} tickers")

def seed_positions_from_streamlit_bundle_once():
    """Baseline Render LONG/CASH from the exact Streamlit bundle on a fresh v16 state file."""
    try:
        if os.path.exists(STATE_FILE):
            return
        opens = set(STREAMLIT_BUNDLE.get("streamlit_open_tickers", []))
        if not opens:
            return
        for t in WATCHLIST:
            positions[t] = "LONG" if t in opens else "CASH"
        save_state()
        print(f"Seeded Render positions from Streamlit bundle: {len(opens)} LONG / {len(WATCHLIST)-len(opens)} CASH")
    except Exception as e:
        print(f"Position seed error: {e}")


def load_signal_lock():
    return _load_json_file(SIGNAL_LOCK_FILE, {})

def save_signal_lock(data):
    _save_json_file(SIGNAL_LOCK_FILE, data)

def apply_signal_lock(ticker, sig, interval=None):
    try:
        if sig is None or len(sig) == 0:
            return sig
        interval = interval or INTERVAL
        tkey = f"{str(ticker).upper()}|{str(interval)}"
        store = load_signal_lock()
        locked = dict(store.get(tkey, {})) if isinstance(store.get(tkey), dict) else {}
        out = sig.copy()
        changed = False
        for dt in out.index:
            key = pd.Timestamp(dt).strftime("%Y-%m-%d %H:%M:%S")
            if key in locked:
                out.loc[dt] = float(locked[key])
            else:
                locked[key] = float(out.loc[dt])
                changed = True
        if changed:
            if len(locked) > 6000:
                for k in sorted(locked.keys())[:len(locked) - 6000]:
                    locked.pop(k, None)
            store[tkey] = locked
            save_signal_lock(store)
        return out
    except Exception as e:
        print(f"Signal lock error {ticker}: {e}")
        return sig

def load_institutional_ledger():
    return _load_json_file(INSTITUTIONAL_LEDGER_FILE, {})

def save_institutional_ledger(data):
    return _save_json_file(INSTITUTIONAL_LEDGER_FILE, data)

# ============================================================
# EXACT STREAMLIT DATA PATH
# ============================================================
def _flatten_yfinance_columns(df, ticker=""):
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.copy()
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        except Exception:
            pass
    return df

def fetch_streamlit_visible_tab_prices(ticker):
    """
    Mirrors the uploaded Streamlit visible 15m tab:
      now_rounded - 30 calendar days -> now_rounded
      auto_adjust=False
      prepost=False
      CT timezone normalization
      add Returns/Log_Returns then dropna()
    """
    try:
        now_rounded = ct_now().replace(second=0, microsecond=0, tzinfo=None)
        start = now_rounded - timedelta(days=LOOKBACK_DAYS)
        end = now_rounded

        df = yf.download(
            str(ticker).upper(),
            start=start,
            end=end,
            interval=INTERVAL,
            progress=False,
            auto_adjust=False,
            prepost=False,
            threads=False,
        )
        if df is None or df.empty:
            return None

        df = _flatten_yfinance_columns(df, ticker)

        try:
            idx = pd.DatetimeIndex(df.index)
            if idx.tz is not None:
                df.index = idx.tz_convert("America/Chicago").tz_localize(None)
            else:
                df.index = idx.tz_localize("America/New_York").tz_convert("America/Chicago").tz_localize(None)
        except Exception:
            if getattr(df.index, "tz", None) is not None:
                df.index = df.index.tz_localize(None)

        if "Close" not in df.columns and "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        if "Close" not in df.columns:
            return None

        df["Returns"] = pd.to_numeric(df["Close"], errors="coerce").pct_change()
        df["Log_Returns"] = np.log(pd.to_numeric(df["Close"], errors="coerce") / pd.to_numeric(df["Close"], errors="coerce").shift(1))
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        px = pd.Series(df["Close"]).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        return px if len(px) >= 80 else None
    except Exception as e:
        print(f"{ticker} data error: {e}")
        return None

# ============================================================
# EXACT STREAMLIT KALMAN / RAIL
# ============================================================
def institutional_adaptive_kalman_trend(prices, fast_gain=0.34, slow_gain=0.055, vol_window=20, polish_span=3):
    try:
        px = pd.Series(prices).astype(float).replace([np.inf, -np.inf], np.nan).ffill().bfill()
        if px.empty:
            return np.array([])
        ret = px.pct_change().abs()
        vol = ret.rolling(int(vol_window), min_periods=max(3, int(vol_window)//3)).median().replace(0, np.nan)
        shock = (ret / (vol + 1e-12)).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 3) / 3.0
        fast_gain = float(fast_gain)
        slow_gain = float(slow_gain)
        gains = (slow_gain + (fast_gain - slow_gain) * shock).clip(min(slow_gain, fast_gain), max(slow_gain, fast_gain))
        out = np.zeros(len(px), dtype=float)
        out[0] = float(px.iloc[0])
        for i in range(1, len(px)):
            out[i] = out[i-1] + float(gains.iloc[i]) * (float(px.iloc[i]) - out[i-1])
        if int(polish_span) > 1:
            out = pd.Series(out, index=px.index).ewm(span=int(polish_span), adjust=False).mean().values
        return out
    except Exception:
        try:
            return pd.Series(prices).ewm(span=8, adjust=False).mean().values
        except Exception:
            return np.array(prices, dtype=float)

def institutional_trend_rail(prices, fast_gain=0.34, slow_gain=0.055, polish_span=3, atr_window=14, atr_mult=1.35):
    try:
        px = pd.Series(prices).astype(float).replace([np.inf, -np.inf], np.nan).ffill().bfill()
        if px.empty:
            return np.array([]), np.array([]), pd.Series(dtype=float)

        center = pd.Series(
            institutional_adaptive_kalman_trend(
                px.values,
                fast_gain=float(fast_gain),
                slow_gain=float(slow_gain),
                vol_window=20,
                polish_span=int(polish_span)
            ),
            index=px.index
        )

        atr = px.diff().abs().ewm(span=int(atr_window), adjust=False).mean()
        atr = atr.replace(0, np.nan).ffill().bfill()
        if atr.isna().all():
            atr = pd.Series(px.std() * 0.02 if len(px) > 2 else 1.0, index=px.index)
        atr = atr.fillna(float(px.iloc[-1]) * 0.015)

        slope = center.diff().ewm(span=5, adjust=False).mean().fillna(0)
        long_state = pd.Series(False, index=px.index)
        rail = pd.Series(index=px.index, dtype=float)

        state = True
        rail.iloc[0] = float(center.iloc[0] - float(atr.iloc[0]) * float(atr_mult))
        long_state.iloc[0] = state

        for i in range(1, len(px)):
            p = float(px.iloc[i])
            c = float(center.iloc[i])
            a = float(atr.iloc[i]) * float(atr_mult)
            sl = float(slope.iloc[i])

            if state:
                candidate = c - a
                if sl >= 0:
                    candidate = max(candidate, float(rail.iloc[i-1]) if np.isfinite(rail.iloc[i-1]) else candidate)
                if p < (float(rail.iloc[i-1]) if np.isfinite(rail.iloc[i-1]) else candidate):
                    state = False
                    candidate = c + a
            else:
                candidate = c + a
                if sl <= 0:
                    candidate = min(candidate, float(rail.iloc[i-1]) if np.isfinite(rail.iloc[i-1]) else candidate)
                if p > (float(rail.iloc[i-1]) if np.isfinite(rail.iloc[i-1]) else candidate):
                    state = True
                    candidate = c - a

            rail.iloc[i] = candidate
            long_state.iloc[i] = state

        rail = rail.ewm(span=2, adjust=False).mean()
        return rail.values, center.values, long_state
    except Exception:
        base = institutional_adaptive_kalman_trend(prices, fast_gain=fast_gain, slow_gain=slow_gain, polish_span=polish_span)
        return base, base, pd.Series([True] * len(base))

def apply_kalman_risk_firewall(prices, signal, trend, max_trade_loss_pct=18.0, trail_stop_pct=22.0, equity_dd_stop_pct=30.0, cooldown_bars=8):
    try:
        px = pd.Series(prices).astype(float).replace([np.inf, -np.inf], np.nan).ffill().bfill()
        sig = pd.Series(signal).reindex(px.index).ffill().fillna(0.0).astype(float).clip(0, 1)
        tr = pd.Series(trend).reindex(px.index).ffill().bfill().astype(float)
        out = pd.Series(0.0, index=px.index)
        in_pos = False
        entry = 0.0
        peak_price = 0.0
        eq = 1.0
        peak_eq = 1.0
        cooldown = 0
        prev_price = None
        max_trade_loss = float(max_trade_loss_pct) / 100.0
        trail_stop = float(trail_stop_pct) / 100.0
        equity_dd_stop = float(equity_dd_stop_pct) / 100.0
        cooldown_bars = int(cooldown_bars)

        for dt in px.index:
            p = float(px.loc[dt])
            desired = float(sig.loc[dt])
            if prev_price is not None and in_pos:
                eq *= (p / float(prev_price))
                peak_eq = max(peak_eq, eq)
            if cooldown > 0:
                cooldown -= 1
            forced_exit = False
            if in_pos:
                peak_price = max(peak_price, p)
                if max_trade_loss > 0 and p <= entry * (1.0 - max_trade_loss):
                    forced_exit = True
                if (not forced_exit) and trail_stop > 0 and p <= peak_price * (1.0 - trail_stop):
                    forced_exit = True
                if (not forced_exit) and equity_dd_stop > 0 and eq <= peak_eq * (1.0 - equity_dd_stop):
                    forced_exit = True
                if (not forced_exit) and p < float(tr.loc[dt]) * 0.985:
                    forced_exit = True
                if forced_exit:
                    in_pos = False
                    cooldown = cooldown_bars
                    out.loc[dt] = 0.0
                elif desired >= 0.5:
                    out.loc[dt] = 1.0
                else:
                    in_pos = False
                    out.loc[dt] = 0.0
            else:
                if cooldown <= 0 and desired >= 0.5:
                    in_pos = True
                    entry = p
                    peak_price = p
                    out.loc[dt] = 1.0
                else:
                    out.loc[dt] = 0.0
            prev_price = p
        return out.ffill().fillna(0).clip(0, 1)
    except Exception:
        return pd.Series(signal).ffill().fillna(0).clip(0, 1)

def build_main_kalman_signal(px, ticker):
    pms = params_for_ticker(ticker)
    rail, center, long_state = institutional_trend_rail(
        px,
        fast_gain=float(pms["fast_gain"]),
        slow_gain=float(pms["slow_gain"]),
        polish_span=int(pms["polish_span"]),
        atr_window=14,
        atr_mult=float(pms["rail_mult"]),
    )
    bt_trend = pd.Series(rail, index=px.index).ffill().bfill()

    buffer_pct = float(pms["buffer_pct"])
    confirm_bars = int(pms["confirm_bars"])
    min_hold_bars = int(pms["min_hold_bars"])
    cooldown_bars = int(pms["cooldown_bars"])

    trend_slope = bt_trend.diff().ewm(span=5, adjust=False).mean().fillna(0)
    close_above = px > bt_trend * (1.0 + buffer_pct)
    close_below = px < bt_trend * (1.0 - buffer_pct)

    if bool(pms["slope_confirm"]):
        entry_cond = close_above & (trend_slope >= 0)
        exit_cond = close_below & (trend_slope <= 0)
    else:
        entry_cond = close_above
        exit_cond = close_below

    if bool(pms["atr_safety"]):
        atr_proxy = px.diff().abs().ewm(span=14, adjust=False).mean().replace(0, np.nan).ffill().bfill()
        safety_exit = px < (bt_trend - 1.25 * atr_proxy)
        exit_cond = exit_cond | safety_exit.fillna(False)

    entry_ready = entry_cond.rolling(confirm_bars, min_periods=confirm_bars).sum().eq(confirm_bars).fillna(False)
    exit_ready = exit_cond.rolling(confirm_bars, min_periods=confirm_bars).sum().eq(confirm_bars).fillna(False)

    sig = pd.Series(0.0, index=px.index)
    in_pos = False
    bars_held = 0
    cooldown_left = 0

    for dt in px.index:
        if cooldown_left > 0:
            cooldown_left -= 1
        if not in_pos:
            if cooldown_left <= 0 and bool(entry_ready.loc[dt]):
                in_pos = True
                bars_held = 0
                sig.loc[dt] = 1.0
            else:
                sig.loc[dt] = 0.0
        else:
            bars_held += 1
            if bars_held >= min_hold_bars and bool(exit_ready.loc[dt]):
                in_pos = False
                cooldown_left = cooldown_bars
                bars_held = 0
                sig.loc[dt] = 0.0
            else:
                sig.loc[dt] = 1.0

    sig = sig.ffill().fillna(0).clip(0, 1)

    if USE_RISK_FIREWALL:
        sig = apply_kalman_risk_firewall(
            px, sig, bt_trend,
            max_trade_loss_pct=TRADE_STOP_PCT,
            trail_stop_pct=TRAIL_STOP_PCT,
            equity_dd_stop_pct=EQUITY_DD_STOP_PCT,
            cooldown_bars=FIREWALL_COOLDOWN,
        )

    sig = apply_signal_lock(ticker, sig, interval=INTERVAL)
    return sig, bt_trend

# ============================================================
# EXACT STREAMLIT BACKTEST TRADE ACCOUNTING
# ============================================================
def run_strategy(prices, signals, initial_capital=10000.0):
    common_idx = prices.index.intersection(signals.index)
    prices = pd.Series(prices.loc[common_idx]).replace([np.inf, -np.inf], np.nan).dropna()
    signals = pd.Series(signals).reindex(prices.index).ffill().fillna(0.0).astype(float).clip(0.0, 1.0)
    if len(prices) == 0:
        return pd.DataFrame()

    trades = []
    position = 0
    entry_price = 0.0
    entry_date = None

    for dt in prices.index:
        price = float(prices.loc[dt])
        desired_signal = float(signals.loc[dt])
        if position == 0 and desired_signal > 0:
            position = 1
            entry_price = price
            entry_date = dt
        elif position == 1 and desired_signal == 0:
            trades.append({
                "Side": "Long",
                "Entry Date": entry_date,
                "Exit Date": dt,
                "Buy Price": float(entry_price),
                "Sell Price": float(price),
                "PnL (%)": ((price - entry_price) / entry_price * 100.0) if entry_price else 0.0,
                "Status": "Closed",
            })
            position = 0
            entry_price = 0.0
            entry_date = None

    if position == 1:
        current_price = float(prices.iloc[-1])
        trades.append({
            "Side": "Long",
            "Entry Date": entry_date,
            "Exit Date": None,
            "Buy Price": float(entry_price),
            "Sell Price": current_price,
            "PnL (%)": ((current_price - entry_price) / entry_price * 100.0) if entry_price else 0.0,
            "Status": "Open",
        })
    return pd.DataFrame(trades)

def _safe_float_or_none(v):
    try:
        if v is None:
            return None
        vv = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
        if pd.isna(vv):
            return None
        return float(vv)
    except Exception:
        return None

def _safe_str_dt(v):
    try:
        if v is None or (pd.isna(v) if np.isscalar(v) else False):
            return ""
    except Exception:
        pass
    try:
        s = str(v).strip()
        if not s or s.lower() in ("nan", "none", "nat"):
            return ""
        if s.lower() == "open":
            return "Open"
        return pd.Timestamp(v).strftime("%Y-%m-%d %H:%M:%S CT")
    except Exception:
        return str(v).strip()

def settings_for_institutional_ledger(ticker):
    p = params_for_ticker(ticker)
    source = param_source_for_ticker(ticker) if str(ticker).upper() in PER_TICKER_PARAMS else "VISIBLE_SLIDER_DEFAULTS"
    model_version = (
        f"{str(ticker).upper()}|Institutional Trend Rail|"
        f"buf{p['buffer_pct']*100:.2f}|conf{int(p['confirm_bars'])}|"
        f"hold{int(p['min_hold_bars'])}|cool{int(p['cooldown_bars'])}|"
        f"fg{p['fast_gain']:.3f}|sg{p['slow_gain']:.3f}|rail{p['rail_mult']:.2f}|"
        f"slope{int(bool(p['slope_confirm']))}|atr{int(bool(p['atr_safety']))}|rf{int(bool(USE_RISK_FIREWALL))}"
    )
    return {
        "ticker": str(ticker).upper(),
        **p,
        "risk_firewall": USE_RISK_FIREWALL,
        "trend_name": "Institutional Trend Rail",
        "source": source,
        "model_version": model_version,
        "saved_ct": fmt_ct_now(),
    }

def trade_row_to_institutional_record(ticker, tr, settings, latest_price=None, baseline=True):
    ticker = str(ticker).upper()
    status_txt = str(tr.get("Status", "")).strip()
    is_open_row = status_txt.upper() == "OPEN"
    entry_dt = tr.get("Entry Date", tr.get("Entry CT", tr.get("Entry", "")))
    exit_dt = tr.get("Exit Date", tr.get("Exit CT", tr.get("Exit", "")))
    entry_ct = _safe_str_dt(entry_dt)
    exit_ct = "Open" if is_open_row else _safe_str_dt(exit_dt)
    entry_px = _safe_float_or_none(tr.get("Buy Price", tr.get("Entry Price", None)))
    exit_px = _safe_float_or_none(tr.get("Sell Price", tr.get("Exit/Current Price", tr.get("Exit Price", None))))
    if is_open_row and latest_price is not None:
        exit_px = float(latest_price)
    pnl = _safe_float_or_none(tr.get("PnL (%)", tr.get("Return (%)", None)))
    if pnl is None and entry_px and exit_px:
        pnl = (float(exit_px) / float(entry_px) - 1.0) * 100.0
    return {
        "Ticker": ticker,
        "Side": "Long",
        "Entry CT": entry_ct,
        "Exit CT": exit_ct,
        "Entry Price": round(entry_px, 4) if entry_px is not None else None,
        "Exit/Current Price": round(exit_px, 4) if exit_px is not None else None,
        "PnL (%)": round(pnl, 4) if pnl is not None else None,
        "Status": "Open" if is_open_row else "Closed",
        "Locked Buffer %": round(float(settings.get("buffer_pct", 0.0)) * 100.0, 4),
        "Locked Confirm": int(settings.get("confirm_bars", 0)),
        "Locked Min Hold": int(settings.get("min_hold_bars", 0)),
        "Locked Cooldown": int(settings.get("cooldown_bars", 0)),
        "Model Version": settings.get("model_version", ""),
        "Settings Source": settings.get("source", ""),
        "Ledger Mode": "Baseline" if baseline else "Live Append",
    }

def apply_institutional_trade_ledger(ticker, candidate_trades, px, settings):
    if not USE_INSTITUTIONAL_LEDGER:
        return candidate_trades

    ticker = str(ticker).upper()
    latest_price = float(pd.Series(px).dropna().iloc[-1])
    latest_time = _safe_str_dt(pd.Series(px).dropna().index[-1])

    ledger = load_institutional_ledger()
    book = ledger.get(ticker, {"trades": [], "last_update_ct": ""})
    rows = book.get("trades", []) if isinstance(book, dict) else []
    if not isinstance(rows, list):
        rows = []
    cand = candidate_trades.copy() if isinstance(candidate_trades, pd.DataFrame) else pd.DataFrame()

    if len(rows) == 0 and not cand.empty:
        rows = [
            trade_row_to_institutional_record(ticker, tr, settings, latest_price=latest_price, baseline=True)
            for _, tr in cand.iterrows()
        ]
    else:
        open_i = None
        for i, r in enumerate(rows):
            if str(r.get("Status", "")).lower() == "open":
                open_i = i
                break

        if open_i is not None:
            open_row = rows[open_i]
            open_entry = str(open_row.get("Entry CT", ""))
            matched_closed = None
            if not cand.empty:
                for _, tr in cand.iterrows():
                    rec = trade_row_to_institutional_record(ticker, tr, settings, latest_price=latest_price, baseline=False)
                    if str(rec.get("Entry CT", "")) == open_entry and str(rec.get("Status", "")).lower() == "closed":
                        matched_closed = rec
                        break
            if matched_closed is not None:
                for k in ["Exit CT", "Exit/Current Price", "PnL (%)", "Status"]:
                    open_row[k] = matched_closed.get(k)
                open_row["Ledger Mode"] = "Live Closed"
                rows[open_i] = open_row
            else:
                ep = _safe_float_or_none(open_row.get("Entry Price"))
                open_row["Exit CT"] = "Open"
                open_row["Exit/Current Price"] = round(latest_price, 4)
                if ep:
                    open_row["PnL (%)"] = round((latest_price / ep - 1.0) * 100.0, 4)
                open_row["Status"] = "Open"
                open_row["Ledger Mode"] = "Live Open"
                rows[open_i] = open_row

        has_open = any(str(r.get("Status", "")).lower() == "open" for r in rows)
        if not has_open and not cand.empty:
            existing_entries = {str(r.get("Entry CT", "")) for r in rows}
            for _, tr in cand.iterrows():
                rec = trade_row_to_institutional_record(ticker, tr, settings, latest_price=latest_price, baseline=False)
                ent = str(rec.get("Entry CT", ""))
                if ent and ent not in existing_entries:
                    rows.append(rec)
                    existing_entries.add(ent)

    ledger[ticker] = {
        "ticker": ticker,
        "trades": rows,
        "last_update_ct": fmt_ct_now(),
        "latest_price": latest_price,
        "latest_time": latest_time,
        "current_settings_seen": settings,
    }
    save_institutional_ledger(ledger)
    return pd.DataFrame(rows)

def _trade_row_is_open(last_row, columns=None):
    try:
        status_val = str(last_row.get("Status", "")).strip().upper()
        if status_val in ("OPEN", "LONG"):
            return True
        if status_val in ("CLOSED", "CLOSE", "STOP LOSS", "TRAILING STOP", "CASH"):
            return False
        for c in ("Exit CT", "Exit Date", "Exit Time", "Exit"):
            if columns is not None and c not in columns:
                continue
            try:
                v = last_row.get(c, "__MISSING__")
            except Exception:
                continue
            if isinstance(v, str) and v == "__MISSING__":
                continue
            try:
                if v is None or (pd.isna(v) if np.isscalar(v) or v is None else False):
                    return True
            except Exception:
                pass
            sv = str(v).strip()
            if sv == "" or sv.lower() in ("nan", "none", "nat", "open"):
                return True
            return False
        row_txt = " ".join([str(x) for x in last_row.values]).upper()
        return ("OPEN" in row_txt) and ("CLOSED" not in row_txt)
    except Exception:
        return False

def latest_kalman_state(ticker):
    px = fetch_streamlit_visible_tab_prices(ticker)
    if px is None or len(px) < 80:
        return None

    sig, rail_s = build_main_kalman_signal(px, ticker)
    latest_sig = int(round(float(sig.iloc[-1])))
    prev_sig = int(round(float(sig.iloc[-2]))) if len(sig) >= 2 else latest_sig
    raw_alert = "BUY" if latest_sig == 1 and prev_sig == 0 else "SELL" if latest_sig == 0 and prev_sig == 1 else "NO NEW ALERT"

    candidate_trades = run_strategy(px, sig)
    settings = settings_for_institutional_ledger(ticker)
    production_trades = apply_institutional_trade_ledger(ticker, candidate_trades, px, settings)

    if isinstance(production_trades, pd.DataFrame) and not production_trades.empty:
        is_open = _trade_row_is_open(production_trades.iloc[-1], columns=production_trades.columns)
        position = "LONG" if is_open else "CASH"
        last_trade = production_trades.iloc[-1].to_dict()
    else:
        position = "LONG" if latest_sig == 1 else "CASH"
        last_trade = None

    last_start = pd.Timestamp(px.index[-1])
    interval_minutes = 15 if INTERVAL == "15m" else 5 if INTERVAL == "5m" else 30 if INTERVAL == "30m" else 60
    candle_close = last_start + pd.Timedelta(minutes=interval_minutes)

    return {
        "ticker": str(ticker).upper(),
        "position": position,
        "raw_alert": raw_alert,
        "raw_signal_position": "LONG" if latest_sig == 1 else "CASH",
        "price": float(px.iloc[-1]),
        "rail": float(rail_s.iloc[-1]),
        "bar_start_ct": last_start,
        "candle_close_ct": candle_close,
        "last_trade": last_trade,
        "params_source": "CUSTOM" if str(ticker).upper() in PER_TICKER_PARAMS else "DEFAULT",
    }

def debug_ticker(ticker):
    ticker = str(ticker).upper().strip()
    try:
        info = latest_kalman_state(ticker)
        if info is None:
            return f"⚠️ {ticker}: no data."
        p = params_for_ticker(ticker)
        bundle_note = "YES" if os.path.exists(BUNDLE_FILE) else "NO"
        lock_store = load_signal_lock()
        inst_store = load_institutional_ledger()
        return "\n".join([
            f"<b>Streamlit Exact Bundle Debug — {ticker}</b>",
            f"Production position: <b>{info['position']}</b>",
            f"Raw signal position: {info['raw_signal_position']} | Latest raw alert: {info['raw_alert']}",
            f"Price: {info['price']:.2f} | Rail: {info['rail']:.2f}",
            f"Latest bar start CT: {info['bar_start_ct'].strftime('%Y-%m-%d %I:%M %p CT')}",
            f"Displayed candle close CT: {info['candle_close_ct'].strftime('%Y-%m-%d %I:%M %p CT')}",
            f"Data: {LOOKBACK_DAYS} days, auto_adjust=False, start/end fetch, Streamlit dropna cleaning",
            f"Params: buffer={p['buffer_pct']}, confirm={p['confirm_bars']}, hold={p['min_hold_bars']}, cool={p['cooldown_bars']}",
            f"Per-ticker params: {'YES' if ticker in PER_TICKER_PARAMS else 'NO'}",
            f"Param source: {param_source_for_ticker(ticker)}",
            f"Streamlit bundle file found: {bundle_note}",
            f"Signal-lock key loaded: {'YES' if f'{ticker}|{INTERVAL}' in lock_store else 'NO'}",
            f"Institutional ledger ticker loaded: {'YES' if ticker in inst_store else 'NO'}",
        ])
    except Exception as e:
        return f"⚠️ {ticker} debug error: {e}"

# ============================================================
# TELEGRAM COMMANDS
# ============================================================
def _load_update_offset():
    global last_update_id
    data = _load_json_file(UPDATE_OFFSET_FILE, {})
    if isinstance(data.get("last_update_id"), int):
        last_update_id = data["last_update_id"]

def _save_update_offset():
    _save_json_file(UPDATE_OFFSET_FILE, {"last_update_id": last_update_id})

def handle_telegram_commands():
    global last_update_id, rescan_requested
    if not BOT_TOKEN:
        return
    try:
        params = {"timeout": 0}
        if last_update_id is not None:
            params["offset"] = str(last_update_id)
        resp = requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates", params=params, timeout=8).json()
        if not resp.get("ok", True):
            print(f"Telegram getUpdates not ok: {resp}")
            return
        for res in resp.get("result", []):
            last_update_id = int(res["update_id"]) + 1
            _save_update_offset()
            msg_obj = res.get("message") or res.get("edited_message") or {}
            raw = str(msg_obj.get("text", "")).strip()
            if not raw.startswith("/"):
                continue
            parts = raw.split(maxsplit=1)
            cmd = parts[0].lower().split("@", 1)[0]
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd == "/status":
                longs = sorted([t for t, s in positions.items() if s == "LONG"])
                cash = sorted([t for t, s in positions.items() if s == "CASH"])
                unknown = sorted([t for t, s in positions.items() if s not in ("LONG", "CASH")])
                msg = (
                    f"📋 <b>Main Kalman Exact Bundle Mirror — {fmt_ct_now()}</b>\n"
                    f"Interval: <b>{INTERVAL}</b> | Visible-tab lookback: <b>{LOOKBACK_DAYS} days</b>\n"
                    f"Full scans: <b>{full_scans_completed}</b> | Params loaded: <b>{len(PER_TICKER_PARAMS)}</b>\n"
                    f"Bundle: <b>{'LOADED' if os.path.exists(BUNDLE_FILE) else 'NOT FOUND'}</b> | "
                    f"Signal-lock keys: <b>{len(load_signal_lock())}</b> | Ledger tickers: <b>{len(load_institutional_ledger())}</b>\n"
                    f"{len(longs)} LONG / {len(cash)} CASH / {len(unknown)} UNKNOWN\n\n"
                    + "<b>LONG</b>\n" + ("".join([f"🟢 {t}\n" for t in longs]) if longs else "None\n")
                    + "\n<b>CASH</b>\n" + ("".join([f"⚪ {t}\n" for t in cash]) if cash else "None\n")
                )
                if unknown:
                    msg += "\n<b>UNKNOWN</b>\n" + "".join([f"❔ {t}\n" for t in unknown])
                send_telegram(msg)

            elif cmd in ("/why", "/debug"):
                send_telegram(debug_ticker(arg))

            elif cmd == "/stateinfo":
                send_telegram(
                    f"<b>Streamlit state loaded</b>\n"
                    f"Bundle found: {os.path.exists(BUNDLE_FILE)}\n"
                    f"Bundle version: {STREAMLIT_BUNDLE.get('bundle_version', 0)}\n"
                    f"Bundle exported CT: {STREAMLIT_BUNDLE.get('exported_ct','')}\n"
                    f"Per-ticker params: {len(PER_TICKER_PARAMS)}\n"
                    f"Bundle open tickers: {len(STREAMLIT_BUNDLE.get('streamlit_open_tickers', []))}\n"
                    f"Signal-lock keys: {len(load_signal_lock())}\n"
                    f"Institutional ledger tickers: {len(load_institutional_ledger())}\n"
                    f"Data path: {LOOKBACK_DAYS}d / {INTERVAL} / auto_adjust=False / prepost=False"
                )

            elif cmd == "/params":
                summary = STREAMLIT_BUNDLE.get("sync_summary", {})
                counts = summary.get("source_counts", {}) if isinstance(summary, dict) else {}
                lines = [
                    "<b>Render Parameter Sync</b>",
                    f"Bundle version: {STREAMLIT_BUNDLE.get('bundle_version', 0)}",
                    f"Bundle exported CT: {STREAMLIT_BUNDLE.get('exported_ct', '')}",
                    f"Per-ticker params loaded: {len(PER_TICKER_PARAMS)}",
                    f"Active Streamlit cache overrides: {summary.get('active_session_overrides', 0) if isinstance(summary, dict) else 0}",
                    f"Live Streamlit captures: {summary.get('live_mirror_captures', 0) if isinstance(summary, dict) else 0}",
                    f"Trusted saved params: {summary.get('trusted_saved_params', 0) if isinstance(summary, dict) else 0}",
                    f"Fallback seed params: {summary.get('fallback_seed_params', 0) if isinstance(summary, dict) else 0}",
                    f"Watchlist tickers: {len(WATCHLIST)}",
                    f"Parameter coverage: {len(WATCHLIST) - len(MISSING_PARAM_TICKERS)}/{len(WATCHLIST)}",
                    f"Full 150 coverage: {PARAM_COVERAGE_OK}",
                ]
                if MISSING_PARAM_TICKERS:
                    lines.append("Missing params: " + ", ".join(MISSING_PARAM_TICKERS))
                if EXTRA_PARAM_TICKERS:
                    lines.append("Extra params: " + ", ".join(EXTRA_PARAM_TICKERS))
                if isinstance(counts, dict) and counts:
                    lines.append("Sources:")
                    for k, v in sorted(counts.items()):
                        lines.append(f"- {k}: {v}")
                send_telegram("\n".join(lines))

            elif cmd == "/rescan":
                rescan_requested = True
                send_telegram("🔄 Full Main Kalman rescan requested.")

            elif cmd == "/ping":
                send_telegram(f"🏓 Bot alive — {fmt_ct_now()}")

            elif cmd == "/help":
                send_telegram("/status\n/why AMD\n/stateinfo\n/params\n/rescan\n/ping")
    except Exception as e:
        print(f"Telegram command error: {e}")

def next_quarter_scan_time(now=None):
    now = now or ct_now()
    interval = max(1, int(SCAN_EVERY_MINUTES))
    base = now.replace(second=0, microsecond=0)
    minute_mod = base.minute % interval
    if minute_mod == 0 and now.second < SCAN_DELAY_SECONDS:
        boundary = base
    else:
        add_min = interval - minute_mod if minute_mod else interval
        boundary = base + timedelta(minutes=add_min)
    return boundary + timedelta(seconds=max(0, SCAN_DELAY_SECONDS))

def sleep_until_scan_time(target_dt):
    global rescan_requested
    while True:
        handle_telegram_commands()
        if rescan_requested:
            return "manual"
        remaining = (target_dt - ct_now()).total_seconds()
        if remaining <= 0:
            return "scheduled"
        time.sleep(min(2.0, remaining))

# ============================================================
# SCAN
# ============================================================
def scan_once():
    global scan_started_at, scan_finished_at, scan_in_progress, full_scans_completed
    scan_in_progress = True
    scan_started_at = fmt_ct_now()
    handle_telegram_commands()

    for ticker in WATCHLIST:
        try:
            info = latest_kalman_state(ticker)
            if info is None:
                positions[ticker] = "UNKNOWN"
                last_error[ticker] = "not enough data"
                last_checked[ticker] = fmt_ct_now()
                print(f"⚠️ {ticker}: no data")
                time.sleep(SLEEP_BETWEEN_TICKERS_SEC)
                continue

            old_state = positions.get(ticker, "UNKNOWN")
            new_state = info["position"]
            raw_alert = info["raw_alert"]
            positions[ticker] = new_state
            last_checked[ticker] = info["candle_close_ct"].strftime("%Y-%m-%d %I:%M %p CT")
            last_error.pop(ticker, None)

            transition = "NO NEW ALERT"
            if old_state == "CASH" and new_state == "LONG":
                transition = "BUY"
            elif old_state == "LONG" and new_state == "CASH":
                transition = "SELL"
            elif old_state == "UNKNOWN" and raw_alert in ("BUY", "SELL"):
                if (raw_alert == "BUY" and new_state == "LONG") or (raw_alert == "SELL" and new_state == "CASH"):
                    transition = raw_alert

            bar_key = f"{ticker}|{transition}|{info['bar_start_ct'].strftime('%Y-%m-%d %H:%M:%S')}"
            if transition in ("BUY", "SELL") and last_alert_bar.get(ticker) != bar_key:
                last_alert_bar[ticker] = bar_key
                emoji = "🟢" if transition == "BUY" else "🔴"
                send_telegram(
                    f"{emoji} <b>PINEHURST MAIN KALMAN {transition}</b>\n"
                    f"Ticker: <b>{ticker}</b>\n"
                    f"Position: <b>{new_state}</b>\n"
                    f"Price: <b>{info['price']:.2f}</b>\n"
                    f"Rail: <b>{info['rail']:.2f}</b>\n"
                    f"Bar Time CT: <b>{info['bar_start_ct'].strftime('%Y-%m-%d %I:%M %p CT')}</b>\n"
                    f"Source: Streamlit Exact Bundle Mirror v16"
                )
                print(f"📊 {ticker}: {old_state} -> {new_state} | {transition}")
            else:
                print(
                    f"{ticker}: {new_state} | raw={info['raw_signal_position']} | "
                    f"price={info['price']:.2f} | previous={old_state} | latest_raw_alert={raw_alert}"
                )

            save_state()
            time.sleep(SLEEP_BETWEEN_TICKERS_SEC)
        except Exception as e:
            positions[ticker] = "UNKNOWN"
            last_error[ticker] = str(e)[:160]
            last_checked[ticker] = fmt_ct_now()
            print(f"{ticker} scan error: {e}")
            time.sleep(SLEEP_BETWEEN_TICKERS_SEC)

    scan_in_progress = False
    scan_finished_at = fmt_ct_now()
    full_scans_completed += 1
    save_state()

if __name__ == "__main__":
    seed_streamlit_state_once()
    load_state()
    seed_positions_from_streamlit_bundle_once()
    _load_update_offset()

    print("🚀 Pinehurst Main Kalman Streamlit Exact Bundle Mirror v16")
    print(f"Data path: {LOOKBACK_DAYS} calendar days / {INTERVAL} / auto_adjust=False")
    print(f"Params loaded: {len(PER_TICKER_PARAMS)}")
    print(f"Bundle found: {os.path.exists(BUNDLE_FILE)}")
    print(f"Watchlist coverage: {len(WATCHLIST) - len(MISSING_PARAM_TICKERS)}/{len(WATCHLIST)} params")
    if MISSING_PARAM_TICKERS:
        print("MISSING PARAM TICKERS: " + ", ".join(MISSING_PARAM_TICKERS))
    if EXTRA_PARAM_TICKERS:
        print("EXTRA PARAM TICKERS: " + ", ".join(EXTRA_PARAM_TICKERS))
    print(f"Signal-lock keys: {len(load_signal_lock())}")
    print(f"Institutional ledger tickers: {len(load_institutional_ledger())}")
    print(f"Scan schedule: every {SCAN_EVERY_MINUTES}m + {SCAN_DELAY_SECONDS}s CT")

    next_scheduled_scan_ct = next_quarter_scan_time()
    send_telegram(
        f"🚀 <b>Pinehurst Main Kalman Exact Bundle Mirror v16 active</b>\n"
        f"Data: {LOOKBACK_DAYS}d / {INTERVAL} / auto_adjust=False\n"
        f"Params: {len(PER_TICKER_PARAMS)} | Bundle: {'LOADED' if os.path.exists(BUNDLE_FILE) else 'NOT FOUND'}\n"
        f"Next scan: <b>{fmt_ct_dt(next_scheduled_scan_ct)}</b>"
    )

    while True:
        next_scheduled_scan_ct = next_quarter_scan_time()
        print(f"⏳ Waiting until {fmt_ct_dt(next_scheduled_scan_ct)}")
        reason = sleep_until_scan_time(next_scheduled_scan_ct)
        if reason == "manual":
            rescan_requested = False
            print("🔄 Manual rescan")
        scan_once()
        save_state()
        handle_telegram_commands()
        print("✅ Full scan complete.")
