"""
TRADE INTELLIGENCE MODULE
7-Layer institutional confirmation system.
Designed to run as a tab inside the main Unified Quant Suite.
Can also be imported standalone.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════
# SECTOR MAP  — ticker → (sector_etf, sector_name, industry)
# ══════════════════════════════════════════════════════════════
SECTOR_ETF_MAP = {
    # Tech
    "AAPL":"XLK","MSFT":"XLK","NVDA":"SOXX","AMD":"SOXX","INTC":"SOXX",
    "AVGO":"SOXX","QCOM":"SOXX","MU":"SOXX","AMAT":"SOXX","LRCX":"SOXX",
    "GOOGL":"XLC","GOOG":"XLC","META":"XLC","NFLX":"XLC","DIS":"XLC",
    "AMZN":"XLY","TSLA":"XLY","HD":"XLY","NKE":"XLY","MCD":"XLY",
    "JPM":"XLF","BAC":"XLF","WFC":"XLF","GS":"XLF","MS":"XLF","V":"XLF","MA":"XLF",
    "JNJ":"XLV","UNH":"XLV","PFE":"XLV","MRK":"XLV","ABBV":"XLV","LLY":"XLV",
    "XOM":"XLE","CVX":"XLE","COP":"XLE","SLB":"XLE","EOG":"XLE",
    "LIN":"XLB","APD":"XLB","NEM":"XLB","FCX":"XLB",
    "NEE":"XLU","DUK":"XLU","SO":"XLU","AEP":"XLU",
    "AMT":"XLRE","PLD":"XLRE","EQIX":"XLRE","SPG":"XLRE",
    "CAT":"XLI","BA":"XLI","HON":"XLI","GE":"XLI","RTX":"XLI","UPS":"XLI",
    "PG":"XLP","KO":"XLP","PEP":"XLP","WMT":"XLP","COST":"XLP","PM":"XLP",
}
SECTOR_NAMES = {
    "XLK":"Technology","SOXX":"Semiconductors","XLC":"Communication",
    "XLY":"Consumer Discret.","XLF":"Financials","XLV":"Healthcare",
    "XLE":"Energy","XLB":"Materials","XLU":"Utilities",
    "XLRE":"Real Estate","XLI":"Industrials","XLP":"Consumer Staples",
}


# ══════════════════════════════════════════════════════════════
# DATA FETCHERS
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def fetch_price_history(ticker: str, period: str = "1y") -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, period=period, interval="1d",
                         auto_adjust=True, progress=False)
        if df.empty: return None
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        df = df[['Open','High','Low','Close','Volume']].copy()
        df.dropna(inplace=True)
        return df
    except: return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_quote(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return info
    except: return {}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_earnings_date(ticker: str):
    """Returns next earnings date or None."""
    try:
        cal = yf.Ticker(ticker).calendar
        if cal is None: return None
        if isinstance(cal, pd.DataFrame):
            if 'Earnings Date' in cal.index:
                val = cal.loc['Earnings Date'].iloc[0]
                return pd.Timestamp(val).date()
        if isinstance(cal, dict):
            ed = cal.get('Earnings Date', [])
            if isinstance(ed, list) and len(ed) > 0:
                return pd.Timestamp(ed[0]).date()
        return None
    except: return None


# ══════════════════════════════════════════════════════════════
# LAYER 1 — MARKET CONDITION
# ══════════════════════════════════════════════════════════════

def analyze_market_condition() -> dict:
    """
    Checks SPY, QQQ, IWM trend + VIX level + market breadth proxy.
    Returns a dict with scores and labels.
    """
    result = {
        'spy_trend': 'Unknown', 'qqq_trend': 'Unknown', 'iwm_trend': 'Unknown',
        'vix_level': None, 'vix_signal': 'Unknown',
        'breadth_score': 0, 'market_score': 0,
        'market_verdict': 'Unknown', 'details': {}
    }

    index_score = 0

    for sym, key in [("SPY","spy"), ("QQQ","qqq"), ("IWM","iwm")]:
        df = fetch_price_history(sym, period="6mo")
        if df is None: continue
        close = df['Close']
        ema20 = close.ewm(span=20).mean()
        ema50 = close.ewm(span=50).mean()
        ema200 = close.rolling(200).mean()
        last = close.iloc[-1]
        above20 = last > ema20.iloc[-1]
        above50 = last > ema50.iloc[-1]
        above200 = last > ema200.iloc[-1] if len(close) >= 200 else True
        slope20 = (ema20.iloc[-1] - ema20.iloc[-5]) / ema20.iloc[-5] * 100
        if above20 and above50 and above200 and slope20 > 0:
            trend = "STRONG UPTREND"; sc = 2
        elif above50 and above200:
            trend = "UPTREND"; sc = 1
        elif not above50 and not above200:
            trend = "DOWNTREND"; sc = -2
        elif not above20:
            trend = "PULLBACK"; sc = 0
        else:
            trend = "MIXED"; sc = 0
        result[f'{key}_trend'] = trend
        result['details'][sym] = {
            'last': round(float(last), 2),
            'ema20': round(float(ema20.iloc[-1]), 2),
            'ema50': round(float(ema50.iloc[-1]), 2),
            'above20': above20, 'above50': above50, 'above200': above200,
            'slope20_pct': round(slope20, 3)
        }
        index_score += sc

    # VIX
    vix_df = fetch_price_history("^VIX", period="6mo")
    if vix_df is not None:
        vix = float(vix_df['Close'].iloc[-1])
        vix_ma20 = float(vix_df['Close'].ewm(span=20).mean().iloc[-1])
        vix_spike = vix > vix_ma20 * 1.3
        result['vix_level'] = round(vix, 2)
        result['details']['VIX'] = {'level': vix, 'ma20': round(vix_ma20, 2), 'spike': vix_spike}
        if vix < 15:
            result['vix_signal'] = "CALM (Low Fear)"; index_score += 1
        elif vix < 20:
            result['vix_signal'] = "NORMAL"; index_score += 0
        elif vix < 30:
            result['vix_signal'] = "ELEVATED (Caution)"; index_score -= 1
        else:
            result['vix_signal'] = "EXTREME FEAR"; index_score -= 2
        if vix_spike:
            result['vix_signal'] += " ⚡ SPIKE"; index_score -= 1

    # Breadth proxy: % of DJIA stocks above 50-EMA
    djia_sample = ["AAPL","MSFT","GS","JPM","HD","MCD","V","DIS","BA","CAT",
                   "MMM","IBM","TRV","WMT","CVX","XOM","PG","JNJ","MRK","UNH"]
    above_50 = 0; checked = 0
    for sym in djia_sample[:10]:  # limit for speed
        df = fetch_price_history(sym, period="3mo")
        if df is None: continue
        ema50 = df['Close'].ewm(span=50).mean()
        if float(df['Close'].iloc[-1]) > float(ema50.iloc[-1]):
            above_50 += 1
        checked += 1
    breadth = (above_50 / checked * 100) if checked > 0 else 50
    result['breadth_score'] = round(breadth, 1)
    result['details']['Breadth'] = {'pct_above_50ema': breadth, 'checked': checked}
    if breadth > 70: index_score += 1
    elif breadth < 40: index_score -= 1

    result['market_score'] = index_score
    if index_score >= 5:
        result['market_verdict'] = "STRONG BULL — High probability environment for longs"
        result['market_color'] = "#00ff88"
    elif index_score >= 2:
        result['market_verdict'] = "BULL BIAS — Favorable for longs, stay selective"
        result['market_color'] = "#44cc66"
    elif index_score >= 0:
        result['market_verdict'] = "NEUTRAL — Mixed signals, reduce size"
        result['market_color'] = "#ffcc00"
    elif index_score >= -2:
        result['market_verdict'] = "BEAR BIAS — Avoid new longs, tight stops"
        result['market_color'] = "#ff8844"
    else:
        result['market_verdict'] = "STRONG BEAR — Cash is king, no new longs"
        result['market_color'] = "#ff4444"

    return result


# ══════════════════════════════════════════════════════════════
# LAYER 2 — SECTOR STRENGTH
# ══════════════════════════════════════════════════════════════

def analyze_sector_strength(ticker: str, stock_df: pd.DataFrame) -> dict:
    sector_etf = SECTOR_ETF_MAP.get(ticker.upper(), None)
    result = {
        'sector_etf': sector_etf,
        'sector_name': SECTOR_NAMES.get(sector_etf, "Unknown") if sector_etf else "Unknown",
        'sector_trend': 'Unknown', 'stock_vs_sector': None,
        'stock_vs_spy': None, 'sector_score': 0,
        'sector_verdict': 'Unknown', 'details': {}
    }
    if sector_etf is None:
        # Try from yfinance info
        try:
            info = fetch_quote(ticker)
            result['sector_name'] = info.get('sector', 'Unknown')
        except: pass
        result['sector_verdict'] = "Sector ETF not mapped — manual check needed"
        return result

    sec_df = fetch_price_history(sector_etf, period="6mo")
    spy_df = fetch_price_history("SPY", period="6mo")
    sc = 0

    if sec_df is not None:
        close = sec_df['Close']
        ema20 = close.ewm(span=20).mean()
        ema50 = close.ewm(span=50).mean()
        last = float(close.iloc[-1])
        above20 = last > float(ema20.iloc[-1])
        above50 = last > float(ema50.iloc[-1])
        momentum_4w = (last / float(close.iloc[-20]) - 1) * 100 if len(close) >= 20 else 0

        if above20 and above50 and momentum_4w > 0:
            result['sector_trend'] = "STRONG"; sc += 2
        elif above50:
            result['sector_trend'] = "MODERATE"; sc += 1
        elif not above20 and not above50:
            result['sector_trend'] = "WEAK"; sc -= 2
        else:
            result['sector_trend'] = "MIXED"; sc += 0

        result['details']['sector_etf'] = {
            'symbol': sector_etf, 'last': round(last, 2),
            'ema20': round(float(ema20.iloc[-1]), 2),
            'ema50': round(float(ema50.iloc[-1]), 2),
            'momentum_4w_pct': round(momentum_4w, 2)
        }

    # Stock vs Sector relative strength
    if sec_df is not None and len(stock_df) > 20 and len(sec_df) > 20:
        common = stock_df.index.intersection(sec_df.index)
        if len(common) >= 20:
            stk = stock_df['Close'].loc[common]
            sec = sec_df['Close'].loc[common]
            rs = (stk / stk.iloc[0]) / (sec / sec.iloc[0])
            rs_slope = (float(rs.iloc[-1]) - float(rs.iloc[-10])) / float(rs.iloc[-10]) * 100
            result['stock_vs_sector'] = round(rs_slope, 2)
            if rs_slope > 2: sc += 1
            elif rs_slope < -2: sc -= 1

    # Stock vs SPY relative strength
    if spy_df is not None and len(stock_df) > 20:
        common = stock_df.index.intersection(spy_df.index)
        if len(common) >= 20:
            stk = stock_df['Close'].loc[common]
            spy = spy_df['Close'].loc[common]
            rs_spy = (stk / stk.iloc[0]) / (spy / spy.iloc[0])
            rs_spy_slope = (float(rs_spy.iloc[-1]) - float(rs_spy.iloc[-10])) / float(rs_spy.iloc[-10]) * 100
            result['stock_vs_spy'] = round(rs_spy_slope, 2)
            if rs_spy_slope > 3: sc += 1
            elif rs_spy_slope < -3: sc -= 1

    result['sector_score'] = sc
    if sc >= 3:
        result['sector_verdict'] = f"STRONG SECTOR TAILWIND ({result['sector_name']} bullish + stock outperforming)"
        result['sector_color'] = "#00ff88"
    elif sc >= 1:
        result['sector_verdict'] = f"MODERATE TAILWIND ({result['sector_name']} holding up)"
        result['sector_color'] = "#44cc66"
    elif sc == 0:
        result['sector_verdict'] = f"NEUTRAL SECTOR ({result['sector_name']} mixed)"
        result['sector_color'] = "#ffcc00"
    else:
        result['sector_verdict'] = f"SECTOR HEADWIND ({result['sector_name']} weak — caution)"
        result['sector_color'] = "#ff4444"

    return result


# ══════════════════════════════════════════════════════════════
# LAYER 3 — PRICE ACTION
# ══════════════════════════════════════════════════════════════

def analyze_price_action(df: pd.DataFrame) -> dict:
    result = {
        'pattern': 'Unknown', 'pattern_score': 0,
        'trend_structure': 'Unknown', 'entry_type': 'No clean entry yet',
        'details': {}
    }
    if df is None or len(df) < 30: return result

    close = df['Close']; high = df['High']; low = df['Low']
    vol = df['Volume']; op = df['Open']

    last = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    last_high = float(high.iloc[-1])
    last_low = float(low.iloc[-1])
    last_open = float(op.iloc[-1])

    sc = 0; patterns = []

    # ── Trend structure: Higher Highs / Higher Lows ──────────────
    lookback = min(20, len(df)-1)
    highs = high.iloc[-lookback:].values
    lows = low.iloc[-lookback:].values

    hh = all(highs[i] >= highs[i-1] for i in range(-5, -1))  # last 5 bars
    hl = all(lows[i] >= lows[i-1] for i in range(-5, -1))
    lh = all(highs[i] <= highs[i-1] for i in range(-5, -1))
    ll = all(lows[i] <= lows[i-1] for i in range(-5, -1))

    if hh and hl:
        result['trend_structure'] = "UPTREND (HH + HL)"; sc += 2
    elif lh and ll:
        result['trend_structure'] = "DOWNTREND (LH + LL)"; sc -= 2
    else:
        result['trend_structure'] = "CHOPPY / SIDEWAYS"

    # ── Close near high of day ────────────────────────────────────
    day_range = last_high - last_low
    if day_range > 0:
        close_pct = (last - last_low) / day_range
        result['details']['close_pct_of_range'] = round(close_pct * 100, 1)
        if close_pct > 0.80:
            patterns.append("Close near HOD ✅"); sc += 1
        elif close_pct < 0.25:
            patterns.append("Close near LOD ⚠️"); sc -= 1

    # ── Rejection wick ────────────────────────────────────────────
    body = abs(last - last_open)
    upper_wick = last_high - max(last, last_open)
    lower_wick = min(last, last_open) - last_low
    if body > 0:
        if lower_wick > body * 1.5 and lower_wick > upper_wick:
            patterns.append("Bullish Hammer / Rejection Wick ✅"); sc += 1
        if upper_wick > body * 1.5 and upper_wick > lower_wick:
            patterns.append("Bearish Shooting Star / Upper Wick ⚠️"); sc -= 1

    # ── Inside Bar ────────────────────────────────────────────────
    if last_high <= float(high.iloc[-2]) and last_low >= float(low.iloc[-2]):
        patterns.append("Inside Bar (Compression — watch for breakout)")

    # ── Breakout above 20-day high ────────────────────────────────
    high_20 = float(high.iloc[-21:-1].max()) if len(df) >= 21 else float(high.iloc[:-1].max())
    low_20 = float(low.iloc[-21:-1].min()) if len(df) >= 21 else float(low.iloc[:-1].min())
    if last > high_20:
        patterns.append("Breakout above 20-day high 🚀"); sc += 2
    if last < low_20:
        patterns.append("Breakdown below 20-day low ⚠️"); sc -= 2

    # ── Pullback to support (EMA) ─────────────────────────────────
    ema20 = float(close.ewm(span=20).mean().iloc[-1])
    ema50 = float(close.ewm(span=50).mean().iloc[-1])
    dist_ema20 = (last - ema20) / ema20 * 100
    dist_ema50 = (last - ema50) / ema50 * 100

    if -1.5 < dist_ema20 < 1.5 and hh:
        patterns.append("Pullback to 20 EMA in uptrend ✅ (Trend continuation entry)"); sc += 2
    if -2.0 < dist_ema50 < 2.0 and hh:
        patterns.append("Pullback to 50 EMA in uptrend ✅ (Deeper support hold)"); sc += 1

    # ── Failed breakout detection ─────────────────────────────────
    if float(high.iloc[-2]) > high_20 and last < high_20 and last < float(close.iloc[-2]):
        patterns.append("Failed breakout (prior bar broke out, today rejected) ⚠️"); sc -= 2

    # ── Volume confirmation ───────────────────────────────────────
    avg_vol = float(vol.rolling(20).mean().iloc[-1])
    last_vol = float(vol.iloc[-1])
    rvol = last_vol / avg_vol if avg_vol > 0 else 1.0
    result['details']['rvol'] = round(rvol, 2)
    if rvol > 1.5 and last > prev:
        patterns.append(f"High RVOL ({rvol:.1f}x) on up day ✅"); sc += 1
    elif rvol > 1.5 and last < prev:
        patterns.append(f"High RVOL ({rvol:.1f}x) on down day ⚠️"); sc -= 1

    result['pattern_score'] = sc
    result['patterns'] = patterns
    result['details'].update({
        'last': last, 'ema20': round(ema20, 2), 'ema50': round(ema50, 2),
        'dist_ema20_pct': round(dist_ema20, 2), 'dist_ema50_pct': round(dist_ema50, 2),
        'high_20d': round(high_20, 2), 'low_20d': round(low_20, 2), 'rvol': round(rvol, 2)
    })

    # Entry type
    if last > high_20 and rvol > 1.3:
        result['entry_type'] = "🚀 Breakout Entry"
    elif -2.0 < dist_ema20 < 0 and hh and hl:
        result['entry_type'] = "📉 Pullback Entry (Trend Continuation)"
    elif -5.0 < dist_ema50 < 0 and close_pct > 0.6:
        result['entry_type'] = "🔄 Mean Reversion Bounce"
    elif last < low_20:
        result['entry_type'] = "🚫 No Entry — Breakdown"
    elif 'Inside Bar' in str(patterns):
        result['entry_type'] = "⏳ No Clean Entry Yet — Inside Bar, Wait for Breakout"
    elif abs(dist_ema20) > 8:
        result['entry_type'] = "⚠️ Overextended — Wait for Pullback"
    elif lh or ll:
        result['entry_type'] = "❌ Lower High Rejection — Avoid Longs"
    else:
        result['entry_type'] = "👀 Watch — No High-Conviction Setup Yet"

    return result


# ══════════════════════════════════════════════════════════════
# LAYER 4 — SUPPORT & RESISTANCE
# ══════════════════════════════════════════════════════════════

def analyze_support_resistance(df: pd.DataFrame) -> dict:
    result = {
        'nearest_support': None, 'nearest_resistance': None,
        'risk_reward': None, 'rr_verdict': 'Unknown',
        'swing_low': None, 'swing_high': None, 'details': {}
    }
    if df is None or len(df) < 30: return result

    close = df['Close']; high = df['High']; low = df['Low']
    last = float(close.iloc[-1])

    # ── Pivot-based S/R (rolling swing highs/lows) ────────────────
    window = 10
    swing_highs, swing_lows = [], []
    for i in range(window, len(df) - window):
        if float(high.iloc[i]) == float(high.iloc[i-window:i+window].max()):
            swing_highs.append(float(high.iloc[i]))
        if float(low.iloc[i]) == float(low.iloc[i-window:i+window].min()):
            swing_lows.append(float(low.iloc[i]))

    # Nearest levels
    supports = sorted([s for s in swing_lows if s < last], reverse=True)
    resistances = sorted([r for r in swing_highs if r > last])

    # Also add key MAs as dynamic S/R
    ema20 = float(close.ewm(span=20).mean().iloc[-1])
    ema50 = float(close.ewm(span=50).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
    high_20d = float(high.iloc[-21:-1].max()) if len(df) >= 21 else None
    low_20d = float(low.iloc[-21:-1].min()) if len(df) >= 21 else None
    high_50d = float(high.iloc[-51:-1].max()) if len(df) >= 51 else None
    low_50d = float(low.iloc[-51:-1].min()) if len(df) >= 51 else None

    support_levels = supports[:3] if supports else []
    resistance_levels = resistances[:3] if resistances else []

    # Add MA levels
    for ma, label in [(ema20,"EMA20"), (ema50,"EMA50"), (ma200,"MA200")]:
        if ma is None: continue
        if ma < last: support_levels.append(ma)
        else: resistance_levels.append(ma)

    support_levels = sorted(support_levels, reverse=True)[:3]
    resistance_levels = sorted(resistance_levels)[:3]

    nearest_sup = support_levels[0] if support_levels else (last * 0.95)
    nearest_res = resistance_levels[0] if resistance_levels else (last * 1.05)

    dist_to_sup = (last - nearest_sup) / last * 100
    dist_to_res = (nearest_res - last) / last * 100
    rr = dist_to_res / dist_to_sup if dist_to_sup > 0 else 0

    result.update({
        'nearest_support': round(nearest_sup, 2),
        'nearest_resistance': round(nearest_res, 2),
        'dist_to_support_pct': round(dist_to_sup, 2),
        'dist_to_resistance_pct': round(dist_to_res, 2),
        'risk_reward': round(rr, 2),
        'support_levels': [round(s, 2) for s in support_levels],
        'resistance_levels': [round(r, 2) for r in resistance_levels],
        'swing_low': round(float(low.iloc[-20:].min()), 2),
        'swing_high': round(float(high.iloc[-20:].max()), 2),
        'high_20d': round(high_20d, 2) if high_20d else None,
        'low_20d': round(low_20d, 2) if low_20d else None,
        'high_50d': round(high_50d, 2) if high_50d else None,
    })

    if rr >= 3.0:
        result['rr_verdict'] = "EXCELLENT R:R (≥3:1) ✅"
        result['rr_color'] = "#00ff88"
    elif rr >= 2.0:
        result['rr_verdict'] = "GOOD R:R (≥2:1) ✅"
        result['rr_color'] = "#44cc66"
    elif rr >= 1.5:
        result['rr_verdict'] = "ACCEPTABLE R:R (≥1.5:1)"
        result['rr_color'] = "#ffcc00"
    elif rr >= 1.0:
        result['rr_verdict'] = "MARGINAL R:R — Reduce size"
        result['rr_color'] = "#ff8844"
    else:
        result['rr_verdict'] = "POOR R:R (<1:1) — Skip this trade ❌"
        result['rr_color'] = "#ff4444"

    return result


# ══════════════════════════════════════════════════════════════
# LAYER 5 — VOLUME & LIQUIDITY
# ══════════════════════════════════════════════════════════════

def analyze_volume_liquidity(df: pd.DataFrame, ticker: str) -> dict:
    result = {
        'avg_dollar_vol': None, 'rvol': None, 'vol_trend': 'Unknown',
        'liquidity_ok': False, 'liquidity_verdict': 'Unknown',
        'options_liquid': False, 'details': {}
    }
    if df is None or len(df) < 20: return result

    close = df['Close']; vol = df['Volume']
    last_price = float(close.iloc[-1])
    last_vol = float(vol.iloc[-1])
    avg_vol_20 = float(vol.rolling(20).mean().iloc[-1])
    avg_dollar_vol = last_price * avg_vol_20
    rvol = last_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0

    # Volume trend: is volume increasing or decreasing over last 5 bars?
    vol_5 = vol.iloc[-5:].mean(); vol_10_15 = vol.iloc[-15:-5].mean()
    vol_trend = "INCREASING" if vol_5 > vol_10_15 * 1.1 else \
                "DECREASING" if vol_5 < vol_10_15 * 0.9 else "STABLE"

    # Dollar volume filter: institutions require >$20M/day average
    liq_ok = avg_dollar_vol >= 20_000_000 and last_price >= 5.0

    # Options liquidity (check if options exist with decent OI)
    options_liq = False
    try:
        tk = yf.Ticker(ticker)
        exps = tk.options
        if exps:
            chain = tk.option_chain(exps[0])
            total_oi = chain.calls['openInterest'].fillna(0).sum() + \
                       chain.puts['openInterest'].fillna(0).sum()
            options_liq = total_oi > 1000
    except: pass

    result.update({
        'avg_dollar_vol': avg_dollar_vol,
        'avg_dollar_vol_fmt': f"${avg_dollar_vol/1e6:.1f}M" if avg_dollar_vol >= 1e6
                              else f"${avg_dollar_vol/1e3:.0f}K",
        'rvol': round(rvol, 2), 'vol_trend': vol_trend,
        'liquidity_ok': liq_ok, 'options_liquid': options_liq,
        'last_price': round(last_price, 2),
        'avg_vol_20': int(avg_vol_20),
        'details': {
            'last_price': round(last_price, 2),
            'avg_dollar_vol_M': round(avg_dollar_vol / 1e6, 2),
            'rvol': round(rvol, 2), 'vol_trend': vol_trend,
            'price_ok': last_price >= 5.0,
            'dollar_vol_ok': avg_dollar_vol >= 20_000_000
        }
    })

    if not liq_ok:
        if last_price < 5.0:
            result['liquidity_verdict'] = "❌ Price below $5 — skip (penny stock risk)"
            result['liquidity_color'] = "#ff4444"
        else:
            result['liquidity_verdict'] = "❌ Avg dollar volume < $20M — too illiquid"
            result['liquidity_color'] = "#ff4444"
    elif rvol > 2.0:
        result['liquidity_verdict'] = f"✅ Liquid + High RVOL ({rvol:.1f}x) — strong interest"
        result['liquidity_color'] = "#00ff88"
    elif rvol > 1.2:
        result['liquidity_verdict'] = f"✅ Liquid + Above-avg volume ({rvol:.1f}x)"
        result['liquidity_color'] = "#44cc66"
    else:
        result['liquidity_verdict'] = f"✅ Liquid but low RVOL ({rvol:.1f}x) — watch for confirmation"
        result['liquidity_color'] = "#ffcc00"

    return result


# ══════════════════════════════════════════════════════════════
# LAYER 6 — EARNINGS & NEWS RISK
# ══════════════════════════════════════════════════════════════

def analyze_earnings_risk(ticker: str) -> dict:
    result = {
        'next_earnings': None, 'days_to_earnings': None,
        'earnings_risk': 'Unknown', 'earnings_color': '#aaaaaa',
        'macro_events': [], 'risk_score': 0, 'details': {}
    }

    ed = fetch_earnings_date(ticker)
    today = datetime.now().date()

    if ed:
        days = (ed - today).days
        result['next_earnings'] = str(ed)
        result['days_to_earnings'] = days

        if 0 <= days <= 3:
            result['earnings_risk'] = f"🚨 EARNINGS IN {days} DAYS — Extreme risk, avoid new entries"
            result['earnings_color'] = "#ff4444"; result['risk_score'] = -3
        elif days <= 7:
            result['earnings_risk'] = f"⚠️ EARNINGS IN {days} DAYS — High IV, binary event risk"
            result['earnings_color'] = "#ff8844"; result['risk_score'] = -2
        elif days <= 14:
            result['earnings_risk'] = f"⚠️ Earnings in {days} days — Be aware, size down"
            result['earnings_color'] = "#ffcc00"; result['risk_score'] = -1
        elif days <= 30:
            result['earnings_risk'] = f"📅 Earnings in {days} days — On radar"
            result['earnings_color'] = "#88aaff"; result['risk_score'] = 0
        else:
            result['earnings_risk'] = f"✅ Next earnings {days} days away — Low near-term risk"
            result['earnings_color'] = "#44cc66"; result['risk_score'] = 1
    else:
        result['earnings_risk'] = "📅 Earnings date unavailable — check manually"
        result['earnings_color'] = "#888888"

    # Fetch basic company info for additional risk flags
    try:
        info = fetch_quote(ticker)
        short_pct = info.get('shortPercentOfFloat', 0) or 0
        if short_pct > 0.20:
            result['macro_events'].append(f"⚠️ High short interest ({short_pct:.1%}) — squeeze or continued pressure risk")
            result['risk_score'] -= 1
        beta = info.get('beta', 1.0) or 1.0
        if beta > 2.0:
            result['macro_events'].append(f"⚡ High beta ({beta:.1f}) — amplified macro sensitivity")
        result['details']['short_pct'] = round(short_pct * 100, 1)
        result['details']['beta'] = round(beta, 2)
        result['details']['sector'] = info.get('sector', 'N/A')
        # Biotech / FDA risk flag
        industry = info.get('industry', '').lower()
        if any(k in industry for k in ['biotech','pharmaceutical','drug','clinical']):
            result['macro_events'].append("🧬 Biotech/Pharma — FDA catalyst risk (binary events possible)")
            result['risk_score'] -= 1
    except: pass

    return result


# ══════════════════════════════════════════════════════════════
# MASTER TRADE INTELLIGENCE SCORER
# ══════════════════════════════════════════════════════════════

def compute_trade_intelligence(ticker: str) -> dict:
    """
    Runs all 7 layers and returns a unified result dict.
    """
    df = fetch_price_history(ticker, period="1y")

    with st.spinner("Layer 1: Market condition..."):
        market = analyze_market_condition()
    with st.spinner("Layer 2: Sector strength..."):
        sector = analyze_sector_strength(ticker, df if df is not None else pd.DataFrame())
    with st.spinner("Layer 3: Price action..."):
        price_action = analyze_price_action(df)
    with st.spinner("Layer 4: Support & resistance..."):
        sr = analyze_support_resistance(df)
    with st.spinner("Layer 5: Volume & liquidity..."):
        volume = analyze_volume_liquidity(df, ticker)
    with st.spinner("Layer 6: Earnings & news risk..."):
        earnings = analyze_earnings_risk(ticker)

    # ── COMPOSITE SCORE ───────────────────────────────────────
    composite = 0
    composite += min(market['market_score'], 4)       # max +4
    composite += min(sector['sector_score'], 3)        # max +3
    composite += min(price_action['pattern_score'], 4) # max +4
    composite += (1 if sr.get('risk_reward', 0) >= 2 else
                  0 if sr.get('risk_reward', 0) >= 1.5 else -1)
    composite += (1 if volume['liquidity_ok'] else -2)
    composite += earnings['risk_score']

    # ── ENTRY TYPE (Layer 7) ─────────────────────────────────
    entry_type = price_action.get('entry_type', 'No clean entry yet')
    if composite >= 10:
        final_verdict = "STRONG BUY"; final_color = "#00ff88"
    elif composite >= 6:
        final_verdict = "BUY"; final_color = "#44cc66"
    elif composite >= 3:
        final_verdict = "WATCH / WAIT"; final_color = "#ffcc00"
    elif composite >= 0:
        final_verdict = "NEUTRAL — SKIP"; final_color = "#aaaaaa"
    else:
        final_verdict = "AVOID"; final_color = "#ff4444"

    # Override: if earnings in <7 days or market is STRONG BEAR, cap at WATCH
    if earnings.get('days_to_earnings') is not None and earnings['days_to_earnings'] <= 7:
        if final_verdict in ["STRONG BUY", "BUY"]:
            final_verdict = "WATCH — EARNINGS RISK"; final_color = "#ff8844"
    if market['market_score'] <= -3 and final_verdict in ["STRONG BUY", "BUY"]:
        final_verdict = "WATCH — MARKET HEADWIND"; final_color = "#ff8844"
    if not volume['liquidity_ok']:
        if final_verdict in ["STRONG BUY", "BUY"]:
            final_verdict = "AVOID — ILLIQUID"; final_color = "#ff4444"

    return {
        'ticker': ticker, 'df': df,
        'market': market, 'sector': sector, 'price_action': price_action,
        'sr': sr, 'volume': volume, 'earnings': earnings,
        'composite_score': composite,
        'final_verdict': final_verdict, 'final_color': final_color,
        'entry_type': entry_type,
    }


# ══════════════════════════════════════════════════════════════
# RENDER FUNCTION — call this inside your Streamlit tab
# ══════════════════════════════════════════════════════════════

def render_trade_intelligence_tab(default_ticker: str = "AAPL"):
    st.write("### 🎯 Trade Intelligence — 7-Layer Institutional Confirmation")
    st.markdown("""
    A signal alone is not enough. This module runs **7 independent confirmation layers** before
    declaring a trade valid — the same logic institutional desks use before sizing into a position.
    """)

    col_in1, col_in2 = st.columns([1, 3])
    with col_in1:
        ti_ticker = st.text_input("Ticker to Analyze", default_ticker.upper(), key="ti_ticker").upper()
        run_ti = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True, key="run_ti")

    if not run_ti:
        st.info("Enter a ticker and click **Run Full Analysis** to see all 7 layers.")
        return

    result = compute_trade_intelligence(ti_ticker)

    # ── MASTER VERDICT BANNER ─────────────────────────────────────────────────
    st.divider()
    verdict_col, score_col, entry_col = st.columns(3)
    with verdict_col:
        st.markdown(f"""
        <div style="background:{result['final_color']}22; border:2px solid {result['final_color']};
                    border-radius:12px; padding:18px; text-align:center;">
            <h2 style="color:{result['final_color']}; margin:0;">{result['final_verdict']}</h2>
            <p style="margin:4px 0; color:#ccc; font-size:0.9em;">{ti_ticker}</p>
        </div>""", unsafe_allow_html=True)
    with score_col:
        sc = result['composite_score']
        bar_pct = min(100, max(0, (sc + 5) / 15 * 100))
        st.markdown(f"""
        <div style="background:#1a1a2e; border-radius:12px; padding:18px; text-align:center;">
            <p style="color:#aaa; margin:0; font-size:0.85em;">COMPOSITE SCORE</p>
            <h2 style="color:{result['final_color']}; margin:4px 0;">{sc} / 15</h2>
            <div style="background:#333; border-radius:4px; height:8px; margin-top:8px;">
                <div style="background:{result['final_color']}; width:{bar_pct:.0f}%;
                            height:8px; border-radius:4px;"></div>
            </div>
        </div>""", unsafe_allow_html=True)
    with entry_col:
        st.markdown(f"""
        <div style="background:#1a1a2e; border-radius:12px; padding:18px; text-align:center;">
            <p style="color:#aaa; margin:0; font-size:0.85em;">ENTRY TYPE</p>
            <h3 style="color:#ffffff; margin:4px 0; font-size:1.1em;">{result['entry_type']}</h3>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── LAYER CARDS ───────────────────────────────────────────────────────────
    with st.expander("📊 Layer 1 — Market Condition", expanded=True):
        mkt = result['market']
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("SPY", mkt['spy_trend'])
        mc2.metric("QQQ", mkt['qqq_trend'])
        mc3.metric("IWM", mkt['iwm_trend'])
        mc4.metric("VIX", f"{mkt['vix_level']}" if mkt['vix_level'] else "N/A",
                   delta=mkt['vix_signal'])

        bc1, bc2 = st.columns(2)
        bc1.metric("Breadth (% stocks above 50 EMA)", f"{mkt['breadth_score']:.0f}%")
        bc2.metric("Market Score", f"{mkt['market_score']}")

        st.markdown(f"""
        <div style="background:{mkt.get('market_color','#333')}22;
                    border-left:4px solid {mkt.get('market_color','#888')};
                    padding:10px; border-radius:4px; margin-top:8px;">
            <b>{mkt['market_verdict']}</b>
        </div>""", unsafe_allow_html=True)

        # Index detail table
        det = mkt.get('details', {})
        rows = []
        for sym in ['SPY','QQQ','IWM']:
            if sym in det:
                d = det[sym]
                rows.append({
                    'Index': sym,
                    'Last': d['last'],
                    'EMA 20': d['ema20'], 'EMA 50': d['ema50'],
                    'Above 20': '✅' if d['above20'] else '❌',
                    'Above 50': '✅' if d['above50'] else '❌',
                    'Above 200': '✅' if d['above200'] else '❌',
                    '20 EMA Slope': f"{d['slope20_pct']:+.2f}%"
                })
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index('Index'), use_container_width=True)

    with st.expander("🏭 Layer 2 — Sector Strength", expanded=True):
        sec = result['sector']
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Sector ETF", sec['sector_etf'] or "N/A")
        sc2.metric("Sector", sec['sector_name'])
        sc3.metric("Sector Trend", sec['sector_trend'])
        s1, s2 = st.columns(2)
        s1.metric(f"Stock vs {sec['sector_etf']} (10d RS)",
                  f"{sec['stock_vs_sector']:+.2f}%" if sec['stock_vs_sector'] is not None else "N/A")
        s2.metric("Stock vs SPY (10d RS)",
                  f"{sec['stock_vs_spy']:+.2f}%" if sec['stock_vs_spy'] is not None else "N/A")
        st.markdown(f"""
        <div style="background:{sec.get('sector_color','#333')}22;
                    border-left:4px solid {sec.get('sector_color','#888')};
                    padding:10px; border-radius:4px;">
            <b>{sec['sector_verdict']}</b>
        </div>""", unsafe_allow_html=True)
        if sec['sector_etf'] and sec['sector_etf'] != 'Unknown':
            sec_df_vis = fetch_price_history(sec['sector_etf'], period="6mo")
            if sec_df_vis is not None and result['df'] is not None:
                common = result['df'].index.intersection(sec_df_vis.index)
                if len(common) > 10:
                    stk_n = (result['df']['Close'].loc[common] /
                             float(result['df']['Close'].loc[common].iloc[0])) * 100
                    sec_n = (sec_df_vis['Close'].loc[common] /
                             float(sec_df_vis['Close'].loc[common].iloc[0])) * 100
                    fig_sec = go.Figure()
                    fig_sec.add_trace(go.Scatter(x=common, y=stk_n, name=ti_ticker,
                                                  line=dict(color='#00f2ff', width=2)))
                    fig_sec.add_trace(go.Scatter(x=common, y=sec_n,
                                                  name=sec['sector_etf'],
                                                  line=dict(color='#ff6b35', width=2)))
                    fig_sec.update_layout(title="Stock vs Sector ETF (Indexed to 100)",
                                          template="plotly_dark", height=300,
                                          yaxis_title="Performance (%)",
                                          hovermode="x unified")
                    st.plotly_chart(fig_sec, use_container_width=True)

    with st.expander("📈 Layer 3 — Price Action", expanded=True):
        pa = result['price_action']
        pa1, pa2, pa3 = st.columns(3)
        pa1.metric("Trend Structure", pa['trend_structure'])
        pa2.metric("Pattern Score", pa['pattern_score'])
        pa3.metric("Entry Type", pa['entry_type'])
        det = pa.get('details', {})
        if det:
            pa4, pa5, pa6 = st.columns(3)
            pa4.metric("Dist from EMA20", f"{det.get('dist_ema20_pct',0):+.2f}%")
            pa5.metric("Dist from EMA50", f"{det.get('dist_ema50_pct',0):+.2f}%")
            pa6.metric("RVOL", f"{det.get('rvol',1):.2f}x")
        if pa.get('patterns'):
            st.write("**Detected patterns:**")
            for p in pa['patterns']:
                st.markdown(f"- {p}")

        # Candlestick chart with EMAs
        if result['df'] is not None and len(result['df']) >= 50:
            df_vis = result['df'].tail(120)
            ema20_s = df_vis['Close'].ewm(span=20).mean()
            ema50_s = df_vis['Close'].ewm(span=50).mean()
            fig_pa = go.Figure()
            fig_pa.add_trace(go.Candlestick(
                x=df_vis.index, open=df_vis['Open'], high=df_vis['High'],
                low=df_vis['Low'], close=df_vis['Close'],
                name='Price', increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'))
            fig_pa.add_trace(go.Scatter(x=df_vis.index, y=ema20_s,
                                         line=dict(color='orange', width=1.5), name='EMA 20'))
            fig_pa.add_trace(go.Scatter(x=df_vis.index, y=ema50_s,
                                         line=dict(color='#a855f7', width=1.5), name='EMA 50'))
            if det.get('high_20d'):
                fig_pa.add_hline(y=det['high_20d'], line_dash="dash",
                                  line_color="#00f2ff", annotation_text="20d High")
            if det.get('low_20d'):
                fig_pa.add_hline(y=det['low_20d'], line_dash="dash",
                                  line_color="#ff8844", annotation_text="20d Low")
            fig_pa.update_layout(title=f"{ti_ticker} — Price Action (120d)",
                                  template="plotly_dark", height=400,
                                  xaxis_rangeslider_visible=False, hovermode="x unified")
            st.plotly_chart(fig_pa, use_container_width=True)

    with st.expander("📍 Layer 4 — Support & Resistance", expanded=True):
        sr = result['sr']
        sr1, sr2, sr3, sr4 = st.columns(4)
        sr1.metric("Nearest Support", f"${sr.get('nearest_support','N/A')}")
        sr2.metric("Nearest Resistance", f"${sr.get('nearest_resistance','N/A')}")
        sr3.metric("Dist to Support", f"{sr.get('dist_to_support_pct',0):.2f}%")
        sr4.metric("Dist to Resistance", f"{sr.get('dist_to_resistance_pct',0):.2f}%")
        st.markdown(f"""
        <div style="background:{sr.get('rr_color','#333')}22;
                    border-left:4px solid {sr.get('rr_color','#888')};
                    padding:12px; border-radius:4px;">
            <b>Risk:Reward = {sr.get('risk_reward','N/A')} : 1 — {sr.get('rr_verdict','')}</b>
        </div>""", unsafe_allow_html=True)

        # S/R levels table
        sup_levels = sr.get('support_levels', [])
        res_levels = sr.get('resistance_levels', [])
        if result['df'] is not None:
            df_sr = result['df'].tail(60)
            fig_sr = go.Figure()
            fig_sr.add_trace(go.Candlestick(
                x=df_sr.index, open=df_sr['Open'], high=df_sr['High'],
                low=df_sr['Low'], close=df_sr['Close'],
                name='Price', increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'))
            for lv in sup_levels:
                fig_sr.add_hline(y=lv, line_dash="dot", line_color="#00ff88",
                                  opacity=0.6, annotation_text=f"SUP {lv:.2f}")
            for lv in res_levels:
                fig_sr.add_hline(y=lv, line_dash="dot", line_color="#ff4444",
                                  opacity=0.6, annotation_text=f"RES {lv:.2f}")
            fig_sr.update_layout(title=f"{ti_ticker} — S/R Levels (60d)",
                                  template="plotly_dark", height=380,
                                  xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_sr, use_container_width=True)

    with st.expander("💧 Layer 5 — Volume & Liquidity", expanded=True):
        vol = result['volume']
        vc1, vc2, vc3, vc4 = st.columns(4)
        vc1.metric("Avg Dollar Vol", vol.get('avg_dollar_vol_fmt', 'N/A'))
        vc2.metric("Relative Volume", f"{vol.get('rvol', 0):.2f}x")
        vc3.metric("Volume Trend", vol.get('vol_trend', 'N/A'))
        vc4.metric("Options Liquid", "✅ Yes" if vol.get('options_liquid') else "❌ No")
        st.markdown(f"""
        <div style="background:{vol.get('liquidity_color','#333')}22;
                    border-left:4px solid {vol.get('liquidity_color','#888')};
                    padding:10px; border-radius:4px;">
            <b>{vol.get('liquidity_verdict','')}</b>
        </div>""", unsafe_allow_html=True)

        # Volume bar chart
        if result['df'] is not None:
            df_vol = result['df'].tail(40)
            avg_20 = float(df_vol['Volume'].rolling(20).mean().iloc[-1])
            bar_colors = ['#00ff88' if c >= o else '#ff4444'
                          for c, o in zip(df_vol['Close'], df_vol['Open'])]
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(x=df_vol.index, y=df_vol['Volume'],
                                      marker_color=bar_colors, name='Volume', opacity=0.8))
            fig_vol.add_hline(y=avg_20, line_dash="dash", line_color="white",
                               annotation_text=f"Avg ({avg_20/1e6:.1f}M)")
            fig_vol.update_layout(title="Volume (40d)", template="plotly_dark",
                                   height=250, yaxis_title="Shares",
                                   margin=dict(t=30, b=10))
            st.plotly_chart(fig_vol, use_container_width=True)

    with st.expander("📅 Layer 6 — Earnings & News Risk", expanded=True):
        earn = result['earnings']
        ec1, ec2 = st.columns(2)
        ec1.metric("Next Earnings", earn.get('next_earnings', 'Unknown'))
        ec2.metric("Days Away", earn.get('days_to_earnings', 'N/A'))
        det_e = earn.get('details', {})
        if det_e:
            ed1, ed2 = st.columns(2)
            ed1.metric("Short Interest", f"{det_e.get('short_pct',0):.1f}%")
            ed2.metric("Beta", det_e.get('beta', 'N/A'))
        st.markdown(f"""
        <div style="background:{earn.get('earnings_color','#333')}22;
                    border-left:4px solid {earn.get('earnings_color','#888')};
                    padding:10px; border-radius:4px;">
            <b>{earn.get('earnings_risk','')}</b>
        </div>""", unsafe_allow_html=True)
        if earn.get('macro_events'):
            st.write("**Additional risk flags:**")
            for ev in earn['macro_events']:
                st.markdown(f"- {ev}")

    with st.expander("🏆 Layer 7 — Entry Quality Summary", expanded=True):
        st.write("#### Final Trade Checklist")
        checks = [
            ("Market environment favorable?",
             result['market']['market_score'] >= 2,
             result['market']['market_verdict']),
            ("Sector tailwind present?",
             result['sector']['sector_score'] >= 1,
             result['sector']['sector_verdict']),
            ("Price action constructive?",
             result['price_action']['pattern_score'] >= 1,
             result['price_action']['trend_structure']),
            ("Clean entry pattern?",
             result['price_action']['entry_type'] not in
             ["No clean entry yet","⏳ No Clean Entry Yet — Inside Bar, Wait for Breakout",
              "⚠️ Overextended — Wait for Pullback","❌ Lower High Rejection — Avoid Longs",
              "🚫 No Entry — Breakdown"],
             result['price_action']['entry_type']),
            ("Risk:Reward ≥ 2:1?",
             (result['sr'].get('risk_reward') or 0) >= 2.0,
             f"R:R = {result['sr'].get('risk_reward','N/A')}"),
            ("Stock is liquid?",
             result['volume']['liquidity_ok'],
             result['volume'].get('liquidity_verdict', '')),
            ("No imminent earnings risk?",
             (result['earnings'].get('days_to_earnings') is None or
              result['earnings'].get('days_to_earnings', 999) > 14),
             result['earnings'].get('earnings_risk', '')),
        ]

        passed = sum(1 for _, ok, _ in checks if ok)
        for label, ok, detail in checks:
            icon = "✅" if ok else "❌"
            color = "#00ff88" if ok else "#ff4444"
            st.markdown(
                f"<div style='display:flex; align-items:center; margin:4px 0; padding:8px;"
                f"background:{'#00ff8811' if ok else '#ff444411'}; border-radius:6px;'>"
                f"<span style='font-size:1.2em; margin-right:10px;'>{icon}</span>"
                f"<div><b style='color:{color}'>{label}</b>"
                f"<br><span style='color:#aaa; font-size:0.85em;'>{detail}</span></div>"
                f"</div>",
                unsafe_allow_html=True
            )

        st.divider()
        pct = passed / len(checks) * 100
        st.markdown(f"""
        <div style='background:{result['final_color']}22; border:2px solid {result['final_color']};
                    border-radius:12px; padding:20px; text-align:center; margin-top:12px;'>
            <h3 style='color:{result['final_color']}; margin:0;'>
                {passed}/{len(checks)} checks passed ({pct:.0f}%)
            </h3>
            <h2 style='color:{result['final_color']}; margin:8px 0;'>{result['final_verdict']}</h2>
            <p style='color:#ccc; margin:0;'>{result['entry_type']}</p>
        </div>""", unsafe_allow_html=True)

        # Composite score breakdown bar chart
        layer_scores = [
            ("Market", min(result['market']['market_score'], 4)),
            ("Sector", min(result['sector']['sector_score'], 3)),
            ("Price Action", min(result['price_action']['pattern_score'], 4)),
            ("R:R", 1 if (result['sr'].get('risk_reward') or 0) >= 2 else
                    0 if (result['sr'].get('risk_reward') or 0) >= 1.5 else -1),
            ("Liquidity", 1 if result['volume']['liquidity_ok'] else -2),
            ("Earnings", result['earnings']['risk_score']),
        ]
        colors_bar = ['#00ff88' if s > 0 else '#ff4444' if s < 0 else '#aaaaaa'
                      for _, s in layer_scores]
        fig_score = go.Figure(go.Bar(
            x=[l for l, _ in layer_scores],
            y=[s for _, s in layer_scores],
            marker_color=colors_bar,
            text=[f"{s:+d}" for _, s in layer_scores],
            textposition='outside'
        ))
        fig_score.add_hline(y=0, line_color="white", opacity=0.5)
        fig_score.update_layout(
            title="Layer Score Breakdown",
            template="plotly_dark", height=280,
            yaxis_title="Score", margin=dict(t=30, b=10)
        )
        st.plotly_chart(fig_score, use_container_width=True)
