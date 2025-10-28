"""
Day Trdr - Streamlit app (real data + AI explanations)

Run:
    pip install streamlit requests python-dotenv pandas numpy plotly
    streamlit run daytrdr_app.py

Environment variables (create a .env file or set in Streamlit Cloud):
    POLYGON_API_KEY=...
    FINNHUB_API_KEY=...
    NEWSAPI_KEY=...
    OPENAI_API_KEY=...   # optional, used for descriptive LLM explanations
    VITE_API_ROOT=...   # not needed here

Notes:
- If both POLYGON and FINNHUB are available the app prefers POLYGON for symbol search and history.
- The OpenAI call is optional and only used for a friendly human-readable explanation.
- This app is for education/demonstration only. Do not trade real capital solely from these signals.
"""
import os
import math
import time
import requests
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

# --- Config / Keys ---
POLYGON_KEY = os.getenv("POLYGON_API_KEY", "") or None
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "") or None
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "") or None
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "") or None

# UI
st.set_page_config(page_title="Day Trdr", layout="wide")
st.title("📈 Day Trdr — Market data + AI trade guidance")
st.caption("Numeric scores + descriptive AI explanations (if OpenAI key provided)")

# Sidebar controls
st.sidebar.header("Search & Settings")
query = st.sidebar.text_input("Ticker or company", value="AAPL")
timeframe = st.sidebar.selectbox("Prediction timeframe", ["scalp", "day", "swing"])
history_period = st.sidebar.selectbox("History period", ["1m", "5m", "1h", "1d", "1w", "1mo", "3mo", "6mo", "YTD", "1y", "5y", "ALL"], index=1)
max_history_points = 200

st.sidebar.markdown("---")
st.sidebar.markdown("**Data sources**")
st.sidebar.text("Polygon: %s" % ("configured" if POLYGON_KEY else "not set"))
st.sidebar.text("Finnhub: %s" % ("configured" if FINNHUB_KEY else "not set"))
st.sidebar.text("NewsAPI: %s" % ("configured" if NEWSAPI_KEY else "not set"))
st.sidebar.text("OpenAI: %s" % ("configured" if OPENAI_KEY else "not set"))
st.sidebar.markdown("---")
st.sidebar.markdown("⚠️ This tool is educational. Always backtest before real trading.")

# ---------------------------
# Helper: API wrappers
# ---------------------------

@st.cache_data(ttl=60)
def search_tickers(q: str):
    q = (q or "").strip()
    if not q:
        return []
    # Polygon lookup
    if POLYGON_KEY:
        try:
            url = "https://api.polygon.io/v3/reference/tickers"
            r = requests.get(url, params={"search": q, "active": "true", "limit": 8, "apiKey": POLYGON_KEY}, timeout=8)
            if r.ok:
                items = r.json().get("results", [])
                return [{"symbol": it.get("ticker"), "name": it.get("name") or ""} for it in items]
        except Exception:
            pass

import os
import requests
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

def get_stock_quote(symbol):
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={d40k10pr01qqo3qiek8gd40k10pr01qqo3qiek90}"
    response = requests.get(url)
    return response.json()
    # Finnhub search
    if FINNHUB_KEY:
        try:
            url = "https://finnhub.io/api/v1/search"
            r = requests.get(url, params={"q": q, "token": FINNHUB_KEY}, timeout=8)
            if r.ok:
                items = r.json().get("result", [])[:8]
                return [{"symbol": it.get("symbol"), "name": it.get("description") or ""} for it in items]
        except Exception:
            pass
    # fallback: echo
    return [{"symbol": q.upper(), "name": ""}]

@st.cache_data(ttl=30)
def fetch_quote_finnhub(symbol: str):
    if not FINNHUB_KEY:
        return None
    try:
        url = "https://finnhub.io/api/v1/quote"
        r = requests.get(url, params={"symbol": symbol, "token": FINNHUB_KEY}, timeout=6)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return None

@st.cache_data(ttl=120)
def fetch_history(symbol: str, period="1mo"):
    """
    Returns DataFrame with Date, o,h,l,c,v
    Tries Polygon aggregates first, then Finnhub candles fallback.
    """
    now = datetime.utcnow()
    if period == "7d":
        start = now - timedelta(days=7)
    elif period == "1mo":
        start = now - timedelta(days=30)
    elif period == "3mo":
        start = now - timedelta(days=90)
    elif period == "1y":
        start = now - timedelta(days=365)
    else:
        start = now - timedelta(days=30)

    # Polygon
    if POLYGON_KEY:
        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start.strftime('%Y-%m-%d')}/{now.strftime('%Y-%m-%d')}"
            r = requests.get(url, params={"adjusted": "true", "sort": "asc", "limit": max_history_points, "apiKey": POLYGON_KEY}, timeout=12)
            if r.ok:
                j = r.json()
                results = j.get("results", [])
                if results:
                    df = pd.DataFrame(results)
                    df['t'] = pd.to_datetime(df['t'], unit='ms')
                    df = df.rename(columns={"t":"Date","o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"})
                    df = df[['Date','Open','High','Low','Close','Volume']]
                    return df
        except Exception:
            pass

    # Finnhub fallback
    if FINNHUB_KEY:
        try:
            to_ts = int(now.timestamp())
            from_ts = int(start.timestamp())
            url = "https://finnhub.io/api/v1/stock/candle"
            r = requests.get(url, params={"symbol": symbol, "resolution":"D", "from": from_ts, "to": to_ts, "token": FINNHUB_KEY}, timeout=10)
            if r.ok:
                j = r.json()
                if j.get("s") == "ok":
                    df = pd.DataFrame({"t": j["t"], "o": j["o"], "h": j["h"], "l": j["l"], "c": j["c"], "v": j["v"]})
                    df['Date'] = pd.to_datetime(df['t'], unit='s')
                    df = df.rename(columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"})
                    df = df[['Date','Open','High','Low','Close','Volume']]
                    return df
        except Exception:
            pass

    return pd.DataFrame()  # empty fallback

@st.cache_data(ttl=120)
def fetch_fundamentals(symbol: str):
    # Use Finnhub metrics if available
    if FINNHUB_KEY:
        try:
            url = "https://finnhub.io/api/v1/stock/metric"
            r = requests.get(url, params={"symbol": symbol, "metric":"all", "token": FINNHUB_KEY}, timeout=8)
            if r.ok:
                return r.json().get("metric", {})
        except Exception:
            pass
    # Polygon has financials endpoints but require extra calls; leave placeholders
    return {}

@st.cache_data(ttl=120)
def fetch_news(symbol: str, limit=8):
    out = []
    # Finnhub company news 7-day
    if FINNHUB_KEY:
        try:
            to = datetime.utcnow().date()
            fr = to - timedelta(days=7)
            url = "https://finnhub.io/api/v1/company-news"
            r = requests.get(url, params={"symbol": symbol, "from": fr.isoformat(), "to": to.isoformat(), "token": FINNHUB_KEY}, timeout=8)
            if r.ok:
                for it in r.json()[:limit]:
                    out.append({"title": it.get("headline"), "source": it.get("source"), "url": it.get("url"), "date": datetime.utcfromtimestamp(it.get("datetime",0)).isoformat()})
        except Exception:
            pass
    # NewsAPI
    if NEWSAPI_KEY:
        try:
            url = "https://newsapi.org/v2/everything"
            r = requests.get(url, params={"q": symbol, "pageSize": limit, "sortBy":"publishedAt", "apiKey": NEWSAPI_KEY}, timeout=8)
            if r.ok:
                for a in r.json().get("articles", [])[:limit]:
                    out.append({"title": a.get("title"), "source": a.get("source",{}).get("name"), "url": a.get("url"), "date": a.get("publishedAt")})
        except Exception:
            pass
    # Deduplicate by title
    seen = set()
    dedup = []
    for n in out:
        k = (n.get("title"), n.get("source"))
        if k in seen: continue
        seen.add(k); dedup.append(n)
    return dedup[:limit]

# ---------------------------
# Indicators / strategies
# ---------------------------

def sma(series, window):
    return series.rolling(window).mean()

def rsi(series, window=14):
    delta = series.diff().dropna()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=window-1, adjust=False).mean()
    ema_down = down.ewm(com=window-1, adjust=False).mean()
    rs = ema_up / (ema_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.reindex(series.index, method='pad').fillna(50)
    return rsi

def atr(df, window=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr

def detect_volume_spike(df, lookback=20, multiplier=2.0):
    if 'Volume' not in df.columns or len(df['Volume']) < lookback+1:
        return False, 0.0
    vol = df['Volume']
    recent = vol.iloc[-1]
    avg = vol.iloc[-lookback:-1].mean()
    if avg == 0 or np.isnan(avg):
        return False, 0.0
    score = min(1.0, (recent / avg) / multiplier)
    return recent > avg * multiplier, float(score)

def detect_ma_crossover(df):
    # short 5, long 20
    if len(df) < 21: return 0.5, "no data"
    short = sma(df['Close'], 5)
    long = sma(df['Close'], 20)
    if short.iloc[-2] < long.iloc[-2] and short.iloc[-1] > long.iloc[-1]:
        return 1.0, "golden_cross"
    if short.iloc[-2] > long.iloc[-2] and short.iloc[-1] < long.iloc[-1]:
        return 0.0, "death_cross"
    # otherwise partial score by distance
    diff = (short.iloc[-1] - long.iloc[-1]) / (long.iloc[-1] + 1e-9)
    score = 0.5 + np.tanh(diff*10)/2  # maps diff to 0..1 coarse
    return float(score), "no_cross"

def detect_wedge_like(df):
    # simplified wedge detection: check if slope of highs and slope of lows converge and trending toward each other
    if len(df) < 10:
        return 0.2, "insufficient data"
    # use linear fit for last N points highs and lows
    N = min(30, len(df))
    y_hi = df['High'].values[-N:]
    y_lo = df['Low'].values[-N:]
    x = np.arange(N)
    # fit linear regression slopes
    A = np.vstack([x, np.ones_like(x)]).T
    try:
        m_hi, b_hi = np.linalg.lstsq(A, y_hi, rcond=None)[0]
        m_lo, b_lo = np.linalg.lstsq(A, y_lo, rcond=None)[0]
    except Exception:
        return 0.2, "regress_fail"
    # if slopes have opposite signs and the distance between lines is decreasing -> wedge forming
    slope_diff = abs(m_hi - m_lo)
    # examine distance between lines at the start and end
    start_gap = (m_hi*0 + b_hi) - (m_lo*0 + b_lo)
    end_gap = (m_hi*(N-1) + b_hi) - (m_lo*(N-1) + b_lo)
    gap_decreasing = abs(end_gap) < abs(start_gap)
    score = 0.5
    reason = "no_wedge"
    if gap_decreasing and slope_diff < abs(0.5* (abs(m_hi) + abs(m_lo)) + 1e-9):
        score = 0.8
        reason = "wedge_like"
    return float(score), reason

# ---------------------------
# Ensemble & prediction
# ---------------------------

def ensemble_predict(df: pd.DataFrame, timeframe: str):
    """
    Returns numeric scores for long and short in 0..1, a suggested stop-loss percent (e.g. 0.03),
    and a short textual reasoning. Uses simple heuristics from multiple strategies.
    """
    # default neutral
    if df.empty or len(df) < 10:
        return {"long": 0.5, "short": 0.5, "confidence": 0.0, "stop_loss_pct": 0.05, "reason": "Not enough history"}

    # compute indicators
    recent_close = float(df['Close'].iloc[-1])
    rsi_series = rsi(df['Close'], 14)
    rsi_val = float(rsi_series.iloc[-1])
    atr_val = float(atr(df, 14).iloc[-1]) if len(df) >= 14 else (df['Close'].pct_change().std() * df['Close'].iloc[-1])
    vol_spike, vol_score = detect_volume_spike(df, lookback=20, multiplier=2.0)
    ma_score, ma_reason = detect_ma_crossover(df)
    wedge_score, wedge_reason = detect_wedge_like(df)

    # simple rule-based scoring:
    # RSI: if RSI <30 -> bias long; if RSI>70 -> bias short
    rsi_long = max(0.0, (50 - rsi_val) / 50)  # 0 when rsi>=50, 1 when rsi=0
    rsi_short = max(0.0, (rsi_val - 50) / 50)  # 0 when rsi<=50, 1 when rsi=100

    # Volume: favors direction of recent candle
    candle_dir = 1 if df['Close'].iloc[-1] > df['Open'].iloc[-1] else -1
    vol_long = vol_score if candle_dir == 1 else 0.0
    vol_short = vol_score if candle_dir == -1 else 0.0

    # MA: ma_score closer to 1 => long bias; closer to 0 => short bias
    ma_long = ma_score
    ma_short = 1 - ma_score

    # Wedge: high wedge_score indicates an imminent breakout; treat as directional uncertain but amplify whichever signal is stronger
    wedge = wedge_score

    # combine with weights (tweak by timeframe)
    if timeframe == "scalp":
        w = {"rsi": 0.25, "ma": 0.15, "vol": 0.35, "wedge": 0.25}
        threshold = 0.6
        sl_mult = 0.6
    elif timeframe == "day":
        w = {"rsi": 0.25, "ma": 0.30, "vol": 0.2, "wedge": 0.25}
        threshold = 0.55
        sl_mult = 1.0
    else:  # swing
        w = {"rsi": 0.2, "ma": 0.4, "vol": 0.15, "wedge": 0.25}
        threshold = 0.5
        sl_mult = 1.6

    long_score = (w["rsi"] * rsi_long) + (w["ma"] * ma_long) + (w["vol"] * vol_long) + (w["wedge"] * wedge)
    short_score = (w["rsi"] * rsi_short) + (w["ma"] * ma_short) + (w["vol"] * vol_short) + (w["wedge"] * wedge * 0.2)  # wedge slightly favors breakout, not necessarily short

    # normalize 0..1
    ssum = long_score + short_score
    if ssum == 0:
        long_n, short_n = 0.5, 0.5
    else:
        long_n, short_n = long_score/ssum, short_score/ssum

    # confidence: how far from 0.5 the top score is
    top = max(long_n, short_n)
    confidence = float((top - 0.5) * 2)  # 0..1
    # adopt threshold logic for recommendation
    recommended = "HOLD"
    if top >= threshold:
        recommended = "LONG" if long_n > short_n else "SHORT"

    # stop-loss percent: base on ATR relative to price, scaled by timeframe multiplier
    if atr_val is None or atr_val == 0:
        base_pct = 0.03
    else:
        base_pct = max(0.005, min(0.2, (atr_val / recent_close) * 2.0))  # ATR -> percent
    stop_loss_pct = base_pct * sl_mult

    # create a simple reasoning summary
    reason_parts = [
        f"RSI={rsi_val:.1f} (rsi_long={rsi_long:.2f}, rsi_short={rsi_short:.2f})",
        f"MA={ma_reason}",
        f"Volume spike score={vol_score:.2f} (spike={vol_spike})",
        f"Wedge score={wedge_score:.2f} ({wedge_reason})",
        f"Ensemble long={long_n:.2f}, short={short_n:.2f}, confidence={confidence:.2f}"
    ]
    reason = "; ".join(reason_parts)

    return {
        "long": round(float(long_n), 3),
        "short": round(float(short_n), 3),
        "confidence": round(float(confidence), 3),
        "recommended": recommended,
        "stop_loss_pct": float(stop_loss_pct),
        "reason": reason,
        "details": {
            "rsi": rsi_val,
            "atr": atr_val,
            "ma_reason": ma_reason,
            "vol_score": vol_score,
            "wedge_score": wedge_score
        }
    }

# ---------------------------
# OpenAI helper (optional)
# ---------------------------

def openai_explain(symbol: str, timeframe: str, df: pd.DataFrame, ensemble: dict):
    if not OPENAI_KEY:
        return None
    # Build a short prompt summarizing indicators
    prompt_lines = [
        f"Analyze the stock {symbol} for a {timeframe} trade.",
        f"Latest close: {df['Close'].iloc[-1]:.4f} (ATR {ensemble['details']['atr']:.4f}, RSI {ensemble['details']['rsi']:.1f}).",
        f"Ensemble recommendation: {ensemble['recommended']} with numeric scores long={ensemble['long']}, short={ensemble['short']}, confidence={ensemble['confidence']:.2f}.",
        "Provide a concise trading note (2-4 sentences), mention a stop-loss percent from ensemble and a rationale referencing RSI, MA crossover, volume, and wedge."
    ]
    prompt = "\n".join(prompt_lines)
    try:
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role":"system", "content":"You are a professional quantitative trader and explain trading signals concisely."},
                {"role":"user", "content": prompt}
            ],
            "max_tokens": 200,
            "temperature": 0.2
        }
        headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type":"application/json"}
        r = requests.post(url, json=payload, headers=headers, timeout=12)
        if r.ok:
            j = r.json()
            txt = j["choices"][0]["message"]["content"].strip()
            return txt
    except Exception:
        pass
    return None

# ---------------------------
# UI: Main interaction
# ---------------------------

col1, col2 = st.columns([1.2, 2])

with col1:
    st.subheader("Search / Select")
    results = search_tickers(query)
    if len(results) == 0:
        st.info("No results — try a different query or enter a symbol (e.g., AAPL).")
    else:
        for r in results:
            if st.button(r["symbol"] + ((" — " + r["name"]) if r.get("name") else ""), key="btn_"+r["symbol"]):
                selected_symbol = r["symbol"]
                st.session_state["selected"] = selected_symbol

    selected = st.session_state.get("selected", results[0]["symbol"] if results else query.upper())
    st.markdown("---")
    st.write("Selected:", selected)

    # run prediction button
    if st.button("Run full analysis"):
        st.session_state["run_analysis"] = time.time()

with col2:
    st.subheader("Chart & Fundamentals")
    df = fetch_history(selected, period=history_period)
    if df.empty:
        st.warning("No historical data available for this symbol with current API keys.")
    else:
        fig = px.line(df, x="Date", y="Close", title=f"{selected} — Close price")
        st.plotly_chart(fig, use_container_width=True)

    # Show quick fundamentals
    with st.expander("Fundamentals"):
        fund = fetch_fundamentals(selected)
        if fund:
            st.json(fund)
        else:
            st.write("No fundamentals available from configured providers.")

    # Show quote
    quote = fetch_quote_finnhub(selected)
    st.write("Quote (latest):")
    if quote:
        st.json(quote)
    else:
        st.write("Quote endpoint not available (no Finnhub key).")

# Right column: news + prediction
col3, col4 = st.columns([1.6, 1])

with col3:
    st.subheader("News")
    news = fetch_news(selected, limit=6)
    if news:
        for n in news:
            st.markdown(f"- [{n['title']}]({n.get('url','') or '#'}) — *{n.get('source')}*  \n  {n.get('date','')}")
    else:
        st.write("No news available.")

with col4:
    st.subheader("AI Ensemble Prediction")
    # trigger analysis on button or page load
    if st.session_state.get("run_analysis") or True:
        ensemble = ensemble_predict(df, timeframe)
        st.metric("Recommendation", ensemble.get("recommended", "HOLD"))
        st.metric("Confidence", f"{ensemble['confidence']*100:.0f}%")
        st.write("Scores — Long / Short:", f"{ensemble['long']}  /  {ensemble['short']}")
        st.write("Stop-loss suggestion (percent):", f"{ensemble['stop_loss_pct']*100:.2f}%")
        if st.button("Show numeric breakdown"):
            st.json(ensemble)

        # Price stop-loss if quote available
        if df is not None and not df.empty:
            price = float(df['Close'].iloc[-1])
            sl_price = price * (1 - ensemble['stop_loss_pct']) if ensemble['recommended'] == "LONG" else price * (1 + ensemble['stop_loss_pct'])
            st.write("Example entry price:", f"${price:.4f}")
            st.write("Suggested stop price:", f"${sl_price:.4f}")

        # LLM explanation (if key present)
        llm_text = openai_explain(selected, timeframe, df, ensemble)
        if llm_text:
            with st.expander("LLM explanation"):
                st.write(llm_text)
        else:
            st.write("LLM explanation not available (provide OPENAI_API_KEY to enable).")

# Footer
st.markdown("---")
st.markdown("**Disclaimer:** This app provides heuristic/ML-assisted signals for educational use. "
            "Do not trade with real capital without proper testing, risk controls and legal review.")
