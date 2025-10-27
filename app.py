"""
Day Trdr - Streamlit app (real data + AI explanations)

Run locally:
    pip install streamlit requests python-dotenv pandas numpy plotly
    streamlit run daytrdr_app.py

Environment variables (create .env or set in Streamlit Cloud):
    POLYGON_API_KEY=...
    FINNHUB_API_KEY=...
    NEWSAPI_KEY=...
    OPENAI_API_KEY=...   # optional (AI explanations)
"""

import os, math, time, requests, numpy as np, pandas as pd
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime, timedelta

# --- Silent install fallback for plotly ---
try:
    import plotly.express as px
except ModuleNotFoundError:
    import sys, subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "plotly"], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import plotly.express as px

# --------------------------
# Config / Environment
# --------------------------
load_dotenv()
POLYGON_KEY = os.getenv("POLYGON_API_KEY", "")
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

st.set_page_config(page_title="Day Trdr", layout="wide")
st.title("📈 Day Trdr — Market data + AI trade guidance")
st.caption("Real-time data, fundamentals, and optional AI explanations (OpenAI API).")

# --------------------------
# Sidebar
# --------------------------
st.sidebar.header("🔍 Search & Settings")
query = st.sidebar.text_input("Ticker or company", value="AAPL")
timeframe = st.sidebar.selectbox("Prediction timeframe", ["scalp", "day", "swing"])
history_period = st.sidebar.selectbox("History period", ["7d", "1mo", "3mo", "1y"], index=1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Sources**")
st.sidebar.text(f"Polygon: {'✅' if POLYGON_KEY else '❌'}")
st.sidebar.text(f"Finnhub: {'✅' if FINNHUB_KEY else '❌'}")
st.sidebar.text(f"NewsAPI: {'✅' if NEWSAPI_KEY else '❌'}")
st.sidebar.text(f"OpenAI: {'✅' if OPENAI_KEY else '❌'}")
st.sidebar.markdown("---")
st.sidebar.caption("⚠️ For educational purposes only — not financial advice.")

# --------------------------
# Helper: API Functions
# --------------------------
@st.cache_data(ttl=60)
def search_tickers(q):
    q = q.strip().upper()
    if not q:
        return []
    try:
        if POLYGON_KEY:
            r = requests.get(
                "https://api.polygon.io/v3/reference/tickers",
                params={"search": q, "active": "true", "limit": 8, "apiKey": POLYGON_KEY},
                timeout=8,
            )
            if r.ok:
                results = r.json().get("results", [])
                return [{"symbol": i["ticker"], "name": i.get("name", "")} for i in results]
    except Exception:
        pass
    try:
        if FINNHUB_KEY:
            r = requests.get(
                "https://finnhub.io/api/v1/search",
                params={"q": q, "token": FINNHUB_KEY},
                timeout=8,
            )
            if r.ok:
                results = r.json().get("result", [])[:8]
                return [{"symbol": i["symbol"], "name": i.get("description", "")} for i in results]
    except Exception:
        pass
    return [{"symbol": q, "name": ""}]

@st.cache_data(ttl=120)
def fetch_history(symbol, period="1mo"):
    now = datetime.utcnow()
    start = now - timedelta(days={"7d":7,"1mo":30,"3mo":90,"1y":365}.get(period,30))
    try:
        if POLYGON_KEY:
            r = requests.get(
                f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start.date()}/{now.date()}",
                params={"adjusted":"true","limit":300,"sort":"asc","apiKey":POLYGON_KEY},
                timeout=10,
            )
            if r.ok:
                j = r.json()
                data = j.get("results", [])
                if data:
                    df = pd.DataFrame(data)
                    df["t"] = pd.to_datetime(df["t"], unit="ms")
                    df.rename(columns={"t":"Date","o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"}, inplace=True)
                    return df[["Date","Open","High","Low","Close","Volume"]]
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=120)
def fetch_news(symbol):
    news = []
    try:
        if NEWSAPI_KEY:
            r = requests.get(
                "https://newsapi.org/v2/everything",
                params={"q": symbol, "pageSize": 6, "sortBy":"publishedAt","apiKey":NEWSAPI_KEY},
                timeout=8,
            )
            if r.ok:
                for n in r.json().get("articles", []):
                    news.append({
                        "title": n["title"], 
                        "url": n["url"],
                        "source": n["source"]["name"],
                        "date": n["publishedAt"][:10],
                    })
    except Exception:
        pass
    return news

# --------------------------
# Indicator Calculations
# --------------------------
def sma(series, w): return series.rolling(w).mean()
def rsi(series, w=14):
    delta = series.diff().dropna()
    up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
    ema_up, ema_down = up.ewm(com=w-1, adjust=False).mean(), down.ewm(com=w-1, adjust=False).mean()
    rs = ema_up / (ema_down + 1e-9)
    r = 100 - (100 / (1 + rs))
    return r.reindex(series.index, method='pad').fillna(50)

def ensemble_predict(df, timeframe):
    if df.empty or len(df) < 10:
        return {"recommended": "HOLD", "confidence": 0, "stop_loss": 0.05}
    recent = df["Close"].iloc[-1]
    rsi_val = rsi(df["Close"]).iloc[-1]
    bias = "LONG" if rsi_val < 40 else "SHORT" if rsi_val > 60 else "HOLD"
    confidence = abs(50 - rsi_val) / 50
    stop_loss = 0.03 if timeframe=="scalp" else 0.05 if timeframe=="day" else 0.08
    return {"recommended": bias, "confidence": confidence, "stop_loss": stop_loss, "rsi": rsi_val, "price": recent}

# --------------------------
# Main UI
# --------------------------
col1, col2 = st.columns([1.2, 2])

with col1:
    st.subheader("Symbol Search")
    results = search_tickers(query)
    for r in results:
        if st.button(r["symbol"], key=f"sym_{r['symbol']}"):
            st.session_state["selected"] = r["symbol"]
    selected = st.session_state.get("selected", results[0]["symbol"] if results else query.upper())
    st.markdown(f"**Selected:** {selected}")
    if st.button("Run Analysis"):
        st.session_state["run_analysis"] = time.time()

with col2:
    st.subheader("Chart")
    df = fetch_history(selected, period=history_period)
    if not df.empty:
        st.plotly_chart(px.line(df, x="Date", y="Close", title=f"{selected} Close Price"), use_container_width=True)
    else:
        st.warning("No historical data available for this ticker.")

# Prediction
st.subheader("🤖 AI Trade Suggestion")
if df is not None and not df.empty:
    pred = ensemble_predict(df, timeframe)
    st.metric("Recommendation", pred["recommended"])
    st.metric("Confidence", f"{pred['confidence']*100:.0f}%")
    st.write(f"RSI: {pred['rsi']:.1f}")
    st.write(f"Suggested Stop Loss: {pred['stop_loss']*100:.1f}%")
else:
    st.info("Enter a valid ticker to get started.")

# News
st.subheader("📰 Latest News")
for n in fetch_news(selected):
    st.markdown(f"- [{n['title']}]({n['url']}) — *{n['source']}* ({n['date']})")

# Footer
st.markdown("---")
st.caption("© 2025 Day Trdr — Educational use only. Not financial advice.")

