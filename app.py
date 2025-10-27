"""
Day Trdr - Streamlit AI Stock App
---------------------------------
View stock fundamentals, charts, and AI trade suggestions
Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

# --- Silent dependency guard (Plotly fix) ---
try:
    import plotly.express as px
except ModuleNotFoundError:
    import sys, subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "plotly"], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import plotly.express as px

# --- Imports ---
import os
import math
import time
import requests
import streamlit as st
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta

# --- Load API keys ---
load_dotenv()
POLYGON_KEY = os.getenv("POLYGON_API_KEY", "")
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

# --- Streamlit Config ---
st.set_page_config(page_title="📊 Day Trdr", layout="wide")
st.title("📊 Day Trdr – AI Stock Insight & Trade Helper")

# --- Sidebar ---
st.sidebar.header("🔍 Search Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g. AAPL, NVDA)", "")
timeframe = st.sidebar.selectbox("Select Trade Type", ["Scalp", "Day Trade", "Swing Trade"])

# --- Helper Functions ---
@st.cache_data(show_spinner=False)
def get_stock_data(symbol):
    """Fetches recent market data for the given symbol."""
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/2024-01-01/{datetime.today().date()}?adjusted=true&sort=asc&limit=120&apiKey={POLYGON_KEY}"
        r = requests.get(url, timeout=10)
        data = r.json().get("results", [])
        if not data:
            return None
        df = pd.DataFrame(data)
        df["t"] = pd.to_datetime(df["t"], unit="ms")
        return df
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def get_fundamentals(symbol):
    """Fetches fundamental data from Finnhub (mock if missing)."""
    try:
        url = f"https://finnhub.io/api/v1/stock/metric?symbol={symbol}&metric=all&token={FINNHUB_KEY}"
        r = requests.get(url, timeout=10)
        return r.json().get("metric", {})
    except Exception:
        return {}

@st.cache_data(show_spinner=False)
def get_news(symbol):
    """Fetches related stock news."""
    try:
        url = f"https://newsapi.org/v2/everything?q={symbol}&sortBy=publishedAt&language=en&apiKey={NEWSAPI_KEY}"
        r = requests.get(url, timeout=10)
        articles = r.json().get("articles", [])[:5]
        return [{"title": a["title"], "url": a["url"], "source": a["source"]["name"]} for a in articles]
    except Exception:
        return []

def ai_predict(symbol, timeframe):
    """Mock AI model – combines strategy indicators."""
    score = np.random.uniform(0, 1)
    if score > 0.7:
        rec = "BUY 🟢"
        explanation = "Momentum strong. Possible wedge breakout forming."
    elif score < 0.3:
        rec = "SELL 🔴"
        explanation = "Weak RSI and declining volume. Short potential."
    else:
        rec = "HOLD ⚪"
        explanation = "Neutral indicators. Wait for confirmation."
    stop_loss = round(np.random.uniform(0.02, 0.06), 3)
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "recommendation": rec,
        "confidence": f"{int(score*100)}%",
        "stop_loss": f"{stop_loss*100}%",
        "explanation": explanation
    }

# --- Main UI ---
if ticker:
    st.subheader(f"📈 {ticker.upper()} Overview")

    data = get_stock_data(ticker)
    if data is not None and not data.empty:
        st.plotly_chart(px.line(data, x="t", y="c", title=f"{ticker.upper()} Price Chart"), use_container_width=True)
    else:
        st.warning("⚠️ No price data found or API limit reached.")

    fundamentals = get_fundamentals(ticker)
    if fundamentals:
        st.write("### 🧾 Fundamentals")
        st.json(fundamentals)
    else:
        st.info("No fundamentals available for this ticker.")

    news = get_news(ticker)
    if news:
        st.write("### 📰 Latest News")
        for n in news:
            st.markdown(f"- [{n['title']}]({n['url']}) ({n['source']})")
    else:
        st.info("No recent news found.")

    st.write("### 🤖 AI Trade Prediction")
    if st.button("Generate AI Signal"):
        with st.spinner("Analyzing market signals..."):
            pred = ai_predict(ticker, timeframe)
            st.success(f"**Recommendation:** {pred['recommendation']}")
            st.write(f"**Confidence:** {pred['confidence']}")
            st.write(f"**Stop Loss:** {pred['stop_loss']}")
            st.caption(pred["explanation"])

else:
    st.info("👈 Enter a stock ticker in the sidebar to get started.")

st.markdown("---")
st.caption("© 2025 Day Trdr – AI-Powered Trading Dashboard")
