"""
Day Trdr - Streamlit frontend (converted from Flask backend)
Run: pip install streamlit requests python-dotenv
Then: streamlit run daytrdr_app.py
"""

import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# --- API KEYS ---
POLYGON_KEY = os.getenv('POLYGON_API_KEY', 'YOUR_POLYGON_KEY')
FINNHUB_KEY = os.getenv('FINNHUB_API_KEY', 'YOUR_FINNHUB_KEY')
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', 'YOUR_NEWSAPI_KEY')
OPENAI_KEY = os.getenv('OPENAI_API_KEY', '')

st.set_page_config(page_title="DayTrdr", layout="wide")
st.title("📊 Day Trdr - Mock Trading Intelligence")

st.sidebar.header("🔍 Search Settings")
query = st.sidebar.text_input("Search for a stock ticker", "")

# --------------------
# Helper functions
# --------------------

@st.cache_data(show_spinner=False)
def search_ticker(q: str):
    if not q.strip():
        return []

    if POLYGON_KEY and POLYGON_KEY != 'YOUR_POLYGON_KEY':
        try:
            url = 'https://api.polygon.io/v3/reference/tickers'
            r = requests.get(url, params={
                'search': q, 'active': 'true', 'limit': 10, 'apiKey': POLYGON_KEY
            }, timeout=8)
            if r.ok:
                res = r.json().get('results', [])
                return [{'symbol': r.get('ticker'), 'name': r.get('name')} for r in res]
        except Exception:
            pass

    # fallback
    return [{'symbol': q.upper(), 'name': q}]

def get_quote(ticker: str):
    if not ticker:
        return {'error': 'missing ticker'}
    # placeholder
    return {'ticker': ticker, 'close': None, 'open': None, 'high': None, 'low': None, 'volume': None}

def get_fundamentals(ticker: str):
    if not ticker:
        return {'error': 'missing ticker'}
    return {'ticker': ticker, 'epsTTM': None, 'pe': None, 'marketCap': None, 'balanceSheet': {}}

def get_news(ticker: str):
    return [{'title': f'Demo news for {ticker}', 'source': 'Demo', 'url': ''}]

def get_prediction(ticker: str, timeframe: str):
    if not ticker:
        return {'error': 'ticker required'}
    return {
        'ticker': ticker,
        'timeframe': timeframe,
        'recommendation': 'HOLD',
        'scores': {'wedge': 0.4, 'rsi': 0.5, 'ma': 0.45, 'volume_spike': 0.2},
        'confidence': '45%',
        'stop_loss_percent': 0.03,
        'explanation': 'Mocked ensemble. Replace with real model & APIs.'
    }

# --------------------
# Streamlit UI
# --------------------

if query:
    st.subheader(f"Search Results for '{query}'")
    results = search_ticker(query)
    for res in results:
        with st.expander(f"{res['symbol']} - {res['name']}"):
            st.write("### 📈 Quote")
            quote = get_quote(res["symbol"])
            st.json(quote)

            st.write("### 🧾 Fundamentals")
            fund = get_fundamentals(res["symbol"])
            st.json(fund)

            st.write("### 📰 News")
            news_items = get_news(res["symbol"])
            for n in news_items:
                st.write(f"- [{n['title']}]({n['url'] or '#'}) ({n['source']})")

            st.write("### 🤖 Prediction")
            tf = st.selectbox(f"Timeframe for {res['symbol']}", ["day", "week", "month"], key=f"tf_{res['symbol']}")
            if st.button(f"Run Prediction for {res['symbol']}", key=f"pred_{res['symbol']}"):
                pred = get_prediction(res["symbol"], tf)
                st.json(pred)
else:
    st.info("👈 Enter a ticker name or symbol in the sidebar to start.")

st.markdown("---")
st.caption("Day Trdr © 2025 – Mock backend adapted to Streamlit UI")

