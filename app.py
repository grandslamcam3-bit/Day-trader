"""Day Trdr - Flask backend (skeleton with real API placeholders)
Run: pip install -r requirements.txt ; python app.py
"""
import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import requests
load_dotenv()

POLYGON_KEY = os.getenv('POLYGON_API_KEY', 'YOUR_POLYGON_KEY')
FINNHUB_KEY = os.getenv('FINNHUB_API_KEY', 'YOUR_FINNHUB_KEY')
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', 'YOUR_NEWSAPI_KEY')
OPENAI_KEY = os.getenv('OPENAI_API_KEY', '')

app = Flask(__name__)

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/api/search')
def search():
    q = request.args.get('q','').strip()
    if not q:
        return jsonify([])
    # Try Polygon lookup if key provided
    if POLYGON_KEY and POLYGON_KEY != 'YOUR_POLYGON_KEY':
        try:
            url = 'https://api.polygon.io/v3/reference/tickers'
            r = requests.get(url, params={'search': q, 'active': 'true', 'limit': 10, 'apiKey': POLYGON_KEY}, timeout=8)
            if r.ok:
                res = r.json().get('results', [])
                out = [{'symbol': r.get('ticker'), 'name': r.get('name')} for r in res]
                if out:
                    return jsonify(out)
        except Exception as e:
            pass
    # Fallback: simple echo
    return jsonify([{'symbol': q.upper(), 'name': q}])

@app.route('/api/quote')
def quote():
    t = request.args.get('t','').upper()
    if not t:
        return jsonify({'error': 'missing ticker'}), 400
    # Implement real provider calls here (Polygon/Finnhub)
    return jsonify({'ticker': t, 'close': None, 'open': None, 'high': None, 'low': None, 'volume': None})

@app.route('/api/fundamentals')
def fundamentals():
    t = request.args.get('t','').upper()
    if not t:
        return jsonify({'error': 'missing ticker'}), 400
    # Implement Polygon or Finnhub fundamentals fetch here
    demo = {'ticker': t, 'epsTTM': None, 'pe': None, 'marketCap': None, 'balanceSheet': {}}
    return jsonify(demo)

@app.route('/api/news')
def news():
    t = request.args.get('t','').upper()
    return jsonify([{'title': f'Demo news for {t}', 'source':'Demo','url':''}])

@app.route('/api/predict', methods=['POST'])
def predict():
    body = request.json or {}
    t = (body.get('ticker') or '').upper()
    timeframe = body.get('timeframe','day')
    # In production, gather OHLC + features, run strategy modules and ensemble + optional LLM
    # Here we return a mocked prediction structure with placeholders
    if not t:
        return jsonify({'error':'ticker required'}), 400
    return jsonify({
        'ticker': t,
        'timeframe': timeframe,
        'recommendation': 'HOLD',
        'scores': {'wedge':0.4,'rsi':0.5,'ma':0.45,'volume_spike':0.2},
        'confidence': '45%',
        'stop_loss_percent': 0.03,
        'explanation': 'Mocked ensemble. Replace with real model & APIs.'
    })

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=False,        # ❌ turn off reloader
        use_reloader=False  # ✅ required in Streamlit / threads
    )
