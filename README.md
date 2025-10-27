# Day Trdr (demo)

This repository is a demo full-stack app for the *day-trdr* product. It includes a Flask backend (placeholders for Polygon/Finnhub/NewsAPI/OpenAI keys) and a minimal React frontend scaffolded with Vite.

## Quickstart (local)

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
# Fill API keys in .env
python app.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

The frontend expects the backend at `http://localhost:5000`. Set `VITE_API_ROOT` environment variable for the frontend if different.

## License
MIT
