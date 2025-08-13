## Why Is The Stock Plummeting? — Project Documentation

A full‑stack app that explains why a stock is dropping, with a touch of humor. The frontend is a React + Vite single‑page app. The backend is a FastAPI service that fetches headlines, composes a witty analysis using an LLM, serves lightweight quote/chart data, generates social preview images, and provides a curated list of the day’s most “interesting” losers.

### High‑Level Architecture
- **Frontend (React + Vite + Tailwind)**: SPA running on `vite` in dev and served via Nginx in prod. Talks to the backend via REST.
- **Backend (FastAPI)**: Provides endpoints for analysis, quotes, charts, curated losers, OG images, and social share pages. Employs caching to keep responses fast and API usage modest.
- **Data Sources**:
  - Headlines via `duckduckgo-search` (news API proxy) with optional on‑disk caching
  - Daily OHLCV via Stooq (free) with Polygon.io as fallback where applicable
  - LLM summaries via OpenAI (graceful offline fallback if no key)

### Project Structure
```text
stock-drop/
  backend/           # FastAPI app and tests
    main.py
    requirements.txt
    test_main.py
  frontend/          # React app (Vite)
    src/
      App.jsx
      components/
        BiggestLosers.jsx
        Header.jsx
        Sparkline.jsx
        StatusBar.jsx
      config.js      # API base URL selection
    vite.config.js   # Dev server on port 8080
    nginx.conf       # Example Nginx site config (container style)
  scripts/
    setup-vps.sh     # One‑shot VPS setup (systemd + Nginx)
  start-local.sh     # Dev helper to start backend + frontend together
```

### How It Works (Data Flow)
1. User enters one or more stock symbols in the UI.
2. Frontend calls `POST /analyze` to get a short, humorous explanation per symbol. In parallel, the UI warms OG images and then fetches quote and chart snapshots.
3. Backend:
   - Searches for recent headlines (`duckduckgo-search`) for each symbol (with optional on‑disk cache).
   - Builds an LLM prompt including recent price context (from quote/chart helpers).
   - Calls OpenAI to produce a concise summary. If no OpenAI key, returns a short “offline snark” fallback.
4. Frontend also calls `GET /quote/{symbol}` and `GET /chart/{symbol}` to display lightweight market context.
5. When there are no results yet, frontend shows `GET /interesting-losers` — a curated EOD list updated daily by a background task in the backend, ranked by an LLM using minimal headline context.
6. For social previews, the backend can generate an OG image `GET /og-image/{symbol}.png` and a share redirect page `GET /s/{symbol}` with proper OpenGraph/Twitter tags.

### Backend API
Base URL is `http://localhost:8000` in dev (or `/api` in production behind Nginx).

- `POST /analyze`
  - Request: `{ "symbols": "AAPL TSLA", "days": 7, "max_results": 8, "tone": "humorous" }`
  - Response: `{ "results": [{ "symbol": "AAPL", "summary": string, "sources": [{ title, url, snippet?, published? }] }] }`
  - Behavior: Finds recent headlines, builds an LLM prompt with price context, returns a concise summary per symbol. Falls back to a generic summary if no OpenAI key.

- `GET /quote/{symbol}`
  - Response: `{ symbol, price?, change?, change_percent?, currency?, market_time?, market_state?, name? }`
  - Source: Prefers Stooq recent daily OHLCV; caches results in‑memory (TTL configurable).

- `GET /chart/{symbol}?range=1mo&interval=1d`
  - Response: `{ symbol, range, interval, timestamps[], opens[], highs[], lows[], closes[], volumes[] }`
  - Source: Prefers Stooq; returns a compact historical series suitable for sparklines; caches in‑memory.

- `GET /interesting-losers?candidates=300&top=12`
  - Response: `{ losers: [{ symbol, price?, change?, change_percent?, volume?, reason? }], last_updated, session }`
  - Behavior: Returns a curated list ranked by an LLM from an EOD losers set computed from Polygon grouped data. Served from cache only; background refresh warms daily. If no Polygon key is set, will log and return an empty list until configured.

- `GET /og-image/{symbol}.png`
  - Returns: PNG bytes with on‑brand OG preview (symbol, price/change, sparkline, succinct commentary). Cached in‑memory by ETag and TTL.

- `POST /og-image/warm/{symbol}`
  - Pre‑generates the OG image and caches it. Returns 201 for new render, 204 if fresh cached.

- `GET /s/{symbol}`
  - Returns: Minimal HTML with OpenGraph/Twitter meta tags for link unfurling, then refresh‑redirects users to the SPA. Uses `PUBLIC_WEB_ORIGIN` to build canonical URLs.

- `GET /`
  - Health/identity endpoint: `{ ok: true, service: "whyisthestockplummeting" }`

### Frontend Behavior
- `App.jsx`
  - Accepts symbols, triggers `POST /analyze` and displays each result (summary + sources)
  - In parallel, warms OG images and fetches `GET /quote/{sym}` + `GET /chart/{sym}` for context
  - Shows `BiggestLosers` grid (from `/interesting-losers`) when no results yet
- `BiggestLosers.jsx`
  - Fetches curated list on mount; refreshes hourly
  - Clicking a ticker auto‑fills the input and starts analysis
- `Sparkline.jsx`
  - Simple SVG sparkline from `ChartResponse.closes`
- `config.js`
  - `API_BASE_URL` is `http://localhost:8000` in dev and `/api` in production
- Dev server runs on port `8080` (see `vite.config.js`)

### Caching & Background Jobs
- In‑memory caches:
  - Quotes: `QUOTE_CACHE_TTL_SECONDS` (default 60s)
  - Charts: `CHART_CACHE_TTL_SECONDS` (default 300s)
  - OG images: `OG_IMAGE_CACHE_TTL_SECONDS` (default 1800s)
  - Curated losers: `INTERESTING_LOSERS_CACHE_TTL_SECONDS` (default 86400s)
- News on‑disk cache (optional): `NEWS_CACHE_DIR` with `NEWS_CACHE_TTL_SECONDS` (default 7 days)
- On startup, a background thread warms and periodically refreshes the curated losers cache.

### Configuration (Environment Variables)
- `OPENAI_API_KEY` (required for LLM summaries; if missing, backend returns a generic fallback)
- `OPENAI_MODEL` (default: `gpt-4o-mini`)
- `POLYGON_API_KEY` (optional; enables EOD losers via grouped data and daily agg fallback)
- `PUBLIC_WEB_ORIGIN` (e.g., `https://whyisthestockplummeting.com`; used in CORS and share pages)
- `OG_FONT_PATH` (optional path to a TTF font for OG rendering; scripts set a good default on Linux)
- `NEWS_CACHE_DIR` (default: `backend/.cache/news`)
- `NEWS_CACHE_TTL_SECONDS` (default: `604800`)
- `QUOTE_CACHE_TTL_SECONDS` (default: `60`)
- `CHART_CACHE_TTL_SECONDS` (default: `300`)
- `OG_IMAGE_CACHE_TTL_SECONDS` (default: `1800`)
- `INTERESTING_LOSERS_CACHE_TTL_SECONDS` (default: `86400`)

### Local Development
Option A: one‑shot helper script
```bash
./start-local.sh
```
- Starts FastAPI on `http://localhost:8000` and Vite on `http://localhost:8080`.
- Reads environment from a top‑level `.env` if present.

Option B: manual
```bash
# Backend
cd backend
python3 -m pip install -r requirements.txt
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

### Deployment (Ubuntu/Debian VPS without Docker)
Use the provided automation to install system packages, build the frontend, configure Nginx, and run FastAPI under systemd.
```bash
sudo bash scripts/setup-vps.sh <domain> [letsencrypt-email]
```
Afterwards:
- Edit `/etc/whyisthestockplummeting.env` to set a real `OPENAI_API_KEY` (and optionally `POLYGON_API_KEY`).
- `sudo systemctl restart whyisthestockplummeting-backend`
- Visit `http://<domain>:8080`

Notes:
- The Nginx site listens on port 8080 by default in the script. TLS via Let’s Encrypt is supported.
- In production, the SPA calls the backend via `/api` and Nginx proxies to `127.0.0.1:8000`.

### Testing
Backend tests are in `backend/test_main.py` and mock external dependencies where needed.
```bash
cd backend
python3 -m pip install -r requirements.txt
pytest -q
```

### Error Handling and Resilience
- LLM calls are wrapped; on failure or missing key, a concise fallback is returned.
- News and polygon fetchers log errors and fail softly; caches prevent hot paths from blocking.
- OG image generation uses a scalable TTF if available; falls back to PIL’s default font.

### Security & Rate‑Limiting Considerations
- The API is open by default; consider adding auth or rate limiting for public deployments.
- The curated losers endpoint is cache‑only to ensure expensive ranking work never happens on the request path.

### Legal/Content Notice
This project provides opinionated summaries of market news for convenience only. No investment advice. Verify information with original sources.


