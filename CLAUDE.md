# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

"Why Is The Stock Plummeting?" - A full-stack application that provides humorous explanations for stock price drops, featuring a React + Vite frontend and FastAPI backend with LLM-powered analysis and mean reversion trading signals.

## Development Commands

### Local Development
```bash
# One-shot start both backend and frontend
./start-local.sh

# Or manually:
# Backend (FastAPI on port 8000)
cd backend
python3 -m pip install -r requirements.txt
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Frontend (Vite on port 8080)
cd frontend
npm install
npm run dev
```

### Testing
```bash
# Backend tests
cd backend
pytest -q

# Frontend linting
cd frontend
npm run lint
```

### Building
```bash
# Frontend production build
cd frontend
npm run build
```

## Architecture

### Backend Structure
- **FastAPI application** (`backend/main.py`) - Main API endpoints
- **Services** (`backend/services/`) - Business logic modules:
  - `market_data.py` - Yahoo Finance/Stooq/Polygon integrations
  - `news.py` - DuckDuckGo news search
  - `analysis.py` - LLM prompt building and OpenAI integration
  - `losers.py` - Curated daily losers with LLM ranking
  - `og_image.py` - Social preview image generation
  - `news_scoring.py` - News event classification
  - `jobs.py` - Async job management
- **Database** (`backend/database/`) - SQLite with SQLAlchemy:
  - `models.py` - Database schemas
  - `repositories/` - Data access layer
- **Analysis** (`backend/analysis/`) - Trading signal logic:
  - `oversold_detector.py` - Market microstructure analysis

### Frontend Structure
- **React + Vite** (`frontend/src/`)
- **Components** (`frontend/src/components/`)
  - `BiggestLosers.jsx` - Curated losers grid
  - `Sparkline.jsx` - SVG price charts
- **Configuration** (`frontend/src/config.js`)
  - Development: API at `http://localhost:8000`
  - Production: API at `/api` (Nginx proxy)

## Key API Endpoints

### Public Endpoints
- `POST /analyze` - Get humorous stock drop explanations
- `GET /quote/{symbol}` - Real-time quote data
- `GET /chart/{symbol}` - Historical price data
- `GET /interesting-losers` - Curated daily losers list
- `GET /og-image/{symbol}.png` - Social preview images

### Premium Features (In Development)
- Trading signals based on mean reversion
- SQLite-backed data persistence
- Background job processing for analysis

## Environment Variables

Required for full functionality:
- `OPENAI_API_KEY` - For LLM summaries
- `POLYGON_API_KEY` - For EOD losers data (optional)
- `PUBLIC_WEB_ORIGIN` - Production domain
- `SQLITE_DB_PATH` - Database location (default: `backend/data/stockdrop.db`)

## Data Flow

1. **User Input** → Frontend accepts stock symbols
2. **Analysis Request** → Frontend calls `POST /analyze`
3. **Data Collection** → Backend fetches news + price data
4. **LLM Processing** → OpenAI generates humorous summary
5. **Response** → Frontend displays results with charts

## Caching Strategy

- **In-memory caches**: Quotes (60s), Charts (300s), OG images (expire at midnight ET)
- **On-disk cache**: News articles (optional, date-based expiry)
- **SQLite database**: Price history, signals, features (persistent)

## Testing Approach

- Backend uses pytest with mocked external dependencies
- Key test files: `backend/test_main.py`
- Tests cover API endpoints, data fetching, and caching logic

## Important Notes

- The Cursor rules in `.cursor/rules/architecture.mdc` contain detailed project documentation
- A comprehensive mean reversion trading feature spec is included but not fully implemented
- The system gracefully degrades when API keys are missing
- Background jobs refresh curated losers daily at 5pm ET
- Development uses SQLite; production deployment supports the same