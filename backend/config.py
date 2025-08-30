"""Central configuration settings for the application."""

import os
from dotenv import load_dotenv

# Load env vars from a local .env if present (dev convenience)
load_dotenv()

# Cache TTL settings
QUOTE_CACHE_TTL_SECONDS = int(os.getenv("QUOTE_CACHE_TTL_SECONDS", "60"))
CHART_CACHE_TTL_SECONDS = int(os.getenv("CHART_CACHE_TTL_SECONDS", "300"))
OG_IMAGE_CACHE_TTL_SECONDS = int(os.getenv("OG_IMAGE_CACHE_TTL_SECONDS", "1800"))
INTERESTING_LOSERS_CACHE_TTL_SECONDS = int(os.getenv("INTERESTING_LOSERS_CACHE_TTL_SECONDS", "86400"))
NEWS_CACHE_TTL_SECONDS = int(os.getenv("NEWS_CACHE_TTL_SECONDS", "604800"))

# Oversold scan settings
OVERSOLD_SCAN_CACHE_TTL_SECONDS = int(os.getenv("OVERSOLD_SCAN_CACHE_TTL_SECONDS", "300"))
OVERSOLD_SCHEDULE_ENABLED = os.getenv("OVERSOLD_SCHEDULE_ENABLED", "1") not in {"0", "false", "False"}
OVERSOLD_UNIVERSE = os.getenv("OVERSOLD_UNIVERSE", "AAPL TSLA NVDA AMD AMZN MSFT META GOOGL NFLX JPM BA").split()

# Application flags
SKIP_CACHE_PREPOPULATION = os.getenv("SKIP_CACHE_PREPOPULATION", "0") == "1"

# Public web origin for CORS and social sharing
PUBLIC_WEB_ORIGIN = os.getenv("PUBLIC_WEB_ORIGIN")

# App started flag
_app_started = False
