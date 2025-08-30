"""Service for generating OpenGraph preview images for social media sharing."""

import os
from io import BytesIO
from datetime import datetime, timedelta
from typing import List, Dict
from zoneinfo import ZoneInfo
from PIL import Image, ImageDraw, ImageFont

from services.market_data import (
    _fetch_stooq_quote_only, _fetch_stooq_chart_only, _get_price_context_stooq_only,
)
from services.news import _search_news_for_symbol
from services.analysis import _build_llm_prompt, _call_openai


# OG images use date-based cache keys, expire at midnight ET
OG_IMAGE_CACHE: Dict[str, bytes] = {}


def _clean_text(text: str) -> str:
    """ASCII-safe text"""
    text = text.replace("\u2014", "-").replace("\u2013", "-")
    return text.encode("ascii", "ignore").decode("ascii")


def _load_font(size: int):
    """Load a scalable TrueType font so size changes actually apply.

    Falls back through common system font paths and an OG_FONT_PATH env var.
    If none are available, uses PIL's default bitmap font (fixed size).
    """
    # Allow override via env var
    env_font = os.getenv("OG_FONT_PATH")
    candidates = [env_font] if env_font else []
    # Common locations
    candidates.extend([
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        # Common on Ubuntu minimal images
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    ])
    for path in candidates:
        try:
            if path and os.path.exists(path) or path and not os.path.isabs(path):
                return ImageFont.truetype(path, size)
        except Exception:
            continue
    # Try PIL's bundled DejaVu
    try:
        import PIL  # type: ignore
        pil_dir = os.path.dirname(PIL.__file__)
        bundled = os.path.join(pil_dir, "fonts", "DejaVuSans.ttf")
        return ImageFont.truetype(bundled, size)
    except Exception:
        pass
    # Last resort (fixed size)
    return ImageFont.load_default()


def _get_full_commentary(symbol: str) -> str:
    """Get actual analysis like the site shows"""
    try:
        # Try to get actual analysis from the backend
        sources = _search_news_for_symbol(symbol, days=7, max_results=8)
        price_ctx = _get_price_context_stooq_only(symbol)
        # Keep OG image preview succinct
        prompt = _build_llm_prompt(symbol, sources, "humorous", price_ctx, max_words=80)
        analysis = _call_openai(prompt)
        if analysis and len(analysis.strip()) > 20:
            return analysis.strip()
    except Exception:
        pass
    return f"The market's been rough on {symbol} lately. Between earnings volatility, analyst downgrades, and general market jitters, it's showing the classic signs of a stock in consolidation mode. Recent price action suggests investors are taking profits and reassessing valuations amid broader market uncertainty."


def _trim_words(text: str, max_words: int) -> str:
    """Safety net: trim to max words to avoid awkward mid-word cuts"""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "…"


def _generate_og_image_png(symbol: str) -> bytes:
    # Match the actual site's look - dark background, detailed commentary
    width, height = 1200, 630
    
    quote = _fetch_stooq_quote_only(symbol)
    chart = _fetch_stooq_chart_only(symbol, range_="5d", interval="1d")
    closes = [c for c in (chart.closes or []) if isinstance(c, (int, float))]

    # Site-matching colors
    bg = (15, 18, 24)  # Dark blue-gray like the site
    white = (255, 255, 255)
    gray = (156, 163, 175)
    red = (248, 113, 113)  # Softer red like site
    green = (34, 197, 94)
    
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    # Font setup
    # Slightly larger fonts so preview text is easier to read
    big_font = _load_font(56)
    med_font = _load_font(36)
    small_font = _load_font(28)
    tiny_font = _load_font(20)

    # Layout: symbol + price on top row, chart on right, commentary filling left
    # Top row: Symbol and price
    padding = 40
    y = 40
    
    # Symbol
    draw.text((padding, y), _clean_text(symbol), font=big_font, fill=white)
    
    # Price on same line, right side
    if quote.price is not None:
        chg = quote.change or 0.0
        chg_pct = quote.change_percent or 0.0
        price_color = red if chg < 0 else green
        price_text = f"${quote.price:.2f}  1d {chg:+.2f} ({chg_pct:+.2f}%)"
        # Measure text to position it right-aligned
        bbox = draw.textbbox((0, 0), _clean_text(price_text), font=med_font)
        price_width = bbox[2] - bbox[0]
        right_x = width - padding
        draw.text((right_x - price_width, y + 5), _clean_text(price_text), font=med_font, fill=price_color)
        # Second line: change percentage matching the chart range
        try:
            if len(closes) >= 2 and closes[0] not in (None, 0):
                m1_pct = ((closes[-1] - closes[0]) / closes[0]) * 100.0
                m1_text = f"{chart.range} {m1_pct:+.2f}%"
                m1_color = red if m1_pct < 0 else green
                m1_bbox = draw.textbbox((0, 0), _clean_text(m1_text), font=small_font)
                m1_width = m1_bbox[2] - m1_bbox[0]
                # Slightly larger offset so it doesn't crowd the top labels
                draw.text((right_x - m1_width, y + 5 + 48), _clean_text(m1_text), font=small_font, fill=m1_color)
        except Exception:
            pass
    
    # Chart area - compact, top right
    # Move chart down to avoid overlapping with the top-right price lines
    chart_y = y + 130
    chart_height = 180
    chart_width = 400
    chart_left = width - padding - chart_width
    chart_right = width - padding
    chart_top = chart_y
    chart_bottom = chart_y + chart_height
    
    if len(closes) >= 2:
        # Sparkline similar to site
        min_close = min(closes)
        max_close = max(closes)
        price_range = max_close - min_close if max_close != min_close else 1
        
        points = []
        for i, price in enumerate(closes):
            x = chart_left + (i / (len(closes) - 1)) * chart_width
            y_norm = (price - min_close) / price_range
            y = chart_bottom - (y_norm * chart_height)
            points.append((x, y))
        
        # Chart line - match site style
        line_color = red if closes[-1] < closes[0] else green
        draw.line(points, fill=line_color, width=3)
        
        # Range label and High/Low labels like site
        draw.text((chart_right - 100, chart_top - 45), f"Range: {chart.range}", font=tiny_font, fill=gray)
        draw.text((chart_right - 100, chart_top - 25), f"High: ${max_close:.2f}", font=tiny_font, fill=gray)
        draw.text((chart_right - 100, chart_bottom + 5), f"Low: ${min_close:.2f}", font=tiny_font, fill=gray)
    else:
        print(f"⚠️ OG chart missing series for {symbol}; history points: {len(closes)}")
    
    # Commentary - get actual analysis like the site shows
    commentary = _clean_text(_get_full_commentary(symbol))
    commentary = _trim_words(commentary, 85)
    
    # Commentary area - left side, flowing around chart
    text_left = padding
    text_right = chart_left - 20  # Leave gap before chart
    text_top = chart_y
    text_width = text_right - text_left
    
    # Wrap text carefully
    wrapped_lines = []
    words = commentary.split()
    current_line = ""
    
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        bbox = draw.textbbox((0, 0), test_line, font=small_font)
        line_width = bbox[2] - bbox[0]
        
        if line_width <= text_width:
            current_line = test_line
        else:
            if current_line:
                wrapped_lines.append(current_line)
            current_line = word
    
    if current_line:
        wrapped_lines.append(current_line)
    
    # Draw commentary lines with dynamic height based on font metrics
    bbox_sample = draw.textbbox((0, 0), "Ag", font=small_font)
    line_height = (bbox_sample[3] - bbox_sample[1]) + 8
    y = text_top
    for line in wrapped_lines:
        if y > height - 60:  # Don't go too close to bottom
            break
        draw.text((text_left, y), line, font=small_font, fill=white)
        y += line_height
    
    # Footer
    draw.text((padding, height - 30), "whyisthestockplummeting.com", font=tiny_font, fill=gray)
    
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _build_share_description(symbol: str) -> str:
    parts: List[str] = []
    price_ctx = _get_price_context_stooq_only(symbol)
    if price_ctx:
        parts.append(price_ctx)
    # Add up to two recent headlines for flavor
    try:
        news = _search_news_for_symbol(symbol, days=7, max_results=3)
        if news:
            titles = [n.title for n in news[:2] if n.title]
            if titles:
                parts.append("; ".join(titles))
    except Exception:
        pass
    if not parts:
        parts.append("Tap to see a quick chart and summary.")
    # Keep description short for social previews
    desc = " — ".join(parts)
    return desc[:280]


def cleanup_old_og_cache():
    """Remove OG image cache entries older than 2 days."""
    try:
        cutoff_date = (datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=2)).isoformat()
        keys_to_remove = []
        for key in list(OG_IMAGE_CACHE.keys()):
            # Key format: SYMBOL_YYYY-MM-DD
            if "_" in key:
                date_part = key.split("_")[-1]
                if date_part < cutoff_date:
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del OG_IMAGE_CACHE[key]
        
        if keys_to_remove:
            print(f"🧹 Cleaned {len(keys_to_remove)} old OG cache entries")
    except Exception as e:
        print(f"OG cache cleanup error: {e}")
