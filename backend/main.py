from datetime import datetime
import os
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import OpenAI
from ddgs import DDGS
from dotenv import load_dotenv

# Load env vars from a local .env if present (dev convenience)
load_dotenv()


app = FastAPI(
    title="Why Is The Stock Plummeting?",
    description="Searches the web and uses an LLM to explain (humorously) why your favorite stock face-planted.",
    version="0.1.0",
)


allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]
public_origin = os.getenv("PUBLIC_WEB_ORIGIN")
if public_origin:
    allowed_origins.append(public_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    symbols: Union[str, List[str]]
    days: int = 7
    max_results: int = 8
    tone: str = "humorous"


class SourceItem(BaseModel):
    title: str
    url: str
    snippet: Optional[str] = None
    published: Optional[str] = None


class SymbolAnalysis(BaseModel):
    symbol: str
    summary: str
    sources: List[SourceItem]


class AnalyzeResponse(BaseModel):
    results: List[SymbolAnalysis]


def _ensure_list_symbols(input_symbols: Union[str, List[str]]) -> List[str]:
    if isinstance(input_symbols, list):
        return [s.strip().upper() for s in input_symbols if s and s.strip()]
    # Comma or whitespace separated
    separators = [",", " "]
    symbols: List[str] = []
    current = input_symbols
    for sep in separators:
        if sep in current:
            parts = [p for p in current.split(sep)]
            symbols = [p.strip().upper() for p in parts if p and p.strip()]
            break
    if not symbols:
        symbols = [current.strip().upper()] if current.strip() else []
    # Deduplicate, preserve order
    seen = set()
    ordered: List[str] = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered


def _search_news_for_symbol(symbol: str, days: int, max_results: int) -> List[SourceItem]:
    # Keep the query broad so we don't accidentally filter out relevant articles
    query = f"{symbol} stock"
    timelimit = f"d{max(1, min(days, 30))}"
    items: List[SourceItem] = []
    # Simple retry loop to handle transient rate limits
    attempts = 0
    while attempts < 3 and len(items) == 0:
        attempts += 1
        try:
            with DDGS() as ddgs:
                # ddgs.news expects the query as the first positional argument (or named as 'query')
                for n in ddgs.news(query, region="us-en", safesearch="moderate", timelimit=timelimit, max_results=max_results):
                    items.append(
                        SourceItem(
                            title=n.get("title") or "",
                            url=n.get("url") or n.get("link") or "",
                            snippet=n.get("excerpt") or n.get("body"),
                            published=n.get("date"),
                        )
                    )
        except Exception as e:
            print(f"DDG news search attempt {attempts} failed for {symbol}: {e}")
            # brief backoff
            import time
            time.sleep(0.8 * attempts)
    cleaned = [i for i in items if i.url]
    if cleaned:
        return cleaned

    # Fallback: Google News RSS
    try:
        print(f"DDG returned no items for {symbol}. Falling back to Google News RSS…")
        from urllib.parse import quote_plus
        import requests
        import re
        import html as _html
        import xml.etree.ElementTree as ET

        # Add recent time window using when:{days}d
        rss_query = quote_plus(f"{symbol} stock when:{max(1, min(days, 30))}d")
        url = f"https://news.google.com/rss/search?q={rss_query}&hl=en-US&gl=US&ceid=US:en"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return []
        root = ET.fromstring(resp.text)
        channel = root.find("channel")
        if channel is None:
            return []
        results: List[SourceItem] = []
        for item in channel.findall("item")[:max_results]:
            title_el = item.find("title")
            link_el = item.find("link")
            pub_el = item.find("pubDate")
            desc_el = item.find("description")
            title = title_el.text if title_el is not None else ""
            link = link_el.text if link_el is not None else ""
            published = pub_el.text if pub_el is not None else None
            snippet_html = desc_el.text if desc_el is not None else None
            snippet = None
            if snippet_html:
                # strip HTML tags and unescape entities
                snippet = re.sub(r"<[^>]+>", "", _html.unescape(snippet_html))
            if link:
                results.append(SourceItem(title=title or link, url=link, snippet=snippet, published=published))
        return results
    except Exception as e:
        print(f"Google News RSS fallback failed for {symbol}: {e}")
        return []


def _build_llm_prompt(symbol: str, sources: List[SourceItem], tone: str) -> str:
    headline_lines = [f"- {s.title} ({s.url})" for s in sources[:10]]
    snippets = [f"{s.title}: {s.snippet}" for s in sources if s.snippet]
    headlines_block = "\n".join(headline_lines) or "(no recent articles found)"
    snippets_block = "\n".join(snippets[:10]) or "(no snippets)"
    prompt = (
        f"You are a witty yet insightful markets analyst. The user asks: Why did {symbol} drop?\n"
        f"Use only the following recent headlines and snippets as context. Summarize the likely reasons "
        f"and deliver it in a concise, {tone} tone. Avoid making up facts beyond the provided links.\n\n"
        f"Headlines:\n{headlines_block}\n\nSnippets:\n{snippets_block}\n\n"
        f"Output guidelines:\n"
        f"- 2–4 short paragraphs max\n- Include 1–2 tongue-in-cheek jokes\n- If uncertainty remains, say so\n"
    )
    return prompt


def _call_openai(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        # Fallback: offline snark mode
        return (
            "LLM key missing, so here's the CliffNotes version: likely earnings jitters, guidance hiccups, "
            "analyst downgrades, or a general case of 'the market woke up on the wrong side of the bed.'"
        )
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a humorous financial analyst who absolutely roasts the stock market and still gets the facts right.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        print(f"OpenAI call failed: {e}")
        return (
            "Couldn't reach the LLM this time. Still, the ingredients for a drop are classic: earnings, "
            "guidance, downgrades, macro dramas, or just vibes."
        )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    symbols = _ensure_list_symbols(request.symbols)
    if not symbols:
        raise HTTPException(status_code=400, detail="No symbols provided")

    results: List[SymbolAnalysis] = []
    for symbol in symbols:
        sources = _search_news_for_symbol(symbol, request.days, request.max_results)
        prompt = _build_llm_prompt(symbol, sources, request.tone)
        summary = _call_openai(prompt)
        results.append(
            SymbolAnalysis(symbol=symbol, summary=summary, sources=sources)
        )

    return AnalyzeResponse(results=results)


@app.get("/")
async def root():
    return {"ok": True, "service": "whyisthestockplummeting"}