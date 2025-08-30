"""Analysis service for LLM-powered stock analysis and news interpretation."""

import os
from typing import List, Optional
from openai import OpenAI
from services.news import SourceItem


def _build_llm_prompt(
    symbol: str,
    sources: List[SourceItem],
    tone: str,
    price_context: Optional[str] = None,
    max_words: Optional[int] = None,
) -> str:
    headline_lines = [f"- {s.title} ({s.url})" for s in sources[:10]]
    snippets = [f"{s.title}: {s.snippet}" for s in sources if s.snippet]
    headlines_block = "\n".join(headline_lines) or "(no recent articles found)"
    snippets_block = "\n".join(snippets[:10]) or "(no snippets)"
    price_block = f"\nRecent price context:\n{price_context}\n" if price_context else "\n"
    prompt = (
        f"You are a witty yet insightful markets analyst. The user asks: Why did {symbol} drop?\n"
        f"Use only the following recent headlines and snippets as context. Summarize the likely reasons "
        f"and deliver it in a concise, {tone} tone. Avoid making up facts beyond the provided links.\n\n"
        f"Headlines:\n{headlines_block}\n\nSnippets:\n{snippets_block}\n{price_block}\n"
        f"Output guidelines:\n"
        f"- 2–4 short paragraphs max\n- Include 1–2 tongue-in-cheek jokes\n- If uncertainty remains, say so\n"
    )
    if max_words is not None:
        prompt += (
            f"- HARD LIMIT: Keep the entire response under {max_words} words; fewer is better for a social preview image.\n"
            f"- Prioritize brevity and clarity. Use short sentences.\n"
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
