"""Common utility functions used across the application."""

from typing import List, Union


def _ensure_list_symbols(input_symbols: Union[str, List[str]]) -> List[str]:
    """Convert input symbols to a clean list of uppercase symbols."""
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


def _sanitize_symbol(symbol: str) -> str:
    """Sanitize symbol for safe use in URLs and filenames."""
    s = (symbol or "").strip().upper()
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-"
    s = "".join(ch for ch in s if ch in allowed)
    return s
