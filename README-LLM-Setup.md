# LLM Setup Notes

The backend ranks “interesting losers” using a lightweight LLM pass. To avoid model hallucinations and bad upstream data:

- Candidate symbols always come from data sources (Polygon grouped EOD). The LLM cannot introduce new symbols; it can only rank those provided.
- Non‑primary equity tickers (rights/warrants/units with fifth‑letter suffix R/W/U/V) are excluded before ranking.
- After ranking, a final sanity filter compares Polygon EOD % to Stooq’s 1‑day %; items with large discrepancies are dropped.

Environment toggles:

- `LOSERS_FINAL_STOOQ_CHECK` (default `1`): enable/disable the final Stooq sanity filter.
- `LOSERS_FINAL_STOOQ_TOLERANCE_PPTS` (default `25.0`): max absolute percentage‑point difference allowed between Polygon EOD % and Stooq %.

These safeguards help prevent mistakes like confusing a rights ticker (e.g., FERAR) with a primary ticker (RACE), or surfacing an extreme drop when Stooq shows only a small move.

