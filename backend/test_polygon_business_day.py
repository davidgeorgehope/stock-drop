#!/usr/bin/env python3
"""Test Polygon with a known business day."""

import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests
from dotenv import load_dotenv

load_dotenv()

def test_business_day():
    api_key = os.getenv("POLYGON_API_KEY")
    
    # Try Friday Aug 30, 2024 (known business day)
    test_date = "2024-08-30"
    
    print(f"🔍 Testing known business day: {test_date}")
    
    url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{test_date}"
    params = {"adjusted": "true", "apiKey": api_key}
    
    response = requests.get(url, params=params)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        count = data.get("resultsCount", 0)
        print(f"✓ Success! Got {count} results")
        if count > 0:
            # Show top 5 biggest losers from that day
            results = data.get("results", [])
            # Calculate percent change and sort by biggest losers
            for r in results:
                if r.get("o") and r.get("c"):
                    r["pct_change"] = ((r["c"] - r["o"]) / r["o"]) * 100
            
            losers = sorted([r for r in results if r.get("pct_change") is not None], 
                          key=lambda x: x["pct_change"])[:10]
            
            print("\nTop 10 losers that day:")
            for loser in losers:
                print(f"  {loser['T']}: {loser['pct_change']:.2f}% (${loser['o']:.2f} → ${loser['c']:.2f})")
    else:
        print(f"❌ Error: {response.text}")

if __name__ == "__main__":
    test_business_day()