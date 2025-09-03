#!/usr/bin/env python3
"""Test script to debug Polygon EOD data fetching."""

import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_polygon_eod():
    """Test fetching EOD data from Polygon."""
    
    # Check if API key is loaded
    api_key = os.getenv("POLYGON_API_KEY")
    print(f"✓ POLYGON_API_KEY loaded: {'Yes' if api_key else 'No'}")
    if api_key:
        print(f"  Key starts with: {api_key[:8]}...")
    else:
        print("❌ No POLYGON_API_KEY found in environment")
        return
    
    # Calculate dates to try
    now_et = datetime.now(ZoneInfo("America/New_York"))
    print(f"\n📅 Current time ET: {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # Try yesterday's date (should always work)
    yesterday = (now_et - timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Try today's date
    today = now_et.strftime("%Y-%m-%d")
    
    # Try dates
    dates_to_try = [
        (yesterday, "Yesterday (should work)"),
        (today, "Today (may fail if before EOD data available)")
    ]
    
    for date_str, description in dates_to_try:
        print(f"\n🔍 Testing {description}: {date_str}")
        
        url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{date_str}"
        params = {
            "adjusted": "true",
            "apiKey": api_key
        }
        
        try:
            response = requests.get(url, params=params)
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "OK":
                    count = data.get("resultsCount", 0)
                    print(f"  ✓ Success! Got {count} results")
                    if count > 0 and "results" in data:
                        # Show a sample result
                        sample = data["results"][0]
                        print(f"  Sample: {sample.get('T')} - Close: ${sample.get('c')}")
                else:
                    print(f"  ⚠️ Response status: {data.get('status')}")
                    print(f"  Message: {data.get('message', 'No message')}")
            else:
                try:
                    error_data = response.json()
                    print(f"  ❌ Error: {error_data.get('error', error_data.get('message', 'Unknown error'))}")
                except:
                    print(f"  ❌ Error: {response.text}")
                    
        except Exception as e:
            print(f"  ❌ Exception: {e}")
    
    # Also test the open-close endpoint for a specific ticker
    print("\n🔍 Testing open-close endpoint for AAPL")
    for date_str, description in dates_to_try:
        url = f"https://api.polygon.io/v1/open-close/AAPL/{date_str}"
        params = {"adjusted": "true", "apiKey": api_key}
        
        try:
            response = requests.get(url, params=params)
            print(f"\n  {description} ({date_str}): Status {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "OK":
                    print(f"    ✓ AAPL close: ${data.get('close')}")
                else:
                    print(f"    Status: {data.get('status')}")
            else:
                try:
                    error_data = response.json()
                    print(f"    Error: {error_data.get('error', error_data.get('message'))}")
                except:
                    print(f"    Error: {response.text}")
        except Exception as e:
            print(f"    Exception: {e}")

if __name__ == "__main__":
    test_polygon_eod()