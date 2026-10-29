#!/usr/bin/env python3
"""
Test script to debug news scoring issues.
Run with: python test_news_scoring.py
"""

import os
import sys
import traceback
import time
from datetime import datetime, timezone
import json

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.connection import get_engine
from services.news_scoring import compute_news_score, build_news_classification_prompt, extract_first_json_block
from services.news import _search_news_for_symbol
from services.analysis import _call_openai
from database.repositories.features_repo import FeaturesRepository
from sqlalchemy.orm import sessionmaker


def test_news_search(symbol: str):
    """Test news search functionality."""
    print(f"\n🔍 Testing news search for {symbol}...")
    try:
        headlines = _search_news_for_symbol(symbol, days=3, max_results=6)
        if headlines:
            print(f"✅ Found {len(headlines)} headlines:")
            for h in headlines[:3]:
                print(f"  - {h.title[:80]}...")
        else:
            print("❌ No headlines found")
        return headlines
    except Exception as e:
        print(f"❌ News search failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def test_openai_call(symbol: str, headlines):
    """Test OpenAI API call."""
    print(f"\n🤖 Testing OpenAI call for {symbol}...")
    
    # Check if API key is set
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        print("❌ XAI_API_KEY not set!")
        return None
    
    print(f"✅ API key found: {api_key[:8]}...")
    
    if not headlines:
        print("⚠️  No headlines to analyze")
        return None
    
    try:
        prompt = build_news_classification_prompt(symbol, headlines)
        print(f"📝 Prompt length: {len(prompt)} chars")
        print(f"📝 Prompt preview: {prompt[:200]}...")
        
        raw = _call_openai(prompt)
        if raw:
            print(f"✅ OpenAI response length: {len(raw)} chars")
            print(f"📝 Response preview: {raw[:200]}...")
            return raw
        else:
            print("❌ Empty OpenAI response")
            return None
    except Exception as e:
        print(f"❌ OpenAI call failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def test_json_extraction(raw_response):
    """Test JSON extraction from LLM response."""
    print(f"\n📊 Testing JSON extraction...")
    
    if not raw_response:
        print("⚠️  No response to extract from")
        return None
    
    try:
        data = extract_first_json_block(raw_response)
        if data:
            print(f"✅ Extracted JSON: {json.dumps(data, indent=2)}")
            return data
        else:
            print("❌ Failed to extract JSON")
            print(f"Raw response: {raw_response}")
            return None
    except Exception as e:
        print(f"❌ JSON extraction failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def test_database_operations(symbol: str):
    """Test database read/write operations."""
    print(f"\n💾 Testing database operations for {symbol}...")
    
    try:
        engine = get_engine()
        print(f"✅ Database engine created")
        
        maker = sessionmaker(bind=engine, autoflush=False, autocommit=False)
        with maker() as db:
            repo = FeaturesRepository(db)
            
            # Test read
            row = repo.latest_for_symbol(symbol, feature_set="oversold_news_score_v1")
            if row:
                print(f"✅ Found cached entry for {symbol}")
                # Handle both timezone-aware and naive timestamps
                if row.timestamp.tzinfo is None:
                    # Assume UTC for naive timestamps
                    age = (datetime.now(timezone.utc) - row.timestamp.replace(tzinfo=timezone.utc)).total_seconds()
                else:
                    age = (datetime.now(timezone.utc) - row.timestamp).total_seconds()
                print(f"   Age: {age/3600:.1f} hours")
                try:
                    payload = json.loads(row.features_json or "{}")
                    score = payload.get("news_score")
                    print(f"   Cached score: {score}")
                except:
                    print(f"   Failed to parse cached data")
            else:
                print(f"ℹ️  No cached entry for {symbol}")
            
            # Test write
            test_payload = {
                "symbol": symbol,
                "news_score": 0.123,
                "test": True,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            repo.add_features(
                symbol=symbol,
                feature_set="test_news_score",
                features_json=json.dumps(test_payload)
            )
            print(f"✅ Successfully wrote test entry")
            
            # Verify write
            test_row = repo.latest_for_symbol(symbol, feature_set="test_news_score")
            if test_row:
                print(f"✅ Verified test entry was written")
            else:
                print(f"❌ Failed to verify test write")
                
        return True
    except Exception as e:
        print(f"❌ Database operation failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def test_full_news_score(symbol: str):
    """Test the full compute_news_score function with detailed logging."""
    print(f"\n🎯 Testing full news score computation for {symbol}...")
    
    start_time = time.time()
    try:
        # Temporarily patch the function to add logging
        import services.news_scoring
        original_func = services.news_scoring.compute_news_score
        
        def logged_compute_news_score(sym):
            print(f"[compute_news_score] Starting for {sym}")
            try:
                result = original_func(sym)
                print(f"[compute_news_score] Completed for {sym}: {result}")
                return result
            except Exception as e:
                print(f"[compute_news_score] Failed for {sym}: {type(e).__name__}: {e}")
                traceback.print_exc()
                raise
        
        services.news_scoring.compute_news_score = logged_compute_news_score
        
        score = compute_news_score(symbol)
        elapsed = time.time() - start_time
        
        if score is not None:
            print(f"✅ News score computed: {score} (took {elapsed:.1f}s)")
        else:
            print(f"❌ News score computation returned None (took {elapsed:.1f}s)")
        
        # Restore original function
        services.news_scoring.compute_news_score = original_func
        
        return score
    except Exception as e:
        print(f"❌ Full computation failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def test_specific_symbols(symbols):
    """Test news scoring for specific symbols."""
    print(f"\n🧪 Testing news scoring for {len(symbols)} symbols...")
    
    results = {}
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Testing {symbol}")
        print(f"{'='*60}")
        
        # Test individual components
        headlines = test_news_search(symbol)
        
        if headlines:
            raw_response = test_openai_call(symbol, headlines)
            if raw_response:
                json_data = test_json_extraction(raw_response)
        
        db_ok = test_database_operations(symbol)
        
        # Test full function
        score = test_full_news_score(symbol)
        results[symbol] = score
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    for symbol, score in results.items():
        status = "✅" if score is not None else "❌"
        print(f"{status} {symbol}: {score}")


def main():
    """Main test runner."""
    print("News Scoring Debug Test")
    print("=" * 60)
    
    # Test environment
    print("\n🔧 Environment Check:")
    print(f"Python: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"XAI_API_KEY: {'✅ Set' if os.environ.get('XAI_API_KEY') else '❌ Not set'}")
    print(f"DATABASE_URL: {os.environ.get('DATABASE_URL', 'Using default SQLite')}")
    
    # Test symbols from the logs
    test_symbols = ["AMZN", "MODV", "COTY", "STFS"]
    
    # You can also test with a single symbol for detailed debugging
    # test_symbols = ["AMZN"]
    
    test_specific_symbols(test_symbols)


if __name__ == "__main__":
    main()
