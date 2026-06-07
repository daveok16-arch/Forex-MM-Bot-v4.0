import requests, json
from datetime import datetime
from typing import Dict, List
import random

class FundamentalFeatures:
    """Economic calendar, news sentiment, interest rate differentials."""
    
    def __init__(self):
        self.news_cache = []
        self.last_update = None
        self.rate_cache = {}
        
    def get_economic_calendar(self, pair: str) -> List[Dict]:
        """Get upcoming high-impact events from free API."""
        try:
            url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
            r = requests.get(url, timeout=10)
            data = r.json()
            country = self._pair_to_country(pair)
            events = []
            for e in data:
                if e.get("impact") == "High" and country in e.get("country", ""):
                    events.append({
                        "title": e.get("title"),
                        "time": e.get("date"),
                        "impact": e.get("impact"),
                        "forecast": e.get("forecast"),
                        "previous": e.get("previous"),
                    })
            return events
        except Exception as e:
            print(f"[Fund] Calendar error: {e}")
            return []
    
    def _pair_to_country(self, pair: str) -> str:
        map = {"EURUSD": "EUR", "GBPUSD": "GBP", "USDJPY": "USD",
               "AUDUSD": "AUD", "USDCAD": "CAD", "USDCHF": "CHF",
               "XAUUSD": "USD", "BTCUSD": "USD"}
        return map.get(pair, "USD")
    
    def get_news_sentiment(self, pair: str) -> Dict:
        """News sentiment analysis."""
        try:
            score = random.uniform(-0.5, 0.5)
            return {
                "score": round(score, 2),
                "bullish": 50 + int(score * 50),
                "bearish": 50 - int(score * 50),
                "neutral": 0,
                "source": "mock"
            }
        except Exception as e:
            print(f"[Fund] News error: {e}")
            return {"score": 0.0, "bullish": 0, "bearish": 0, "neutral": 0}
    
    def get_interest_rate_diff(self, pair: str) -> float:
        """Interest rate differential between pair currencies."""
        try:
            rates = {"USD": 5.50, "EUR": 4.50, "GBP": 5.25,
                     "JPY": 0.10, "AUD": 4.35, "CAD": 5.00, "CHF": 1.75}
            base, quote = pair[:3], pair[3:]
            if base in ["XAU", "BTC"]: base_rate = 0.0
            else: base_rate = rates.get(base, 0.0)
            quote_rate = rates.get(quote, 0.0)
            return base_rate - quote_rate
        except Exception as e:
            print(f"[Fund] Rate error: {e}")
            return 0.0
    
    def get_feature_vector(self, pair: str) -> Dict:
        """Return all fundamental features."""
        calendar = self.get_economic_calendar(pair)
        sentiment = self.get_news_sentiment(pair)
        rate_diff = self.get_interest_rate_diff(pair)
        
        event_risk = 0.0
        for e in calendar:
            try:
                et = datetime.strptime(e["time"], "%Y-%m-%d %H:%M:%S")
                if (et - datetime.utcnow()).total_seconds() < 86400:
                    event_risk = 1.0
                    break
            except: pass
        
        return {
            "event_risk": event_risk,
            "sentiment_score": sentiment["score"],
            "sentiment_bullish": sentiment["bullish"] / 100.0,
            "rate_differential": rate_diff / 10.0,
            "calendar_events": len(calendar),
        }
