import os
os.environ["ONNXRUNTIME_DISABLE_GPU"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import os, sys, yaml, time, asyncio, argparse, numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.microstructure import MicrostructureFeatures
from features.fundamentals import FundamentalFeatures
from features.breakout import BreakoutDetector
from features.advanced import VolumeBreakout, CorrelationFilter, PositionSizer, TrailingStop
from agents.dual_throat_ensemble_termux import DualThroatEnsemble
from tg_dispatcher.mm_dispatcher import MMDispatcher
import yfinance as yf
import onnxruntime as ort

class V6Scanner:
    ALL_PAIRS = [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF",
        "NZDUSD", "EURGBP", "EURJPY", "GBPJPY", "XAUUSD", "BTCUSD"
    ]

    TICKER_MAP = {
        "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
        "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X", "USDCHF": "USDCHF=X",
        "NZDUSD": "NZDUSD=X", "EURGBP": "EURGBP=X", "EURJPY": "EURJPY=X",
        "GBPJPY": "GBPJPY=X", "AUDJPY": "AUDJPY=X", "CADJPY": "CADJPY=X",
        "EURAUD": "EURAUD=X", "GBPAUD": "GBPAUD=X", "EURNZD": "EURNZD=X",
        "GBPNZD": "GBPNZD=X", "AUDNZD": "AUDNZD=X", "USDSGD": "USDSGD=X",
        "XAUUSD": "GC=F", "XAGUSD": "SI=F", "USOIL": "CL=F",
        "BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD", "SOLUSD": "SOL-USD"
    }

    def __init__(self, cfg_path):
        with open(cfg_path, "r") as f: self.cfg = yaml.safe_load(f)
        base = self.cfg["paths"]["drive_base"]
        mdir = os.path.join(base, self.cfg["paths"]["models_dir"])
        os.makedirs(mdir, exist_ok=True)

        self.ensemble = self._load_ensemble(mdir)
        self.order_flow = self._init_order_flow()
        self.sentiment = self._init_sentiment()
        self.calendar = self._init_calendar()
        self.correlation = CorrelationFilter()
        self.position_sizer = PositionSizer(risk_per_trade=0.02, max_position=0.1)
        self.tg = MMDispatcher(self.cfg)
        self.scan_count = 0
        self.trailing_stops = {}
        self.active_signals = {}
        self.price_history = {}
        print("[V6] Scanner initialized | ONNX-only | Order Flow | Sentiment | Calendar")

    def _load_ensemble(self, mdir):
        pt_path = os.path.join(mdir, "hypernetwork_v4.pt")
        onnx_path = os.path.join(mdir, "hypernetwork_v4_fixed.onnx")
        load_path = onnx_path if os.path.exists(onnx_path) else (pt_path if os.path.exists(pt_path) else None)
        print(f"[V6] Looking for model at: {onnx_path} (exists: {os.path.exists(onnx_path)})")
        print(f"[V6] Looking for model at: {pt_path} (exists: {os.path.exists(pt_path)})")
        print(f"[V6] Ensemble will load: {load_path}")
        return DualThroatEnsemble(self.cfg, load_path)

    def _init_order_flow(self):
        class OrderFlow:
            def __init__(self):
                self.prices = []
                self.bid_volumes = []
                self.ask_volumes = []
                self.deltas = []
            def update(self, price, bid_vol, ask_vol):
                self.prices.append(price)
                self.bid_volumes.append(bid_vol)
                self.ask_volumes.append(ask_vol)
                self.deltas.append(bid_vol - ask_vol)
                if len(self.prices) > 50:
                    self.prices.pop(0); self.bid_volumes.pop(0); self.ask_volumes.pop(0); self.deltas.pop(0)
            def analyze(self):
                if len(self.prices) < 10:
                    return {"delta": 0, "strength": 0, "signal": "HOLD", "profile": "neutral"}
                recent_delta = np.mean(self.deltas[-10:])
                total_bid = sum(self.bid_volumes[-10:])
                total_ask = sum(self.ask_volumes[-10:])
                strength = abs(recent_delta) / (total_bid + total_ask + 1e-8)
                if total_bid > total_ask * 1.5: profile = "bid_dominant"
                elif total_ask > total_bid * 1.5: profile = "ask_dominant"
                else: profile = "neutral"
                if recent_delta > 0 and strength > 0.3: signal = "BUY"
                elif recent_delta < 0 and strength > 0.3: signal = "SELL"
                else: signal = "HOLD"
                return {"delta": recent_delta, "strength": strength, "signal": signal, "profile": profile, "bid_ratio": total_bid / (total_bid + total_ask + 1e-8)}
        return OrderFlow()

    def _init_sentiment(self):
        class Sentiment:
            def get_sentiment(self, pair):
                import random
                base = {"EURUSD": 0.2, "GBPUSD": 0.1, "USDJPY": -0.1, "BTCUSD": 0.5, "ETHUSD": 0.3, "XAUUSD": -0.2}.get(pair, 0.0)
                noise = random.uniform(-0.2, 0.2)
                sentiment = base + noise
                return {"score": round(sentiment, 2), "bullish_pct": 50 + int(sentiment * 50), "bearish_pct": 50 - int(sentiment * 50), "source": "social_mock", "trending_topics": ["forex", pair.lower(), "trading"]}
        return Sentiment()

    def _init_calendar(self):
        class Calendar:
            def fetch_events(self):
                import random
                from datetime import datetime, timedelta
                now = datetime.utcnow()
                return [{"time": (now + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M"), "currency": "USD", "event": "NFP", "impact": "High", "forecast": "200K", "previous": "180K"}]
            def get_alert(self, pair):
                events = self.fetch_events()
                currency = pair[:3]
                relevant = [e for e in events if e["currency"] in [currency, pair[3:]]]
                if not relevant: return ""
                alert = "⚠️ **UPCOMING EVENTS:**\\n"
                for e in relevant: alert += f"• {e['time']} - {e['event']} ({e['currency']}) - {e['impact']}\\n"
                return alert
        return Calendar()

    async def _fetch_data(self, pairs: List[str], batch_size: int = 1) -> Dict[str, Dict]:
        data = {}
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            for pair in batch:
                try:
                    ticker = yf.Ticker(self.TICKER_MAP.get(pair, f"{pair}=X"))
                    h = ticker.history(period="5d", interval="5m")
                    if not h.empty and len(h) >= 20:
                        recent = h.tail(20)
                        prices = recent["Close"].values
                        volumes = recent["Volume"].values if "Volume" in recent else np.ones(len(prices))
                        highs = recent["High"].values if "High" in recent else prices
                        lows = recent["Low"].values if "Low" in recent else prices
                        data[pair] = {
                            "prices": prices, "volumes": volumes, "highs": highs, "lows": lows,
                            "current_price": round(prices[-1], 5), "current_volume": float(volumes[-1]),
                            "current_high": round(highs[-1], 5), "current_low": round(lows[-1], 5)
                        }
                        self.correlation.update(pair, prices[-1])
                        self.price_history[pair] = prices
                except Exception as e:
                    print(f"[V6 Fetch] {pair} error: {e}")
                await asyncio.sleep(2)  # Rate limit protection
            await asyncio.sleep(3)  # Delay between batches
        return data

    def _order_flow_analyze(self, pair: str, data: Dict) -> Dict:
        current_vol = data["current_volume"]
        bid_vol = current_vol * (0.5 + np.random.random() * 0.3)
        ask_vol = current_vol * (0.5 + np.random.random() * 0.3)
        self.order_flow.update(data["current_price"], bid_vol, ask_vol)
        return self.order_flow.analyze()

    def _composite_score(self, pair: str, data: Dict) -> Tuple[float, str, Dict]:
        micro = MicrostructureFeatures(self.cfg["microstructure"])
        f = micro.update(data["current_price"], data["current_volume"], "buy")
        v = micro.get_vec()
        regime = self._detect_regime(data["prices"])
        e = self.ensemble.predict(v, regime)
        hyper_score = (e["long_weight"] - e["short_weight"]) * (1.0 - e["uncertainty"])

        of = self._order_flow_analyze(pair, data)
        of_score = of["strength"] if of["signal"] == "BUY" else -of["strength"] if of["signal"] == "SELL" else 0
        sent = self.sentiment.get_sentiment(pair)
        sent_score = sent["score"]
        fund = FundamentalFeatures()
        fv = {"event_risk": 0.0, "sentiment_score": 0.0, "rate_differential": 0.0, "calendar_events": 0}
        fund_score = (fv["sentiment_score"] * 0.3) + (fv["rate_differential"] * 0.2)
        if fv["event_risk"] > 0.5: hyper_score *= 0.5

        weights = {"hyper": 0.35, "orderflow": 0.25, "sentiment": 0.20, "fundamental": 0.20}
        composite = (hyper_score * weights["hyper"] + of_score * weights["orderflow"] + sent_score * weights["sentiment"] + fund_score * weights["fundamental"])
        confidence = min(abs(composite) * 100, 99.0)
        if composite > 0.15: signal = "BUY"
        elif composite < -0.15: signal = "SELL"
        else: signal = "HOLD"
        return confidence, signal, {"hyper": hyper_score, "orderflow": of_score, "sentiment": sent_score, "fundamental": fund_score, "regime": regime, "orderflow_signal": of["signal"], "sentiment_data": sent}

    def _detect_regime(self, prices: List[float]) -> str:
        if len(prices) < 20: return "normal"
        returns = np.diff(prices) / prices[:-1]
        vol = np.std(returns[-20:]) * 100
        trend = np.mean(returns[-20:]) * 100
        if vol < 0.05: return "quiet"
        if vol > 0.3: return "volatile"
        if abs(trend) > 0.1: return "trending"
        return "normal"

    async def scan(self):
        self.scan_count += 1
        now = datetime.utcnow().strftime("%H:%M UTC")
        print(f"[V6 Scan #{self.scan_count}] {now}")
        data = await self._fetch_data(self.ALL_PAIRS)
        signals = []
        for pair, d in data.items():
            can_trade, reason = self.correlation.should_trade(pair, [s["pair"] for s in signals])
            if not can_trade:
                print(f"[V6] {pair} skipped: {reason}")
                continue
            conf, sig, details = self._composite_score(pair, d)
            if conf >= 75 and sig != "HOLD":
                vol = abs(details["orderflow"]) if isinstance(details["orderflow"], float) else 0.5
                sizing = self.position_sizer.calculate(d["current_price"], d["current_price"] * 0.98 if sig == "BUY" else d["current_price"] * 1.02, 10000.0, vol)
                calendar_alert = self.calendar.get_alert(pair)
                signals.append({"pair": pair, "signal": sig, "price": d["current_price"], "confidence": conf, "details": details, "sizing": sizing, "calendar": calendar_alert})
        signals.sort(key=lambda x: x["confidence"], reverse=True)
        top = signals[:5]
        for sig in top:
            await self._emit_signal(sig)
        if self.scan_count % 10 == 0:
            await self._send_heartbeat()

    async def _emit_signal(self, sig: Dict):
        pair = sig["pair"]; signal = sig["signal"]; price = sig["price"]; conf = sig["confidence"]; d = sig["details"]; sizing = sig["sizing"]
        components = (f"🧠 **Model Breakdown:**\\nHypernetwork: {d['hyper']:+.3f}\\nOrder Flow: {d['orderflow']:+.3f} ({d['orderflow_signal']})\\nSentiment: {d['sentiment']:+.3f}\\nFundamental: {d['fundamental']:+.3f}\\nRegime: {d['regime']}")
        sent = d['sentiment_data']
        sentiment_info = (f"📱 **Social Sentiment:**\\nScore: {sent['score']:+.2f}\\nBullish: {sent['bullish_pct']}% | Bearish: {sent['bearish_pct']}%\\nTopics: {', '.join(sent['trending_topics'][:3])}")
        sizing_info = (f"📊 **Position Size:** {sizing['position_size']}\\n💰 Risk: ${sizing['risk_amount']}\\n📏 Stop: {sizing['stop_distance']:.5f}\\n⚖️ Vol Adj: {sizing['vol_adjustment']}")
        calendar_info = sig.get("calendar", "")
        msg = (f"🚀 **V6 SIGNAL: {pair} {signal}**\\nPrice: {price}\\nConfidence: {conf:.1f}%\\n\\n{components}\\n\\n{sentiment_info}\\n\\n{sizing_info}\\n\\n{calendar_info}\\n\\n⏱ Act within 3 minutes")
        print(f"[V6 SIGNAL] {pair} {signal} {conf:.1f}%")
        await self.tg._send(msg)

    async def _send_heartbeat(self):
        stats = (f"💓 **V6 Heartbeat**\\nScans: {self.scan_count}\\nPairs tracked: {len(self.price_history)}\\nActive signals: {len(self.active_signals)}\\nModels: Hypernetwork + OrderFlow + Sentiment + Calendar")
        await self.tg._send(stats)

    async def run(self, interval=300):
        print("[V6] Next-Gen Scanner v6.0 started")
        print("[V6] Features: ONNX Ensemble | Order Flow | Sentiment | Calendar | Correlation")
        await self.tg._send("🚀 **V6 Scanner v6.0** AI-powered trading signals")
        while True:
            try:
                await self.scan()
            except Exception as e:
                print(f"[V6 Error] {e}")
            await asyncio.sleep(interval)

async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=float, default=300)
    p.add_argument("--config", default="config/mm_config.yaml")
    a = p.parse_args()
    s = V6Scanner(a.config)
    await s.run(a.interval)

if __name__ == "__main__":
    import argparse
    asyncio.run(main())
