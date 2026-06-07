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

class MarketSession:
    SESSIONS = {
        "asian": {"start": 0, "end": 8, "pairs": ["USDJPY", "AUDUSD", "NZDUSD", "USDCAD", "XAUUSD"], "weight": 1.0},
        "london": {"start": 8, "end": 16, "pairs": ["EURUSD", "GBPUSD", "USDCHF", "EURGBP", "EURJPY", "GBPJPY"], "weight": 1.2},
        "ny": {"start": 13, "end": 21, "pairs": ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "XAUUSD", "BTCUSD", "ETHUSD"], "weight": 1.3},
        "overlap": {"start": 13, "end": 16, "pairs": ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"], "weight": 1.5},
        "weekend": {"start": 21, "end": 24, "pairs": ["BTCUSD", "ETHUSD"], "weight": 1.0}
    }
    
    def __init__(self):
        self.current_session = None
        self.active_pairs = []
        self.session_weight = 1.0
    
    def update(self):
        now = datetime.utcnow()
        weekday = now.weekday()
        hour = now.hour
        
        if weekday >= 5:
            self.current_session = "weekend"
            self.active_pairs = self.SESSIONS["weekend"]["pairs"]
            self.session_weight = self.SESSIONS["weekend"]["weight"]
            return
        
        if 13 <= hour < 16:
            self.current_session = "overlap"
            self.active_pairs = self.SESSIONS["overlap"]["pairs"]
            self.session_weight = self.SESSIONS["overlap"]["weight"]
            return
        
        for session, data in self.SESSIONS.items():
            if session in ["overlap", "weekend"]:
                continue
            if data["start"] <= hour < data["end"]:
                self.current_session = session
                self.active_pairs = data["pairs"]
                self.session_weight = data["weight"]
                return
        
        self.current_session = "transition"
        self.active_pairs = ["EURUSD", "GBPUSD", "USDJPY", "BTCUSD"]
        self.session_weight = 0.8
    
    def get_session_header(self) -> str:
        now = datetime.utcnow().strftime("%H:%M UTC")
        emoji = {"asian": "🌏", "london": "🇬🇧", "ny": "🇺🇸", "overlap": "🔥", "weekend": "🌙", "transition": "⏳"}
        return f"{emoji.get(self.current_session, '⏳')} <b>{self.current_session.upper()}</b> Session | {now} | Weight: {self.session_weight}x"

class SignalLifecycle:
    def __init__(self):
        self.signals = {}
        self.cooldowns = {}
        self.confirmed = {}
        self.rejected = {}
    
    def generate_id(self, pair: str) -> str:
        ts = datetime.utcnow().strftime("%H%M%S")
        seq = len(self.signals) + 1
        return f"SIG_{pair}_{ts}_{seq:04d}"
    
    def can_emit(self, pair: str) -> bool:
        if pair in self.cooldowns:
            if (datetime.utcnow() - self.cooldowns[pair]).total_seconds() < 300:
                return False
        return True
    
    def emit(self, pair: str, signal: str, price: float, confidence: float) -> str:
        sid = self.generate_id(pair)
        self.signals[sid] = {
            "pair": pair,
            "signal": signal,
            "price": price,
            "confidence": confidence,
            "time": datetime.utcnow(),
            "status": "pending",
            "ttl": 180
        }
        self.cooldowns[pair] = datetime.utcnow()
        return sid
    
    def check_ttl(self, sid: str) -> bool:
        if sid not in self.signals:
            return False
        sig = self.signals[sid]
        age = (datetime.utcnow() - sig["time"]).total_seconds()
        if age > sig["ttl"]:
            sig["status"] = "expired"
            return False
        return True
    
    def confirm(self, sid: str):
        if sid in self.signals:
            self.signals[sid]["status"] = "confirmed"
            self.confirmed[sid] = self.signals[sid]
    
    def reject(self, sid: str):
        if sid in self.signals:
            self.signals[sid]["status"] = "rejected"
            self.rejected[sid] = self.signals[sid]

class MultiPairScanner:
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
        
        self.session = MarketSession()
        self.lifecycle = SignalLifecycle()
        self.ensemble = self._load_ensemble(mdir)
        self.tg = MMDispatcher(self.cfg)
        
        # Advanced features
        self.vol_breakout = VolumeBreakout(lookback=20, volume_mult=2.0)
        self.correlation = CorrelationFilter()
        self.position_sizer = PositionSizer(risk_per_trade=0.02, max_position=0.1)
        self.trailing_stops = {}
        
        self.scan_count = 0
        self.last_session = None
        self.active_signals = []
    
    def _load_ensemble(self, mdir):
        pt_path = os.path.join(mdir, "hypernetwork_v4.pt")
        onnx_path = os.path.join(mdir, "hypernetwork_v4_fixed.onnx")
        load_path = onnx_path if os.path.exists(onnx_path) else (pt_path if os.path.exists(pt_path) else None)
        print(f"[Ensemble] Loading: {load_path}")
        return DualThroatEnsemble(self.cfg, load_path)
    
    async def _fetch_batch(self, pairs: List[str], batch_size: int = 5) -> Dict[str, Dict]:
        data = {}
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            for pair in batch:
                try:
                    ticker = yf.Ticker(self.TICKER_MAP.get(pair, f"{pair}=X"))
                    h = ticker.history(period="1d", interval="5m")
                    if not h.empty:
                        last = h.iloc[-1]
                        data[pair] = {
                            "price": round(last["Close"], 5),
                            "volume": float(last.get("Volume", 1.0)),
                            "high": round(last.get("High", last["Close"]), 5),
                            "low": round(last.get("Low", last["Close"]), 5)
                        }
                        self.correlation.update(pair, data[pair]["price"])
                except Exception as e:
                    print(f"[Fetch] {pair} error: {e}")
            await asyncio.sleep(1)
        return data
    
    def _calculate_confidence(self, pair: str, data: Dict, regime: str) -> Tuple[float, str, Dict]:
        price = data["price"]
        volume = data["volume"]
        high = data["high"]
        low = data["low"]
        
        # Technical score
        micro = MicrostructureFeatures(self.cfg["microstructure"])
        f = micro.update(price, volume, "buy")
        v = micro.get_vec()
        e = self.ensemble.predict(v, regime)
        tech_score = (e["long_weight"] - e["short_weight"]) * (1.0 - e["uncertainty"])
        
        # Volume breakout
        self.vol_breakout.update(price, volume, high, low)
        vb = self.vol_breakout.detect()
        
        # Fundamental score
        fund = FundamentalFeatures()
        fv = fund.get_feature_vector(pair)
        fund_score = (fv["sentiment_score"] * 0.3) + (fv["rate_differential"] * 0.2)
        if fv["event_risk"] > 0.5:
            tech_score *= 0.5
        
        # Breakout boost
        if vb["breakout"]:
            if vb["direction"] == "UP":
                tech_score += vb["strength"] * 0.02
            else:
                tech_score -= vb["strength"] * 0.02
        
        # Composite
        composite = abs(tech_score + fund_score) * 100
        composite = min(composite, 99.0)
        
        if tech_score + fund_score > 0:
            signal = "BUY"
        else:
            signal = "SELL"
        
        return composite, signal, {
            "tech": tech_score,
            "fund": fund_score,
            "vol_breakout": vb,
            "regime": regime,
            "source": e["source"]
        }
    
    def _get_regime(self, prices: List[float]) -> str:
        if len(prices) < 20:
            return "normal"
        returns = np.diff(prices) / prices[:-1]
        vol = np.std(returns[-20:]) * 100
        trend = np.mean(returns[-20:]) * 100
        
        if vol < 0.05: return "quiet"
        if vol > 0.3: return "volatile"
        if abs(trend) > 0.1: return "trending"
        return "normal"
    
    async def scan(self):
        self.scan_count += 1
        self.session.update()
        
        if self.session.current_session != self.last_session:
            header = self.session.get_session_header()
            print(f"[Session] {header}")
            await self.tg._send(header)
            self.last_session = self.session.current_session
        
        active_pairs = self.session.active_pairs
        print(f"[Scan #{self.scan_count}] {len(active_pairs)} pairs | {self.session.current_session}")
        
        data = await self._fetch_batch(active_pairs)
        
        signals = []
        for pair, d in data.items():
            if not self.lifecycle.can_emit(pair):
                continue
            
            # Correlation check
            can_trade, reason = self.correlation.should_trade(pair, [s["pair"] for s in signals])
            if not can_trade:
                print(f"[{pair}] Skipped: {reason}")
                continue
            
            regime = self._get_regime([d["price"]])
            conf, sig, details = self._calculate_confidence(pair, d, regime)
            
            conf *= self.session.session_weight
            conf = min(conf, 99.0)
            
            if conf >= 72:
                sid = self.lifecycle.emit(pair, sig, d["price"], conf)
                
                # Position sizing
                vol = details["vol_breakout"].get("strength", 0.5) / 100
                sizing = self.position_sizer.calculate(
                    d["price"], 
                    d["price"] * 0.99 if sig == "BUY" else d["price"] * 1.01,
                    10000.0,  # Default balance
                    vol
                )
                
                # Initialize trailing stop
                ts = TrailingStop(mode="atr", atr_mult=2.0)
                ts.set_entry(d["price"], sig)
                self.trailing_stops[sid] = ts
                
                signals.append({
                    "id": sid,
                    "pair": pair,
                    "signal": sig,
                    "price": d["price"],
                    "confidence": conf,
                    "details": details,
                    "session": self.session.current_session,
                    "sizing": sizing,
                    "trailing_stop": ts
                })
        
        signals.sort(key=lambda x: x["confidence"], reverse=True)
        top_signals = signals[:5]
        
        for sig in top_signals:
            await self._emit_signal(sig)
        
        if self.scan_count % 30 == 0:
            await self._send_heartbeat()
    
    async def _emit_signal(self, sig: Dict):
        pair = sig["pair"]
        sid = sig["id"]
        conf = sig["confidence"]
        signal = sig["signal"]
        price = sig["price"]
        d = sig["details"]
        sizing = sig["sizing"]
        
        # Risk notes
        risk_notes = []
        if sig["session"] == "weekend" and pair not in ["BTCUSD", "ETHUSD", "SOLUSD"]:
            risk_notes.append("⚠️ Weekend - Forex closed")
        if conf > 85:
            risk_notes.append("🔥 Strong signal")
        if d["vol_breakout"].get("breakout"):
            risk_notes.append(f"🔥 Volume Breakout {d['vol_breakout']['direction']} ({d['vol_breakout']['strength']:.2f}%)")
        if d["vol_breakout"].get("volume_confirmed"):
            risk_notes.append("✅ Volume Confirmed")
        
        risk_text = "\n".join(risk_notes) if risk_notes else "✅ Standard risk"
        
        # Position sizing info
        sizing_text = (f"📊 <b>Position Size:</b> {sizing['position_size']}\n"
                      f"💰 Risk: ${sizing['risk_amount']}\n"
                      f"📏 Stop Distance: {sizing['stop_distance']:.5f}\n"
                      f"⚖️ Vol Adjustment: {sizing['vol_adjustment']}")
        
        msg = (f"📊 <b>{sid}</b>\n"
               f"<b>{pair} {signal}</b> @ {price}\n"
               f"Confidence: {conf:.1f}%\n"
               f"Regime: {d['regime']}\n"
               f"Source: {d['source']}\n\n"
               f"📈 <b>Breakdown:</b>\n"
               f"Technical: {d['tech']:+.3f}\n"
               f"Fundamental: {d['fund']:+.3f}\n\n"
               f"{sizing_text}\n\n"
               f"⚠️ <b>Risk Notes:</b>\n"
               f"{risk_text}\n\n"
               f"⏱ TTL: 3 minutes\n"
               f"Reply CONFIRM {sid} or REJECT {sid}")
        
        print(f"[SIGNAL] {sid} {pair} {signal} {conf:.1f}%")
        await self.tg._send(msg)
    
    async def _update_trailing_stops(self):
        """Update all active trailing stops."""
        for sid, ts in list(self.trailing_stops.items()):
            if sid not in self.lifecycle.signals:
                continue
            
            sig = self.lifecycle.signals[sid]
            pair = sig["pair"]
            
            # Get current price
            try:
                ticker = yf.Ticker(self.TICKER_MAP.get(pair, f"{pair}=X"))
                h = ticker.history(period="1d", interval="1m")
                if not h.empty:
                    current = h.iloc[-1]["Close"]
                    result = ts.update(current)
                    
                    if result["stopped"]:
                        msg = (f"🛑 <b>TRAILING STOP HIT</b>\n"
                               f"Signal: {sid}\n"
                               f"Pair: {pair}\n"
                               f"Entry: {result['entry']}\n"
                               f"Stop: {result['current_stop']}\n"
                               f"Profit: {result['profit_pct']}%")
                        await self.tg._send(msg)
                        del self.trailing_stops[sid]
            except:
                pass
    
    async def _send_heartbeat(self):
        stats = (f"💓 <b>Scanner Heartbeat</b>\n"
                 f"Cycles: {self.scan_count}\n"
                 f"Session: {self.session.current_session}\n"
                 f"Active pairs: {len(self.session.active_pairs)}\n"
                 f"Signals: {len(self.lifecycle.signals)}\n"
                 f"Confirmed: {len(self.lifecycle.confirmed)}\n"
                 f"Rejected: {len(self.lifecycle.rejected)}\n"
                 f"Trailing Stops: {len(self.trailing_stops)}")
        await self.tg._send(stats)
    
    async def run(self, interval=300):
        print("[Scanner] Multi-Pair Scanner v5.1 started")
        print("[Scanner] Features: Volume Breakout, Correlation Filter, Position Sizing, Trailing Stops")
        await self.tg._send("🚀 <b>Multi-Pair Scanner v5.1</b> Advanced features active")
        
        while True:
            try:
                await self.scan()
                await self._update_trailing_stops()
            except Exception as e:
                print(f"[Scanner Error] {e}")
            await asyncio.sleep(interval)

async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=float, default=300)
    p.add_argument("--config", default="/data/data/com.termux/files/home/Forex-MM-Bot-v4.0/config/mm_config.yaml")
    a = p.parse_args()
    s = MultiPairScanner(a.config)
    await s.run(a.interval)

if __name__ == "__main__":
    import argparse
    asyncio.run(main())
