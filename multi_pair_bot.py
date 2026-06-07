import os, sys, yaml, time, asyncio, argparse, numpy as np
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features.microstructure import MicrostructureFeatures
from features.fundamentals import FundamentalFeatures
from features.breakout import BreakoutDetector
from agents.dual_throat_ensemble_termux import DualThroatEnsemble
from tg_dispatcher.mm_dispatcher import MMDispatcher
import yfinance as yf

class MultiPairBot:
    PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "XAUUSD", "BTCUSD"]
    TICKER_MAP = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X", "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X", "USDCHF": "USDCHF=X", "XAUUSD": "GC=F", "BTCUSD": "BTC-USD"}
    
    def __init__(self, cfg_path):
        with open(cfg_path, "r") as f: self.cfg = yaml.safe_load(f)
        base = self.cfg["paths"]["drive_base"]
        mdir = os.path.join(base, self.cfg["paths"]["models_dir"])
        os.makedirs(mdir, exist_ok=True)
        self.micro = MicrostructureFeatures(self.cfg["microstructure"])
        self.fund = FundamentalFeatures()
        self.pairs = {}
        for p in self.PAIRS:
            self.pairs[p] = {
                "ticker": yf.Ticker(self.TICKER_MAP.get(p, f"{p}=X")),
                "micro": MicrostructureFeatures(self.cfg["microstructure"]),
                "breakout": BreakoutDetector(lookback=20),
                "prices": [],
                "returns": [],
                "last_sig": None,
                "last_sig_time": 0,
                "sig_count": 0
            }
        pt_path = os.path.join(mdir, "hypernetwork_v4.pt")
        onnx_path = os.path.join(mdir, "hypernetwork_v4_fixed.onnx")
        load_path = onnx_path if os.path.exists(onnx_path) else (pt_path if os.path.exists(pt_path) else None)
        print(f"[Ensemble] Loading: {load_path}")
        self.ensemble = DualThroatEnsemble(self.cfg, load_path)
        self.tg = MMDispatcher(self.cfg)
    
    def _get_price(self, pair_data):
        try:
            h = pair_data["ticker"].history(period="1d", interval="5m")
            if h.empty: return None
            last = h.iloc[-1]
            return round(last["Close"], 5)
        except: return None
    
    def _regime(self, pair_data, price):
        pd = pair_data
        pd["prices"].append(price)
        if len(pd["prices"]) > 1: pd["returns"].append((price - pd["prices"][-2]) / pd["prices"][-2])
        if len(pd["returns"]) < 20: return "normal"
        rr = np.array(pd["returns"][-20:])
        v = np.std(rr) * 100
        t = np.mean(rr) * 100
        if v < 0.05: return "quiet"
        if v > 0.3: return "volatile"
        if abs(t) > 0.1: return "trending"
        return "normal"
    
    def _signal(self, pair_data, price, regime, pair):
        pd = pair_data
        f = pd["micro"].update(price, 1.0, "buy")
        v = pd["micro"].get_vec()
        e = self.ensemble.predict(v, regime)
        lw = e["long_weight"]
        sw = e["short_weight"]
        conf = 1.0 - e["uncertainty"]
        src = e["source"]
        
        # Breakout detection
        pd["breakout"].update(price)
        br = pd["breakout"].detect()
        
        # Fundamental features
        fund = self.fund.get_feature_vector(pair)
        event_risk = fund["event_risk"]
        sentiment = fund["sentiment_score"]
        rate_diff = fund["rate_differential"]
        
        # Adjust score with fundamentals
        fund_boost = (sentiment * 0.3) + (rate_diff * 0.2)
        if event_risk > 0.5:
            conf *= 0.5
        
        score = (lw - sw) * conf + fund_boost
        
        # STRICT THRESHOLDS
        if score > 0.2 and conf > 0.70:
            sig = "BUY"
        elif score < -0.2 and conf > 0.70:
            sig = "SELL"
        else:
            return None, conf, lw, sw, src, fund, br
        
        # CHECK BREAKOUT — flip signal if breakout detected
        should_flip, new_sig = pd["breakout"].should_flip_signal(sig, br)
        if should_flip:
            print(f"[BREAKOUT] {pair} Signal flipped: {sig} → {new_sig} (strength: {br['strength']:.2f}%)")
            sig = new_sig
            conf *= 0.9  # Reduce confidence slightly after flip
        
        return sig, conf, lw, sw, src, fund, br
    
    async def check_pair(self, pair):
        pd = self.pairs[pair]
        price = self._get_price(pd)
        if price is None: return
        regime = self._regime(pd, price)
        sig, conf, lw, sw, src, fund, br = self._signal(pd, price, regime, pair)
        
        now = time.time()
        cooldown = 1800  # 30 minutes
        
        if sig and (sig != pd["last_sig"] or (now - pd["last_sig_time"]) > cooldown):
            if pd["sig_count"] >= 5:
                print(f"[{pair}] Daily signal limit reached")
                return
            
            # Breakout alert in message
            breakout_msg = ""
            if br["breakout"]:
                breakout_msg = f"\n🔥 <b>BREAKOUT {br['direction']}</b> (strength: {br['strength']:.2f}%)"
            
            msg = (f"📊 <b>{pair} SIGNAL: {sig}</b>\n"
                   f"Price: {price}\n"
                   f"Confidence: {conf:.2%}\n"
                   f"Hyper: Long {lw:.2f} | Short {sw:.2f}\n"
                   f"Regime: {regime}\n"
                   f"Source: {src}{breakout_msg}\n\n"
                   f"📈 <b>Fundamentals:</b>\n"
                   f"Event Risk: {'⚠️ HIGH' if fund['event_risk'] > 0.5 else '✅ Low'}\n"
                   f"Sentiment: {fund['sentiment_score']:+.2f}\n"
                   f"Rate Diff: {fund['rate_differential']:.2f}")
            print(f"[SIGNAL] {pair} {sig} @ {price} conf={conf:.2f}")
            await self.tg._send(msg)
            pd["last_sig"] = sig
            pd["last_sig_time"] = now
            pd["sig_count"] += 1
        else:
            print(f"[{pair}] {price} {regime} {'no_signal' if sig is None else 'cooldown'}")
    
    async def run(self, dur=3600, interval=300):
        print(f"[MultiPairBot] Started {len(self.pairs)} pairs | Breakout detection ON")
        await self.tg._send(f"🚀 <b>Multi-Pair Bot v4.3</b> Breakout detection active")
        start = time.time()
        while time.time() - start < dur:
            tasks = [self.check_pair(p) for p in self.pairs]
            await asyncio.gather(*tasks)
            await asyncio.sleep(interval)
        print(f"[MultiPairBot] Done")

async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=3600)
    p.add_argument("--interval", type=float, default=300)
    p.add_argument("--config", default="/data/data/com.termux/files/home/Forex-MM-Bot-v4.0/config/mm_config.yaml")
    a = p.parse_args()
    b = MultiPairBot(a.config)
    await b.run(a.duration, a.interval)

if __name__ == "__main__":
    import argparse
    asyncio.run(main())
