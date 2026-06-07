import os, sys, yaml, time, asyncio, argparse, numpy as np
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features.microstructure import MicrostructureFeatures
from agents.dual_throat_ensemble import DualThroatEnsemble
from telegram.mm_dispatcher import MMDispatcher
import yfinance as yf
class SignalBot:
    TICKER_MAP = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X", "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X", "USDCHF": "USDCHF=X", "XAUUSD": "GC=F", "BTCUSD": "BTC-USD"}
    def __init__(self, cfg_path, pair):
        with open(cfg_path, "r") as f: self.cfg = yaml.safe_load(f)
        self.pair = pair
        self.ticker = yf.Ticker(self.TICKER_MAP.get(pair, f"{pair}=X"))
        base = self.cfg["paths"]["drive_base"]
        mdir = os.path.join(base, self.cfg["paths"]["models_dir"])
        os.makedirs(mdir, exist_ok=True)
        self.micro = MicrostructureFeatures(self.cfg["microstructure"])
        self.p = []
        self.r = []
        mp = os.path.join(mdir, "hypernetwork_v4.onnx")
        mp = mp if os.path.exists(mp) else (os.path.join(mdir, "hypernetwork_v4.pt") if os.path.exists(os.path.join(mdir, "hypernetwork_v4.pt")) else None)
        self.ensemble = DualThroatEnsemble(self.cfg, mp)
        self.tg = MMDispatcher(self.cfg)
        self.last_sig = None
    def _get_price(self):
        try:
            h = self.ticker.history(period="1d", interval="1m")
            if h.empty: return None
            last = h.iloc[-1]
            return round(last["Close"], 5)
        except: return None
    def _regime(self, price):
        self.p.append(price)
        if len(self.p) > 1: self.r.append((price - self.p[-2]) / self.p[-2])
        if len(self.r) < 20: return "normal"
        rr = np.array(self.r[-20:])
        v = np.std(rr) * 100
        t = np.mean(rr) * 100
        if v < 0.05: return "quiet"
        if v > 0.3: return "volatile"
        if abs(t) > 0.1: return "trending"
        return "normal"
    def _signal(self, price, regime):
        f = self.micro.update(price, 1.0, "buy")
        v = self.micro.get_vec()
        e = self.ensemble.predict(v, regime)
        lw = e["long_weight"]
        sw = e["short_weight"]
        conf = 1.0 - e["uncertainty"]
        src = e["source"]
        if lw > sw + 0.05 and conf > 0.3:
            return "BUY", conf, lw, sw, src
        elif sw > lw + 0.05 and conf > 0.3:
            return "SELL", conf, lw, sw, src
        return None, conf, lw, sw, src
    async def run(self, dur=300, interval=60):
        print(f"[SignalBot] {self.pair} started")
        await self.tg._send(f"🚀 <b>Signal Bot v4.0</b> Pair:{self.pair}")
        start = time.time()
        while time.time() - start < dur:
            price = self._get_price()
            if price is None:
                await asyncio.sleep(5)
                continue
            regime = self._regime(price)
            sig, conf, lw, sw, src = self._signal(price, regime)
            if sig and sig != self.last_sig:
                msg = f"📊 <b>{self.pair} SIGNAL: {sig}</b>\nPrice: {price}\nConfidence: {conf:.2%}\nLong: {lw:.2f} | Short: {sw:.2f}\nRegime: {regime}\nSource: {src}"
                print(f"[SIGNAL] {self.pair} {sig} @ {price} conf={conf:.2f}")
                await self.tg._send(msg)
                self.last_sig = sig
            else:
                print(f"[{self.pair}] {price} no_signal regime={regime}")
            await asyncio.sleep(interval)
        print(f"[SignalBot] {self.pair} done")
async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="EURUSD")
    p.add_argument("--duration", type=float, default=300)
    p.add_argument("--interval", type=float, default=60)
    p.add_argument("--config", default="/content/drive/MyDrive/Forex-MM-Bot-v4.0/config/mm_config.yaml")
    a = p.parse_args()
    b = SignalBot(a.config, a.pair)
    await b.run(a.duration, a.interval)
if __name__ == "__main__":
    asyncio.run(main())
