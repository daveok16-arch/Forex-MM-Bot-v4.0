import argparse
import os, sys, yaml, time, asyncio, numpy as np
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features.microstructure import MicrostructureFeatures
from agents.dual_throat_ensemble_termux import DualThroatEnsemble
from tg_dispatcher.mm_dispatcher import MMDispatcher
import yfinance as yf
import onnxruntime as ort
class SignalBotTermux:
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
        pt_path = os.path.join(mdir, "hypernetwork_v4.pt")
        onnx_path = os.path.join(mdir, "hypernetwork_v4.onnx")
        load_path = pt_path if os.path.exists(pt_path) else (onnx_path if os.path.exists(onnx_path) else None)
        print(f"[Hyper] Loading: {load_path}")
        self.ensemble = DualThroatEnsemble(self.cfg, load_path)
        self.tg = MMDispatcher(self.cfg)
        self.last_sig = None
        cnn_onnx = os.path.join(mdir, "cnn_pattern.onnx")
        cnn_pt = os.path.join(mdir, "cnn_pattern.pt")
        if os.path.exists(cnn_onnx):
            self.cnn = ort.InferenceSession(cnn_onnx)
            print(f"[CNN] ONNX loaded: {cnn_onnx}")
        elif os.path.exists(cnn_pt):
            print(f"[CNN] PT found but no ONNX: {cnn_pt}")
            print("[CNN] CNN disabled - using hypernetwork only")
            self.cnn = None
        else:
            print("[CNN] No model found")
            self.cnn = None
    def _get_price(self):
        try:
            h = self.ticker.history(period="5d", interval="1m")
            if h.empty: return None, None
            last = h.iloc[-1]
            prices = h["Close"].values[-60:]
            return round(last["Close"], 5), prices
        except: return None, None
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
    def _cnn_signal(self, prices):
        if self.cnn is None or len(prices) < 60: return 0.33, 0.33, 0.33
        p = (prices - np.mean(prices)) / (np.std(prices) + 1e-8)
        inp = p.astype(np.float32).reshape(1, -1)
        o = self.cnn.run(None, {"input": inp})
        return o[0][0]
    def _signal(self, price, regime, prices):
        f = self.micro.update(price, 1.0, "buy")
        v = self.micro.get_vec()
        e = self.ensemble.predict(v, regime)
        lw = e["long_weight"]
        sw = e["short_weight"]
        conf = 1.0 - e["uncertainty"]
        src = e["source"]
        cb, ch, cs = self._cnn_signal(prices)
        score = (lw - sw) * conf + (cb - cs) * 0.3
        final_conf = min(conf + abs(cb - cs) * 0.2, 0.95)
        if score > 0.1 and final_conf > 0.4:
            return "BUY", final_conf, lw, sw, cb, cs, src
        elif score < -0.1 and final_conf > 0.4:
            return "SELL", final_conf, lw, sw, cb, cs, src
        return None, final_conf, lw, sw, cb, cs, src
    async def run(self, dur=300, interval=60):
        print(f"[SignalBotTermux] {self.pair} started")
        await self.tg._send(f"🚀 <b>Signal Bot Termux</b> Pair:{self.pair}")
        start = time.time()
        while time.time() - start < dur:
            price, prices = self._get_price()
            if price is None:
                await asyncio.sleep(5)
                continue
            regime = self._regime(price)
            sig, conf, lw, sw, cb, cs, src = self._signal(price, regime, prices)
            if sig and sig != self.last_sig:
                msg = f"📊 <b>{self.pair} SIGNAL: {sig}</b>\nPrice: {price}\nConfidence: {conf:.2%}\nHyper: Long {lw:.2f} | Short {sw:.2f}\nCNN: Buy {cb:.2f} | Sell {cs:.2f}\nRegime: {regime}\nSource: {src}"
                print(f"[SIGNAL] {self.pair} {sig} @ {price} conf={conf:.2f}")
                await self.tg._send(msg)
                self.last_sig = sig
            else:
                print(f"[{self.pair}] {price} no_signal regime={regime}")
            await asyncio.sleep(interval)
        print(f"[SignalBotTermux] {self.pair} done")
async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="EURUSD")
    p.add_argument("--duration", type=float, default=300)
    p.add_argument("--interval", type=float, default=60)
    p.add_argument("--config", default="/data/data/com.termux/files/home/Forex-MM-Bot-v4.0/config/mm_config.yaml")
    a = p.parse_args()
    b = SignalBotTermux(a.config, a.pair)
    await b.run(a.duration, a.interval)
if __name__ == "__main__":
    asyncio.run(main())
