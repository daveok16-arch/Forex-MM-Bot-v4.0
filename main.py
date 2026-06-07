import os, sys, yaml, time, asyncio, argparse, numpy as np
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features.microstructure import MicrostructureFeatures
from agents.dual_throat_ensemble import DualThroatEnsemble
from mm_engine.inventory_tracker import InventoryTracker
from mm_engine.quote_generator import QuoteGenerator
from mm_engine.risk_guard import RiskGuard
from execution.order_manager import OrderManager
from telegram.mm_dispatcher import MMDispatcher
import yfinance as yf
class RegimeDetector:
    def __init__(self, lb=20):
        self.lb = lb
        self.p = []
        self.r = []
    def update(self, price):
        self.p.append(price)
        if len(self.p) > 1: self.r.append((price - self.p[-2]) / self.p[-2])
        if len(self.r) < self.lb: return "normal"
        rr = np.array(self.r[-self.lb:])
        v = np.std(rr) * 100
        t = np.mean(rr) * 100
        if v < 0.05: return "quiet"
        if v > 0.3: return "volatile"
        if abs(t) > 0.1: return "trending"
        return "normal"
    def get_vol(self):
        return np.std(self.r[-self.lb:]) * 100 if len(self.r) >= 2 else 0.0
    def get_atr(self):
        return np.mean(np.abs(np.diff(self.p[-self.lb:]))) if len(self.p) >= self.lb else 0.0001
class ForexMMBot:
    TICKER_MAP = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X", "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X", "USDCHF": "USDCHF=X", "XAUUSD": "GC=F", "BTCUSD": "BTC-USD"}
    def __init__(self, cfg_path, pair, mode="paper"):
        with open(cfg_path, "r") as f: self.cfg = yaml.safe_load(f)
        self.pair = pair
        self.mode = mode
        self.running = False
        self.ticker = yf.Ticker(self.TICKER_MAP.get(pair, f"{pair}=X"))
        base = self.cfg["paths"]["drive_base"]
        mdir = os.path.join(base, self.cfg["paths"]["models_dir"])
        ldir = os.path.join(base, self.cfg["paths"]["logs_dir"])
        os.makedirs(mdir, exist_ok=True)
        os.makedirs(ldir, exist_ok=True)
        self.micro = MicrostructureFeatures(self.cfg["microstructure"])
        self.regime = RegimeDetector()
        mp = os.path.join(mdir, "hypernetwork_v4.onnx")
        mp = mp if os.path.exists(mp) else (os.path.join(mdir, "hypernetwork_v4.pt") if os.path.exists(os.path.join(mdir, "hypernetwork_v4.pt")) else None)
        self.ensemble = DualThroatEnsemble(self.cfg, mp)
        self.inv = InventoryTracker(self.cfg, os.path.join(ldir, f"inv_{pair}.json"))
        self.qg = QuoteGenerator(self.cfg)
        self.risk = RiskGuard(self.cfg)
        self.om = OrderManager()
        self.tg = MMDispatcher(self.cfg)
        self.ticks = 0
        self.fills = 0
        self.quotes = 0
        self.last_price = None
    def _tick(self):
        try:
            hist = self.ticker.history(period="1d", interval="1m")
            if hist.empty:
                if self.last_price: return {"price": self.last_price, "volume": 1.0, "side": "buy"}
                return {"price": 1.1000, "volume": 1.0, "side": "buy"}
            last = hist.iloc[-1]
            price = round(last["Close"], 5)
            volume = float(last["Volume"]) if "Volume" in last else 1.0
            change = last["Close"] - last["Open"] if "Open" in last else 0
            side = "buy" if change >= 0 else "sell"
            self.last_price = price
            return {"price": price, "volume": volume, "side": side}
        except Exception as e:
            print(f"[Tick Error] {e}")
            if self.last_price: return {"price": self.last_price, "volume": 1.0, "side": "buy"}
            return {"price": 1.1000, "volume": 1.0, "side": "buy"}
    def _process(self, tick):
        self.ticks += 1
        price = tick["price"]
        vol = tick["volume"]
        side = tick["side"]
        feats = self.micro.update(price, vol, side)
        vec = self.micro.get_vec()
        regime = self.regime.update(price)
        volatility = self.regime.get_vol()
        atr = self.regime.get_atr()
        ens = self.ensemble.predict(vec, regime)
        skew = self.inv.get_skew(self.pair)
        pnl = self.inv._pos_pnl(self.pair, price)
        aq = self.qg.get_active()
        cq = aq.get(self.pair, {})
        qsp = (cq.get("spread_pips", self.cfg["spread"]["base_spread"]) * 0.0001) / price * 100
        bsp = self.cfg["spread"]["base_spread"] * 0.0001 / price * 100
        halt, kills = self.risk.check(self.pair, price, skew, pnl, volatility, atr, qsp, bsp)
        if halt:
            self.om.cancel_pair(self.pair)
            return {"action": "halt", "kills": kills, "price": price}
        lw = ens["long_weight"]
        sw = ens["short_weight"]
        esk = (lw - sw) * 0.3
        csk = skew + esk
        quote = self.qg.generate(self.pair, price, volatility, regime, csk, feats)
        self.quotes += 1
        self.om.cancel_pair(self.pair)
        self.om.place(self.pair, "buy", quote["bid"], 1.0 * lw, 5.0)
        self.om.place(self.pair, "sell", quote["ask"], 1.0 * sw, 5.0)
        fills = self.om.check_fills(self.pair, price)
        fr = []
        for fill in fills:
            qp = fill["quote_price"]
            sp = abs(fill["price"] - qp)
            iu = self.inv.update_fill(self.pair, fill["side"], fill["price"], fill["size"], qp)
            self.risk.record_fill(self.pair, fill["side"], fill["price"], sp)
            self.fills += 1
            fr.append({"fill": fill, "spnl": sp, "inv": iu})
        self.om.expire()
        return {"action": "quote", "price": price, "regime": regime, "vol": volatility, "ens": ens, "quote": quote, "fills": fr, "skew": skew}
    async def run(self, dur=300.0, interval=60.0):
        self.running = True
        print(f"[Bot] MM v4.0 {self.pair} Mode:{self.mode}")
        await self.tg.send_start()
        start = time.time()
        while self.running and (time.time() - start) < dur:
            tick = self._tick()
            r = self._process(tick)
            if r["action"] == "halt":
                print(f"[HALT] {self.pair} {r['kills']} @ {r['price']}")
                await self.tg.send_risk(r["kills"], self.pair)
                await asyncio.sleep(5.0)
                self.risk.reset()
            else:
                if self.ticks % 1 == 0:
                    print(f"[T{self.ticks}] {self.pair} {r['price']:.5f} {r['regime']} S:{r['quote']['spread_pips']:.2f}p F:{self.fills} PnL:{self.inv.total_pnl:+.4f}")
                if self.quotes % 5 == 0: await self.tg.send_quote(r["quote"])
                for f in r["fills"]: await self.tg.send_fill(f["fill"], f["spnl"])
            await self.tg.send_hb({"total_pnl": self.inv.total_pnl, "active_quotes": len(self.qg.get_active()), "open_orders": len(self.om.get_open())})
            await asyncio.sleep(interval)
        self.running = False
        print(f"[Bot] Done. Ticks:{self.ticks} Fills:{self.fills} PnL:{self.inv.total_pnl:+.4f}")
        rep = self.inv.get_report()
        await self.tg.send_inv(rep)
        return rep
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="EURUSD")
    p.add_argument("--mode", default="paper", choices=["paper", "live"])
    p.add_argument("--duration", type=float, default=300.0)
    p.add_argument("--tick", type=float, default=60.0)
    p.add_argument("--config", default="/content/drive/MyDrive/Forex-MM-Bot-v4.0/config/mm_config.yaml")
    a = p.parse_args()
    b = ForexMMBot(a.config, a.pair, a.mode)
    try: asyncio.run(b.run(a.duration, a.tick))
    except KeyboardInterrupt: print("[Bot] Interrupted"); b.running = False
if __name__ == "__main__": main()
