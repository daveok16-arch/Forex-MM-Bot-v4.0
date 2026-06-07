import json, os, numpy as np
from datetime import datetime
class InventoryTracker:
    def __init__(self, config, state_file=None):
        self.max_u = config["inventory"]["max_position_units"]
        self.max_usd = config["inventory"]["max_position_usd"]
        self.skew_f = config["inventory"]["skew_factor"]
        self.flat_t = config["inventory"]["flatten_threshold"]
        self.emer_t = config["inventory"]["emergency_flatten"]
        self.state_file = state_file
        self.positions = {}
        self.daily_pnl = {}
        self.total_pnl = 0.0
        if state_file and os.path.exists(state_file): self.load()
    def update_fill(self, pair, side, price, size, quote_price):
        if pair not in self.positions:
            self.positions[pair] = {"units":0.0, "avg_price":0.0, "fills":[]}
            self.daily_pnl[pair] = 0.0
        pos = self.positions[pair]
        if side=="long":
            sp = quote_price - price
            pos["units"] += size
        else:
            sp = price - quote_price
            pos["units"] -= size
        if pos["units"] != 0:
            tc = pos["avg_price"] * (pos["units"] - size if side=="long" else pos["units"] + size)
            tc += price * size
            pos["avg_price"] = tc / abs(pos["units"])
        pos["fills"].append({"time":datetime.utcnow().isoformat(), "side":side, "price":price, "size":size, "spread_pnl":sp})
        self.daily_pnl[pair] += sp
        self.total_pnl += sp
        self.save()
        return {"pair":pair, "side":side, "spread_pnl":sp, "position_units":pos["units"], "position_pnl":self._pos_pnl(pair, price), "inventory_skew":self.get_skew(pair)}
    def _pos_pnl(self, pair, price):
        if pair not in self.positions: return 0.0
        pos = self.positions[pair]
        if pos["units"]==0: return 0.0
        return pos["units"] * (price - pos["avg_price"])
    def get_skew(self, pair):
        if pair not in self.positions: return 0.0
        return np.clip(self.positions[pair]["units"] / self.max_u, -1.0, 1.0)
    def get_global_skew(self):
        t = sum(abs(p["units"]) for p in self.positions.values())
        if t==0: return 0.0
        return np.clip(sum(p["units"] for p in self.positions.values()) / self.max_u, -1.0, 1.0)
    def should_flatten(self, pair):
        return abs(self.get_skew(pair)) >= self.flat_t
    def emergency_flatten(self, pair):
        return abs(self.get_skew(pair)) >= self.emer_t
    def get_quote_skew(self, pair):
        return -self.get_skew(pair) * self.skew_f
    def save(self):
        if not self.state_file: return
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump({"positions":self.positions, "daily_pnl":self.daily_pnl, "total_pnl":self.total_pnl, "timestamp":datetime.utcnow().isoformat()}, f)
    def load(self):
        with open(self.state_file, "r") as f:
            s = json.load(f)
        self.positions = s.get("positions", {})
        self.daily_pnl = s.get("daily_pnl", {})
        self.total_pnl = s.get("total_pnl", 0.0)
    def reset_daily(self):
        self.daily_pnl = {k:0.0 for k in self.daily_pnl}
    def get_report(self):
        return {"total_pnl":round(self.total_pnl,4), "daily_pnl":{k:round(v,4) for k,v in self.daily_pnl.items()}, "positions":{k:{"units":round(v["units"],4), "avg_price":round(v["avg_price"],5)} for k,v in self.positions.items()}, "global_skew":round(self.get_global_skew(),4)}
