import numpy as np
from datetime import datetime
class QuoteGenerator:
    def __init__(self, config):
        s = config["spread"]
        self.base = s["base_spread"]
        self.min_s = s["min_spread"]
        self.max_s = s["max_spread"]
        self.vol_m = s["volatility_multiplier"]
        self.reg_t = s["regime_tighten"]
        self.reg_w = s["regime_widen"]
        q = config["quotes"]
        self.refresh = q["quote_refresh_ms"]
        self.half = q["time_decay_halflife"]
        self.part = q["partial_fill_prob"]
        self.slip = q["slippage_pips"]
        self.last_q = {}
        self.active = {}
    def generate(self, pair, mid, vol, regime, inv_skew, micro=None):
        mult = self.reg_t if regime=="quiet" else self.reg_w if regime=="volatile" else 1.0
        half = (self.base * mult * (1.0 + vol * self.vol_m)) / 2.0
        half = np.clip(half, self.min_s/2.0, self.max_s/2.0)
        skew = inv_skew * half * 0.5
        mo = 0.0
        if micro:
            mo = micro.get("toxicity", 0.0) * 0.3 * half
        bid = mid - (half + skew + mo) * 0.0001
        ask = mid + (half - skew + mo) * 0.0001
        if bid >= ask:
            sp = max(ask - bid, self.min_s * 0.0001)
            m = (bid + ask) / 2
            bid = m - sp/2
            ask = m + sp/2
        q = {"pair":pair, "mid":round(mid,5), "bid":round(bid,5), "ask":round(ask,5), "spread_pips":round((ask-bid)/0.0001,2), "regime":regime, "inventory_skew":round(inv_skew,4), "timestamp":datetime.utcnow().isoformat()}
        self.active[pair] = q
        self.last_q[pair] = datetime.utcnow()
        return q
    def check_fill(self, pair, price, side, rng=None):
        if pair not in self.active: return False, 0.0
        rng = rng or np.random.default_rng()
        q = self.active[pair]
        if side=="long" and price <= q["bid"]:
            fp = price + self.slip * 0.0001 if rng.random() < self.part else q["bid"]
            return True, round(fp, 5)
        if side=="short" and price >= q["ask"]:
            fp = price - self.slip * 0.0001 if rng.random() < self.part else q["ask"]
            return True, round(fp, 5)
        return False, 0.0
    def expire(self, max_age=5.0):
        now = datetime.utcnow()
        for p in list(self.active.keys()):
            if p in self.last_q and (now - self.last_q[p]).total_seconds() > max_age:
                del self.active[p]
                del self.last_q[p]
    def get_active(self):
        return self.active.copy()
