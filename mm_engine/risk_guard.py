import numpy as np
from datetime import datetime
from collections import deque
class RiskGuard:
    def __init__(self, config):
        r = config["risk"]
        self.max_dd = r["max_drawdown_pct"]
        self.max_skew = r["max_inventory_skew"]
        self.stale = r["stale_position_minutes"]
        self.adv_w = r["adverse_selection_window"]
        self.adv_t = r["adverse_selection_threshold"]
        self.max_vol = r["max_volatility_atr_multiple"]
        self.max_spread = r["max_spread_widening_pct"]
        self.init_bal = 0.0
        self.peak = 0.0
        self.kills = {"drawdown":False, "inventory":False, "stale":False, "adverse_selection":False, "volatility":False, "spread":False, "manual":False}
        self.fills = deque(maxlen=100)
        self.adv = {}
        self.last_trade = {}
    def set_balance(self, bal):
        self.init_bal = bal
        self.peak = bal
    def check(self, pair, price, skew, pnl, vol, atr, q_spread, b_spread):
        t = []
        cur = self.init_bal + pnl
        self.peak = max(self.peak, cur)
        dd = (self.peak - cur) / max(self.init_bal, 1.0) * 100
        if dd >= self.max_dd:
            self.kills["drawdown"] = True
            t.append(f"drawdown:{dd:.1f}%")
        if abs(skew) >= self.max_skew:
            self.kills["inventory"] = True
            t.append(f"skew:{skew:.2f}")
        if pair in self.last_trade:
            stale = (datetime.utcnow() - self.last_trade[pair]).total_seconds() / 60
            if stale >= self.stale and abs(skew) > 0.1:
                self.kills["stale"] = True
                t.append(f"stale:{stale:.0f}min")
        if self._check_adv(pair):
            self.kills["adverse_selection"] = True
            t.append("adverse")
        if atr > 0 and vol / atr >= self.max_vol:
            self.kills["volatility"] = True
            t.append(f"vol:{vol/atr:.1f}x")
        if b_spread > 0 and (q_spread / b_spread - 1) * 100 >= self.max_spread:
            self.kills["spread"] = True
            t.append(f"spread:{(q_spread/b_spread-1)*100:.0f}%")
        return len(t) > 0, t
    def record_fill(self, pair, side, price, pnl):
        ts = datetime.utcnow()
        self.fills.append({"pair":pair, "side":side, "price":price, "pnl":pnl, "timestamp":ts})
        self.last_trade[pair] = ts
        if pair not in self.adv: self.adv[pair] = deque(maxlen=self.adv_w)
        self.adv[pair].append(pnl)
    def _check_adv(self, pair):
        if pair not in self.adv or len(self.adv[pair]) < self.adv_w // 2: return False
        recent = list(self.adv[pair])
        losses = sum(1 for p in recent if p < 0)
        return losses / len(recent) >= self.adv_t
    def manual_kill(self, reason="manual"):
        self.kills["manual"] = True
        return True, [f"manual:{reason}"]
    def reset(self, switch=None):
        if switch: self.kills[switch] = False
        else:
            for k in self.kills: self.kills[k] = False
    def get_status(self):
        return {"kills":self.kills.copy(), "any":any(self.kills.values()), "fills":len(self.fills), "adv":{k:len(v) for k,v in self.adv.items()}}
