import numpy as np
from collections import deque
class MicrostructureFeatures:
    def __init__(self, cfg):
        self.vpin_w = cfg.get("vpin_window",50)
        self.imb_d = cfg.get("imbalance_depth",10)
        self.ent_w = cfg.get("tick_entropy_window",20)
        self.acc_l = cfg.get("acceleration_lookback",5)
        self.buys = deque(maxlen=self.vpin_w)
        self.sells = deque(maxlen=self.vpin_w)
        self.vols = deque(maxlen=self.vpin_w)
        self.prices = deque(maxlen=self.ent_w)
    def update(self, price, volume, side):
        self.prices.append(price)
        self.vols.append(volume)
        if side=="buy": self.buys.append(volume); self.sells.append(0.0)
        else: self.buys.append(0.0); self.sells.append(volume)
        return {"vpin":self._vpin(),"toxicity":self._tox(),"imbalance":self._imb(),"tick_entropy":self._ent(),"acceleration":self._acc()}
    def _vpin(self):
        tv = sum(self.vols)
        return abs(sum(self.buys)-sum(self.sells))/tv if tv>0 else 0.0
    def _tox(self):
        return min(self._vpin()*2.0, 1.0)
    def _imb(self):
        if len(self.buys)<self.imb_d: return 0.0
        rb = sum(list(self.buys)[-self.imb_d:])
        rs = sum(list(self.sells)[-self.imb_d:])
        t = rb+rs
        return (rb-rs)/t if t>0 else 0.0
    def _ent(self):
        if len(self.prices)<3: return 0.0
        c = np.diff(list(self.prices))
        if len(c)==0: return 0.0
        b = np.digitize(c, [-0.0001, 0.0001])
        p = np.bincount(b, minlength=3)
        p = p/p.sum()
        p = p[p>0]
        e = -np.sum(p*np.log2(p))
        return e/np.log2(3)
    def _acc(self):
        if len(self.prices)<self.acc_l+2: return 0.0
        p = list(self.prices)[-self.acc_l:]
        v1 = (p[-1]-p[0])/len(p)
        if len(self.prices)>=self.acc_l*2+2:
            pp = list(self.prices)[-(self.acc_l*2):-self.acc_l]
            v0 = (pp[-1]-pp[0])/len(pp)
            return v1-v0
        return v1
    def get_vec(self):
        f = self.update(self.prices[-1] if self.prices else 1.0, 0.0, "buy")
        return np.array([f["vpin"],f["toxicity"],f["imbalance"],f["tick_entropy"],f["acceleration"]], dtype=np.float32)
