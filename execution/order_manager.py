import uuid
from datetime import datetime, timedelta
class Order:
    def __init__(self, pair, side, price, size, expiry=300.0):
        self.id = str(uuid.uuid4())[:8]
        self.pair = pair
        self.side = side
        self.price = price
        self.size = size
        self.filled = 0.0
        self.status = "open"
        self.created = datetime.utcnow()
        self.expiry = self.created + timedelta(seconds=expiry)
        self.fills = []
    def expired(self):
        return datetime.utcnow() > self.expiry
    def remain(self):
        return self.size - self.filled
    def fill(self, amount, price):
        fa = min(amount, self.remain())
        self.filled += fa
        self.fills.append({"time":datetime.utcnow().isoformat(), "price":price, "amount":fa})
        if self.filled >= self.size * 0.99: self.status = "filled"
        elif self.filled > 0: self.status = "partial"
        return self.fills[-1]
    def cancel(self):
        if self.status in ["open", "partial"]: self.status = "cancelled"
    def to_dict(self):
        return {"id":self.id, "pair":self.pair, "side":self.side, "price":self.price, "size":self.size, "filled":self.filled, "remain":self.remain(), "status":self.status, "created":self.created.isoformat(), "expiry":self.expiry.isoformat(), "fills":self.fills}
class OrderManager:
    def __init__(self):
        self.orders = {}
        self.history = []
    def place(self, pair, side, price, size, expiry=300.0):
        o = Order(pair, side, price, size, expiry)
        self.orders[o.id] = o
        return o
    def cancel(self, oid):
        if oid in self.orders: self.orders[oid].cancel(); return True
        return False
    def cancel_pair(self, pair, side=None):
        c = 0
        for o in self.orders.values():
            if o.pair == pair and o.status in ["open", "partial"]:
                if side is None or o.side == side: o.cancel(); c += 1
        return c
    def check_fills(self, pair, price):
        fills = []
        for o in list(self.orders.values()):
            if o.pair != pair or o.status not in ["open", "partial"]: continue
            if o.expired(): o.status = "expired"; continue
            if o.side == "buy" and price <= o.price:
                f = o.fill(o.remain(), price)
                fills.append({"oid":o.id, "pair":pair, "side":"long", "price":f["price"], "size":f["amount"], "quote_price":o.price})
            elif o.side == "sell" and price >= o.price:
                f = o.fill(o.remain(), price)
                fills.append({"oid":o.id, "pair":pair, "side":"short", "price":f["price"], "size":f["amount"], "quote_price":o.price})
        return fills
    def expire(self):
        for o in self.orders.values():
            if o.status in ["open", "partial"] and o.expired(): o.status = "expired"
    def get_open(self, pair=None):
        return [o for o in self.orders.values() if o.status in ["open", "partial"] and (pair is None or o.pair == pair)]
    def get(self, oid):
        return self.orders.get(oid)
    def stats(self):
        s = {}
        for o in self.orders.values():
            s[o.status] = s.get(o.status, 0) + 1
        return {"total":len(self.orders), "open":s.get("open",0)+s.get("partial",0), "filled":s.get("filled",0), "cancelled":s.get("cancelled",0), "expired":s.get("expired",0)}
