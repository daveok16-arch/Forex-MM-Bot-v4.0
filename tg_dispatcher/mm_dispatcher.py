import os, asyncio, aiohttp
from datetime import datetime
class MMDispatcher:
    def __init__(self, config):
        t = config.get("telegram", {})
        self.enabled = t.get("enabled", False)
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat = os.environ.get("TELEGRAM_CHAT_ID", "")
        self.hb = t.get("heartbeat_interval_min", 5)
        self.alert_fill = t.get("alert_on_fill", True)
        self.alert_risk = t.get("alert_on_risk_kill", True)
        self.alert_inv = t.get("alert_on_inventory", True)
        self.url = f"https://api.telegram.org/bot{self.token}"
        self.last_hb = datetime.utcnow()
        self.msg_count = 0
    async def _send(self, text):
        if not self.enabled or not self.token or not self.chat:
            print(f"[TG] {text}")
            return False
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(f"{self.url}/sendMessage", json={"chat_id":self.chat, "text":text, "parse_mode":"HTML"}, timeout=aiohttp.ClientTimeout(total=10)) as r:
                    if r.status == 200: self.msg_count += 1; return True
                    else: print(f"[TG ERR] {r.status}"); return False
        except Exception as e: print(f"[TG ERR] {e}"); return False
    async def send_quote(self, q):
        await self._send(f"📊 <b>{q['pair']}</b> Mid:{q['mid']} Bid:{q['bid']} Ask:{q['ask']} Spread:{q['spread_pips']}p Regime:{q['regime']}")
    async def send_fill(self, fill, pnl):
        if not self.alert_fill: return
        e = "🟢" if pnl > 0 else "🔴"
        await self._send(f"{e} <b>FILL {fill['pair']}</b> {fill['side'].upper()} @ {fill['price']} Size:{fill['size']} PnL:{pnl:+.4f}")
    async def send_inv(self, report):
        if not self.alert_inv: return
        p = "\n".join([f"  {k}:{v['units']}@{v['avg_price']}" for k,v in report.get("positions",{}).items()]) or "  No positions"
        await self._send(f"📦 <b>Inventory</b> Total:{report['total_pnl']:+.4f} Skew:{report['global_skew']:.4f}\n{p}")
    async def send_risk(self, kills, pair):
        if not self.alert_risk: return
        k = "\n".join([f"  ❌ {x}" for x in kills])
        await self._send(f"🛑 <b>RISK KILL {pair}</b>\nHALTED\n{k}")
    async def send_hb(self, stats):
        if (datetime.utcnow() - self.last_hb).total_seconds() < self.hb * 60: return
        self.last_hb = datetime.utcnow()
        await self._send(f"💓 <b>HB</b> PnL:{stats.get('total_pnl',0):+.4f} Quotes:{stats.get('active_quotes',0)} Orders:{stats.get('open_orders',0)}")
    async def send_start(self):
        await self._send(f"🚀 <b>MM Bot v4.0</b> Mode:PAPER Time:{datetime.utcnow().isoformat()}")
