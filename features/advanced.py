import numpy as np
from typing import Dict, List, Tuple
from collections import deque

class VolumeBreakout:
    """Real-time volume breakout detection with confirmation."""
    
    def __init__(self, lookback: int = 20, volume_mult: float = 2.0):
        self.lookback = lookback
        self.volume_mult = volume_mult
        self.volumes = deque(maxlen=lookback)
        self.prices = deque(maxlen=lookback)
        self.highs = deque(maxlen=lookback)
        self.lows = deque(maxlen=lookback)
    
    def update(self, price: float, volume: float, high: float, low: float):
        self.prices.append(price)
        self.volumes.append(volume)
        self.highs.append(high)
        self.lows.append(low)
    
    def detect(self) -> Dict:
        if len(self.prices) < self.lookback:
            return {"breakout": False, "direction": None, "strength": 0.0, "volume_confirmed": False}
        
        avg_vol = np.mean(list(self.volumes)[:-5])
        current_vol = self.volumes[-1]
        vol_spike = current_vol > avg_vol * self.volume_mult
        
        recent_high = max(list(self.highs)[:-5])
        recent_low = min(list(self.lows)[:-5])
        current = self.prices[-1]
        
        # Breakout up
        if current > recent_high and vol_spike:
            strength = (current - recent_high) / recent_high * 100
            return {
                "breakout": True,
                "direction": "UP",
                "strength": strength,
                "volume_confirmed": True,
                "avg_volume": avg_vol,
                "current_volume": current_vol
            }
        
        # Breakout down
        if current < recent_low and vol_spike:
            strength = (recent_low - current) / recent_low * 100
            return {
                "breakout": True,
                "direction": "DOWN",
                "strength": strength,
                "volume_confirmed": True,
                "avg_volume": avg_vol,
                "current_volume": current_vol
            }
        
        return {
            "breakout": False,
            "direction": None,
            "strength": 0.0,
            "volume_confirmed": False,
            "avg_volume": avg_vol if self.volumes else 0,
            "current_volume": current_vol
        }

class CorrelationFilter:
    """Filter correlated pairs to avoid overexposure."""
    
    def __init__(self):
        self.price_history = {}
        self.correlation_matrix = {}
    
    def update(self, pair: str, price: float):
        if pair not in self.price_history:
            self.price_history[pair] = deque(maxlen=50)
        self.price_history[pair].append(price)
    
    def get_correlation(self, pair1: str, pair2: str) -> float:
        if pair1 not in self.price_history or pair2 not in self.price_history:
            return 0.0
        
        p1 = list(self.price_history[pair1])
        p2 = list(self.price_history[pair2])
        
        if len(p1) < 10 or len(p2) < 10:
            return 0.0
        
        # Calculate returns
        r1 = np.diff(p1) / p1[:-1]
        r2 = np.diff(p2) / p2[:-1]
        
        min_len = min(len(r1), len(r2))
        if min_len < 5:
            return 0.0
        
        # Pearson correlation
        corr = np.corrcoef(r1[-min_len:], r2[-min_len:])[0, 1]
        return corr if not np.isnan(corr) else 0.0
    
    def should_trade(self, pair: str, active_pairs: List[str], threshold: float = 0.8) -> Tuple[bool, str]:
        for other in active_pairs:
            if other == pair:
                continue
            corr = self.get_correlation(pair, other)
            if abs(corr) > threshold:
                return False, f"Correlated with {other} ({corr:.2f})"
        return True, "No correlation"

class PositionSizer:
    """Dynamic position sizing based on volatility and account balance."""
    
    def __init__(self, risk_per_trade: float = 0.02, max_position: float = 0.1):
        self.risk_per_trade = risk_per_trade  # 2% risk per trade
        self.max_position = max_position  # 10% max position
    
    def calculate(self, price: float, stop_loss: float, account_balance: float, volatility: float) -> Dict:
        # Risk amount
        risk_amount = account_balance * self.risk_per_trade
        
        # Stop distance in pips/points
        stop_distance = abs(price - stop_loss)
        if stop_distance == 0:
            stop_distance = price * 0.01  # Default 1%
        
        # Position size
        position_size = risk_amount / stop_distance
        
        # Adjust for volatility (reduce size in high volatility)
        vol_adjustment = 1.0 / (1.0 + volatility * 10)
        position_size *= vol_adjustment
        
        # Cap at max position
        max_units = (account_balance * self.max_position) / price
        position_size = min(position_size, max_units)
        
        return {
            "position_size": round(position_size, 4),
            "risk_amount": round(risk_amount, 2),
            "stop_distance": round(stop_distance, 5),
            "vol_adjustment": round(vol_adjustment, 2),
            "max_units": round(max_units, 4)
        }

class TrailingStop:
    """Trailing stop logic with multiple modes."""
    
    def __init__(self, mode: str = "atr", atr_mult: float = 2.0, fixed_points: float = 50.0):
        self.mode = mode
        self.atr_mult = atr_mult
        self.fixed_points = fixed_points
        self.entry_price = None
        self.current_stop = None
        self.highest_price = None
        self.lowest_price = None
        self.prices = deque(maxlen=20)
    
    def set_entry(self, price: float, direction: str):
        self.entry_price = price
        self.highest_price = price
        self.lowest_price = price
        self.direction = direction
        self.prices.clear()
        self.prices.append(price)
        
        # Initial stop
        if direction == "BUY":
            self.current_stop = price * 0.99  # 1% below entry
        else:
            self.current_stop = price * 1.01  # 1% above entry
    
    def update(self, price: float) -> Dict:
        self.prices.append(price)
        
        # Update extremes
        if price > self.highest_price:
            self.highest_price = price
        if price < self.lowest_price:
            self.lowest_price = price
        
        # Calculate new stop
        if self.mode == "atr":
            atr = self._calculate_atr()
            if self.direction == "BUY":
                new_stop = self.highest_price - (atr * self.atr_mult)
                self.current_stop = max(self.current_stop, new_stop)
            else:
                new_stop = self.lowest_price + (atr * self.atr_mult)
                self.current_stop = min(self.current_stop, new_stop)
        
        elif self.mode == "fixed":
            if self.direction == "BUY":
                self.current_stop = max(self.current_stop, price - self.fixed_points * 0.0001)
            else:
                self.current_stop = min(self.current_stop, price + self.fixed_points * 0.0001)
        
        # Check if stopped
        stopped = False
        if self.direction == "BUY" and price <= self.current_stop:
            stopped = True
        elif self.direction == "SELL" and price >= self.current_stop:
            stopped = True
        
        return {
            "entry": self.entry_price,
            "current_stop": round(self.current_stop, 5),
            "highest": self.highest_price,
            "lowest": self.lowest_price,
            "stopped": stopped,
            "profit_pct": round((price - self.entry_price) / self.entry_price * 100 * (1 if self.direction == "BUY" else -1), 2)
        }
    
    def _calculate_atr(self) -> float:
        if len(self.prices) < 2:
            return self.entry_price * 0.01
        returns = np.diff(list(self.prices))
        return np.mean(np.abs(returns)) if len(returns) > 0 else self.entry_price * 0.01
