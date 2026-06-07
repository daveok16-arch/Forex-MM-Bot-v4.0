import numpy as np
from typing import Dict, Tuple

class BreakoutDetector:
    """Detect price breakouts to avoid false signals."""
    
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.highs = []
        self.lows = []
        self.volumes = []
    
    def update(self, price: float, volume: float = 1.0):
        self.highs.append(price)
        self.lows.append(price)
        self.volumes.append(volume)
        
        # Keep only lookback period
        if len(self.highs) > self.lookback:
            self.highs.pop(0)
            self.lows.pop(0)
            self.volumes.pop(0)
    
    def detect(self) -> Dict:
        if len(self.highs) < self.lookback:
            return {"breakout": False, "direction": None, "strength": 0.0}
        
        recent_high = max(self.highs[:-5])  # Exclude last 5 bars
        recent_low = min(self.lows[:-5])
        current = self.highs[-1]
        prev = self.highs[-2] if len(self.highs) > 1 else current
        
        # Volume check
        avg_vol = np.mean(self.volumes[:-5]) if len(self.volumes) > 5 else 1.0
        current_vol = self.volumes[-1]
        vol_spike = current_vol > avg_vol * 1.5
        
        # Breakout up
        if current > recent_high and vol_spike:
            strength = (current - recent_high) / recent_high * 100
            return {"breakout": True, "direction": "UP", "strength": strength}
        
        # Breakout down
        if current < recent_low and vol_spike:
            strength = (recent_low - current) / recent_low * 100
            return {"breakout": True, "direction": "DOWN", "strength": strength}
        
        # Near breakout
        range_pct = (recent_high - recent_low) / recent_low * 100
        if range_pct < 0.5:  # Tight range
            if current > recent_high * 0.998:
                return {"breakout": False, "direction": "POTENTIAL_UP", "strength": 0.3}
            if current < recent_low * 1.002:
                return {"breakout": False, "direction": "POTENTIAL_DOWN", "strength": 0.3}
        
        return {"breakout": False, "direction": None, "strength": 0.0}
    
    def should_flip_signal(self, signal: str, breakout: Dict) -> Tuple[bool, str]:
        """Check if signal should be flipped due to breakout."""
        if not breakout["breakout"]:
            return False, signal
        
        # If breakout UP but signal is SELL → flip to BUY
        if breakout["direction"] == "UP" and signal == "SELL":
            return True, "BUY"
        
        # If breakout DOWN but signal is BUY → flip to SELL
        if breakout["direction"] == "DOWN" and signal == "BUY":
            return True, "SELL"
        
        return False, signal
