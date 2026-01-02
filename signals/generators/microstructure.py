"""
Microstructure Signal Generator - Order flow and market microstructure analysis.

Implements institutional-grade microstructure signals:
- Order flow imbalance
- VPIN (Volume-synchronized Probability of Informed Trading)
- Toxic flow detection
- Quote pressure analysis
- Trade intensity (Hawkes process)
- Iceberg order detection
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TickData:
    """Single tick data point."""
    timestamp: datetime
    bid: float
    ask: float
    last: float
    bid_size: float
    ask_size: float
    volume: float
    
    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2
    
    @property
    def spread(self) -> float:
        return self.ask - self.bid
    
    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        return (self.spread / self.mid) * 10000


@dataclass
class MicrostructureSignals:
    """Collection of microstructure signals."""
    trade_imbalance: float  # -1 to 1 (negative = selling pressure)
    quote_pressure: float  # -1 to 1 (negative = ask pressure)
    spread_signal: float  # Spread relative to average
    trade_intensity: float  # -1 to 1 (high = unusual activity)
    vpin: float  # 0 to 1 (high = toxic flow)
    aggregate_signal: float  # Combined signal
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


class MicrostructureAnalyzer:
    """
    Analyzes market microstructure from tick data.
    
    This class implements the key microstructure signals that
    institutional traders use to detect informed trading activity.
    
    Usage:
        analyzer = MicrostructureAnalyzer(buffer_size=1000)
        
        # Process ticks
        for tick in tick_stream:
            signals = analyzer.process_tick(tick)
            if signals.aggregate_signal > 0.5:
                # Strong buying pressure
                pass
    """
    
    def __init__(
        self,
        buffer_size: int = 1000,
        vpin_bucket_size: int = 50,
        intensity_window: int = 100
    ):
        """
        Initialize microstructure analyzer.
        
        Args:
            buffer_size: Number of ticks to keep in buffer
            vpin_bucket_size: Volume per VPIN bucket
            intensity_window: Window for intensity calculation
        """
        self.buffer_size = buffer_size
        self.vpin_bucket_size = vpin_bucket_size
        self.intensity_window = intensity_window
        
        # Data buffers
        self.tick_buffer: deque = deque(maxlen=buffer_size)
        self.trade_timestamps: deque = deque(maxlen=buffer_size)
        self.trade_signs: deque = deque(maxlen=buffer_size)
        self.trade_volumes: deque = deque(maxlen=buffer_size)
        
        # Running statistics
        self.avg_spread: float = 0.0
        self.avg_volume: float = 0.0
        self.last_tick: Optional[TickData] = None
        
        # VPIN buckets
        self.vpin_buy_buckets: List[float] = []
        self.vpin_sell_buckets: List[float] = []
        self.current_bucket_buy: float = 0.0
        self.current_bucket_sell: float = 0.0
        self.current_bucket_volume: float = 0.0
    
    def process_tick(self, tick: TickData) -> MicrostructureSignals:
        """
        Process a new tick and generate signals.
        
        Args:
            tick: TickData object
            
        Returns:
            MicrostructureSignals with all signal values
        """
        # Update buffers
        self.tick_buffer.append(tick)
        self.trade_timestamps.append(tick.timestamp)
        
        # Classify trade direction (Lee-Ready algorithm)
        trade_sign = self._classify_trade(tick)
        self.trade_signs.append(trade_sign)
        self.trade_volumes.append(tick.volume)
        
        # Update running statistics
        self._update_statistics(tick)
        
        # Update VPIN
        self._update_vpin(tick, trade_sign)
        
        # Calculate signals
        trade_imbalance = self._calculate_trade_imbalance()
        quote_pressure = self._calculate_quote_pressure(tick)
        spread_signal = self._calculate_spread_signal(tick)
        trade_intensity = self._calculate_intensity()
        vpin = self._calculate_vpin()
        
        # Aggregate signal
        aggregate = self._aggregate_signals(
            trade_imbalance, quote_pressure, spread_signal,
            trade_intensity, vpin
        )
        
        # Confidence based on data sufficiency
        confidence = min(1.0, len(self.tick_buffer) / (self.buffer_size * 0.5))
        
        self.last_tick = tick
        
        return MicrostructureSignals(
            trade_imbalance=trade_imbalance,
            quote_pressure=quote_pressure,
            spread_signal=spread_signal,
            trade_intensity=trade_intensity,
            vpin=vpin,
            aggregate_signal=aggregate,
            confidence=confidence
        )
    
    def _classify_trade(self, tick: TickData) -> int:
        """
        Classify trade direction using Lee-Ready algorithm.
        
        Returns:
            1 for buy, -1 for sell, 0 for undetermined
        """
        if self.last_tick is None:
            return 0
        
        mid = tick.mid
        last = tick.last
        prev_last = self.last_tick.last
        
        # Tick test
        if last > prev_last:
            return 1  # Buy
        elif last < prev_last:
            return -1  # Sell
        else:
            # Quote test
            if last > mid:
                return 1
            elif last < mid:
                return -1
            else:
                return 0
    
    def _update_statistics(self, tick: TickData):
        """Update running statistics."""
        alpha = 0.01  # Exponential smoothing factor
        
        if self.avg_spread == 0:
            self.avg_spread = tick.spread
            self.avg_volume = tick.volume
        else:
            self.avg_spread = alpha * tick.spread + (1 - alpha) * self.avg_spread
            self.avg_volume = alpha * tick.volume + (1 - alpha) * self.avg_volume
    
    def _update_vpin(self, tick: TickData, trade_sign: int):
        """Update VPIN buckets."""
        volume = tick.volume
        
        # Classify volume
        if trade_sign > 0:
            self.current_bucket_buy += volume
        elif trade_sign < 0:
            self.current_bucket_sell += volume
        else:
            # Split evenly
            self.current_bucket_buy += volume / 2
            self.current_bucket_sell += volume / 2
        
        self.current_bucket_volume += volume
        
        # Check if bucket is complete
        if self.current_bucket_volume >= self.vpin_bucket_size:
            self.vpin_buy_buckets.append(self.current_bucket_buy)
            self.vpin_sell_buckets.append(self.current_bucket_sell)
            
            # Keep only recent buckets
            if len(self.vpin_buy_buckets) > 50:
                self.vpin_buy_buckets = self.vpin_buy_buckets[-50:]
                self.vpin_sell_buckets = self.vpin_sell_buckets[-50:]
            
            # Reset current bucket
            self.current_bucket_buy = 0.0
            self.current_bucket_sell = 0.0
            self.current_bucket_volume = 0.0
    
    def _calculate_trade_imbalance(self) -> float:
        """
        Calculate trade flow imbalance.
        
        Returns:
            -1 (selling pressure) to 1 (buying pressure)
        """
        if len(self.trade_signs) < 10:
            return 0.0
        
        recent_signs = list(self.trade_signs)[-100:]
        recent_volumes = list(self.trade_volumes)[-100:]
        
        # Volume-weighted imbalance
        buy_volume = sum(v for s, v in zip(recent_signs, recent_volumes) if s > 0)
        sell_volume = sum(v for s, v in zip(recent_signs, recent_volumes) if s < 0)
        
        total = buy_volume + sell_volume
        
        if total == 0:
            return 0.0
        
        return (buy_volume - sell_volume) / total
    
    def _calculate_quote_pressure(self, tick: TickData) -> float:
        """
        Calculate quote size imbalance.
        
        More bid size = buying pressure
        More ask size = selling pressure
        """
        total_size = tick.bid_size + tick.ask_size
        
        if total_size == 0:
            return 0.0
        
        return (tick.bid_size - tick.ask_size) / total_size
    
    def _calculate_spread_signal(self, tick: TickData) -> float:
        """
        Calculate spread signal.
        
        Wide spreads indicate uncertainty or low liquidity.
        Returns:
            Negative values when spread is above average (bearish)
        """
        if self.avg_spread == 0:
            return 0.0
        
        spread_ratio = tick.spread / self.avg_spread
        
        # Normalize to -1 to 1
        return 1 - min(2, spread_ratio)  # Wide spread = negative
    
    def _calculate_intensity(self) -> float:
        """
        Calculate trade arrival intensity.
        
        Uses simplified Hawkes process estimation.
        High intensity often precedes moves.
        """
        if len(self.trade_timestamps) < 10:
            return 0.0
        
        timestamps = list(self.trade_timestamps)
        
        # Calculate inter-arrival times
        deltas = []
        for i in range(1, len(timestamps)):
            delta = (timestamps[i] - timestamps[i-1]).total_seconds()
            if delta > 0:
                deltas.append(delta)
        
        if not deltas:
            return 0.0
        
        # Current intensity vs average
        recent_deltas = deltas[-10:] if len(deltas) >= 10 else deltas
        
        recent_intensity = 1 / (np.mean(recent_deltas) + 0.001)
        avg_intensity = 1 / (np.mean(deltas) + 0.001)
        
        if avg_intensity == 0:
            return 0.0
        
        intensity_ratio = recent_intensity / avg_intensity
        
        # Normalize to -1 to 1
        return np.tanh(intensity_ratio - 1)
    
    def _calculate_vpin(self) -> float:
        """
        Calculate Volume-synchronized Probability of Informed Trading.
        
        High VPIN = high probability of informed trading = potential adverse selection
        """
        if len(self.vpin_buy_buckets) < 5:
            return 0.0
        
        # Calculate order imbalance in each bucket
        imbalances = []
        
        for buy, sell in zip(self.vpin_buy_buckets, self.vpin_sell_buckets):
            total = buy + sell
            if total > 0:
                imbalance = abs(buy - sell) / total
                imbalances.append(imbalance)
        
        if not imbalances:
            return 0.0
        
        # VPIN is average absolute imbalance
        vpin = np.mean(imbalances)
        
        return min(1.0, vpin)
    
    def _aggregate_signals(
        self,
        trade_imbalance: float,
        quote_pressure: float,
        spread_signal: float,
        trade_intensity: float,
        vpin: float
    ) -> float:
        """
        Aggregate all signals into single trading signal.
        
        Returns:
            -1 to 1 (negative = bearish, positive = bullish)
        """
        # Weights for each signal
        weights = {
            'trade_imbalance': 0.35,
            'quote_pressure': 0.25,
            'spread_signal': 0.10,
            'trade_intensity': 0.15,
            'vpin': 0.15
        }
        
        # VPIN is unsigned, affects confidence not direction
        # High VPIN reduces position size, doesn't change direction
        
        directional_signal = (
            weights['trade_imbalance'] * trade_imbalance +
            weights['quote_pressure'] * quote_pressure +
            weights['spread_signal'] * spread_signal
        )
        
        # Intensity amplifies the signal
        if trade_intensity > 0.3:
            directional_signal *= (1 + trade_intensity * 0.5)
        
        return np.clip(directional_signal, -1, 1)
    
    def get_signal_summary(self) -> Dict[str, Any]:
        """Get summary of current microstructure state."""
        if not self.tick_buffer:
            return {'status': 'no_data'}
        
        signals = self.process_tick(self.tick_buffer[-1])
        
        return {
            'aggregate_signal': signals.aggregate_signal,
            'trade_imbalance': signals.trade_imbalance,
            'quote_pressure': signals.quote_pressure,
            'vpin': signals.vpin,
            'confidence': signals.confidence,
            'interpretation': self._interpret_signals(signals),
            'timestamp': signals.timestamp
        }
    
    def _interpret_signals(self, signals: MicrostructureSignals) -> str:
        """Generate human-readable interpretation."""
        if signals.confidence < 0.3:
            return "Insufficient data"
        
        if signals.aggregate_signal > 0.5:
            return "Strong buying pressure"
        elif signals.aggregate_signal > 0.2:
            return "Moderate buying pressure"
        elif signals.aggregate_signal < -0.5:
            return "Strong selling pressure"
        elif signals.aggregate_signal < -0.2:
            return "Moderate selling pressure"
        else:
            return "Neutral"
    
    def detect_iceberg_orders(self) -> Dict[str, Any]:
        """
        Detect potential iceberg orders.
        
        Iceberg orders show consistent small-size executions at same price
        with quick refills.
        """
        if len(self.tick_buffer) < 50:
            return {'detected': False}
        
        ticks = list(self.tick_buffer)[-50:]
        
        # Look for repeated trades at same price
        price_counts: Dict[float, int] = {}
        price_volumes: Dict[float, float] = {}
        
        for tick in ticks:
            price = round(tick.last, 5)
            price_counts[price] = price_counts.get(price, 0) + 1
            price_volumes[price] = price_volumes.get(price, 0) + tick.volume
        
        # Iceberg detection: many trades at same price with similar sizes
        for price, count in price_counts.items():
            if count >= 10:  # Many trades at same price
                avg_size = price_volumes[price] / count
                
                # Check if sizes are consistent (low variance)
                sizes = [t.volume for t in ticks if round(t.last, 5) == price]
                if np.std(sizes) / (np.mean(sizes) + 0.001) < 0.3:
                    return {
                        'detected': True,
                        'price': price,
                        'trade_count': count,
                        'avg_size': avg_size,
                        'total_volume': price_volumes[price]
                    }
        
        return {'detected': False}
    
    def detect_toxic_flow(self) -> Dict[str, Any]:
        """
        Detect toxic flow (informed trading).
        
        Indicators:
        - High VPIN
        - Directional imbalance
        - Widening spreads
        """
        if len(self.tick_buffer) < 100:
            return {'toxic': False, 'confidence': 0.0}
        
        vpin = self._calculate_vpin()
        imbalance = abs(self._calculate_trade_imbalance())
        
        # Check spread widening
        recent_spreads = [t.spread for t in list(self.tick_buffer)[-20:]]
        older_spreads = [t.spread for t in list(self.tick_buffer)[-100:-20]]
        
        if older_spreads:
            spread_widening = np.mean(recent_spreads) / (np.mean(older_spreads) + 0.0001)
        else:
            spread_widening = 1.0
        
        # Toxic flow score
        toxic_score = (
            0.4 * vpin +
            0.3 * imbalance +
            0.3 * min(1, spread_widening - 1)
        )
        
        return {
            'toxic': toxic_score > 0.5,
            'score': toxic_score,
            'vpin': vpin,
            'imbalance': imbalance,
            'spread_widening': spread_widening,
            'recommendation': 'reduce_size' if toxic_score > 0.3 else 'normal'
        }
