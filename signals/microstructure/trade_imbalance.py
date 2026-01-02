"""
Trade Imbalance Detection Module

Detects significant order flow imbalances that may indicate:
- Large institutional orders
- Momentum ignition attempts
- Liquidity imbalances
- Price manipulation

Key features:
- Real-time imbalance tracking
- Multi-timeframe imbalance analysis
- Imbalance persistence detection
- Sweep detection (aggressive orders hitting multiple price levels)
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


logger = logging.getLogger(__name__)


class ImbalanceType(Enum):
    """Type of detected imbalance."""
    NONE = "none"
    MINOR_BUY = "minor_buy"
    MINOR_SELL = "minor_sell"
    SIGNIFICANT_BUY = "significant_buy"
    SIGNIFICANT_SELL = "significant_sell"
    SWEEP_UP = "sweep_up"  # Aggressive buying sweep
    SWEEP_DOWN = "sweep_down"  # Aggressive selling sweep
    ABSORPTION = "absorption"  # Large passive orders absorbing flow


@dataclass
class TradeRecord:
    """Record of a single trade."""
    timestamp: datetime
    price: float
    volume: float
    side: int  # 1 = buy, -1 = sell, 0 = unknown
    is_aggressive: bool = False  # Hit the spread

    @property
    def dollar_volume(self) -> float:
        return self.price * self.volume


@dataclass
class TradeImbalanceSignal:
    """Trade imbalance detection signal."""
    imbalance_type: ImbalanceType
    imbalance_ratio: float  # -1 to 1 (sell to buy)
    imbalance_volume: float  # Net volume difference
    confidence: float  # 0 to 1

    # Analysis
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    total_volume: float = 0.0

    # Persistence
    is_persistent: bool = False  # Imbalance sustained over time
    duration_seconds: float = 0.0

    # Sweep detection
    is_sweep: bool = False
    sweep_levels: int = 0  # Number of price levels swept

    # Price impact
    price_impact: float = 0.0  # % price move during imbalance

    # Trading signal
    direction: float = 0.0  # -1 to 1
    signal_strength: float = 0.0  # 0 to 1

    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_action(self) -> str:
        """Get recommended trading action."""
        if self.imbalance_type == ImbalanceType.NONE:
            return "neutral"
        elif self.imbalance_type in [ImbalanceType.SWEEP_UP, ImbalanceType.SIGNIFICANT_BUY]:
            return "long"
        elif self.imbalance_type in [ImbalanceType.SWEEP_DOWN, ImbalanceType.SIGNIFICANT_SELL]:
            return "short"
        elif self.imbalance_type == ImbalanceType.ABSORPTION:
            return "fade"  # Trade against the flow being absorbed
        else:
            return "monitor"


class TradeImbalanceDetector:
    """
    Detects trade flow imbalances for trading signals.

    Monitors incoming trades to identify significant imbalances
    between buying and selling pressure. Large imbalances often
    precede or accompany price moves.

    Usage:
        detector = TradeImbalanceDetector(
            window_seconds=60,
            significant_threshold=0.6
        )

        # Process trades
        for trade in trade_stream:
            signal = detector.process_trade(
                price=trade.price,
                volume=trade.volume,
                side='buy' or 'sell',
                timestamp=trade.time
            )

            if signal.is_sweep:
                # Aggressive institutional flow detected
                pass
    """

    def __init__(
        self,
        window_seconds: float = 60,
        buffer_size: int = 5000,
        minor_threshold: float = 0.3,
        significant_threshold: float = 0.6,
        sweep_threshold: float = 0.8,
        persistence_seconds: float = 10
    ):
        """
        Initialize trade imbalance detector.

        Args:
            window_seconds: Time window for imbalance calculation
            buffer_size: Maximum trades to keep in buffer
            minor_threshold: Threshold for minor imbalance
            significant_threshold: Threshold for significant imbalance
            sweep_threshold: Threshold for sweep detection
            persistence_seconds: Duration for persistence detection
        """
        self.window_seconds = window_seconds
        self.buffer_size = buffer_size
        self.minor_threshold = minor_threshold
        self.significant_threshold = significant_threshold
        self.sweep_threshold = sweep_threshold
        self.persistence_seconds = persistence_seconds

        # Trade buffer
        self.trades: deque = deque(maxlen=buffer_size)

        # State tracking
        self.last_price: Optional[float] = None
        self.imbalance_start: Optional[datetime] = None
        self.imbalance_direction: int = 0

        # Price levels touched (for sweep detection)
        self.price_levels: Dict[float, float] = {}  # price -> volume

        # Statistics
        self.avg_volume_per_trade: float = 0.0
        self.avg_trades_per_minute: float = 0.0

    def process_trade(
        self,
        price: float,
        volume: float,
        side: Optional[str] = None,
        timestamp: datetime = None,
        is_aggressive: bool = None
    ) -> TradeImbalanceSignal:
        """
        Process a trade and check for imbalances.

        Args:
            price: Trade price
            volume: Trade volume
            side: Trade side ('buy', 'sell', or None)
            timestamp: Trade timestamp
            is_aggressive: Whether trade was aggressive (hit spread)

        Returns:
            TradeImbalanceSignal with detection results
        """
        timestamp = timestamp or datetime.now()

        # Determine trade side
        if side:
            trade_side = 1 if side.lower() == 'buy' else -1
        else:
            # Infer from price movement
            trade_side = self._infer_side(price)

        # Determine if aggressive
        if is_aggressive is None:
            is_aggressive = self._infer_aggressive(price, trade_side)

        # Create trade record
        trade = TradeRecord(
            timestamp=timestamp,
            price=price,
            volume=volume,
            side=trade_side,
            is_aggressive=is_aggressive
        )

        self.trades.append(trade)
        self._update_statistics(trade)

        # Track price levels
        self._track_price_level(price, volume * trade_side)

        # Calculate imbalance
        signal = self._calculate_imbalance()

        # Update state
        self.last_price = price

        return signal

    def _infer_side(self, price: float) -> int:
        """Infer trade side from price movement."""
        if self.last_price is None:
            return 0

        if price > self.last_price:
            return 1  # Likely buy
        elif price < self.last_price:
            return -1  # Likely sell
        else:
            return 0

    def _infer_aggressive(self, price: float, side: int) -> bool:
        """Infer if trade was aggressive based on price movement."""
        if self.last_price is None:
            return False

        # Aggressive if price moved in trade direction
        price_move = (price - self.last_price) / self.last_price if self.last_price else 0

        if side > 0 and price_move > 0:
            return True  # Buy that moved price up
        elif side < 0 and price_move < 0:
            return True  # Sell that moved price down

        return False

    def _update_statistics(self, trade: TradeRecord):
        """Update running statistics."""
        alpha = 0.01

        # Volume per trade
        if self.avg_volume_per_trade == 0:
            self.avg_volume_per_trade = trade.volume
        else:
            self.avg_volume_per_trade = (
                alpha * trade.volume +
                (1 - alpha) * self.avg_volume_per_trade
            )

    def _track_price_level(self, price: float, signed_volume: float):
        """Track volume at each price level for sweep detection."""
        # Round to tick
        rounded_price = round(price, 4)

        if rounded_price in self.price_levels:
            self.price_levels[rounded_price] += signed_volume
        else:
            self.price_levels[rounded_price] = signed_volume

        # Prune old levels
        if len(self.price_levels) > 100:
            # Keep only extreme prices
            prices = sorted(self.price_levels.keys())
            for p in prices[25:75]:
                del self.price_levels[p]

    def _calculate_imbalance(self) -> TradeImbalanceSignal:
        """Calculate current trade imbalance."""
        now = datetime.now()

        # Get trades within window
        window_trades = [
            t for t in self.trades
            if (now - t.timestamp).total_seconds() <= self.window_seconds
        ]

        if len(window_trades) < 5:
            return TradeImbalanceSignal(
                imbalance_type=ImbalanceType.NONE,
                imbalance_ratio=0.0,
                imbalance_volume=0.0,
                confidence=len(window_trades) / 10
            )

        # Calculate volumes
        buy_volume = sum(t.volume for t in window_trades if t.side > 0)
        sell_volume = sum(t.volume for t in window_trades if t.side < 0)
        total_volume = buy_volume + sell_volume

        if total_volume == 0:
            return TradeImbalanceSignal(
                imbalance_type=ImbalanceType.NONE,
                imbalance_ratio=0.0,
                imbalance_volume=0.0,
                confidence=0.5
            )

        # Imbalance ratio
        imbalance_ratio = (buy_volume - sell_volume) / total_volume
        imbalance_volume = buy_volume - sell_volume

        # Determine imbalance type
        imbalance_type = self._classify_imbalance(
            imbalance_ratio, window_trades
        )

        # Check persistence
        is_persistent, duration = self._check_persistence(imbalance_ratio)

        # Check for sweep
        is_sweep, sweep_levels = self._detect_sweep(window_trades)

        # If sweep, override type
        if is_sweep:
            if imbalance_ratio > 0:
                imbalance_type = ImbalanceType.SWEEP_UP
            else:
                imbalance_type = ImbalanceType.SWEEP_DOWN

        # Calculate price impact
        if len(window_trades) >= 2:
            first_price = window_trades[0].price
            last_price = window_trades[-1].price
            price_impact = (last_price - first_price) / first_price
        else:
            price_impact = 0.0

        # Signal direction and strength
        direction = imbalance_ratio
        signal_strength = abs(imbalance_ratio) * (0.5 + 0.5 * is_persistent)

        # Confidence
        confidence = min(1.0, len(window_trades) / 50)

        return TradeImbalanceSignal(
            imbalance_type=imbalance_type,
            imbalance_ratio=imbalance_ratio,
            imbalance_volume=imbalance_volume,
            confidence=confidence,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            total_volume=total_volume,
            is_persistent=is_persistent,
            duration_seconds=duration,
            is_sweep=is_sweep,
            sweep_levels=sweep_levels,
            price_impact=price_impact,
            direction=direction,
            signal_strength=signal_strength,
            metadata={
                'trade_count': len(window_trades),
                'aggressive_trades': sum(1 for t in window_trades if t.is_aggressive)
            }
        )

    def _classify_imbalance(
        self,
        ratio: float,
        trades: List[TradeRecord]
    ) -> ImbalanceType:
        """Classify the type of imbalance."""
        abs_ratio = abs(ratio)

        # Check for absorption (large passive orders)
        aggressive_ratio = (
            sum(1 for t in trades if t.is_aggressive) / len(trades)
            if trades else 0
        )

        # Absorption: lots of aggressive trades but price not moving
        if aggressive_ratio > 0.7 and abs_ratio < 0.3:
            return ImbalanceType.ABSORPTION

        # Standard classification
        if abs_ratio >= self.significant_threshold:
            return (
                ImbalanceType.SIGNIFICANT_BUY if ratio > 0
                else ImbalanceType.SIGNIFICANT_SELL
            )
        elif abs_ratio >= self.minor_threshold:
            return (
                ImbalanceType.MINOR_BUY if ratio > 0
                else ImbalanceType.MINOR_SELL
            )
        else:
            return ImbalanceType.NONE

    def _check_persistence(self, current_ratio: float) -> Tuple[bool, float]:
        """
        Check if imbalance is persistent.

        Returns:
            (is_persistent, duration_in_seconds)
        """
        now = datetime.now()
        current_direction = 1 if current_ratio > 0 else (-1 if current_ratio < 0 else 0)

        if current_direction == 0:
            self.imbalance_start = None
            self.imbalance_direction = 0
            return False, 0.0

        # Check if same direction as before
        if current_direction == self.imbalance_direction:
            if self.imbalance_start:
                duration = (now - self.imbalance_start).total_seconds()
                is_persistent = duration >= self.persistence_seconds
                return is_persistent, duration
            else:
                self.imbalance_start = now
                return False, 0.0
        else:
            # Direction changed, reset
            self.imbalance_start = now
            self.imbalance_direction = current_direction
            return False, 0.0

    def _detect_sweep(
        self,
        trades: List[TradeRecord]
    ) -> Tuple[bool, int]:
        """
        Detect if aggressive orders are sweeping multiple price levels.

        Returns:
            (is_sweep, number_of_levels)
        """
        if len(trades) < 5:
            return False, 0

        # Get aggressive trades
        aggressive = [t for t in trades if t.is_aggressive]

        if len(aggressive) < 3:
            return False, 0

        # Check if aggressive trades hit multiple price levels
        prices = set(round(t.price, 4) for t in aggressive)
        num_levels = len(prices)

        # Check if prices are sequential (indicating sweep)
        if num_levels >= 3:
            sorted_prices = sorted(prices)
            direction_consistent = True

            # Check if all moves are in same direction
            for i in range(1, len(sorted_prices)):
                if i > 1:
                    prev_move = sorted_prices[i-1] - sorted_prices[i-2]
                    curr_move = sorted_prices[i] - sorted_prices[i-1]
                    if np.sign(prev_move) != np.sign(curr_move):
                        direction_consistent = False
                        break

            if direction_consistent:
                # Calculate aggressiveness
                aggressive_volume = sum(t.volume for t in aggressive)
                total_volume = sum(t.volume for t in trades)
                aggressive_ratio = aggressive_volume / total_volume if total_volume > 0 else 0

                if aggressive_ratio >= self.sweep_threshold:
                    return True, num_levels

        return False, num_levels

    def get_current_state(self) -> Dict[str, Any]:
        """Get current imbalance state."""
        signal = self._calculate_imbalance()

        return {
            'imbalance_type': signal.imbalance_type.value,
            'imbalance_ratio': round(signal.imbalance_ratio, 4),
            'buy_volume': signal.buy_volume,
            'sell_volume': signal.sell_volume,
            'is_persistent': signal.is_persistent,
            'is_sweep': signal.is_sweep,
            'sweep_levels': signal.sweep_levels,
            'signal_strength': round(signal.signal_strength, 4),
            'action': signal.get_action()
        }

    def get_multi_timeframe(self) -> Dict[str, TradeImbalanceSignal]:
        """
        Calculate imbalances over multiple time windows.

        Returns:
            Dict with timeframe -> signal
        """
        now = datetime.now()
        windows = {
            '10s': 10,
            '30s': 30,
            '1m': 60,
            '5m': 300
        }

        results = {}

        for name, seconds in windows.items():
            # Get trades for this window
            window_trades = [
                t for t in self.trades
                if (now - t.timestamp).total_seconds() <= seconds
            ]

            if len(window_trades) < 3:
                results[name] = TradeImbalanceSignal(
                    imbalance_type=ImbalanceType.NONE,
                    imbalance_ratio=0.0,
                    imbalance_volume=0.0,
                    confidence=0.0
                )
                continue

            buy_vol = sum(t.volume for t in window_trades if t.side > 0)
            sell_vol = sum(t.volume for t in window_trades if t.side < 0)
            total = buy_vol + sell_vol

            if total == 0:
                ratio = 0.0
            else:
                ratio = (buy_vol - sell_vol) / total

            results[name] = TradeImbalanceSignal(
                imbalance_type=self._classify_imbalance(ratio, window_trades),
                imbalance_ratio=ratio,
                imbalance_volume=buy_vol - sell_vol,
                confidence=min(1.0, len(window_trades) / 20),
                buy_volume=buy_vol,
                sell_volume=sell_vol,
                total_volume=total
            )

        return results

    def reset(self):
        """Reset detector state."""
        self.trades.clear()
        self.last_price = None
        self.imbalance_start = None
        self.imbalance_direction = 0
        self.price_levels.clear()
        self.avg_volume_per_trade = 0.0
