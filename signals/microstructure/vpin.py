"""
VPIN (Volume-Synchronized Probability of Informed Trading)

Implements the VPIN metric for detecting informed trading activity.
VPIN measures order flow toxicity by analyzing the imbalance between
buy and sell volumes in volume-synchronized buckets.

Key concepts:
- Volume buckets: Fixed-volume bars instead of time bars
- Trade classification: Lee-Ready tick rule or bulk classification
- Order imbalance: Absolute difference between buy/sell volumes
- Toxicity: High VPIN indicates high probability of informed trading

Reference: Easley, Lopez de Prado, O'Hara (2011)
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class VPINBucket:
    """A single volume bucket for VPIN calculation."""
    bucket_id: int
    total_volume: float
    buy_volume: float
    sell_volume: float
    start_time: datetime
    end_time: datetime

    @property
    def imbalance(self) -> float:
        """Absolute order imbalance."""
        return abs(self.buy_volume - self.sell_volume)

    @property
    def imbalance_ratio(self) -> float:
        """Order imbalance as ratio of total volume."""
        if self.total_volume == 0:
            return 0.0
        return self.imbalance / self.total_volume

    @property
    def buy_ratio(self) -> float:
        """Proportion of buy volume."""
        if self.total_volume == 0:
            return 0.5
        return self.buy_volume / self.total_volume

    @property
    def direction(self) -> float:
        """Net direction: positive = buying, negative = selling."""
        if self.total_volume == 0:
            return 0.0
        return (self.buy_volume - self.sell_volume) / self.total_volume


@dataclass
class VPINSignal:
    """VPIN trading signal output."""
    vpin: float  # 0 to 1, higher = more toxic
    direction: float  # -1 to 1, net order flow direction
    toxicity_level: str  # "low", "moderate", "high", "extreme"
    confidence: float  # 0 to 1

    # Bucket information
    num_buckets: int = 0
    avg_bucket_imbalance: float = 0.0

    # Trend
    vpin_trend: float = 0.0  # Change in VPIN over recent period
    is_increasing: bool = False

    # Action recommendation
    recommended_action: str = "normal"  # "normal", "reduce_size", "avoid"

    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class VPINCalculator:
    """
    Calculates VPIN (Volume-synchronized Probability of Informed Trading).

    VPIN is a real-time measure of order flow toxicity. High VPIN values
    indicate a higher probability of informed trading, which often precedes
    large price moves.

    Usage:
        vpin = VPINCalculator(
            bucket_size=1000,  # Volume per bucket
            num_buckets=50     # Buckets for VPIN calculation
        )

        # Process trades
        for trade in trade_stream:
            signal = vpin.process_trade(
                price=trade.price,
                volume=trade.volume,
                side=trade.side,  # 'buy', 'sell', or None
                timestamp=trade.time
            )

            if signal.vpin > 0.7:
                # High toxicity - reduce position sizes
                pass
    """

    def __init__(
        self,
        bucket_size: float = 1000,
        num_buckets: int = 50,
        classification_method: str = "tick",  # "tick" or "bulk"
        toxicity_thresholds: Dict[str, float] = None
    ):
        """
        Initialize VPIN calculator.

        Args:
            bucket_size: Volume per bucket
            num_buckets: Number of buckets to use for VPIN calculation
            classification_method: "tick" for tick rule, "bulk" for bulk volume classification
            toxicity_thresholds: Dict with "moderate", "high", "extreme" thresholds
        """
        self.bucket_size = bucket_size
        self.num_buckets = num_buckets
        self.classification_method = classification_method

        self.toxicity_thresholds = toxicity_thresholds or {
            'moderate': 0.4,
            'high': 0.6,
            'extreme': 0.8
        }

        # Completed buckets
        self.buckets: deque = deque(maxlen=num_buckets)

        # Current bucket (accumulating)
        self.current_buy: float = 0.0
        self.current_sell: float = 0.0
        self.current_volume: float = 0.0
        self.current_start: datetime = datetime.now()
        self.bucket_count: int = 0

        # Price tracking for tick rule
        self.last_price: Optional[float] = None
        self.last_trade_side: int = 0  # 1 = buy, -1 = sell, 0 = unknown

        # VPIN history for trend
        self.vpin_history: deque = deque(maxlen=20)

    def process_trade(
        self,
        price: float,
        volume: float,
        side: Optional[str] = None,
        timestamp: datetime = None
    ) -> VPINSignal:
        """
        Process a trade and update VPIN.

        Args:
            price: Trade price
            volume: Trade volume
            side: Trade side ('buy', 'sell', or None for auto-classification)
            timestamp: Trade timestamp

        Returns:
            VPINSignal with current VPIN metrics
        """
        timestamp = timestamp or datetime.now()

        # Classify trade direction
        if side:
            trade_direction = 1 if side.lower() == 'buy' else -1
        else:
            trade_direction = self._classify_trade(price)

        # Update last price
        self.last_price = price
        self.last_trade_side = trade_direction

        # Distribute volume to buy/sell
        if self.classification_method == "bulk":
            # Bulk volume classification
            buy_vol, sell_vol = self._bulk_classify(volume, trade_direction)
        else:
            # Standard classification
            if trade_direction > 0:
                buy_vol, sell_vol = volume, 0.0
            elif trade_direction < 0:
                buy_vol, sell_vol = 0.0, volume
            else:
                # Unknown: split evenly
                buy_vol, sell_vol = volume / 2, volume / 2

        # Add to current bucket
        self.current_buy += buy_vol
        self.current_sell += sell_vol
        self.current_volume += volume

        # Check if bucket is complete
        while self.current_volume >= self.bucket_size:
            self._complete_bucket(timestamp)

        # Calculate VPIN
        return self._calculate_signal()

    def _classify_trade(self, price: float) -> int:
        """
        Classify trade direction using tick rule.

        Returns:
            1 for buy, -1 for sell, 0 for unknown
        """
        if self.last_price is None:
            return 0

        if price > self.last_price:
            return 1  # Uptick = buy
        elif price < self.last_price:
            return -1  # Downtick = sell
        else:
            # Zero tick: use last classification
            return self.last_trade_side

    def _bulk_classify(
        self,
        volume: float,
        direction: int
    ) -> Tuple[float, float]:
        """
        Bulk volume classification using probability.

        Based on CDF of standard normal for price changes.

        Returns:
            (buy_volume, sell_volume)
        """
        if direction == 0:
            return volume / 2, volume / 2

        # Probability of buy (simplified)
        if direction > 0:
            p_buy = 0.7  # 70% likely to be buyer-initiated
        else:
            p_buy = 0.3  # 30% likely to be buyer-initiated

        buy_vol = volume * p_buy
        sell_vol = volume * (1 - p_buy)

        return buy_vol, sell_vol

    def _complete_bucket(self, timestamp: datetime):
        """Complete current bucket and start new one."""
        # Create bucket with proportional volume
        excess = self.current_volume - self.bucket_size

        # Ratio of volume for this bucket
        if self.current_volume > 0:
            ratio = self.bucket_size / self.current_volume
        else:
            ratio = 1.0

        bucket = VPINBucket(
            bucket_id=self.bucket_count,
            total_volume=self.bucket_size,
            buy_volume=self.current_buy * ratio,
            sell_volume=self.current_sell * ratio,
            start_time=self.current_start,
            end_time=timestamp
        )

        self.buckets.append(bucket)
        self.bucket_count += 1

        # Carry over excess to new bucket
        remaining_ratio = 1 - ratio
        self.current_buy = self.current_buy * remaining_ratio
        self.current_sell = self.current_sell * remaining_ratio
        self.current_volume = excess
        self.current_start = timestamp

    def _calculate_signal(self) -> VPINSignal:
        """Calculate current VPIN signal."""
        if len(self.buckets) < 5:
            return VPINSignal(
                vpin=0.0,
                direction=0.0,
                toxicity_level="low",
                confidence=len(self.buckets) / self.num_buckets,
                num_buckets=len(self.buckets),
                recommended_action="normal"
            )

        # Calculate VPIN
        buckets = list(self.buckets)
        imbalances = [b.imbalance_ratio for b in buckets]
        vpin = np.mean(imbalances)

        # Track VPIN history
        self.vpin_history.append(vpin)

        # Calculate direction (net order flow)
        directions = [b.direction for b in buckets]
        direction = np.mean(directions)

        # VPIN trend
        if len(self.vpin_history) >= 5:
            recent = list(self.vpin_history)
            vpin_trend = recent[-1] - np.mean(recent[:-1])
            is_increasing = vpin_trend > 0.02
        else:
            vpin_trend = 0.0
            is_increasing = False

        # Determine toxicity level
        if vpin >= self.toxicity_thresholds['extreme']:
            toxicity_level = "extreme"
            recommended_action = "avoid"
        elif vpin >= self.toxicity_thresholds['high']:
            toxicity_level = "high"
            recommended_action = "reduce_size"
        elif vpin >= self.toxicity_thresholds['moderate']:
            toxicity_level = "moderate"
            recommended_action = "reduce_size"
        else:
            toxicity_level = "low"
            recommended_action = "normal"

        # Confidence based on data sufficiency
        confidence = min(1.0, len(self.buckets) / self.num_buckets)

        return VPINSignal(
            vpin=vpin,
            direction=direction,
            toxicity_level=toxicity_level,
            confidence=confidence,
            num_buckets=len(self.buckets),
            avg_bucket_imbalance=np.mean(imbalances),
            vpin_trend=vpin_trend,
            is_increasing=is_increasing,
            recommended_action=recommended_action,
            metadata={
                'bucket_count': self.bucket_count,
                'current_bucket_fill': self.current_volume / self.bucket_size,
                'thresholds': self.toxicity_thresholds.copy()
            }
        )

    def get_toxicity_for_position_sizing(self) -> float:
        """
        Get toxicity multiplier for position sizing.

        Returns:
            Multiplier 0-1 (1 = full size, lower = reduce size)
        """
        signal = self._calculate_signal()

        if signal.toxicity_level == "extreme":
            return 0.0
        elif signal.toxicity_level == "high":
            return 0.3
        elif signal.toxicity_level == "moderate":
            return 0.6
        else:
            return 1.0

    def reset(self):
        """Reset calculator state."""
        self.buckets.clear()
        self.current_buy = 0.0
        self.current_sell = 0.0
        self.current_volume = 0.0
        self.current_start = datetime.now()
        self.bucket_count = 0
        self.last_price = None
        self.last_trade_side = 0
        self.vpin_history.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get VPIN statistics."""
        signal = self._calculate_signal()

        return {
            'vpin': signal.vpin,
            'toxicity_level': signal.toxicity_level,
            'direction': signal.direction,
            'num_buckets': len(self.buckets),
            'total_buckets_created': self.bucket_count,
            'current_bucket_fill': self.current_volume / self.bucket_size,
            'vpin_trend': signal.vpin_trend,
            'is_increasing': signal.is_increasing,
            'recommended_action': signal.recommended_action
        }
