"""
Order Flow Analysis Module

Analyzes order flow dynamics to detect institutional activity,
liquidity changes, and adverse selection conditions.

Key signals:
- Bid-ask volume differential
- Trade arrival intensity (simplified Hawkes process)
- Spread dynamics and widening detection
- Liquidity withdrawal detection
- Adverse selection scoring
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


logger = logging.getLogger(__name__)


class LiquidityState(Enum):
    """Market liquidity state."""
    NORMAL = "normal"
    THIN = "thin"
    WITHDRAWING = "withdrawing"
    STRESSED = "stressed"
    RECOVERING = "recovering"


@dataclass
class QuoteUpdate:
    """Single quote/tick update."""
    timestamp: datetime
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    last_price: float
    last_volume: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        if self.mid == 0:
            return 0.0
        return (self.spread / self.mid) * 10000

    @property
    def quote_imbalance(self) -> float:
        """Quote size imbalance: positive = bid-heavy, negative = ask-heavy."""
        total = self.bid_size + self.ask_size
        if total == 0:
            return 0.0
        return (self.bid_size - self.ask_size) / total


@dataclass
class OrderFlowSignal:
    """Order flow analysis signal."""
    direction: float  # -1 to 1 (selling to buying pressure)
    intensity: float  # 0 to 1 (trade arrival intensity)
    liquidity_score: float  # 0 to 1 (higher = more liquid)
    adverse_selection: float  # 0 to 1 (probability of informed flow)

    # State
    liquidity_state: LiquidityState = LiquidityState.NORMAL
    is_spread_widening: bool = False

    # Quote analysis
    quote_imbalance: float = 0.0
    avg_spread_bps: float = 0.0
    spread_zscore: float = 0.0

    # Trade analysis
    trade_imbalance: float = 0.0
    net_buy_volume: float = 0.0

    # Confidence
    confidence: float = 0.0

    # Recommendation
    recommended_action: str = "normal"  # "normal", "reduce_size", "widen_spread", "avoid"

    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class OrderFlowAnalyzer:
    """
    Analyzes order flow for trading signals.

    Combines quote data and trade data to detect:
    - Directional order flow (buying vs selling pressure)
    - Trade intensity patterns
    - Liquidity conditions
    - Adverse selection risk

    Usage:
        analyzer = OrderFlowAnalyzer(buffer_size=1000)

        # Process quote updates
        signal = analyzer.process_quote(
            bid=99.90, ask=100.10,
            bid_size=100, ask_size=80,
            last_price=100.05, last_volume=10
        )

        if signal.adverse_selection > 0.7:
            # High adverse selection - reduce market making
            pass
    """

    def __init__(
        self,
        buffer_size: int = 1000,
        intensity_window: int = 50,
        spread_window: int = 100,
        adverse_threshold: float = 0.6
    ):
        """
        Initialize order flow analyzer.

        Args:
            buffer_size: Number of quotes to keep in buffer
            intensity_window: Window for intensity calculation
            spread_window: Window for spread statistics
            adverse_threshold: Threshold for adverse selection warning
        """
        self.buffer_size = buffer_size
        self.intensity_window = intensity_window
        self.spread_window = spread_window
        self.adverse_threshold = adverse_threshold

        # Data buffers
        self.quote_buffer: deque = deque(maxlen=buffer_size)
        self.trade_signs: deque = deque(maxlen=buffer_size)
        self.trade_volumes: deque = deque(maxlen=buffer_size)
        self.trade_times: deque = deque(maxlen=buffer_size)

        # Running statistics
        self.avg_spread: float = 0.0
        self.spread_std: float = 0.0
        self.avg_intensity: float = 0.0

        # Hawkes process state (simplified)
        self.hawkes_intensity: float = 0.0
        self.hawkes_decay: float = 0.95

        # Last quote
        self.last_quote: Optional[QuoteUpdate] = None

    def process_quote(
        self,
        bid: float,
        ask: float,
        bid_size: float,
        ask_size: float,
        last_price: float,
        last_volume: float,
        timestamp: datetime = None
    ) -> OrderFlowSignal:
        """
        Process a quote update and generate signal.

        Args:
            bid: Best bid price
            ask: Best ask price
            bid_size: Bid volume
            ask_size: Ask volume
            last_price: Last trade price
            last_volume: Last trade volume
            timestamp: Quote timestamp

        Returns:
            OrderFlowSignal with analysis results
        """
        timestamp = timestamp or datetime.now()

        quote = QuoteUpdate(
            timestamp=timestamp,
            bid=bid,
            ask=ask,
            bid_size=bid_size,
            ask_size=ask_size,
            last_price=last_price,
            last_volume=last_volume
        )

        # Classify trade direction
        trade_sign = self._classify_trade(quote)
        self.trade_signs.append(trade_sign)
        self.trade_volumes.append(last_volume)
        self.trade_times.append(timestamp)

        # Update buffers and statistics
        self.quote_buffer.append(quote)
        self._update_statistics(quote)

        # Calculate signals
        direction = self._calculate_direction()
        intensity = self._calculate_intensity()
        liquidity_score, liquidity_state = self._analyze_liquidity(quote)
        adverse_selection = self._calculate_adverse_selection()

        # Quote analysis
        quote_imbalance = quote.quote_imbalance
        spread_zscore = self._calculate_spread_zscore(quote)
        is_spread_widening = spread_zscore > 1.5

        # Trade analysis
        trade_imbalance = self._calculate_trade_imbalance()
        net_buy_volume = self._calculate_net_buy_volume()

        # Confidence
        confidence = min(1.0, len(self.quote_buffer) / 100)

        # Recommendation
        recommended_action = self._get_recommendation(
            adverse_selection, liquidity_state, spread_zscore
        )

        self.last_quote = quote

        return OrderFlowSignal(
            direction=direction,
            intensity=intensity,
            liquidity_score=liquidity_score,
            adverse_selection=adverse_selection,
            liquidity_state=liquidity_state,
            is_spread_widening=is_spread_widening,
            quote_imbalance=quote_imbalance,
            avg_spread_bps=self.avg_spread * 10000 / quote.mid if quote.mid > 0 else 0,
            spread_zscore=spread_zscore,
            trade_imbalance=trade_imbalance,
            net_buy_volume=net_buy_volume,
            confidence=confidence,
            recommended_action=recommended_action,
            metadata={
                'buffer_size': len(self.quote_buffer),
                'hawkes_intensity': self.hawkes_intensity
            }
        )

    def _classify_trade(self, quote: QuoteUpdate) -> int:
        """
        Classify trade as buy or sell using Lee-Ready algorithm.

        Returns:
            1 for buy, -1 for sell, 0 for undetermined
        """
        if self.last_quote is None:
            return 0

        # Tick test
        if quote.last_price > self.last_quote.last_price:
            return 1  # Uptick = buy
        elif quote.last_price < self.last_quote.last_price:
            return -1  # Downtick = sell

        # Quote test (for zero-tick)
        if quote.last_price > quote.mid:
            return 1  # Above mid = buy
        elif quote.last_price < quote.mid:
            return -1  # Below mid = sell

        return 0  # At mid = undetermined

    def _update_statistics(self, quote: QuoteUpdate):
        """Update running statistics."""
        alpha = 0.02  # Smoothing factor

        # Update spread statistics
        if self.avg_spread == 0:
            self.avg_spread = quote.spread
            self.spread_std = 0.0
        else:
            # Exponential moving average
            self.avg_spread = alpha * quote.spread + (1 - alpha) * self.avg_spread

            # Running standard deviation
            deviation = quote.spread - self.avg_spread
            self.spread_std = alpha * abs(deviation) + (1 - alpha) * self.spread_std

        # Update Hawkes intensity
        self._update_hawkes_intensity()

    def _update_hawkes_intensity(self):
        """Update simplified Hawkes process intensity."""
        if len(self.trade_times) < 2:
            return

        # Decay existing intensity
        self.hawkes_intensity *= self.hawkes_decay

        # Add jump for new trade
        self.hawkes_intensity += 0.1

        # Cap intensity
        self.hawkes_intensity = min(1.0, self.hawkes_intensity)

    def _calculate_direction(self) -> float:
        """
        Calculate net order flow direction.

        Returns:
            -1 to 1 (selling to buying pressure)
        """
        if len(self.trade_signs) < 10:
            return 0.0

        signs = list(self.trade_signs)[-100:]
        volumes = list(self.trade_volumes)[-100:]

        # Volume-weighted direction
        buy_volume = sum(v for s, v in zip(signs, volumes) if s > 0)
        sell_volume = sum(v for s, v in zip(signs, volumes) if s < 0)
        total = buy_volume + sell_volume

        if total == 0:
            return 0.0

        return (buy_volume - sell_volume) / total

    def _calculate_intensity(self) -> float:
        """
        Calculate trade arrival intensity.

        Returns:
            0 to 1 (relative intensity)
        """
        if len(self.trade_times) < 10:
            return 0.0

        times = list(self.trade_times)[-self.intensity_window:]

        # Calculate inter-arrival times
        deltas = []
        for i in range(1, len(times)):
            delta = (times[i] - times[i-1]).total_seconds()
            if delta > 0:
                deltas.append(delta)

        if not deltas:
            return 0.0

        # Current rate vs average
        recent_deltas = deltas[-10:] if len(deltas) >= 10 else deltas
        current_rate = 1 / (np.mean(recent_deltas) + 0.001)
        avg_rate = 1 / (np.mean(deltas) + 0.001)

        if avg_rate == 0:
            return 0.0

        # Relative intensity (capped)
        intensity_ratio = current_rate / avg_rate

        return min(1.0, intensity_ratio / 3)  # Normalize assuming 3x is max

    def _analyze_liquidity(
        self,
        quote: QuoteUpdate
    ) -> Tuple[float, LiquidityState]:
        """
        Analyze liquidity conditions.

        Returns:
            (liquidity_score, liquidity_state)
        """
        if len(self.quote_buffer) < 10:
            return 0.5, LiquidityState.NORMAL

        quotes = list(self.quote_buffer)

        # Spread analysis
        spreads = [q.spread for q in quotes[-50:]]
        current_spread = quote.spread
        avg_spread = np.mean(spreads)
        spread_ratio = current_spread / (avg_spread + 1e-10)

        # Size analysis
        total_sizes = [q.bid_size + q.ask_size for q in quotes[-50:]]
        current_size = quote.bid_size + quote.ask_size
        avg_size = np.mean(total_sizes)
        size_ratio = current_size / (avg_size + 1e-10)

        # Liquidity score (higher = more liquid)
        # Narrow spread and large size = more liquid
        liquidity_score = 0.5 * (1 / spread_ratio) + 0.5 * size_ratio
        liquidity_score = min(1.0, max(0.0, liquidity_score / 2))

        # Determine state
        if spread_ratio > 2.0 and size_ratio < 0.5:
            state = LiquidityState.STRESSED
        elif spread_ratio > 1.5 or size_ratio < 0.6:
            # Check if recovering
            if len(quotes) >= 20:
                recent_spreads = [q.spread for q in quotes[-10:]]
                older_spreads = [q.spread for q in quotes[-20:-10]]
                if np.mean(recent_spreads) < np.mean(older_spreads):
                    state = LiquidityState.RECOVERING
                else:
                    state = LiquidityState.THIN
            else:
                state = LiquidityState.THIN
        elif spread_ratio > 1.2:
            state = LiquidityState.WITHDRAWING
        else:
            state = LiquidityState.NORMAL

        return liquidity_score, state

    def _calculate_adverse_selection(self) -> float:
        """
        Calculate probability of adverse selection.

        Adverse selection occurs when market makers are picked off
        by informed traders.

        Returns:
            0 to 1 (probability of informed trading)
        """
        if len(self.quote_buffer) < 20:
            return 0.0

        quotes = list(self.quote_buffer)

        # Component 1: Trade-quote correlation
        # If trades hit the same side repeatedly, may be informed
        signs = list(self.trade_signs)[-50:]
        if signs:
            consecutive_same = 0
            max_consecutive = 0
            prev_sign = 0
            for sign in signs:
                if sign == prev_sign and sign != 0:
                    consecutive_same += 1
                    max_consecutive = max(max_consecutive, consecutive_same)
                else:
                    consecutive_same = 0
                prev_sign = sign

            trade_correlation = min(1.0, max_consecutive / 10)
        else:
            trade_correlation = 0.0

        # Component 2: Spread widening after trades
        spreads = [q.spread for q in quotes[-50:]]
        recent_spreads = spreads[-10:] if len(spreads) >= 10 else spreads
        older_spreads = spreads[:-10] if len(spreads) >= 10 else [self.avg_spread]

        if older_spreads:
            spread_widening = np.mean(recent_spreads) / (np.mean(older_spreads) + 1e-10) - 1
            spread_component = min(1.0, max(0.0, spread_widening * 5))
        else:
            spread_component = 0.0

        # Component 3: Size depletion
        sizes = [q.bid_size + q.ask_size for q in quotes[-50:]]
        recent_sizes = sizes[-10:] if len(sizes) >= 10 else sizes
        older_sizes = sizes[:-10] if len(sizes) >= 10 else sizes

        if older_sizes:
            size_depletion = 1 - np.mean(recent_sizes) / (np.mean(older_sizes) + 1e-10)
            size_component = min(1.0, max(0.0, size_depletion * 2))
        else:
            size_component = 0.0

        # Combine components
        adverse_selection = (
            0.4 * trade_correlation +
            0.35 * spread_component +
            0.25 * size_component
        )

        return min(1.0, adverse_selection)

    def _calculate_spread_zscore(self, quote: QuoteUpdate) -> float:
        """Calculate z-score of current spread."""
        if self.spread_std == 0:
            return 0.0

        return (quote.spread - self.avg_spread) / (self.spread_std + 1e-10)

    def _calculate_trade_imbalance(self) -> float:
        """Calculate trade volume imbalance."""
        if len(self.trade_signs) < 10:
            return 0.0

        signs = list(self.trade_signs)[-100:]
        volumes = list(self.trade_volumes)[-100:]

        buy_vol = sum(v for s, v in zip(signs, volumes) if s > 0)
        sell_vol = sum(v for s, v in zip(signs, volumes) if s < 0)
        total = buy_vol + sell_vol

        if total == 0:
            return 0.0

        return (buy_vol - sell_vol) / total

    def _calculate_net_buy_volume(self) -> float:
        """Calculate net buy volume (buy - sell)."""
        if len(self.trade_signs) < 5:
            return 0.0

        signs = list(self.trade_signs)[-100:]
        volumes = list(self.trade_volumes)[-100:]

        return sum(v * s for s, v in zip(signs, volumes))

    def _get_recommendation(
        self,
        adverse_selection: float,
        liquidity_state: LiquidityState,
        spread_zscore: float
    ) -> str:
        """Get trading recommendation based on conditions."""
        if adverse_selection > 0.8 or liquidity_state == LiquidityState.STRESSED:
            return "avoid"
        elif adverse_selection > 0.6 or spread_zscore > 2.0:
            return "reduce_size"
        elif adverse_selection > 0.4 or liquidity_state == LiquidityState.THIN:
            return "widen_spread"
        else:
            return "normal"

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of current order flow state."""
        if not self.quote_buffer:
            return {'status': 'no_data'}

        signal = self.process_quote(
            bid=self.last_quote.bid if self.last_quote else 0,
            ask=self.last_quote.ask if self.last_quote else 0,
            bid_size=self.last_quote.bid_size if self.last_quote else 0,
            ask_size=self.last_quote.ask_size if self.last_quote else 0,
            last_price=self.last_quote.last_price if self.last_quote else 0,
            last_volume=0
        ) if self.last_quote else None

        if not signal:
            return {'status': 'no_data'}

        return {
            'direction': signal.direction,
            'intensity': signal.intensity,
            'liquidity_score': signal.liquidity_score,
            'liquidity_state': signal.liquidity_state.value,
            'adverse_selection': signal.adverse_selection,
            'trade_imbalance': signal.trade_imbalance,
            'spread_zscore': signal.spread_zscore,
            'recommended_action': signal.recommended_action
        }

    def reset(self):
        """Reset analyzer state."""
        self.quote_buffer.clear()
        self.trade_signs.clear()
        self.trade_volumes.clear()
        self.trade_times.clear()
        self.avg_spread = 0.0
        self.spread_std = 0.0
        self.hawkes_intensity = 0.0
        self.last_quote = None
