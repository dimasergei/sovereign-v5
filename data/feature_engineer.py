"""
Feature Engineer - Generates trading features from OHLCV data.

All indicator periods are derived from MarketCalibrator, not hardcoded.
Implements the Lossless Principle.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.lossless import FullCalibrationResult


logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Generates comprehensive trading features from OHLCV data.
    
    All indicator periods are derived from the calibrator, implementing
    the Lossless Principle - no hardcoded magic numbers.
    
    Usage:
        calibrator = MarketCalibrator()
        calibration = calibrator.calibrate_all(df)
        
        engineer = FeatureEngineer.from_calibration(calibration)
        df_with_features = engineer.add_all_features(df)
    """
    
    def __init__(
        self,
        fast_period: int = 8,
        slow_period: int = 21,
        signal_period: int = 9,
        atr_period: int = 14,
        overbought: float = 70.0,
        oversold: float = 30.0,
    ):
        """
        Initialize feature engineer with calibrated parameters.
        
        Args:
            fast_period: Fast indicator period (from calibrator)
            slow_period: Slow indicator period (from calibrator)
            signal_period: Signal smoothing period (from calibrator)
            atr_period: ATR calculation period (from calibrator)
            overbought: Overbought threshold (from calibrator)
            oversold: Oversold threshold (from calibrator)
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.atr_period = atr_period
        self.overbought = overbought
        self.oversold = oversold
        
        logger.info(
            f"FeatureEngineer initialized: fast={fast_period}, slow={slow_period}, "
            f"signal={signal_period}, atr={atr_period}"
        )
    
    @classmethod
    def from_calibration(cls, calibration: FullCalibrationResult) -> "FeatureEngineer":
        """
        Create FeatureEngineer from CalibrationResult.
        
        Args:
            calibration: Result from MarketCalibrator.calibrate_all()
            
        Returns:
            Configured FeatureEngineer instance
        """
        return cls(
            fast_period=int(calibration.fast_period),
            slow_period=int(calibration.slow_period),
            signal_period=int(calibration.signal_period),
            atr_period=int(calibration.atr_period),
            overbought=calibration.overbought_threshold,
            oversold=calibration.oversold_threshold,
        )
    
    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all trading features to OHLCV DataFrame.
        
        Args:
            df: DataFrame with open, high, low, close, volume columns
            
        Returns:
            DataFrame with all features added
        """
        df = df.copy()
        
        # Normalize column names
        df.columns = df.columns.str.lower()
        
        # Validate required columns
        required = {"open", "high", "low", "close"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"DataFrame must have columns: {required}")
        
        # Add volume if missing
        if "volume" not in df.columns:
            df["volume"] = 0
        
        logger.debug(f"Adding features to {len(df)} rows")
        
        # Add all feature categories
        df = self._add_returns(df)
        df = self._add_volatility(df)
        df = self._add_momentum(df)
        df = self._add_trend(df)
        df = self._add_volume_features(df)
        df = self._add_bands(df)
        df = self._add_market_structure(df)
        df = self._add_derived_signals(df)
        
        logger.debug(f"Added {len(df.columns)} total columns")
        return df
    
    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return-based features."""
        close = df["close"]
        
        df["returns"] = close.pct_change()
        df["log_returns"] = np.log(close / close.shift(1))
        
        for period in [self.fast_period, self.slow_period]:
            df[f"returns_{period}"] = close.pct_change(period)
        
        return df
    
    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        open_ = df["open"]
        
        # Rolling volatility
        df["volatility"] = df["log_returns"].rolling(self.atr_period).std()
        
        # True Range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR
        df["atr"] = df["tr"].rolling(self.atr_period).mean()
        df["atr_pct"] = df["atr"] / close * 100
        
        # Parkinson volatility
        hl_ratio = np.log(high / low)
        df["parkinson_vol"] = np.sqrt(
            (1 / (4 * np.log(2))) * (hl_ratio ** 2).rolling(self.atr_period).mean()
        )
        
        # Garman-Klass volatility
        hl = np.log(high / low) ** 2
        co = np.log(close / open_) ** 2
        df["garman_klass_vol"] = np.sqrt(
            (0.5 * hl - (2 * np.log(2) - 1) * co).rolling(self.atr_period).mean()
        )
        
        return df
    
    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        
        # RSI
        df["rsi"] = self._calculate_rsi(close, self.atr_period)
        
        # Stochastic
        stoch_k, stoch_d = self._calculate_stochastic(
            high, low, close, self.atr_period, self.signal_period
        )
        df["stochastic_k"] = stoch_k
        df["stochastic_d"] = stoch_d
        
        # MACD
        macd, signal, hist = self._calculate_macd(
            close, self.fast_period, self.slow_period, self.signal_period
        )
        df["macd"] = macd
        df["macd_signal"] = signal
        df["macd_hist"] = hist
        
        # Williams %R
        df["williams_r"] = self._calculate_williams_r(high, low, close, self.atr_period)
        
        # CCI
        df["cci"] = self._calculate_cci(high, low, close, self.atr_period)
        
        # ROC and Momentum
        df["roc"] = (close / close.shift(self.atr_period) - 1) * 100
        df["momentum"] = close - close.shift(self.atr_period)
        
        return df
    
    def _add_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        
        # Moving Averages
        df["sma_fast"] = close.rolling(self.fast_period).mean()
        df["sma_slow"] = close.rolling(self.slow_period).mean()
        df["ema_fast"] = close.ewm(span=self.fast_period, adjust=False).mean()
        df["ema_slow"] = close.ewm(span=self.slow_period, adjust=False).mean()
        
        # ADX
        adx, plus_di, minus_di = self._calculate_adx(high, low, close, self.atr_period)
        df["adx"] = adx
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di
        
        # Supertrend
        df["supertrend"], df["supertrend_direction"] = self._calculate_supertrend(
            high, low, close, self.atr_period, multiplier=3.0
        )
        
        # Price vs MAs
        df["price_vs_sma_fast"] = (close - df["sma_fast"]) / df["sma_fast"] * 100
        df["price_vs_sma_slow"] = (close - df["sma_slow"]) / df["sma_slow"] * 100
        df["ma_cross"] = np.where(df["sma_fast"] > df["sma_slow"], 1, -1)
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        
        df["volume_sma"] = volume.rolling(self.slow_period).mean()
        df["volume_ratio"] = volume / (df["volume_sma"] + 1e-10)
        df["obv"] = self._calculate_obv(close, volume)
        df["vwap"] = self._calculate_vwap(high, low, close, volume)
        df["mfi"] = self._calculate_mfi(high, low, close, volume, self.atr_period)
        
        return df
    
    def _add_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add band-based indicators."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        atr = df["atr"]
        
        # Bollinger Bands
        df["bb_mid"] = close.rolling(self.slow_period).mean()
        bb_std = close.rolling(self.slow_period).std()
        df["bb_upper"] = df["bb_mid"] + 2 * bb_std
        df["bb_lower"] = df["bb_mid"] - 2 * bb_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"] * 100
        df["bb_pct"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)
        
        # Keltner Channels
        ema = close.ewm(span=self.slow_period, adjust=False).mean()
        df["keltner_upper"] = ema + 2 * atr
        df["keltner_lower"] = ema - 2 * atr
        
        # Donchian Channels
        df["donchian_high"] = high.rolling(self.slow_period).max()
        df["donchian_low"] = low.rolling(self.slow_period).min()
        df["donchian_mid"] = (df["donchian_high"] + df["donchian_low"]) / 2
        df["donchian_pct"] = (close - df["donchian_low"]) / (
            df["donchian_high"] - df["donchian_low"] + 1e-10
        )
        
        return df
    
    def _add_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market structure features."""
        high = df["high"].values
        low = df["low"].values
        n = len(df)
        
        # Higher highs and lower lows
        higher_high = np.zeros(n)
        lower_low = np.zeros(n)
        
        for i in range(1, n):
            if high[i] > high[i - 1]:
                higher_high[i] = 1
            if low[i] < low[i - 1]:
                lower_low[i] = 1
        
        df["higher_high"] = higher_high
        df["lower_low"] = lower_low
        
        # Swing highs and lows
        lookback = max(3, self.fast_period // 2)
        swing_high = np.zeros(n)
        swing_low = np.zeros(n)
        
        for i in range(lookback, n - lookback):
            if high[i] == max(high[i - lookback:i + lookback + 1]):
                swing_high[i] = 1
            if low[i] == min(low[i - lookback:i + lookback + 1]):
                swing_low[i] = 1
        
        df["swing_high"] = swing_high
        df["swing_low"] = swing_low
        
        # Structure bias
        swing_high_count = pd.Series(swing_high).rolling(self.slow_period).sum()
        swing_low_count = pd.Series(swing_low).rolling(self.slow_period).sum()
        df["structure_bias"] = swing_high_count - swing_low_count
        
        return df
    
    def _add_derived_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived trading signals."""
        # RSI extremes
        df["rsi_overbought"] = (df["rsi"] > self.overbought).astype(int)
        df["rsi_oversold"] = (df["rsi"] < self.oversold).astype(int)
        
        # BB extremes
        df["bb_overbought"] = (df["bb_pct"] > 1.0).astype(int)
        df["bb_oversold"] = (df["bb_pct"] < 0.0).astype(int)
        
        # Trend strength composite
        df["trend_strength"] = (
            df["adx"] / 100 * 0.4 +
            np.abs(df["macd_hist"]) / (df["close"] + 1e-10) * 1000 * 0.3 +
            np.abs(df["price_vs_sma_slow"]) / 10 * 0.3
        ).clip(0, 1)
        
        # Volatility regime
        vol_median = df["atr_pct"].rolling(self.slow_period * 5).median()
        df["high_volatility"] = (df["atr_pct"] > vol_median * 1.5).astype(int)
        df["low_volatility"] = (df["atr_pct"] < vol_median * 0.5).astype(int)
        
        return df
    
    # ==================== INDICATOR CALCULATIONS ====================
    
    def _calculate_rsi(self, close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def _calculate_stochastic(
        self, high: pd.Series, low: pd.Series, close: pd.Series,
        k_period: int, d_period: int
    ) -> Tuple[pd.Series, pd.Series]:
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        stoch_d = stoch_k.rolling(d_period).mean()
        
        return stoch_k, stoch_d
    
    def _calculate_macd(
        self, close: pd.Series, fast: int, slow: int, signal: int
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        
        return macd, macd_signal, macd_hist
    
    def _calculate_williams_r(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int
    ) -> pd.Series:
        highest_high = high.rolling(period).max()
        lowest_low = low.rolling(period).min()
        
        return -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    def _calculate_cci(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int
    ) -> pd.Series:
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(period).mean()
        mean_dev = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        return (tp - sma_tp) / (0.015 * mean_dev + 1e-10)
    
    def _calculate_adx(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        tr_smooth = pd.Series(tr).ewm(span=period, adjust=False).mean()
        plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, adjust=False).mean()
        minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, adjust=False).mean()
        
        plus_di = 100 * plus_dm_smooth / (tr_smooth + 1e-10)
        minus_di = 100 * minus_dm_smooth / (tr_smooth + 1e-10)
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return adx, plus_di, minus_di
    
    def _calculate_supertrend(
        self, high: pd.Series, low: pd.Series, close: pd.Series,
        period: int, multiplier: float
    ) -> Tuple[pd.Series, pd.Series]:
        hl2 = (high + low) / 2
        
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr
        
        n = len(close)
        supertrend = np.zeros(n)
        direction = np.zeros(n)
        
        supertrend[0] = upper_band.iloc[0]
        direction[0] = 1
        
        for i in range(1, n):
            if close.iloc[i] > supertrend[i-1]:
                supertrend[i] = max(lower_band.iloc[i], supertrend[i-1]) if direction[i-1] == 1 else lower_band.iloc[i]
                direction[i] = 1
            else:
                supertrend[i] = min(upper_band.iloc[i], supertrend[i-1]) if direction[i-1] == -1 else upper_band.iloc[i]
                direction[i] = -1
        
        return pd.Series(supertrend, index=close.index), pd.Series(direction, index=close.index)
    
    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        direction = np.sign(close.diff())
        return (direction * volume).cumsum()
    
    def _calculate_vwap(
        self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
    ) -> pd.Series:
        tp = (high + low + close) / 3
        cumulative_tp_vol = (tp * volume).cumsum()
        cumulative_vol = volume.cumsum()
        return cumulative_tp_vol / (cumulative_vol + 1e-10)
    
    def _calculate_mfi(
        self, high: pd.Series, low: pd.Series, close: pd.Series,
        volume: pd.Series, period: int
    ) -> pd.Series:
        tp = (high + low + close) / 3
        mf = tp * volume
        
        positive_mf = mf.where(tp > tp.shift(1), 0)
        negative_mf = mf.where(tp < tp.shift(1), 0)
        
        positive_mf_sum = positive_mf.rolling(period).sum()
        negative_mf_sum = negative_mf.rolling(period).sum()
        
        return 100 - (100 / (1 + positive_mf_sum / (negative_mf_sum + 1e-10)))
