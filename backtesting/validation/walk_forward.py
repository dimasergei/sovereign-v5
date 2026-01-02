"""
Walk-Forward Validation - Robust out-of-sample testing.

Implements rolling window validation to test strategy robustness
and detect overfitting.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Result from a single walk-forward window."""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    # Performance metrics
    train_sharpe: float
    test_sharpe: float
    train_return: float
    test_return: float
    test_max_dd: float
    
    # Trade metrics
    n_trades: int
    win_rate: float
    
    # Model info
    model_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WalkForwardAnalysis:
    """Complete walk-forward analysis results."""
    windows: List[WalkForwardResult]
    
    # Aggregate metrics
    avg_test_sharpe: float
    avg_test_return: float
    avg_test_max_dd: float
    consistency_ratio: float  # % of profitable windows
    
    # Degradation analysis
    train_test_sharpe_ratio: float  # Avg test sharpe / train sharpe
    stability_score: float  # 1 - std(returns) / mean(returns)
    
    # Combined equity
    combined_equity: pd.Series = None
    
    def is_robust(self, min_consistency: float = 0.6, min_ratio: float = 0.5) -> bool:
        """Check if strategy is robust."""
        return (
            self.consistency_ratio >= min_consistency and
            self.train_test_sharpe_ratio >= min_ratio and
            self.avg_test_sharpe > 0.5
        )


class WalkForwardValidator:
    """
    Walk-forward validation framework.
    
    Splits data into train/test windows and validates strategy
    performance on out-of-sample data.
    
    Usage:
        validator = WalkForwardValidator(
            n_windows=5,
            train_pct=0.7,
            gap_periods=5
        )
        
        results = validator.validate(
            df,
            train_func=lambda train_df: model.fit(train_df),
            predict_func=lambda test_df, model: model.predict(test_df)
        )
        
        if results.is_robust():
            print("Strategy is robust!")
    """
    
    def __init__(
        self,
        n_windows: int = 5,
        train_pct: float = 0.7,
        gap_periods: int = 0,
        expanding: bool = False
    ):
        """
        Initialize validator.
        
        Args:
            n_windows: Number of walk-forward windows
            train_pct: Percentage of each window for training
            gap_periods: Gap between train and test to avoid lookahead
            expanding: If True, use expanding window (all prior data)
        """
        self.n_windows = n_windows
        self.train_pct = train_pct
        self.gap_periods = gap_periods
        self.expanding = expanding
    
    def validate(
        self,
        df: pd.DataFrame,
        train_func: Callable[[pd.DataFrame], Any],
        predict_func: Callable[[pd.DataFrame, Any], pd.Series],
        metric_func: Callable[[pd.Series, pd.Series], Dict[str, float]] = None
    ) -> WalkForwardAnalysis:
        """
        Run walk-forward validation.
        
        Args:
            df: OHLCV DataFrame
            train_func: Function to train model on training data
            predict_func: Function to generate signals on test data
            metric_func: Optional custom metric function
            
        Returns:
            WalkForwardAnalysis with all results
        """
        n = len(df)
        window_size = n // self.n_windows
        
        results = []
        all_test_returns = []
        
        for i in range(self.n_windows):
            # Calculate indices
            if self.expanding:
                train_start_idx = 0
            else:
                train_start_idx = i * window_size
            
            train_end_idx = (i + 1) * window_size
            
            # Adjust for train percentage
            train_size = int((train_end_idx - train_start_idx) * self.train_pct)
            actual_train_end = train_start_idx + train_size
            
            test_start_idx = actual_train_end + self.gap_periods
            test_end_idx = train_end_idx
            
            if test_start_idx >= test_end_idx:
                continue
            
            # Split data
            train_df = df.iloc[train_start_idx:actual_train_end].copy()
            test_df = df.iloc[test_start_idx:test_end_idx].copy()
            
            if len(train_df) < 50 or len(test_df) < 10:
                continue
            
            try:
                # Train model
                model = train_func(train_df)
                
                # Generate signals
                train_signals = predict_func(train_df, model)
                test_signals = predict_func(test_df, model)
                
                # Calculate returns
                train_returns = self._calculate_returns(train_df, train_signals)
                test_returns = self._calculate_returns(test_df, test_signals)
                
                all_test_returns.append(test_returns)
                
                # Calculate metrics
                train_sharpe = self._calculate_sharpe(train_returns)
                test_sharpe = self._calculate_sharpe(test_returns)
                
                train_total_return = (1 + train_returns).prod() - 1
                test_total_return = (1 + test_returns).prod() - 1
                
                test_max_dd = self._calculate_max_drawdown(test_returns)
                
                # Trade metrics
                n_trades = (test_signals.diff().abs() > 0).sum()
                win_rate = (test_returns > 0).mean() * 100
                
                result = WalkForwardResult(
                    window_id=i,
                    train_start=df.index[train_start_idx] if isinstance(df.index, pd.DatetimeIndex) else datetime.now(),
                    train_end=df.index[actual_train_end] if isinstance(df.index, pd.DatetimeIndex) else datetime.now(),
                    test_start=df.index[test_start_idx] if isinstance(df.index, pd.DatetimeIndex) else datetime.now(),
                    test_end=df.index[test_end_idx-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.now(),
                    train_sharpe=train_sharpe,
                    test_sharpe=test_sharpe,
                    train_return=train_total_return * 100,
                    test_return=test_total_return * 100,
                    test_max_dd=test_max_dd * 100,
                    n_trades=n_trades,
                    win_rate=win_rate
                )
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Window {i} failed: {e}")
                continue
        
        if not results:
            return WalkForwardAnalysis(
                windows=[],
                avg_test_sharpe=0,
                avg_test_return=0,
                avg_test_max_dd=0,
                consistency_ratio=0,
                train_test_sharpe_ratio=0,
                stability_score=0
            )
        
        # Calculate aggregate metrics
        avg_test_sharpe = np.mean([r.test_sharpe for r in results])
        avg_test_return = np.mean([r.test_return for r in results])
        avg_test_max_dd = np.mean([r.test_max_dd for r in results])
        
        profitable_windows = sum(1 for r in results if r.test_return > 0)
        consistency_ratio = profitable_windows / len(results)
        
        avg_train_sharpe = np.mean([r.train_sharpe for r in results])
        train_test_ratio = avg_test_sharpe / avg_train_sharpe if avg_train_sharpe > 0 else 0
        
        test_returns_array = [r.test_return for r in results]
        if np.mean(test_returns_array) != 0:
            stability = 1 - abs(np.std(test_returns_array) / np.mean(test_returns_array))
        else:
            stability = 0
        
        # Combine equity curves
        if all_test_returns:
            combined_returns = pd.concat(all_test_returns)
            combined_equity = (1 + combined_returns).cumprod()
        else:
            combined_equity = None
        
        return WalkForwardAnalysis(
            windows=results,
            avg_test_sharpe=avg_test_sharpe,
            avg_test_return=avg_test_return,
            avg_test_max_dd=avg_test_max_dd,
            consistency_ratio=consistency_ratio,
            train_test_sharpe_ratio=train_test_ratio,
            stability_score=max(0, stability),
            combined_equity=combined_equity
        )
    
    def _calculate_returns(
        self,
        df: pd.DataFrame,
        signals: pd.Series
    ) -> pd.Series:
        """Calculate strategy returns from signals."""
        price_returns = df['close'].pct_change().fillna(0)
        strategy_returns = signals.shift(1).fillna(0) * price_returns
        return strategy_returns
    
    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio."""
        if returns.std() == 0:
            return 0.0
        return np.sqrt(252) * returns.mean() / returns.std()
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdown = (rolling_max - cum_returns) / rolling_max
        return drawdown.max()


class ParameterStability:
    """
    Test parameter stability across different time periods.
    
    A robust strategy should have stable optimal parameters.
    """
    
    def __init__(self, n_periods: int = 5):
        """Initialize stability tester."""
        self.n_periods = n_periods
    
    def test_stability(
        self,
        df: pd.DataFrame,
        optimize_func: Callable[[pd.DataFrame], Dict[str, float]],
        param_names: List[str]
    ) -> Dict[str, Any]:
        """
        Test parameter stability across periods.
        
        Args:
            df: OHLCV DataFrame
            optimize_func: Function that returns optimal parameters
            param_names: Names of parameters to track
            
        Returns:
            Dict with stability metrics for each parameter
        """
        n = len(df)
        period_size = n // self.n_periods
        
        param_values = {name: [] for name in param_names}
        
        for i in range(self.n_periods):
            start_idx = i * period_size
            end_idx = (i + 1) * period_size if i < self.n_periods - 1 else n
            
            period_df = df.iloc[start_idx:end_idx]
            
            try:
                optimal_params = optimize_func(period_df)
                
                for name in param_names:
                    if name in optimal_params:
                        param_values[name].append(optimal_params[name])
            except Exception as e:
                logger.warning(f"Optimization failed for period {i}: {e}")
        
        # Calculate stability metrics
        stability_results = {}
        
        for name, values in param_values.items():
            if len(values) < 2:
                continue
            
            values = np.array(values)
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            # Coefficient of variation
            cv = std_val / abs(mean_val) if mean_val != 0 else float('inf')
            
            stability_results[name] = {
                'mean': float(mean_val),
                'std': float(std_val),
                'cv': float(cv),
                'min': float(values.min()),
                'max': float(values.max()),
                'is_stable': cv < 0.3  # Less than 30% variation
            }
        
        return stability_results


class OverfitDetector:
    """
    Detect potential overfitting in trading strategies.
    """
    
    @staticmethod
    def deflated_sharpe_ratio(
        sharpe: float,
        n_trials: int,
        n_observations: int,
        skew: float = 0,
        kurtosis: float = 3
    ) -> float:
        """
        Calculate deflated Sharpe ratio accounting for multiple testing.
        
        Args:
            sharpe: Observed Sharpe ratio
            n_trials: Number of strategies/parameters tested
            n_observations: Number of data points
            skew: Return skewness
            kurtosis: Return kurtosis
            
        Returns:
            Deflated Sharpe ratio (probability of being false positive)
        """
        from scipy.stats import norm
        
        # Expected maximum Sharpe under null
        euler = 0.5772156649
        expected_max_sharpe = (
            (1 - euler) * norm.ppf(1 - 1/n_trials) +
            euler * norm.ppf(1 - 1/(n_trials * np.e))
        )
        
        # Variance of Sharpe estimator
        var_sharpe = (
            1 + 0.5 * sharpe**2 - skew * sharpe +
            (kurtosis - 3) / 4 * sharpe**2
        ) / n_observations
        
        # Deflated Sharpe
        deflated = norm.cdf(
            (sharpe - expected_max_sharpe) / np.sqrt(var_sharpe)
        )
        
        return deflated
    
    @staticmethod
    def minimum_track_record(
        sharpe: float,
        target_sharpe: float = 1.0,
        confidence: float = 0.95
    ) -> int:
        """
        Calculate minimum track record length needed.
        
        Args:
            sharpe: Observed Sharpe ratio
            target_sharpe: Target Sharpe to distinguish from
            confidence: Confidence level
            
        Returns:
            Minimum number of periods needed
        """
        from scipy.stats import norm
        
        z = norm.ppf(confidence)
        
        # Minimum track record (Bailey & Lopez de Prado)
        min_track = (z / (sharpe - target_sharpe)) ** 2
        
        return int(np.ceil(min_track * 12))  # In months
