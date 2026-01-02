"""
Pairs Trading - Statistical arbitrage using cointegration.

Finds pairs of assets that move together and trades
deviations from their equilibrium relationship.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

try:
    from statsmodels.tsa.stattools import coint, adfuller
    from statsmodels.regression.linear_model import OLS
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("statsmodels not installed")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.base import BaseModel, ModelPrediction


class CointegrationTest:
    """Cointegration testing utilities."""
    
    @staticmethod
    def test_cointegration(series1: np.ndarray, series2: np.ndarray) -> Dict[str, Any]:
        """
        Test for cointegration between two series.
        
        Uses Engle-Granger two-step method.
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required")
        
        # Engle-Granger test
        score, pvalue, _ = coint(series1, series2)
        
        # Also run ADF on the spread
        # OLS regression to find hedge ratio
        series1 = series1.reshape(-1, 1) if len(series1.shape) == 1 else series1
        model = OLS(series2, series1).fit()
        hedge_ratio = model.params[0]
        
        spread = series2 - hedge_ratio * series1.flatten()
        adf_stat, adf_pvalue, *_ = adfuller(spread)
        
        # Half-life of mean reversion
        spread_lag = spread[:-1]
        spread_ret = np.diff(spread)
        model_hl = OLS(spread_ret, spread_lag).fit()
        
        if model_hl.params[0] < 0:
            half_life = -np.log(2) / model_hl.params[0]
        else:
            half_life = float('inf')
        
        return {
            'is_cointegrated': pvalue < 0.05,
            'coint_pvalue': float(pvalue),
            'coint_statistic': float(score),
            'adf_pvalue': float(adf_pvalue),
            'hedge_ratio': float(hedge_ratio),
            'half_life': float(half_life),
            'spread_mean': float(np.mean(spread)),
            'spread_std': float(np.std(spread))
        }
    
    @staticmethod
    def find_cointegrated_pairs(
        prices_df: pd.DataFrame,
        pvalue_threshold: float = 0.05
    ) -> List[Dict[str, Any]]:
        """
        Find all cointegrated pairs in a DataFrame of prices.
        
        Args:
            prices_df: DataFrame with asset prices as columns
            pvalue_threshold: Maximum p-value for cointegration
            
        Returns:
            List of cointegrated pairs with test results
        """
        pairs = []
        columns = prices_df.columns.tolist()
        
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                try:
                    result = CointegrationTest.test_cointegration(
                        prices_df[col1].values,
                        prices_df[col2].values
                    )
                    
                    if result['is_cointegrated']:
                        pairs.append({
                            'asset1': col1,
                            'asset2': col2,
                            **result
                        })
                except Exception as e:
                    logger.warning(f"Error testing {col1}/{col2}: {e}")
        
        # Sort by p-value
        pairs.sort(key=lambda x: x['coint_pvalue'])
        
        return pairs


class PairsTradingModel(BaseModel):
    """
    Pairs trading model using cointegration.
    
    Trades mean-reversion of the spread between two cointegrated assets.
    
    Usage:
        model = PairsTradingModel("BTCUSD", "ETHUSD")
        model.fit(prices_df)
        
        signal = model.predict(recent_prices)
    """
    
    def __init__(
        self,
        name: str = "pairs_trading",
        asset1: str = None,
        asset2: str = None,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        stop_zscore: float = 4.0
    ):
        """
        Initialize pairs trading model.
        
        Args:
            name: Model name
            asset1: First asset symbol
            asset2: Second asset symbol
            entry_zscore: Z-score threshold for entry
            exit_zscore: Z-score threshold for exit
            stop_zscore: Z-score threshold for stop loss
        """
        super().__init__(name)
        
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required")
        
        self.asset1 = asset1
        self.asset2 = asset2
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.stop_zscore = stop_zscore
        
        # Fitted parameters
        self.hedge_ratio: float = 1.0
        self.spread_mean: float = 0.0
        self.spread_std: float = 1.0
        self.half_life: float = 20.0
        self.coint_pvalue: float = 1.0
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        prices_df: pd.DataFrame = None
    ) -> 'PairsTradingModel':
        """
        Fit pairs trading model.
        
        Args:
            X: Price series for asset1 (or DataFrame with both)
            y: Price series for asset2 (if X is array)
            prices_df: Alternative DataFrame input
        """
        if prices_df is not None:
            prices1 = prices_df[self.asset1].values
            prices2 = prices_df[self.asset2].values
        elif y is not None:
            prices1 = X.flatten() if len(X.shape) > 1 else X
            prices2 = y.flatten() if len(y.shape) > 1 else y
        else:
            raise ValueError("Need two price series")
        
        # Test cointegration
        result = CointegrationTest.test_cointegration(prices1, prices2)
        
        self.hedge_ratio = result['hedge_ratio']
        self.spread_mean = result['spread_mean']
        self.spread_std = result['spread_std']
        self.half_life = result['half_life']
        self.coint_pvalue = result['coint_pvalue']
        
        self.is_trained = True
        self.training_date = datetime.now()
        
        self.metadata = {
            'is_cointegrated': result['is_cointegrated'],
            'coint_pvalue': result['coint_pvalue'],
            'half_life': result['half_life']
        }
        
        logger.info(
            f"Pairs model fitted: hedge={self.hedge_ratio:.4f}, "
            f"halflife={self.half_life:.1f}, pvalue={self.coint_pvalue:.4f}"
        )
        
        return self
    
    def predict(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        prices_df: pd.DataFrame = None
    ) -> ModelPrediction:
        """
        Generate trading signal.
        
        Returns signal for asset2:
        - Positive = long asset2, short asset1 (spread too low)
        - Negative = short asset2, long asset1 (spread too high)
        """
        if not self.is_trained:
            raise RuntimeError("Model not fitted")
        
        if prices_df is not None:
            price1 = prices_df[self.asset1].iloc[-1]
            price2 = prices_df[self.asset2].iloc[-1]
        elif y is not None:
            price1 = X[-1] if len(X.shape) == 1 else X[-1, 0]
            price2 = y[-1] if len(y.shape) == 1 else y[-1, 0]
        else:
            raise ValueError("Need two price series")
        
        # Calculate current spread
        spread = price2 - self.hedge_ratio * price1
        
        # Z-score
        zscore = (spread - self.spread_mean) / (self.spread_std + 1e-8)
        
        # Generate signal
        if zscore > self.entry_zscore:
            # Spread too high - short asset2, long asset1
            direction = -1.0
            confidence = min(1.0, abs(zscore) / 4)
        elif zscore < -self.entry_zscore:
            # Spread too low - long asset2, short asset1
            direction = 1.0
            confidence = min(1.0, abs(zscore) / 4)
        elif abs(zscore) < self.exit_zscore:
            # Near mean - close position
            direction = 0.0
            confidence = 0.0
        else:
            # In between - reduce position
            direction = -np.sign(zscore) * 0.5
            confidence = 0.3
        
        # Check for stop loss
        if abs(zscore) > self.stop_zscore:
            confidence = 0.0  # Signal to close
        
        # Adjust confidence by cointegration strength
        confidence *= (1 - self.coint_pvalue)
        
        return ModelPrediction(
            model_name=self.name,
            direction=direction,
            magnitude=abs(zscore) * self.spread_std / price2,  # Expected move
            confidence=confidence,
            metadata={
                'zscore': float(zscore),
                'spread': float(spread),
                'hedge_ratio': self.hedge_ratio,
                'action': 'entry' if abs(direction) > 0.5 else 'exit',
                'pair': f"{self.asset1}/{self.asset2}"
            }
        )
    
    def get_spread_series(
        self,
        prices_df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Get spread and z-score series.
        
        Returns:
            Tuple of (spread, zscore) Series
        """
        if not self.is_trained:
            raise RuntimeError("Model not fitted")
        
        spread = prices_df[self.asset2] - self.hedge_ratio * prices_df[self.asset1]
        zscore = (spread - self.spread_mean) / (self.spread_std + 1e-8)
        
        return spread, zscore
    
    def _get_state(self) -> Dict[str, Any]:
        return {
            'asset1': self.asset1,
            'asset2': self.asset2,
            'entry_zscore': self.entry_zscore,
            'exit_zscore': self.exit_zscore,
            'stop_zscore': self.stop_zscore,
            'hedge_ratio': self.hedge_ratio,
            'spread_mean': self.spread_mean,
            'spread_std': self.spread_std,
            'half_life': self.half_life,
            'coint_pvalue': self.coint_pvalue
        }
    
    def _set_state(self, state: Dict[str, Any]):
        self.asset1 = state['asset1']
        self.asset2 = state['asset2']
        self.entry_zscore = state['entry_zscore']
        self.exit_zscore = state['exit_zscore']
        self.stop_zscore = state['stop_zscore']
        self.hedge_ratio = state['hedge_ratio']
        self.spread_mean = state['spread_mean']
        self.spread_std = state['spread_std']
        self.half_life = state['half_life']
        self.coint_pvalue = state['coint_pvalue']


class PairsPortfolio:
    """
    Manages multiple pairs trading strategies.
    
    Features:
    - Automatic pair discovery
    - Position sizing across pairs
    - Correlation management
    """
    
    def __init__(
        self,
        max_pairs: int = 10,
        position_limit_per_pair: float = 0.2
    ):
        """
        Initialize pairs portfolio.
        
        Args:
            max_pairs: Maximum number of pairs to trade
            position_limit_per_pair: Max allocation per pair
        """
        self.max_pairs = max_pairs
        self.position_limit = position_limit_per_pair
        
        self.pairs: List[PairsTradingModel] = []
        self.pair_results: Dict[str, Dict] = {}
    
    def discover_pairs(
        self,
        prices_df: pd.DataFrame,
        min_observations: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Discover tradeable pairs from price data.
        
        Args:
            prices_df: DataFrame with asset prices
            min_observations: Minimum data points required
            
        Returns:
            List of discovered pairs with statistics
        """
        if len(prices_df) < min_observations:
            logger.warning("Insufficient data for pair discovery")
            return []
        
        # Find cointegrated pairs
        cointegrated = CointegrationTest.find_cointegrated_pairs(prices_df)
        
        # Filter and rank
        valid_pairs = []
        
        for pair in cointegrated[:self.max_pairs]:
            # Check half-life is reasonable (5-50 periods)
            if 5 <= pair['half_life'] <= 50:
                valid_pairs.append(pair)
                
                # Create and fit model
                model = PairsTradingModel(
                    name=f"pair_{pair['asset1']}_{pair['asset2']}",
                    asset1=pair['asset1'],
                    asset2=pair['asset2']
                )
                model.fit(prices_df=prices_df)
                self.pairs.append(model)
        
        logger.info(f"Discovered {len(valid_pairs)} tradeable pairs")
        
        return valid_pairs
    
    def get_portfolio_signals(
        self,
        prices_df: pd.DataFrame
    ) -> Dict[str, ModelPrediction]:
        """
        Get signals for all pairs.
        
        Returns:
            Dict mapping pair name to prediction
        """
        signals = {}
        
        for model in self.pairs:
            try:
                signal = model.predict(prices_df=prices_df)
                signals[model.name] = signal
            except Exception as e:
                logger.warning(f"Error getting signal for {model.name}: {e}")
        
        return signals
