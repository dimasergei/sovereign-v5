"""
Hurst Exponent Analysis for Regime Classification.

The Hurst exponent measures long-term memory in time series:
- H < 0.5: Mean-reverting (anti-persistent)
- H = 0.5: Random walk
- H > 0.5: Trending (persistent)
"""

import numpy as np
from typing import Tuple, Optional, List
import logging


logger = logging.getLogger(__name__)


def hurst_exponent(prices: np.ndarray, max_lag: int = None) -> float:
    """
    Calculate Hurst exponent using R/S (Rescaled Range) analysis.
    
    Args:
        prices: Array of prices
        max_lag: Maximum lag for analysis
        
    Returns:
        Hurst exponent (0 to 1)
    """
    n = len(prices)
    if n < 20:
        return 0.5
    
    if max_lag is None:
        max_lag = min(n // 4, 100)
    
    prices = np.array(prices, dtype=float)
    
    # Calculate returns
    returns = np.diff(np.log(prices + 1e-10))
    returns = returns[np.isfinite(returns)]
    
    if len(returns) < 20:
        return 0.5
    
    # R/S analysis for different time scales
    max_k = min(int(np.log2(len(returns))) - 1, 8)
    if max_k < 2:
        return 0.5
    
    rs_values = []
    ns = []
    
    for k in range(2, max_k + 1):
        subset_size = int(2 ** k)
        num_subsets = len(returns) // subset_size
        
        if num_subsets == 0:
            continue
        
        rs_list = []
        
        for i in range(num_subsets):
            subset = returns[i * subset_size:(i + 1) * subset_size]
            
            if len(subset) < 2:
                continue
            
            # Mean-adjusted cumulative sum
            mean_adj = subset - np.mean(subset)
            cumsum = np.cumsum(mean_adj)
            
            # Range
            R = np.max(cumsum) - np.min(cumsum)
            
            # Standard deviation
            S = np.std(subset, ddof=1)
            
            if S > 0:
                rs_list.append(R / S)
        
        if len(rs_list) > 0:
            rs_values.append(np.mean(rs_list))
            ns.append(subset_size)
    
    if len(rs_values) < 3:
        return 0.5
    
    # Linear regression on log-log plot
    log_n = np.log(ns)
    log_rs = np.log(rs_values)
    
    valid = np.isfinite(log_n) & np.isfinite(log_rs)
    if np.sum(valid) < 3:
        return 0.5
    
    try:
        H, _ = np.polyfit(log_n[valid], log_rs[valid], 1)
    except Exception:
        return 0.5
    
    # Clip to valid range
    H = max(0.0, min(1.0, H))
    
    logger.debug(f"Hurst exponent (R/S): {H:.3f}")
    return H


def hurst_exponent_dfa(prices: np.ndarray) -> float:
    """
    Calculate Hurst exponent using Detrended Fluctuation Analysis.
    
    DFA is often more robust than R/S for non-stationary data.
    H = DFA_alpha for stationary processes.
    
    Args:
        prices: Array of prices
        
    Returns:
        Hurst exponent
    """
    from .fractal import detrended_fluctuation_analysis
    
    alpha = detrended_fluctuation_analysis(prices)
    
    # For stationary processes, H ≈ alpha
    # For non-stationary, relationship is more complex
    H = alpha
    
    # Clip to valid Hurst range
    H = max(0.0, min(1.0, H))
    
    return H


def hurst_exponent_variance(prices: np.ndarray, max_lag: int = 50) -> float:
    """
    Calculate Hurst exponent using variance ratio method.
    
    Based on the fact that Var(sum of H-self-similar process) scales as n^(2H).
    
    Args:
        prices: Array of prices
        max_lag: Maximum lag to consider
        
    Returns:
        Hurst exponent
    """
    n = len(prices)
    if n < 20:
        return 0.5
    
    prices = np.array(prices, dtype=float)
    returns = np.diff(np.log(prices + 1e-10))
    returns = returns[np.isfinite(returns)]
    
    if len(returns) < 20:
        return 0.5
    
    lags = []
    variances = []
    
    for lag in range(1, min(max_lag, len(returns) // 4)):
        # Sum returns over lag period
        summed = []
        for i in range(0, len(returns) - lag, lag):
            summed.append(np.sum(returns[i:i + lag]))
        
        if len(summed) > 2:
            var = np.var(summed)
            if var > 0:
                lags.append(lag)
                variances.append(var)
    
    if len(lags) < 3:
        return 0.5
    
    # Fit: log(var) = 2H * log(lag) + const
    log_lags = np.log(lags)
    log_vars = np.log(variances)
    
    valid = np.isfinite(log_lags) & np.isfinite(log_vars)
    if np.sum(valid) < 3:
        return 0.5
    
    try:
        slope, _ = np.polyfit(log_lags[valid], log_vars[valid], 1)
        H = slope / 2
    except Exception:
        return 0.5
    
    H = max(0.0, min(1.0, H))
    
    logger.debug(f"Hurst exponent (variance): {H:.3f}")
    return H


def rolling_hurst(
    prices: np.ndarray,
    window: int = 100,
    step: int = 10
) -> np.ndarray:
    """
    Calculate rolling Hurst exponent.
    
    Useful for detecting regime changes.
    
    Args:
        prices: Array of prices
        window: Rolling window size
        step: Step size between calculations
        
    Returns:
        Array of Hurst values (shorter than input)
    """
    n = len(prices)
    if n < window:
        return np.array([hurst_exponent(prices)])
    
    hurst_values = []
    
    for i in range(0, n - window + 1, step):
        window_data = prices[i:i + window]
        H = hurst_exponent(window_data)
        hurst_values.append(H)
    
    return np.array(hurst_values)


def regime_from_hurst(H: float) -> str:
    """
    Classify market regime from Hurst exponent.
    
    Args:
        H: Hurst exponent value
        
    Returns:
        Regime classification string
    """
    if H < 0.35:
        return "strong_mean_reversion"
    elif H < 0.45:
        return "mild_mean_reversion"
    elif H < 0.55:
        return "random_walk"
    elif H < 0.65:
        return "mild_trending"
    else:
        return "strong_trending"


def expected_range(
    H: float,
    volatility: float,
    periods: int
) -> Tuple[float, float]:
    """
    Calculate expected price range based on Hurst exponent.
    
    For H-self-similar process:
    E[Range] ∝ σ * n^H
    
    Args:
        H: Hurst exponent
        volatility: Per-period volatility
        periods: Number of periods ahead
        
    Returns:
        Tuple of (expected_range, range_std)
    """
    # Expected range scales as n^H
    expected = volatility * (periods ** H)
    
    # Standard deviation of range (approximation)
    range_std = expected * 0.5 * (1 - abs(H - 0.5))
    
    return expected, range_std


def optimal_holding_period(H: float, base_period: int = 10) -> int:
    """
    Calculate optimal holding period based on Hurst exponent.
    
    - Trending (H > 0.5): Longer holds are better
    - Mean-reverting (H < 0.5): Shorter holds, trade reversals
    
    Args:
        H: Hurst exponent
        base_period: Base holding period
        
    Returns:
        Optimal holding period
    """
    if H > 0.5:
        # Trending: extend holding period
        multiplier = 1 + (H - 0.5) * 4  # 1x to 3x
    else:
        # Mean-reverting: reduce holding period
        multiplier = 1 - (0.5 - H) * 1.5  # 0.25x to 1x
    
    period = int(base_period * multiplier)
    return max(1, period)


def mean_reversion_halflife(prices: np.ndarray) -> float:
    """
    Calculate mean-reversion half-life using Ornstein-Uhlenbeck estimation.
    
    For mean-reverting processes, this tells us how long it takes
    for the price to revert halfway to the mean.
    
    Args:
        prices: Array of prices
        
    Returns:
        Half-life in periods (inf if not mean-reverting)
    """
    if len(prices) < 20:
        return float('inf')
    
    prices = np.array(prices, dtype=float)
    
    # Fit AR(1) model to log prices
    log_prices = np.log(prices + 1e-10)
    
    y = log_prices[1:]
    x = log_prices[:-1]
    
    if len(y) < 2:
        return float('inf')
    
    # OLS regression: y = alpha + beta * x
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sum((x - x_mean) ** 2)
    
    if den == 0:
        return float('inf')
    
    beta = num / den
    
    # Half-life from beta
    if beta >= 1 or beta <= 0:
        return float('inf')  # Not mean-reverting
    
    halflife = -np.log(2) / np.log(beta)
    
    # Bound to reasonable range
    halflife = max(1, min(len(prices) * 2, halflife))
    
    logger.debug(f"Mean reversion half-life: {halflife:.1f} periods")
    return halflife


def derive_period_from_hurst(
    prices: np.ndarray,
    min_period: int = 5,
    max_period: int = 200
) -> int:
    """
    Derive optimal indicator period based on Hurst exponent.
    
    Args:
        prices: Array of prices
        min_period: Minimum period
        max_period: Maximum period
        
    Returns:
        Optimal period
    """
    H = hurst_exponent(prices)
    
    if H < 0.5:
        # Mean-reverting: use half-life as basis
        halflife = mean_reversion_halflife(prices)
        if np.isfinite(halflife):
            period = int(halflife)
        else:
            period = min_period
    else:
        # Trending: longer periods work better
        period = int(min_period + (H - 0.5) * 2 * (max_period - min_period))
    
    period = max(min_period, min(max_period, period))
    
    logger.debug(f"Period from Hurst ({H:.2f}): {period}")
    return period


def hurst_confidence(prices: np.ndarray, n_bootstrap: int = 100) -> Tuple[float, float]:
    """
    Calculate Hurst exponent with bootstrap confidence interval.
    
    Args:
        prices: Array of prices
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Tuple of (mean_hurst, std_hurst)
    """
    n = len(prices)
    if n < 50:
        H = hurst_exponent(prices)
        return H, 0.1  # High uncertainty with small sample
    
    hurst_samples = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n, size=n, replace=True)
        sample = prices[indices]
        
        H = hurst_exponent(sample)
        hurst_samples.append(H)
    
    mean_H = np.mean(hurst_samples)
    std_H = np.std(hurst_samples)
    
    return mean_H, std_H


def adaptive_hurst(
    prices: np.ndarray,
    methods: List[str] = None
) -> float:
    """
    Calculate Hurst exponent using multiple methods and average.
    
    More robust than single-method estimation.
    
    Args:
        prices: Array of prices
        methods: List of methods to use ('rs', 'dfa', 'variance')
        
    Returns:
        Average Hurst exponent
    """
    if methods is None:
        methods = ['rs', 'variance']  # DFA requires fractal module
    
    results = []
    
    if 'rs' in methods:
        results.append(hurst_exponent(prices))
    
    if 'variance' in methods:
        results.append(hurst_exponent_variance(prices))
    
    if 'dfa' in methods:
        results.append(hurst_exponent_dfa(prices))
    
    if not results:
        return 0.5
    
    # Weighted average (could weight by confidence in future)
    return np.mean(results)
