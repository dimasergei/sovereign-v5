"""
Entropy Analysis for Parameter Derivation.

Information-theoretic approach to deriving optimal parameters
from market data without hardcoding.
"""

import numpy as np
from scipy.stats import entropy as scipy_entropy
from typing import Tuple, Optional
import logging


logger = logging.getLogger(__name__)


def market_entropy(prices: np.ndarray, bins: int = 50) -> float:
    """
    Calculate Shannon entropy of price return distribution.
    
    Higher entropy = more random/unpredictable market
    Lower entropy = more structured/predictable market
    
    Args:
        prices: Array of prices
        bins: Number of histogram bins
        
    Returns:
        Shannon entropy value
    """
    if len(prices) < 2:
        return 0.0
    
    # Calculate returns
    returns = np.diff(np.log(prices + 1e-10))
    returns = returns[np.isfinite(returns)]
    
    if len(returns) < 10:
        return 0.0
    
    # Create histogram
    hist, _ = np.histogram(returns, bins=bins, density=True)
    hist = hist[hist > 0]  # Remove zeros for log
    
    if len(hist) == 0:
        return 0.0
    
    # Normalize to probability distribution
    hist = hist / hist.sum()
    
    return scipy_entropy(hist)


def conditional_entropy(
    prices: np.ndarray,
    condition_length: int,
    bins: int = 30
) -> float:
    """
    Calculate conditional entropy H(future|past).
    
    Lower conditional entropy means past better predicts future.
    
    Args:
        prices: Array of prices
        condition_length: Length of conditioning window
        bins: Number of histogram bins
        
    Returns:
        Conditional entropy value
    """
    if len(prices) < condition_length * 3:
        return float('inf')
    
    returns = np.diff(np.log(prices + 1e-10))
    returns = returns[np.isfinite(returns)]
    
    if len(returns) < condition_length * 2:
        return float('inf')
    
    # Create past-future pairs
    past = returns[:condition_length]
    future = returns[condition_length:condition_length * 2]
    
    if len(future) < 10:
        return float('inf')
    
    # Joint entropy
    joint = np.concatenate([past, future])
    H_joint = market_entropy(np.exp(np.cumsum(joint)), bins)
    
    # Past entropy
    H_past = market_entropy(np.exp(np.cumsum(past)), bins)
    
    # Conditional entropy = H(joint) - H(past)
    return max(0, H_joint - H_past)


def optimal_lookback_from_entropy(
    prices: np.ndarray,
    min_period: int = 5,
    max_period: int = 200,
    step: int = 5
) -> int:
    """
    Find optimal lookback period by minimizing prediction entropy.
    
    The optimal period is where past data best predicts future data,
    measured by conditional entropy.
    
    Args:
        prices: Array of prices
        min_period: Minimum period to test
        max_period: Maximum period to test
        step: Step size for testing
        
    Returns:
        Optimal lookback period
    """
    if len(prices) < max_period * 3:
        max_period = len(prices) // 3
    
    if max_period < min_period:
        return min_period
    
    returns = np.diff(np.log(prices + 1e-10))
    returns = returns[np.isfinite(returns)]
    
    best_period = min_period
    min_entropy = float('inf')
    
    for period in range(min_period, min(max_period, len(returns) // 3), step):
        try:
            # Calculate conditional entropy for this period
            cond_entropy = 0
            n_windows = min(10, (len(returns) - period * 2) // period)
            
            if n_windows < 3:
                continue
            
            for i in range(n_windows):
                start = i * period
                past = returns[start:start + period]
                future = returns[start + period:start + period * 2]
                
                if len(future) < period // 2:
                    continue
                
                # Simple conditional entropy approximation
                past_std = np.std(past)
                future_std = np.std(future)
                correlation = np.corrcoef(
                    past[-min(len(past), len(future)):],
                    future[:min(len(past), len(future))]
                )[0, 1] if len(past) > 1 and len(future) > 1 else 0
                
                if not np.isfinite(correlation):
                    correlation = 0
                
                # Lower entropy when correlation is higher
                window_entropy = future_std * (1 - abs(correlation))
                cond_entropy += window_entropy
            
            cond_entropy /= n_windows
            
            if cond_entropy < min_entropy:
                min_entropy = cond_entropy
                best_period = period
                
        except Exception as e:
            logger.debug(f"Error testing period {period}: {e}")
            continue
    
    logger.debug(f"Optimal lookback from entropy: {best_period}")
    return best_period


def sample_entropy(
    data: np.ndarray,
    m: int = 2,
    r: Optional[float] = None
) -> float:
    """
    Calculate Sample Entropy (regularity measure).
    
    Lower SampEn = more regular/predictable
    Higher SampEn = more random
    
    Args:
        data: Time series data
        m: Embedding dimension
        r: Tolerance (default: 0.2 * std)
        
    Returns:
        Sample entropy value
    """
    N = len(data)
    if N < 10:
        return 0.0
    
    if r is None:
        r = 0.2 * np.std(data)
    
    if r == 0:
        return 0.0
    
    def _count_matches(template_length: int) -> int:
        count = 0
        for i in range(N - template_length):
            for j in range(i + 1, N - template_length):
                if np.max(np.abs(
                    data[i:i + template_length] - data[j:j + template_length]
                )) < r:
                    count += 1
        return count
    
    try:
        A = _count_matches(m + 1)
        B = _count_matches(m)
        
        if B == 0:
            return 0.0
        
        return -np.log(A / B) if A > 0 else 0.0
    except Exception:
        return 0.0


def approximate_entropy(
    data: np.ndarray,
    m: int = 2,
    r: Optional[float] = None
) -> float:
    """
    Calculate Approximate Entropy.
    
    Similar to sample entropy but includes self-matches.
    
    Args:
        data: Time series data
        m: Embedding dimension
        r: Tolerance (default: 0.2 * std)
        
    Returns:
        Approximate entropy value
    """
    N = len(data)
    if N < 10:
        return 0.0
    
    if r is None:
        r = 0.2 * np.std(data)
    
    if r == 0:
        return 0.0
    
    def _phi(m_val: int) -> float:
        patterns = np.array([
            data[i:i + m_val] for i in range(N - m_val + 1)
        ])
        
        C = []
        for i, pattern in enumerate(patterns):
            matches = np.sum(
                np.max(np.abs(patterns - pattern), axis=1) < r
            )
            C.append(matches / (N - m_val + 1))
        
        return np.mean(np.log(np.array(C) + 1e-10))
    
    try:
        return _phi(m) - _phi(m + 1)
    except Exception:
        return 0.0


def permutation_entropy(
    data: np.ndarray,
    order: int = 3,
    delay: int = 1
) -> float:
    """
    Calculate Permutation Entropy.
    
    Measures complexity based on ordinal patterns.
    
    Args:
        data: Time series data
        order: Pattern length
        delay: Time delay
        
    Returns:
        Normalized permutation entropy (0-1)
    """
    from math import factorial
    
    N = len(data)
    if N < order * delay:
        return 0.0
    
    # Extract ordinal patterns
    patterns = []
    for i in range(N - (order - 1) * delay):
        pattern = tuple(
            np.argsort(data[i:i + order * delay:delay])
        )
        patterns.append(pattern)
    
    if not patterns:
        return 0.0
    
    # Count pattern frequencies
    from collections import Counter
    pattern_counts = Counter(patterns)
    
    # Calculate entropy
    total = len(patterns)
    probs = np.array([count / total for count in pattern_counts.values()])
    
    H = -np.sum(probs * np.log2(probs + 1e-10))
    
    # Normalize by maximum entropy
    H_max = np.log2(factorial(order))
    
    return H / H_max if H_max > 0 else 0.0


def multiscale_entropy(
    data: np.ndarray,
    scales: list = None,
    m: int = 2,
    r: Optional[float] = None
) -> np.ndarray:
    """
    Calculate Multiscale Entropy.
    
    Computes sample entropy at multiple time scales.
    
    Args:
        data: Time series data
        scales: List of scale factors (default: [1, 2, 4, 8, 16])
        m: Embedding dimension
        r: Tolerance
        
    Returns:
        Array of entropy values at each scale
    """
    if scales is None:
        scales = [1, 2, 4, 8, 16]
    
    if r is None:
        r = 0.2 * np.std(data)
    
    mse = []
    
    for scale in scales:
        # Coarse-grain the time series
        N = len(data)
        n_segments = N // scale
        
        if n_segments < 10:
            mse.append(np.nan)
            continue
        
        coarse = np.array([
            np.mean(data[i * scale:(i + 1) * scale])
            for i in range(n_segments)
        ])
        
        # Calculate sample entropy
        se = sample_entropy(coarse, m, r)
        mse.append(se)
    
    return np.array(mse)


def entropy_based_regime(prices: np.ndarray, window: int = 50) -> str:
    """
    Detect market regime based on entropy characteristics.
    
    Args:
        prices: Price array
        window: Analysis window
        
    Returns:
        Regime string: "trending", "mean_reverting", "random", or "chaotic"
    """
    if len(prices) < window:
        return "unknown"
    
    recent_prices = prices[-window:]
    
    # Calculate various entropy measures
    shannon = market_entropy(recent_prices)
    perm = permutation_entropy(recent_prices)
    
    # Calculate Hurst-like measure from entropy
    # Low permutation entropy + low Shannon = trending
    # High permutation entropy + high Shannon = random/chaotic
    
    if perm < 0.6 and shannon < 2.0:
        return "trending"
    elif perm > 0.8 and shannon > 3.0:
        return "chaotic"
    elif perm > 0.7:
        return "random"
    else:
        return "mean_reverting"
