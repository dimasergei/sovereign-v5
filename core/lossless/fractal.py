"""
Fractal Analysis for Market Regime Detection.

Fractal dimension measures the "roughness" or complexity of price movements,
helping to distinguish between trending and mean-reverting regimes.
"""

import numpy as np
from typing import Tuple, Optional
import logging


logger = logging.getLogger(__name__)


def fractal_dimension(prices: np.ndarray, max_k: int = 10) -> float:
    """
    Calculate fractal dimension using box-counting method.
    
    Interpretation:
    - FD ≈ 1.0: Straight line (strong trend)
    - FD ≈ 1.5: Random walk (Brownian motion)
    - FD ≈ 2.0: Space-filling (choppy, mean-reverting)
    
    Args:
        prices: Array of prices
        max_k: Maximum number of scales to use
        
    Returns:
        Fractal dimension (typically 1.0 to 2.0)
    """
    if len(prices) < 20:
        return 1.5  # Default to random walk
    
    # Normalize prices to [0, 1]
    prices = np.array(prices, dtype=float)
    p_min = np.min(prices)
    p_max = np.max(prices)
    
    if p_max == p_min:
        return 1.0  # Flat line
    
    normalized = (prices - p_min) / (p_max - p_min)
    
    # Box counting at different scales
    scales = []
    counts = []
    
    n_points = len(normalized)
    
    for k in range(1, min(max_k, int(np.log2(n_points)))):
        box_size = 1.0 / (2 ** k)
        
        if box_size * n_points < 1:
            break
        
        # Count boxes
        time_boxes = int(np.ceil(n_points * box_size))
        price_boxes = int(np.ceil(1.0 / box_size))
        
        if time_boxes == 0 or price_boxes == 0:
            continue
        
        grid = np.zeros((time_boxes, price_boxes), dtype=bool)
        
        for i, p in enumerate(normalized):
            t_idx = min(int(i * box_size), time_boxes - 1)
            p_idx = min(int(p / box_size), price_boxes - 1)
            grid[t_idx, p_idx] = True
        
        scales.append(box_size)
        counts.append(np.sum(grid))
    
    if len(scales) < 3:
        return 1.5
    
    # Linear regression on log-log plot
    log_scales = np.log(scales)
    log_counts = np.log(counts)
    
    # Filter out any inf/nan
    valid = np.isfinite(log_scales) & np.isfinite(log_counts)
    if np.sum(valid) < 3:
        return 1.5
    
    log_scales = log_scales[valid]
    log_counts = log_counts[valid]
    
    try:
        slope, _ = np.polyfit(log_scales, log_counts, 1)
        fd = -slope
    except Exception:
        return 1.5
    
    # Bound to valid range
    fd = max(1.0, min(2.0, fd))
    
    logger.debug(f"Fractal dimension: {fd:.3f}")
    return fd


def higuchi_fractal_dimension(prices: np.ndarray, k_max: int = 10) -> float:
    """
    Calculate fractal dimension using Higuchi's method.
    
    More robust than box-counting for time series data.
    
    Args:
        prices: Array of prices
        k_max: Maximum interval
        
    Returns:
        Fractal dimension
    """
    N = len(prices)
    if N < k_max * 4:
        k_max = N // 4
    
    if k_max < 2:
        return 1.5
    
    prices = np.array(prices, dtype=float)
    
    L = []
    x = np.arange(1, k_max + 1)
    
    for k in x:
        Lk = []
        for m in range(1, k + 1):
            # Number of elements in subseries
            N_m = int((N - m) / k)
            
            if N_m < 1:
                continue
            
            # Calculate length for this subseries
            L_mk = 0
            for i in range(1, N_m):
                idx1 = m + i * k - 1
                idx2 = m + (i - 1) * k - 1
                
                if idx1 < N and idx2 >= 0:
                    L_mk += abs(prices[idx1] - prices[idx2])
            
            if N_m > 0:
                L_mk = (L_mk * (N - 1)) / (k * N_m * k)
                Lk.append(L_mk)
        
        if Lk:
            L.append(np.mean(Lk))
        else:
            L.append(np.nan)
    
    L = np.array(L)
    valid = np.isfinite(L) & (L > 0)
    
    if np.sum(valid) < 3:
        return 1.5
    
    # Linear regression of log(L) vs log(1/k)
    log_k = np.log(1.0 / x[valid])
    log_L = np.log(L[valid])
    
    try:
        slope, _ = np.polyfit(log_k, log_L, 1)
        hfd = slope
    except Exception:
        return 1.5
    
    hfd = max(1.0, min(2.0, hfd))
    
    logger.debug(f"Higuchi FD: {hfd:.3f}")
    return hfd


def katz_fractal_dimension(prices: np.ndarray) -> float:
    """
    Calculate fractal dimension using Katz's method.
    
    Simpler and faster than Higuchi, good for quick estimates.
    
    Args:
        prices: Array of prices
        
    Returns:
        Fractal dimension
    """
    n = len(prices)
    if n < 10:
        return 1.5
    
    prices = np.array(prices, dtype=float)
    
    # Total length of curve
    L = np.sum(np.abs(np.diff(prices)))
    
    if L == 0:
        return 1.0
    
    # Maximum distance from first point
    d = np.max(np.abs(prices - prices[0]))
    
    if d == 0:
        return 1.0
    
    # Katz FD
    n_steps = n - 1
    kfd = np.log10(n_steps) / (np.log10(d / L) + np.log10(n_steps))
    
    kfd = max(1.0, min(2.0, kfd))
    
    return kfd


def petrosian_fractal_dimension(prices: np.ndarray) -> float:
    """
    Calculate fractal dimension using Petrosian's method.
    
    Based on the number of sign changes in the derivative.
    
    Args:
        prices: Array of prices
        
    Returns:
        Fractal dimension
    """
    n = len(prices)
    if n < 10:
        return 1.5
    
    # Calculate first difference
    diff = np.diff(prices)
    
    # Count sign changes
    n_delta = np.sum(diff[:-1] * diff[1:] < 0)
    
    # Petrosian FD
    pfd = np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * n_delta)))
    
    return max(1.0, min(2.0, pfd))


def regime_from_fractal_dimension(fd: float) -> str:
    """
    Classify market regime from fractal dimension.
    
    Args:
        fd: Fractal dimension value
        
    Returns:
        Regime classification string
    """
    if fd < 1.3:
        return "strong_trending"
    elif fd < 1.45:
        return "mild_trending"
    elif fd < 1.55:
        return "random_walk"
    elif fd < 1.7:
        return "mild_mean_reversion"
    else:
        return "strong_mean_reversion"


def fractal_efficiency_ratio(prices: np.ndarray, period: int = None) -> float:
    """
    Calculate Fractal Efficiency Ratio.
    
    Similar to Kaufman's Efficiency Ratio but fractal-based.
    Higher values indicate more efficient (trending) moves.
    
    Args:
        prices: Array of prices
        period: Lookback period (default: len(prices))
        
    Returns:
        Efficiency ratio (0 to 1)
    """
    if period is None:
        period = len(prices)
    
    if len(prices) < period:
        period = len(prices)
    
    if period < 2:
        return 0.5
    
    prices = np.array(prices[-period:], dtype=float)
    
    # Net change (direction)
    net_change = abs(prices[-1] - prices[0])
    
    # Total path length
    path_length = np.sum(np.abs(np.diff(prices)))
    
    if path_length == 0:
        return 0.5
    
    efficiency = net_change / path_length
    
    return max(0, min(1, efficiency))


def derive_period_from_fractal(
    prices: np.ndarray,
    min_period: int = 5,
    max_period: int = 200
) -> int:
    """
    Derive optimal indicator period based on fractal characteristics.
    
    - Trending markets (low FD): use longer periods
    - Mean-reverting markets (high FD): use shorter periods
    
    Args:
        prices: Array of prices
        min_period: Minimum period
        max_period: Maximum period
        
    Returns:
        Optimal period
    """
    fd = fractal_dimension(prices)
    
    # Map FD to period
    # FD=1.0 (trend) -> longer period
    # FD=2.0 (MR) -> shorter period
    
    # Linear mapping
    fd_normalized = (fd - 1.0)  # 0 to 1
    
    period_range = max_period - min_period
    period = int(max_period - fd_normalized * period_range)
    
    period = max(min_period, min(max_period, period))
    
    logger.debug(f"Period from FD ({fd:.2f}): {period}")
    return period


def detrended_fluctuation_analysis(
    prices: np.ndarray,
    min_box: int = 4,
    max_box: int = None
) -> float:
    """
    Perform Detrended Fluctuation Analysis (DFA).
    
    Returns the scaling exponent alpha:
    - alpha < 0.5: Anti-correlated (mean-reverting)
    - alpha = 0.5: Random walk
    - alpha > 0.5: Correlated (trending)
    - alpha = 1.0: 1/f noise
    - alpha > 1.0: Non-stationary
    
    Args:
        prices: Array of prices
        min_box: Minimum box size
        max_box: Maximum box size
        
    Returns:
        DFA scaling exponent (alpha)
    """
    N = len(prices)
    
    if max_box is None:
        max_box = N // 4
    
    if N < min_box * 4:
        return 0.5
    
    prices = np.array(prices, dtype=float)
    
    # Integrate the series
    mean_price = np.mean(prices)
    y = np.cumsum(prices - mean_price)
    
    # Box sizes
    box_sizes = np.unique(np.logspace(
        np.log10(min_box),
        np.log10(max_box),
        num=20
    ).astype(int))
    
    fluctuations = []
    
    for n in box_sizes:
        # Number of boxes
        n_boxes = N // n
        
        if n_boxes < 2:
            continue
        
        F_n = []
        
        for i in range(n_boxes):
            # Box boundaries
            start = i * n
            end = start + n
            
            if end > N:
                break
            
            # Fit linear trend in box
            segment = y[start:end]
            x = np.arange(n)
            
            try:
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                
                # Calculate fluctuation
                F_n.append(np.sqrt(np.mean((segment - trend) ** 2)))
            except Exception:
                continue
        
        if F_n:
            fluctuations.append((n, np.mean(F_n)))
    
    if len(fluctuations) < 3:
        return 0.5
    
    # Linear fit on log-log scale
    log_n = np.log([f[0] for f in fluctuations])
    log_F = np.log([f[1] for f in fluctuations])
    
    valid = np.isfinite(log_n) & np.isfinite(log_F)
    if np.sum(valid) < 3:
        return 0.5
    
    try:
        alpha, _ = np.polyfit(log_n[valid], log_F[valid], 1)
    except Exception:
        return 0.5
    
    logger.debug(f"DFA alpha: {alpha:.3f}")
    return alpha
