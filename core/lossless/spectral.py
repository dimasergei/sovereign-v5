"""
Spectral Analysis for Parameter Derivation.

Fourier-based analysis to identify dominant cycles and
derive optimal indicator periods from market data.
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Tuple, List, Optional
import logging


logger = logging.getLogger(__name__)


def spectral_density(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density of price series.
    
    Args:
        prices: Array of prices
        
    Returns:
        Tuple of (frequencies, power spectral density)
    """
    if len(prices) < 10:
        return np.array([0]), np.array([0])
    
    # Detrend to focus on cycles
    try:
        detrended = signal.detrend(prices.astype(float))
    except Exception:
        detrended = prices - np.mean(prices)
    
    # Apply window to reduce spectral leakage
    n = len(detrended)
    window = signal.windows.hann(n)
    windowed = detrended * window
    
    # Compute FFT
    yf = fft(windowed)
    xf = fftfreq(n, 1)[:n // 2]
    
    # Power spectral density
    psd = 2.0 / n * np.abs(yf[0:n // 2])
    
    return xf, psd


def dominant_cycle_period(prices: np.ndarray, min_period: int = 5) -> int:
    """
    Find the dominant cycle period in the price series.
    
    Args:
        prices: Price array
        min_period: Minimum period to consider
        
    Returns:
        Dominant cycle period in bars
    """
    if len(prices) < min_period * 3:
        return min_period
    
    freqs, psd = spectral_density(prices)
    
    if len(freqs) < 2:
        return len(prices) // 4
    
    # Find peak (excluding DC component and very low frequencies)
    psd_filtered = psd.copy()
    
    # Zero out frequencies corresponding to periods > len/3
    min_freq = 3.0 / len(prices)
    psd_filtered[freqs < min_freq] = 0
    
    # Zero out frequencies corresponding to periods < min_period
    max_freq = 1.0 / min_period if min_period > 0 else 0.5
    psd_filtered[freqs > max_freq] = 0
    
    if np.sum(psd_filtered) == 0:
        return len(prices) // 4
    
    peak_idx = np.argmax(psd_filtered)
    peak_freq = freqs[peak_idx]
    
    if peak_freq > 0:
        period = int(1 / peak_freq)
    else:
        period = len(prices) // 4
    
    # Bound to reasonable range
    period = max(min_period, min(len(prices) // 3, period))
    
    logger.debug(f"Dominant cycle period: {period} bars")
    return period


def find_all_cycles(
    prices: np.ndarray,
    min_period: int = 5,
    max_period: Optional[int] = None,
    significance_threshold: float = 0.05
) -> List[Tuple[int, float]]:
    """
    Find all significant cycles in the price series.
    
    Args:
        prices: Price array
        min_period: Minimum period to consider
        max_period: Maximum period to consider
        significance_threshold: Minimum power fraction to consider significant
        
    Returns:
        List of (period, relative_power) tuples, sorted by power
    """
    if max_period is None:
        max_period = len(prices) // 3
    
    if len(prices) < min_period * 3:
        return []
    
    freqs, psd = spectral_density(prices)
    
    if len(freqs) < 2:
        return []
    
    # Find peaks in PSD
    try:
        peaks, properties = signal.find_peaks(psd, prominence=np.std(psd) * 0.5)
    except Exception:
        return []
    
    cycles = []
    total_power = np.sum(psd)
    
    if total_power == 0:
        return []
    
    for peak in peaks:
        if peak >= len(freqs):
            continue
            
        freq = freqs[peak]
        power = psd[peak] / total_power
        
        if freq <= 0:
            continue
        
        period = int(1 / freq)
        
        if min_period <= period <= max_period and power >= significance_threshold:
            cycles.append((period, float(power)))
    
    # Sort by power (descending)
    cycles.sort(key=lambda x: x[1], reverse=True)
    
    logger.debug(f"Found {len(cycles)} significant cycles")
    return cycles


def derive_fast_period(prices: np.ndarray, min_period: int = 3) -> int:
    """
    Derive fast indicator period from spectral analysis.
    
    The fast period corresponds to the shortest significant cycle.
    
    Args:
        prices: Price array
        min_period: Minimum period to consider
        
    Returns:
        Fast period in bars
    """
    cycles = find_all_cycles(prices, min_period=min_period)
    
    if not cycles:
        # Fallback: use data-derived estimate
        return max(min_period, len(prices) // 50)
    
    # Get shortest significant cycle
    periods = [c[0] for c in cycles]
    fast_period = min(periods)
    
    # Ensure it's at least min_period
    fast_period = max(min_period, fast_period)
    
    logger.debug(f"Derived fast period: {fast_period}")
    return fast_period


def derive_slow_period(prices: np.ndarray, fast_period: int = None) -> int:
    """
    Derive slow indicator period from spectral analysis.
    
    The slow period corresponds to the dominant (highest power) cycle.
    
    Args:
        prices: Price array
        fast_period: Fast period (slow must be larger)
        
    Returns:
        Slow period in bars
    """
    dominant = dominant_cycle_period(prices)
    
    if fast_period:
        # Ensure slow > fast by at least 2x
        dominant = max(dominant, fast_period * 2)
    
    # Bound to reasonable range
    dominant = min(dominant, len(prices) // 4)
    
    logger.debug(f"Derived slow period: {dominant}")
    return dominant


def spectral_coherence(
    prices1: np.ndarray,
    prices2: np.ndarray,
    nperseg: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate spectral coherence between two price series.
    
    High coherence at a frequency means the series are correlated
    at that cycle length.
    
    Args:
        prices1: First price series
        prices2: Second price series
        nperseg: Segment length for Welch's method
        
    Returns:
        Tuple of (frequencies, coherence values)
    """
    min_len = min(len(prices1), len(prices2))
    if min_len < nperseg:
        nperseg = min_len // 4
    
    if nperseg < 8:
        return np.array([0]), np.array([0])
    
    try:
        freqs, coh = signal.coherence(
            prices1[:min_len].astype(float),
            prices2[:min_len].astype(float),
            nperseg=nperseg
        )
        return freqs, coh
    except Exception as e:
        logger.debug(f"Coherence calculation failed: {e}")
        return np.array([0]), np.array([0])


def hilbert_instantaneous_frequency(prices: np.ndarray) -> np.ndarray:
    """
    Calculate instantaneous frequency using Hilbert transform.
    
    Useful for detecting frequency modulation in price series.
    
    Args:
        prices: Price array
        
    Returns:
        Array of instantaneous frequencies
    """
    from scipy.signal import hilbert
    
    if len(prices) < 10:
        return np.zeros(len(prices))
    
    # Detrend
    try:
        detrended = signal.detrend(prices.astype(float))
    except Exception:
        detrended = prices - np.mean(prices)
    
    # Hilbert transform
    try:
        analytic_signal = hilbert(detrended)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi)
        
        # Pad to original length
        instantaneous_frequency = np.concatenate([[instantaneous_frequency[0]], instantaneous_frequency])
        
        return instantaneous_frequency
    except Exception as e:
        logger.debug(f"Hilbert transform failed: {e}")
        return np.zeros(len(prices))


def wavelet_decomposition(
    prices: np.ndarray,
    wavelet: str = 'db4',
    level: int = None
) -> List[np.ndarray]:
    """
    Perform wavelet decomposition of price series.
    
    Each level represents different frequency bands.
    
    Args:
        prices: Price array
        wavelet: Wavelet type
        level: Decomposition level (auto if None)
        
    Returns:
        List of coefficient arrays [cA_n, cD_n, cD_n-1, ..., cD_1]
    """
    try:
        import pywt
    except ImportError:
        logger.warning("PyWavelets not installed, skipping wavelet decomposition")
        return [prices]
    
    if level is None:
        level = min(8, int(np.log2(len(prices))) - 1)
    
    if level < 1:
        return [prices]
    
    try:
        coeffs = pywt.wavedec(prices.astype(float), wavelet, level=level)
        return coeffs
    except Exception as e:
        logger.debug(f"Wavelet decomposition failed: {e}")
        return [prices]


def adaptive_period_from_spectrum(
    prices: np.ndarray,
    target_type: str = "momentum"  # "momentum", "mean_reversion", "trend"
) -> int:
    """
    Derive adaptive period based on current spectral characteristics.
    
    Different trading styles need different periods:
    - Momentum: faster periods from high-frequency cycles
    - Mean reversion: periods matching dominant cycles
    - Trend: slower periods from low-frequency cycles
    
    Args:
        prices: Price array
        target_type: Trading style to optimize for
        
    Returns:
        Optimal period for the target trading style
    """
    cycles = find_all_cycles(prices, min_period=3)
    
    if not cycles:
        return dominant_cycle_period(prices)
    
    periods = [c[0] for c in cycles]
    powers = [c[1] for c in cycles]
    
    if target_type == "momentum":
        # Use fastest significant cycle
        return min(periods)
    
    elif target_type == "mean_reversion":
        # Use cycle with highest power
        max_power_idx = np.argmax(powers)
        return periods[max_power_idx]
    
    elif target_type == "trend":
        # Use slowest significant cycle
        return max(periods)
    
    else:
        return dominant_cycle_period(prices)
