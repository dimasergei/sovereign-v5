#!/usr/bin/env python3
"""
Backtest Runner - Command-line interface for running strategy backtests.

Fetches historical data from Yahoo Finance and runs backtests using the
signal generation and backtesting modules.

Usage:
    python scripts/run_backtest.py --symbol BTCUSD --days 365
    python scripts/run_backtest.py --symbol XAUUSD --days 730 --max-dd 8.5 --output reports/xauusd.html
    python scripts/run_backtest.py --symbol NAS100 --days 365 --capital 50000
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not installed. Install with: pip install yfinance")

# Import project modules
from backtesting import (
    VectorizedBacktester,
    BacktestConfig,
    BacktestResults,
    TearsheetGenerator,
    DrawdownAnalyzer,
    generate_html_report,
)
from core.lossless import MarketCalibrator
from data import FeatureEngineer
from models import RegimeDetector, EnsembleMetaLearner
from signals.generator import SignalGenerator, TradingSignal
from signals.trend_filter import TrendFilter, TrendDirection


# Symbol mappings to Yahoo Finance tickers
SYMBOL_MAP = {
    # Crypto
    'BTCUSD': 'BTC-USD',
    'BTCUSD.x': 'BTC-USD',
    'ETHUSD': 'ETH-USD',
    'ETHUSD.x': 'ETH-USD',
    'SOLUSD': 'SOL-USD',
    'SOLUSD.x': 'SOL-USD',
    'XRPUSD': 'XRP-USD',
    'XRPUSD.x': 'XRP-USD',
    'LTCUSD': 'LTC-USD',
    'LTCUSD.x': 'LTC-USD',
    # Gold
    'XAUUSD': 'GC=F',
    'XAUUSD.x': 'GC=F',
    # Forex
    'EURUSD': 'EURUSD=X',
    'EURUSD.x': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X',
    'GBPUSD.x': 'GBPUSD=X',
    'USDJPY': 'USDJPY=X',
    'USDJPY.x': 'USDJPY=X',
    'AUDUSD': 'AUDUSD=X',
    'AUDUSD.x': 'AUDUSD=X',
    'USDCAD': 'USDCAD=X',
    'USDCAD.x': 'USDCAD=X',
    # Indices
    'NAS100': '^IXIC',
    'NAS100.x': '^IXIC',
    'US30': '^DJI',
    'US30.x': '^DJI',
    'SPX500': '^GSPC',
    'SPX500.x': '^GSPC',
    'US500': '^GSPC',
    'US500.x': '^GSPC',
}

logger = logging.getLogger(__name__)


def get_yahoo_ticker(symbol: str) -> str:
    """Convert trading symbol to Yahoo Finance ticker."""
    # Strip .x suffix if present
    clean_symbol = symbol.replace('.x', '').upper()

    # Check mapping
    if symbol in SYMBOL_MAP:
        return SYMBOL_MAP[symbol]
    if clean_symbol in SYMBOL_MAP:
        return SYMBOL_MAP[clean_symbol]

    # Default: assume it's already a valid ticker
    return symbol


def fetch_data(symbol: str, days: int) -> pd.DataFrame:
    """
    Fetch historical data from Yahoo Finance.

    Args:
        symbol: Trading symbol (e.g., BTCUSD, XAUUSD)
        days: Number of days of history

    Returns:
        DataFrame with OHLCV data
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance is required. Install with: pip install yfinance")

    ticker = get_yahoo_ticker(symbol)
    logger.info(f"Fetching {days} days of data for {symbol} (Yahoo: {ticker})")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    try:
        data = yf.download(
            ticker,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            progress=False
        )

        if data.empty:
            raise ValueError(f"No data returned for {ticker}")

        # Handle MultiIndex columns (yfinance returns tuples like ('Open', 'BTC-USD'))
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten MultiIndex - take first level (the OHLCV names)
            data.columns = data.columns.get_level_values(0)

        # Standardize column names (handle both string and tuple cases)
        data.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in data.columns]

        # Rename adj close to close if needed
        if 'adj close' in data.columns:
            data['close'] = data['adj close']
            data = data.drop('adj close', axis=1)

        # Ensure we have required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        # Add time column if index is datetime
        if isinstance(data.index, pd.DatetimeIndex):
            data['time'] = data.index

        logger.info(f"Fetched {len(data)} bars from {data.index[0]} to {data.index[-1]}")
        return data

    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        raise


def generate_signals(
    df: pd.DataFrame,
    symbol: str,
    calibration_pct: float = 0.2
) -> pd.Series:
    """
    Generate trading signals using SignalGenerator.

    Args:
        df: OHLCV DataFrame
        symbol: Trading symbol
        calibration_pct: Percentage of data to use for calibration

    Returns:
        Series of signals (-1, 0, 1)
    """
    # Split data
    calibration_size = int(len(df) * calibration_pct)
    calibration_df = df.iloc[:calibration_size].copy()
    trading_df = df.iloc[calibration_size:].copy()

    logger.info(f"Calibrating on {calibration_size} bars, trading on {len(trading_df)} bars")

    # Initialize components
    calibrator = MarketCalibrator(min_calibration_bars=50)  # Lower for backtesting
    feature_engineer = FeatureEngineer()
    regime_detector = RegimeDetector()

    # Calibrate on first portion
    try:
        calibration_result = calibrator.calibrate_all(calibration_df)
        logger.info(f"Calibration: regime={calibration_result.current_regime}, "
                   f"fast={calibration_result.fast_period}, slow={calibration_result.slow_period}")
    except Exception as e:
        logger.warning(f"Calibration failed: {e}, using defaults")

    # Initialize signal generator with lower min_confidence for backtesting
    signal_gen = SignalGenerator(
        calibrator=calibrator,
        feature_engineer=feature_engineer,
        regime_detector=regime_detector,
        min_confidence=0.3  # Lower threshold for backtesting
    )

    # Generate signals for trading period
    signals = []
    errors = 0

    # We need context - at least 100 bars for signal generator, prefer 200
    # Use all available historical data up to window_size
    window_size = min(200, len(df) - 1)

    min_context_bars = 100  # Minimum bars needed for signal generation

    for i in range(len(trading_df)):
        # Get context window - include all available history up to current bar
        current_idx = calibration_size + i
        start_idx = max(0, current_idx - window_size)
        end_idx = current_idx + 1
        context_df = df.iloc[start_idx:end_idx].copy()

        # Skip if not enough history yet
        if len(context_df) < min_context_bars:
            signals.append(0)
            continue

        try:
            signal = signal_gen.generate_signal(symbol, context_df)

            if signal.action == 'long':
                signals.append(1)
            elif signal.action == 'short':
                signals.append(-1)
            else:
                signals.append(0)

            # Debug first few signals
            if i < 5:
                logger.debug(f"Bar {i}: action={signal.action}, direction={signal.direction:.3f}, "
                           f"confidence={signal.confidence:.3f}")
        except Exception as e:
            errors += 1
            if errors <= 5:
                logger.warning(f"Signal generation failed at bar {i}: {e}")
            signals.append(0)

    if errors > 0:
        logger.warning(f"Total signal generation errors: {errors}/{len(trading_df)}")

    signal_series = pd.Series(signals, index=trading_df.index)

    # Log signal distribution
    long_signals = (signal_series == 1).sum()
    short_signals = (signal_series == -1).sum()
    neutral_signals = (signal_series == 0).sum()
    logger.info(f"Signals: Long={long_signals}, Short={short_signals}, Neutral={neutral_signals}")

    # Log trend filter stats (CRITICAL - this shows counter-trend prevention)
    blocked_counter_trend = signal_gen.blocked_signals.get("counter_trend", 0)
    blocked_low_conf = signal_gen.blocked_signals.get("low_confidence", 0)
    logger.info(f"Trend Filter: Blocked {blocked_counter_trend} counter-trend signals, "
               f"{blocked_low_conf} low-confidence signals")

    # Calculate long/short ratio (CRITICAL for trend following)
    if short_signals > 0:
        long_short_ratio = long_signals / short_signals
        logger.info(f"Long/Short Ratio: {long_short_ratio:.2f}")
    else:
        logger.info(f"Long/Short Ratio: N/A (no shorts)")

    return signal_series


def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    capital: float,
    max_dd: float
) -> BacktestResults:
    """
    Run vectorized backtest.

    CRITICAL: Uses GUARDIAN threshold (1% below max) to simulate real trading
    where we stop BEFORE hitting the actual limit.

    Args:
        df: OHLCV DataFrame (must match signals index)
        signals: Signal series (-1, 0, 1)
        capital: Initial capital
        max_dd: Maximum drawdown limit (actual firm limit)

    Returns:
        BacktestResults
    """
    # Align data with signals
    df_aligned = df.loc[signals.index].copy()

    # Use GUARDIAN threshold - stop 1% before actual limit
    # This is critical for prop firm safety
    guardian_threshold = max_dd - 1.0
    logger.info(f"Using guardian threshold: {guardian_threshold}% (actual limit: {max_dd}%)")

    config = BacktestConfig(
        initial_capital=capital,
        commission_pct=0.001,  # 0.1%
        slippage_pct=0.0005,  # 0.05%
        max_drawdown_pct=guardian_threshold,  # GUARDIAN threshold, not actual limit
        max_daily_loss_pct=guardian_threshold / 2,  # Half of guardian
    )

    backtester = VectorizedBacktester(config)
    results = backtester.run(df_aligned, signals=signals)

    return results


def check_prop_firm_safety(
    results: BacktestResults,
    max_dd: float
) -> Dict[str, Any]:
    """
    Check if strategy is safe for prop firm trading.

    Args:
        results: Backtest results
        max_dd: Maximum allowed drawdown

    Returns:
        Safety analysis dict
    """
    guardian_threshold = max_dd * 0.85  # Guardian at 85% of limit

    analyzer = DrawdownAnalyzer(guardian_threshold_pct=guardian_threshold)

    # Analyze drawdowns
    analysis = analyzer.analyze(
        equity_curve=results.equity_curve,
        trades=results.trades
    )

    # Safety checks
    is_safe = results.max_drawdown < max_dd
    breached_guardian = results.max_drawdown >= guardian_threshold

    return {
        'is_safe': is_safe,
        'max_drawdown': results.max_drawdown,
        'max_allowed': max_dd,
        'guardian_threshold': guardian_threshold,
        'breached_guardian': breached_guardian,
        'safety_margin': max_dd - results.max_drawdown,
        'drawdown_analysis': analysis
    }


def print_summary(results: BacktestResults, safety: Dict[str, Any], symbol: str):
    """Print backtest summary to console."""
    print("\n" + "=" * 60)
    print(f"BACKTEST RESULTS: {symbol}")
    print("=" * 60)

    # Performance
    print("\n📈 PERFORMANCE:")
    print(f"  Total Return:     ${results.total_return:,.2f} ({results.total_return_pct:+.2f}%)")
    print(f"  CAGR:             {results.cagr:.2f}%")
    print(f"  Sharpe Ratio:     {results.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio:    {results.sortino_ratio:.2f}")
    print(f"  Calmar Ratio:     {results.calmar_ratio:.2f}")

    # Risk
    print("\n⚠️  RISK:")
    print(f"  Max Drawdown:     {results.max_drawdown:.2f}%")
    print(f"  Max DD Duration:  {results.max_drawdown_duration} bars")

    # Prop firm safety
    safety_icon = "✅" if safety['is_safe'] else "❌"
    print(f"\n{safety_icon} PROP FIRM SAFETY:")
    print(f"  Max Allowed DD:   {safety['max_allowed']:.1f}%")
    print(f"  Guardian Level:   {safety['guardian_threshold']:.1f}%")
    print(f"  Safety Margin:    {safety['safety_margin']:.2f}%")
    if safety['breached_guardian']:
        print("  ⚠️  WARNING: Breached guardian threshold!")

    # Trade stats
    print("\n📊 TRADE STATISTICS:")
    print(f"  Total Trades:     {results.total_trades}")
    print(f"  Win Rate:         {results.win_rate:.1f}%")
    print(f"  Profit Factor:    {results.profit_factor:.2f}")
    print(f"  Avg Trade PnL:    ${results.avg_trade_pnl:,.2f}")
    print(f"  Avg Winner:       ${results.avg_winner:,.2f}")
    print(f"  Avg Loser:        ${results.avg_loser:,.2f}")
    print(f"  Largest Winner:   ${results.largest_winner:,.2f}")
    print(f"  Largest Loser:    ${results.largest_loser:,.2f}")
    print(f"  Avg Hold Time:    {results.avg_hold_time:.1f} bars")

    print("\n" + "=" * 60)


def save_html_report(
    results: BacktestResults,
    symbol: str,
    output_path: str,
    df: pd.DataFrame
):
    """Generate and save HTML tearsheet report."""
    # Create reports directory if needed
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Generate tearsheet
    generator = TearsheetGenerator(
        guardian_threshold_pct=results.config.max_drawdown_pct * 0.85,
        max_drawdown_pct=results.config.max_drawdown_pct
    )

    report = generator.generate(
        returns=results.returns_series,
        equity_curve=results.equity_curve,
        market_data=df,
        trades=results.trades,
        strategy_name=f"{symbol} Strategy"
    )

    # Generate HTML
    html_content = generate_html_report(report)

    # Save to file
    with open(output_file, 'w') as f:
        f.write(html_content)

    logger.info(f"Saved HTML report to {output_path}")
    print(f"\n📄 Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run strategy backtest on historical data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_backtest.py --symbol BTCUSD --days 365
    python scripts/run_backtest.py --symbol XAUUSD --days 730 --max-dd 8.5
    python scripts/run_backtest.py --symbol NAS100 --days 365 --capital 50000 --output reports/nas100.html

Supported Symbols:
    Crypto:  BTCUSD, ETHUSD, SOLUSD, XRPUSD, LTCUSD
    Gold:    XAUUSD
    Forex:   EURUSD, GBPUSD, USDJPY, AUDUSD
    Indices: NAS100, US30, SPX500
        """
    )

    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        help='Trading symbol (e.g., BTCUSD, XAUUSD, EURUSD, NAS100)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Number of days of historical data (default: 365)'
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=10000.0,
        help='Initial capital (default: 10000)'
    )
    parser.add_argument(
        '--max-dd',
        type=float,
        default=6.0,
        help='Maximum drawdown limit %% (default: 6.0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save HTML report (optional)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Suppress yfinance noise
    logging.getLogger('yfinance').setLevel(logging.WARNING)

    print(f"\n🚀 Running backtest for {args.symbol}")
    print(f"   Days: {args.days}, Capital: ${args.capital:,.0f}, Max DD: {args.max_dd}%\n")

    try:
        # 1. Fetch data
        print("📥 Fetching historical data...")
        df = fetch_data(args.symbol, args.days)

        # 2. Generate signals
        print("🔮 Generating trading signals...")
        signals = generate_signals(df, args.symbol)

        # 3. Run backtest
        print("⚡ Running backtest...")
        results = run_backtest(
            df=df,
            signals=signals,
            capital=args.capital,
            max_dd=args.max_dd
        )

        # 4. Check prop firm safety
        safety = check_prop_firm_safety(results, args.max_dd)

        # 5. Print summary
        print_summary(results, safety, args.symbol)

        # 6. Save HTML report if requested
        if args.output:
            print("\n📝 Generating HTML report...")
            save_html_report(results, args.symbol, args.output, df)

        # Return exit code based on safety
        return 0 if safety['is_safe'] else 1

    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=args.debug)
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
