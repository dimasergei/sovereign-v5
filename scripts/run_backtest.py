#!/usr/bin/env python3
"""
Backtest Runner - Multi-Alpha High-Frequency Strategy.

Fetches historical data from Yahoo Finance and runs backtests using the
Renaissance-style multi-alpha signal generation.

Key metrics to track:
- Number of trades (target: 80+)
- Win rate (target: 50%+)
- Average win vs average loss (target: 2:1 R:R)
- Total return (target: +30%+)
- Max drawdown (must stay under 4%)
- Sharpe ratio (target: >1.5)

Usage:
    python scripts/run_backtest.py --symbol BTCUSD --days 365 --verbose
    python scripts/run_backtest.py --symbol XAUUSD --days 730 --max-dd 8.5
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
from signals.generator import SignalGenerator, TradingSignal
from config.trading_params import get_params


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
    clean_symbol = symbol.replace('.x', '').upper()

    if symbol in SYMBOL_MAP:
        return SYMBOL_MAP[symbol]
    if clean_symbol in SYMBOL_MAP:
        return SYMBOL_MAP[clean_symbol]

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

        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Standardize column names
        data.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in data.columns]

        if 'adj close' in data.columns:
            data['close'] = data['adj close']
            data = data.drop('adj close', axis=1)

        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

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
    calibration_pct: float = 0.1,  # Reduced calibration for more trading data
    verbose: bool = False
) -> pd.Series:
    """
    Generate trading signals using Multi-Alpha SignalGenerator.

    Args:
        df: OHLCV DataFrame
        symbol: Trading symbol
        calibration_pct: Percentage of data to use for calibration (reduced to 10%)
        verbose: Print detailed signal info

    Returns:
        Series of signals (-1, 0, 1)
    """
    # Get trading params
    params = get_params(symbol)

    # Split data - use less for calibration, more for trading
    calibration_size = int(len(df) * calibration_pct)
    trading_df = df.iloc[calibration_size:].copy()

    logger.info(f"Using {calibration_size} bars for warmup, trading on {len(trading_df)} bars")

    # Initialize new multi-alpha signal generator
    # Lower confidence threshold for more trades
    signal_gen = SignalGenerator(min_confidence=params.get('min_confidence', 0.40))

    # Generate signals for trading period
    signals = []
    entry_reasons = []
    strategy_counts = {}
    errors = 0

    # Context window - need at least 50 bars
    window_size = min(200, len(df) - 1)
    min_context_bars = 50

    for i in range(len(trading_df)):
        current_idx = calibration_size + i
        start_idx = max(0, current_idx - window_size)
        end_idx = current_idx + 1
        context_df = df.iloc[start_idx:end_idx].copy()

        if len(context_df) < min_context_bars:
            signals.append(0)
            entry_reasons.append("")
            continue

        try:
            signal = signal_gen.generate_signal(context_df, symbol)

            if signal.action == 'long':
                signals.append(1)
            elif signal.action == 'short':
                signals.append(-1)
            else:
                signals.append(0)

            # Track entry reasons and strategies
            reason = signal.entry_reason if signal.entry_reason else "no_signal"
            entry_reasons.append(reason)

            if signal.primary_strategy and signal.action != 'neutral':
                strategy_counts[signal.primary_strategy] = strategy_counts.get(signal.primary_strategy, 0) + 1

            # Debug output
            if verbose and i < 10 and signal.action != 'neutral':
                logger.info(f"Bar {i}: {signal.action} | conf={signal.confidence:.2f} | "
                           f"strategy={signal.primary_strategy} | reason={signal.entry_reason}")

        except Exception as e:
            errors += 1
            if errors <= 5:
                logger.warning(f"Signal generation failed at bar {i}: {e}")
            signals.append(0)
            entry_reasons.append("error")

    if errors > 0:
        logger.warning(f"Total signal generation errors: {errors}/{len(trading_df)}")

    # Log entry reason distribution
    reason_counts = {}
    for reason in entry_reasons:
        if reason and reason != "no_signal":
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    if reason_counts:
        logger.info(f"Entry reasons: {reason_counts}")

    # Log strategy distribution
    if strategy_counts:
        logger.info(f"Strategy signals: {strategy_counts}")

    signal_series = pd.Series(signals, index=trading_df.index)

    # Log signal distribution
    long_signals = (signal_series == 1).sum()
    short_signals = (signal_series == -1).sum()
    neutral_signals = (signal_series == 0).sum()
    total_signals = long_signals + short_signals

    logger.info(f"Signals: Long={long_signals}, Short={short_signals}, Neutral={neutral_signals}")
    logger.info(f"Total entry signals: {total_signals}")

    # Calculate approximate trades per year
    trading_days = len(trading_df)
    if trading_days > 0:
        signals_per_day = total_signals / trading_days
        approx_trades_per_year = signals_per_day * 252
        logger.info(f"Approximate trades/year: {approx_trades_per_year:.0f}")

    # Long/short ratio
    if short_signals > 0:
        long_short_ratio = long_signals / short_signals
        logger.info(f"Long/Short Ratio: {long_short_ratio:.2f}")
    else:
        logger.info(f"Long/Short Ratio: N/A (no shorts)")

    # Get signal stats from generator
    stats = signal_gen.get_signal_stats()
    logger.info(f"Signal generator stats: {stats}")

    return signal_series


def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    capital: float,
    max_dd: float
) -> BacktestResults:
    """
    Run vectorized backtest.

    Args:
        df: OHLCV DataFrame (must match signals index)
        signals: Signal series (-1, 0, 1)
        capital: Initial capital
        max_dd: Maximum drawdown limit (actual firm limit)

    Returns:
        BacktestResults
    """
    df_aligned = df.loc[signals.index].copy()

    # Use GUARDIAN threshold - stop 2.0% before actual limit
    guardian_threshold = max_dd - 2.0  # 4.0% for 6% limit
    logger.info(f"Using guardian threshold: {guardian_threshold}% (actual limit: {max_dd}%)")

    config = BacktestConfig(
        initial_capital=capital,
        commission_pct=0.001,  # 0.1%
        slippage_pct=0.0005,  # 0.05%
        max_drawdown_pct=guardian_threshold,
        max_daily_loss_pct=guardian_threshold / 2,
    )

    backtester = VectorizedBacktester(config)
    results = backtester.run(df_aligned, signals=signals)

    return results


def check_prop_firm_safety(
    results: BacktestResults,
    max_dd: float
) -> Dict[str, Any]:
    """Check if strategy is safe for prop firm trading."""
    guardian_threshold = max_dd * 0.85

    analyzer = DrawdownAnalyzer(guardian_threshold_pct=guardian_threshold)

    analysis = analyzer.analyze(
        equity_curve=results.equity_curve,
        trades=results.trades
    )

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
    """Print backtest summary with multi-alpha specific metrics."""
    print("\n" + "=" * 70)
    print(f"  MULTI-ALPHA BACKTEST RESULTS: {symbol}")
    print("=" * 70)

    # Key metrics for Renaissance-style trading
    print("\n🎯 TARGET vs ACTUAL:")
    print("-" * 40)

    # Trades
    trades_target = 80
    trades_icon = "✅" if results.total_trades >= trades_target else "❌"
    print(f"  Trades:          {results.total_trades:>6} (target: {trades_target}+) {trades_icon}")

    # Win rate
    win_rate_target = 50
    win_rate_icon = "✅" if results.win_rate >= win_rate_target else "❌"
    print(f"  Win Rate:        {results.win_rate:>5.1f}% (target: {win_rate_target}%+) {win_rate_icon}")

    # Avg Win / Avg Loss ratio
    if results.avg_loser != 0:
        win_loss_ratio = abs(results.avg_winner / results.avg_loser)
    else:
        win_loss_ratio = float('inf') if results.avg_winner > 0 else 0
    rr_target = 2.0
    rr_icon = "✅" if win_loss_ratio >= rr_target else "❌"
    print(f"  Win/Loss Ratio:  {win_loss_ratio:>5.2f}x (target: {rr_target}x+) {rr_icon}")

    # Return
    return_target = 30
    return_icon = "✅" if results.total_return_pct >= return_target else "❌"
    print(f"  Total Return:    {results.total_return_pct:>+5.1f}% (target: +{return_target}%+) {return_icon}")

    # Max DD
    dd_target = 4.0
    dd_icon = "✅" if results.max_drawdown <= dd_target else "❌"
    print(f"  Max Drawdown:    {results.max_drawdown:>5.2f}% (limit: {dd_target}%) {dd_icon}")

    # Sharpe
    sharpe_target = 1.5
    sharpe_icon = "✅" if results.sharpe_ratio >= sharpe_target else "❌"
    print(f"  Sharpe Ratio:    {results.sharpe_ratio:>5.2f} (target: {sharpe_target}+) {sharpe_icon}")

    # Detailed Performance
    print("\n📈 DETAILED PERFORMANCE:")
    print("-" * 40)
    print(f"  Initial Capital: ${results.config.initial_capital:,.0f}")
    print(f"  Final Equity:    ${results.config.initial_capital + results.total_return:,.0f}")
    print(f"  Total Return:    ${results.total_return:,.2f} ({results.total_return_pct:+.2f}%)")
    print(f"  CAGR:            {results.cagr:.2f}%")
    print(f"  Sortino Ratio:   {results.sortino_ratio:.2f}")
    print(f"  Calmar Ratio:    {results.calmar_ratio:.2f}")

    # Trade Statistics
    print("\n📊 TRADE STATISTICS:")
    print("-" * 40)
    print(f"  Total Trades:    {results.total_trades}")
    print(f"  Winning Trades:  {int(results.total_trades * results.win_rate / 100)}")
    print(f"  Losing Trades:   {int(results.total_trades * (100 - results.win_rate) / 100)}")
    print(f"  Profit Factor:   {results.profit_factor:.2f}")
    print(f"  Avg Trade PnL:   ${results.avg_trade_pnl:,.2f}")
    print(f"  Avg Winner:      ${results.avg_winner:,.2f}")
    print(f"  Avg Loser:       ${results.avg_loser:,.2f}")
    print(f"  Largest Winner:  ${results.largest_winner:,.2f}")
    print(f"  Largest Loser:   ${results.largest_loser:,.2f}")
    print(f"  Avg Hold Time:   {results.avg_hold_time:.1f} bars")

    # Risk
    print("\n⚠️  RISK METRICS:")
    print("-" * 40)
    print(f"  Max Drawdown:    {results.max_drawdown:.2f}%")
    print(f"  Max DD Duration: {results.max_drawdown_duration} bars")

    # Prop firm safety
    safety_icon = "✅" if safety['is_safe'] else "❌"
    print(f"\n{safety_icon} PROP FIRM SAFETY:")
    print("-" * 40)
    print(f"  Max Allowed DD:  {safety['max_allowed']:.1f}%")
    print(f"  Guardian Level:  {safety['guardian_threshold']:.1f}%")
    print(f"  Safety Margin:   {safety['safety_margin']:.2f}%")
    if safety['breached_guardian']:
        print("  ⚠️  WARNING: Breached guardian threshold!")

    # Score
    print("\n📋 RENAISSANCE SCORE:")
    print("-" * 40)
    score = 0
    if results.total_trades >= trades_target:
        score += 1
    if results.win_rate >= win_rate_target:
        score += 1
    if win_loss_ratio >= rr_target:
        score += 1
    if results.total_return_pct >= return_target:
        score += 1
    if results.max_drawdown <= dd_target:
        score += 1
    if results.sharpe_ratio >= sharpe_target:
        score += 1

    score_icon = "🏆" if score >= 5 else "🔄" if score >= 3 else "❌"
    print(f"  {score_icon} Score: {score}/6 targets met")

    if score < 4:
        print("\n  Recommendations:")
        if results.total_trades < trades_target:
            print("    - Lower signal thresholds to generate more trades")
        if results.win_rate < win_rate_target:
            print("    - Review entry criteria quality")
        if win_loss_ratio < rr_target:
            print("    - Improve stop/target ratios")
        if results.max_drawdown > dd_target:
            print("    - Reduce position sizes")

    print("\n" + "=" * 70)


def save_html_report(
    results: BacktestResults,
    symbol: str,
    output_path: str,
    df: pd.DataFrame
):
    """Generate and save HTML tearsheet report."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    generator = TearsheetGenerator(
        guardian_threshold_pct=results.config.max_drawdown_pct * 0.85,
        max_drawdown_pct=results.config.max_drawdown_pct
    )

    report = generator.generate(
        returns=results.returns_series,
        equity_curve=results.equity_curve,
        market_data=df,
        trades=results.trades,
        strategy_name=f"{symbol} Multi-Alpha Strategy"
    )

    html_content = generate_html_report(report)

    with open(output_file, 'w') as f:
        f.write(html_content)

    logger.info(f"Saved HTML report to {output_path}")
    print(f"\n📄 Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-alpha strategy backtest on historical data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_backtest.py --symbol BTCUSD --days 365 --verbose
    python scripts/run_backtest.py --symbol XAUUSD --days 730 --max-dd 8.5
    python scripts/run_backtest.py --symbol NAS100 --days 365 --capital 50000

Target Metrics (Renaissance-style):
    - Trades per symbol: 80+
    - Win Rate: 50-55%
    - Avg Win / Avg Loss: >2.0
    - Annual Return: >30%
    - Max Drawdown: <4%
    - Sharpe Ratio: >1.5

Supported Symbols:
    Crypto:  BTCUSD, ETHUSD, SOLUSD
    Gold:    XAUUSD
    Forex:   EURUSD, GBPUSD
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
        '--verbose',
        action='store_true',
        help='Enable verbose logging (shows signal details)'
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

    print(f"\n🚀 Multi-Alpha Backtest: {args.symbol}")
    print(f"   Days: {args.days}, Capital: ${args.capital:,.0f}, Max DD: {args.max_dd}%\n")

    try:
        # 1. Fetch data
        print("📥 Fetching historical data...")
        df = fetch_data(args.symbol, args.days)

        # 2. Generate signals
        print("🔮 Generating multi-alpha signals...")
        signals = generate_signals(df, args.symbol, verbose=args.verbose)

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

        return 0 if safety['is_safe'] else 1

    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=args.debug)
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
