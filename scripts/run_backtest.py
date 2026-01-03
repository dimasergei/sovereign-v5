#!/usr/bin/env python3
"""
Event-Driven Backtest Runner - Proper Stop/Target Execution.

The vectorized backtester ignores stop_loss and take_profit from signals,
resulting in W/L ratio of ~1.0x instead of the target 2.5x.

This script uses the EventDrivenBacktester which properly executes:
- Stop losses (using bar LOW for longs, HIGH for shorts)
- Take profits (using bar HIGH for longs, LOW for shorts)
- Trailing stops (breakeven at 1R, trail at 1.5R)

Target metrics (Renaissance-style):
- Trades per symbol: 80+
- Win Rate: 50-55%
- Avg Win / Avg Loss: 2.0x+ (CRITICAL - this is why we need event-driven)
- Annual Return: 30%+
- Max Drawdown: <4%
- Sharpe Ratio: >1.5

Usage:
    python scripts/run_backtest.py --symbol BTCUSD --days 365
    python scripts/run_backtest.py --all --days 365
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

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
from backtesting.engine.event_driven import EventDrivenBacktester, BacktestResult
from signals.generator import SignalGenerator
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
    # Commodities
    'XAUUSD': 'GC=F',      # Gold futures
    'XAUUSD.x': 'GC=F',
    'XAGUSD': 'SI=F',      # Silver futures
    'XAGUSD.x': 'SI=F',
    'USOIL': 'CL=F',       # WTI Crude Oil futures
    'USOIL.x': 'CL=F',
    # Forex Majors
    'EURUSD': 'EURUSD=X',
    'EURUSD.x': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X',
    'GBPUSD.x': 'GBPUSD=X',
    'USDJPY': 'JPY=X',     # Note: Yahoo uses inverse
    'USDJPY.x': 'JPY=X',
    'USDCHF': 'CHF=X',
    'USDCHF.x': 'CHF=X',
    'AUDUSD': 'AUDUSD=X',
    'AUDUSD.x': 'AUDUSD=X',
    'USDCAD': 'CAD=X',
    'USDCAD.x': 'CAD=X',
    'NZDUSD': 'NZDUSD=X',
    'NZDUSD.x': 'NZDUSD=X',
    # Forex Crosses
    'EURGBP': 'EURGBP=X',
    'EURGBP.x': 'EURGBP=X',
    'EURJPY': 'EURJPY=X',
    'EURJPY.x': 'EURJPY=X',
    'GBPJPY': 'GBPJPY=X',
    'GBPJPY.x': 'GBPJPY=X',
    'AUDJPY': 'AUDJPY=X',
    'AUDJPY.x': 'AUDJPY=X',
    # Indices
    'NAS100': '^IXIC',
    'NAS100.x': '^IXIC',
    'US30': '^DJI',
    'US30.x': '^DJI',
    'SPX500': '^GSPC',
    'SPX500.x': '^GSPC',
    'GER40': '^GDAXI',     # DAX
    'GER40.x': '^GDAXI',
    'UK100': '^FTSE',      # FTSE 100
    'UK100.x': '^FTSE',
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
    """Fetch historical data from Yahoo Finance."""
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance is required. Install with: pip install yfinance")

    ticker = get_yahoo_ticker(symbol)
    logger.info(f"Fetching {days} days of data for {symbol} (Yahoo: {ticker})")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 50)  # Extra for warmup

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
                if col == 'volume':
                    data['volume'] = 0
                else:
                    raise ValueError(f"Missing required column: {col}")

        if isinstance(data.index, pd.DatetimeIndex):
            data['time'] = data.index

        logger.info(f"Fetched {len(data)} bars from {data.index[0]} to {data.index[-1]}")
        return data

    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        raise


def run_backtest(symbol: str, days: int = 365, verbose: bool = False) -> BacktestResult:
    """Run event-driven backtest with proper stop/target execution."""

    print(f"\n{'='*70}")
    print(f"  EVENT-DRIVEN BACKTEST: {symbol}")
    print(f"{'='*70}\n")

    # Fetch data
    print("Fetching historical data...")
    df = fetch_data(symbol, days)
    logger.info(f"Loaded {len(df)} bars")

    # Get trading params
    params = get_params(symbol)

    # Initialize components
    print("Initializing signal generator...")
    signal_generator = SignalGenerator(min_confidence=params.get('min_confidence', 0.40))

    backtester = EventDrivenBacktester(
        initial_capital=10000.0,
        max_drawdown_pct=6.0,
        guardian_buffer_pct=2.0,  # Guardian at 4%
        base_risk_pct=params.get('base_risk_pct', 0.5),
        max_positions=params.get('max_positions', 4),
        use_trailing_stops=True,
        breakeven_trigger_r=1.0,
        trail_trigger_r=1.5,
        trail_distance_atr=2.0
    )

    # Run backtest
    print("Running event-driven backtest...")
    result = backtester.run(df, signal_generator, symbol)

    # Print results
    print(f"\n{'='*50}")
    print(f"RESULTS: {symbol}")
    print(f"{'='*50}")

    print(f"\n📈 PERFORMANCE:")
    print(f"-" * 40)
    print(f"  Total Return:     {result.total_return_pct:+.2f}%")
    print(f"  Max Drawdown:     {result.max_drawdown_pct:.2f}%")
    print(f"  Sharpe Ratio:     {result.sharpe_ratio:.2f}")
    print(f"  Profit Factor:    {result.profit_factor:.2f}")

    print(f"\n📊 TRADE STATISTICS:")
    print(f"-" * 40)
    print(f"  Total Trades:     {result.total_trades}")
    print(f"  Win Rate:         {result.win_rate:.1f}%")
    print(f"  Winning Trades:   {result.winning_trades}")
    print(f"  Losing Trades:    {result.losing_trades}")
    print(f"  Longs:            {result.long_trades}")
    print(f"  Shorts:           {result.short_trades}")
    if result.short_trades > 0:
        print(f"  L/S Ratio:        {result.long_trades/result.short_trades:.2f}")
    else:
        print(f"  L/S Ratio:        Longs only")

    print(f"\n💰 WIN/LOSS ANALYSIS:")
    print(f"-" * 40)
    print(f"  Avg Win:          {result.avg_win_pct:+.2f}%")
    print(f"  Avg Loss:         {result.avg_loss_pct:-.2f}%")
    print(f"  Win/Loss Ratio:   {result.win_loss_ratio:.2f}x")  # THIS IS THE KEY METRIC
    print(f"  Avg Bars Held:    {result.avg_bars_held:.1f}")

    if result.guardian_triggered:
        print(f"\n⚠️  Guardian triggered at bar {result.guardian_bar}")

    # Strategy breakdown
    if result.strategy_breakdown:
        print(f"\n🎯 STRATEGY BREAKDOWN:")
        print(f"-" * 40)
        for strat, stats in result.strategy_breakdown.items():
            print(f"  {strat}:")
            print(f"    Trades:    {stats['trades']}")
            print(f"    Win Rate:  {stats['win_rate']:.1f}%")
            print(f"    Total PnL: ${stats['total_pnl']:.2f}")
            print(f"    Avg PnL:   {stats['avg_pnl_pct']:.2f}%")

    # Target comparison
    print(f"\n{'='*70}")
    print(f"TARGET COMPARISON (Renaissance-style)")
    print(f"{'='*70}")
    print(f"{'Metric':<20} {'Result':<15} {'Target':<15} {'Status'}")
    print(f"-" * 70)

    checks = [
        ("Trades", f"{result.total_trades}", "80+",
         "✅" if result.total_trades >= 80 else "❌"),
        ("Win Rate", f"{result.win_rate:.1f}%", "50%+",
         "✅" if result.win_rate >= 50 else "❌"),
        ("W/L Ratio", f"{result.win_loss_ratio:.2f}x", "2.0x+",
         "✅" if result.win_loss_ratio >= 2.0 else "❌"),
        ("Return", f"{result.total_return_pct:+.1f}%", "+30%+",
         "✅" if result.total_return_pct >= 30 else "❌"),
        ("Max DD", f"{result.max_drawdown_pct:.1f}%", "<4%",
         "✅" if result.max_drawdown_pct < 4 else "❌"),
        ("Sharpe", f"{result.sharpe_ratio:.2f}", ">1.5",
         "✅" if result.sharpe_ratio > 1.5 else "❌"),
    ]

    for metric, actual, target, status in checks:
        print(f"{metric:<20} {actual:<15} {target:<15} {status}")

    passed = sum(1 for _, _, _, s in checks if s == "✅")
    score_icon = "🏆" if passed >= 5 else "🔄" if passed >= 3 else "❌"
    print(f"\n{score_icon} Score: {passed}/6 targets met")

    if passed < 4:
        print(f"\nRecommendations:")
        if result.total_trades < 80:
            print("  - Lower signal thresholds to generate more trades")
        if result.win_rate < 50:
            print("  - Review entry criteria quality")
        if result.win_loss_ratio < 2.0:
            print("  - Verify stops/targets are being executed properly")
        if result.max_drawdown_pct >= 4:
            print("  - Reduce position sizes or tighten stops")

    print(f"\n{'='*70}")

    return result


def run_all_symbols(days: int = 365, verbose: bool = False) -> Dict[str, BacktestResult]:
    """Run backtest on elite portfolio symbols (top 6 by Sharpe ratio)."""
    # Elite Portfolio - optimized for maximum Sharpe-weighted returns
    symbols = ['XAUUSD', 'XAGUSD', 'NAS100', 'UK100', 'SPX500', 'EURUSD']
    results = {}

    for symbol in symbols:
        try:
            results[symbol] = run_backtest(symbol, days, verbose)
        except Exception as e:
            logger.error(f"Failed to backtest {symbol}: {e}")
            print(f"\n❌ Error backtesting {symbol}: {e}")

    # Summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY - ALL SYMBOLS")
    print(f"{'='*80}")
    print(f"{'Symbol':<10} {'Return':<12} {'Max DD':<10} {'Sharpe':<10} {'Trades':<10} {'Win Rate':<12} {'W/L Ratio'}")
    print(f"-" * 80)

    total_return = 0
    for sym, r in results.items():
        print(f"{sym:<10} {r.total_return_pct:+.2f}%{'':<6} {r.max_drawdown_pct:.2f}%{'':<5} "
              f"{r.sharpe_ratio:.2f}{'':<6} {r.total_trades:<10} {r.win_rate:.1f}%{'':<7} "
              f"{r.win_loss_ratio:.2f}x")
        total_return += r.total_return_pct

    print(f"-" * 80)
    print(f"Combined return (simple): {total_return:+.2f}%")
    print(f"Average per symbol:       {total_return/len(results):+.2f}%")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Event-Driven Backtest with Proper Stop/Target Execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_backtest.py --symbol BTCUSD --days 365
    python scripts/run_backtest.py --all --days 365

Target Metrics (Renaissance-style):
    - Trades per symbol: 80+
    - Win Rate: 50-55%
    - Avg Win / Avg Loss: 2.0x+ (requires event-driven backtest!)
    - Annual Return: 30%+
    - Max Drawdown: <4%
    - Sharpe Ratio: >1.5

Supported Symbols:
    Crypto:  BTCUSD, ETHUSD
    Gold:    XAUUSD
    Forex:   EURUSD, GBPUSD
    Indices: NAS100, US30, SPX500
        """
    )

    parser.add_argument(
        '--symbol',
        type=str,
        default='BTCUSD',
        help='Trading symbol (e.g., BTCUSD, XAUUSD, NAS100)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Number of days of historical data (default: 365)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run backtest on all major symbols'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
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

    try:
        if args.all:
            run_all_symbols(args.days, args.verbose)
        else:
            run_backtest(args.symbol, args.days, args.verbose)
        return 0
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=args.debug)
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
