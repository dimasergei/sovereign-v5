#!/usr/bin/env python3
"""Quick status viewer for paper trading accounts."""

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STATE_DIR = "storage/state"
ACCOUNTS = ["GFT_1", "GFT_2", "GFT_3", "THE5ERS_1"]

def main():
    print("\n" + "=" * 70)
    print("  SOVEREIGN V5 - ACCOUNT STATUS")
    print("=" * 70)

    total_equity = 0
    total_pnl = 0
    total_positions = 0

    for account in ACCOUNTS:
        state_file = os.path.join(STATE_DIR, f"{account}_paper.json")

        if not os.path.exists(state_file):
            print(f"\n[{account}] No state file found")
            continue

        with open(state_file, 'r') as f:
            state = json.load(f)

        balance = state.get('balance', 0)
        equity = state.get('equity', balance)
        initial = state.get('initial_balance', balance)
        pnl = equity - initial
        positions = state.get('open_positions', [])
        trades = state.get('trade_history', [])

        total_equity += equity
        total_pnl += pnl
        total_positions += len(positions)

        pnl_pct = (pnl / initial * 100) if initial > 0 else 0
        emoji = "🟢" if pnl >= 0 else "🔴"

        print(f"\n{emoji} [{account}]")
        print(f"   Balance: ${balance:,.2f} | Equity: ${equity:,.2f}")
        print(f"   P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
        print(f"   Total Trades: {len(trades)} | Open: {len(positions)}")

        if positions:
            print(f"\n   Open Positions:")
            for pos in positions:
                symbol = pos.get('symbol', 'N/A')
                direction = pos.get('direction', 'N/A')
                size = pos.get('size', 0)
                entry = pos.get('entry_price', 0)
                unrealized = pos.get('unrealized_pnl', 0)
                print(f"     • {symbol} {direction.upper()} x{size:.2f} @ {entry:.2f} | P&L: ${unrealized:+.2f}")

        if trades:
            print(f"\n   Recent Trades (last 5):")
            for trade in trades[-5:]:
                symbol = trade.get('symbol', 'N/A')
                direction = trade.get('direction', 'N/A')
                realized = trade.get('realized_pnl', 0)
                status = trade.get('status', 'N/A')
                emoji_t = "✅" if realized >= 0 else "❌"
                print(f"     {emoji_t} {symbol} {direction.upper()} ${realized:+.2f} ({status})")

    # Summary
    print("\n" + "=" * 70)
    total_initial = 35000  # 3x$10K + 1x$5K
    total_pnl_pct = (total_pnl / total_initial * 100) if total_initial > 0 else 0
    emoji = "🟢" if total_pnl >= 0 else "🔴"

    print(f"  TOTAL: ${total_equity:,.2f} | {emoji} P&L: ${total_pnl:+,.2f} ({total_pnl_pct:+.2f}%)")
    print(f"  Open Positions: {total_positions}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
