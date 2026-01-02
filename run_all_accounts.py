#!/usr/bin/env python3
"""
Run All Accounts - Multi-Account Trading Launcher.

Launches all configured trading accounts simultaneously in separate processes.

Usage:
    python run_all_accounts.py           # Live trading all accounts
    python run_all_accounts.py --paper   # Paper trading all accounts
    python run_all_accounts.py --debug   # Enable debug logging

Account Configurations:
    - config/gft_account_1.py   (GFT Crypto)
    - config/gft_account_2.py   (GFT Crypto)
    - config/gft_account_3.py   (GFT Crypto)
    - config/the5ers_account.py (The5ers Forex)
"""

import argparse
import logging
import multiprocessing
import os
import signal
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


# Account configurations
ACCOUNTS = [
    {
        'name': 'GFT Account 1',
        'config': 'config/gft_account_1.py',
        'bot_module': 'bots.gft_bot',
        'bot_class': 'GFTBot',
    },
    {
        'name': 'GFT Account 2',
        'config': 'config/gft_account_2.py',
        'bot_module': 'bots.gft_bot',
        'bot_class': 'GFTBot',
    },
    {
        'name': 'GFT Account 3',
        'config': 'config/gft_account_3.py',
        'bot_module': 'bots.gft_bot',
        'bot_class': 'GFTBot',
    },
    {
        'name': 'The5ers Account',
        'config': 'config/the5ers_account.py',
        'bot_module': 'bots.the5ers_bot',
        'bot_class': 'The5ersBot',
    },
]


def run_bot_process(account: Dict[str, Any], paper_mode: bool, debug: bool):
    """
    Run a single bot in a separate process.

    Args:
        account: Account configuration dict
        paper_mode: If True, run in paper trading mode
        debug: If True, enable debug logging
    """
    import importlib

    # Setup logging for this process
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format=f'%(asctime)s - [{account["name"]}] - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Starting {account['name']}...")
        logger.info(f"Config: {account['config']}")
        logger.info(f"Paper Mode: {paper_mode}")

        # Import the bot module
        bot_module = importlib.import_module(account['bot_module'])

        # Load config
        config = bot_module.load_config(account['config'])

        # Get bot class
        bot_class = getattr(bot_module, account['bot_class'])

        # Create and run bot
        bot = bot_class(config, paper_mode=paper_mode)
        bot.start()

    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}")
        logger.error(f"Please create {account['config']} with your account settings")
    except Exception as e:
        logger.critical(f"Fatal error in {account['name']}: {e}", exc_info=True)
        raise


class MultiAccountManager:
    """Manages multiple trading bot processes."""

    def __init__(self, accounts: List[Dict], paper_mode: bool = False, debug: bool = False):
        """
        Initialize the multi-account manager.

        Args:
            accounts: List of account configurations
            paper_mode: If True, run all accounts in paper mode
            debug: If True, enable debug logging
        """
        self.accounts = accounts
        self.paper_mode = paper_mode
        self.debug = debug
        self.processes: List[multiprocessing.Process] = []
        self.is_running = False

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logging.getLogger(__name__).info(f"Received signal {signum}, shutting down all accounts...")
        self.stop_all()

    def start_all(self):
        """Start all trading accounts in separate processes."""
        logger = logging.getLogger(__name__)

        logger.info("=" * 70)
        logger.info("MULTI-ACCOUNT TRADING LAUNCHER")
        logger.info("=" * 70)
        logger.info(f"Mode: {'PAPER TRADING' if self.paper_mode else 'LIVE TRADING'}")
        logger.info(f"Accounts: {len(self.accounts)}")
        logger.info("=" * 70)

        self.is_running = True

        for account in self.accounts:
            logger.info(f"Launching {account['name']}...")

            # Create process for this account
            process = multiprocessing.Process(
                target=run_bot_process,
                args=(account, self.paper_mode, self.debug),
                name=account['name']
            )
            process.start()
            self.processes.append(process)

            # Small delay between launches to avoid MT5 conflicts
            time.sleep(2)

        logger.info("=" * 70)
        logger.info(f"All {len(self.processes)} accounts launched")
        logger.info("Press Ctrl+C to stop all accounts")
        logger.info("=" * 70)

        # Monitor processes
        self._monitor_processes()

    def _monitor_processes(self):
        """Monitor running processes and handle failures."""
        logger = logging.getLogger(__name__)

        while self.is_running:
            try:
                # Check each process
                for i, process in enumerate(self.processes):
                    if not process.is_alive():
                        account = self.accounts[i]
                        exit_code = process.exitcode

                        if exit_code != 0:
                            logger.error(
                                f"{account['name']} exited with code {exit_code}"
                            )
                            # Optionally restart failed processes
                            # self._restart_process(i)
                        else:
                            logger.info(f"{account['name']} exited normally")

                # Check every 5 seconds
                time.sleep(5)

            except KeyboardInterrupt:
                break

        self.stop_all()

    def stop_all(self):
        """Stop all running processes."""
        logger = logging.getLogger(__name__)
        self.is_running = False

        logger.info("Stopping all accounts...")

        for process in self.processes:
            if process.is_alive():
                logger.info(f"Terminating {process.name}...")
                process.terminate()

        # Wait for processes to finish
        for process in self.processes:
            process.join(timeout=10)

            if process.is_alive():
                logger.warning(f"Force killing {process.name}...")
                process.kill()

        logger.info("All accounts stopped")

    def get_status(self) -> Dict[str, str]:
        """Get status of all processes."""
        status = {}
        for i, process in enumerate(self.processes):
            account = self.accounts[i]
            if process.is_alive():
                status[account['name']] = 'running'
            else:
                status[account['name']] = f'stopped (exit: {process.exitcode})'
        return status


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Account Trading Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_all_accounts.py           # Live trading all accounts
    python run_all_accounts.py --paper   # Paper trading (simulation)
    python run_all_accounts.py --debug   # Enable debug logging

Account Configs:
    Edit the config files in config/ directory with your credentials:
    - config/gft_account_1.py
    - config/gft_account_2.py
    - config/gft_account_3.py
    - config/the5ers_account.py
        """
    )
    parser.add_argument(
        '--paper',
        action='store_true',
        help='Paper trading mode - simulate trades without executing'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--accounts',
        type=str,
        nargs='+',
        choices=['gft1', 'gft2', 'gft3', 'the5ers', 'all'],
        default=['all'],
        help='Which accounts to run (default: all)'
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)

    # Filter accounts based on selection
    if 'all' in args.accounts:
        selected_accounts = ACCOUNTS
    else:
        account_map = {
            'gft1': 0,
            'gft2': 1,
            'gft3': 2,
            'the5ers': 3,
        }
        selected_accounts = [ACCOUNTS[account_map[a]] for a in args.accounts if a in account_map]

    if not selected_accounts:
        logger.error("No accounts selected")
        sys.exit(1)

    # Verify config files exist
    missing_configs = []
    for account in selected_accounts:
        config_path = Path(account['config'])
        if not config_path.exists():
            missing_configs.append(account['config'])

    if missing_configs:
        logger.warning("Missing config files:")
        for config in missing_configs:
            logger.warning(f"  - {config}")
        logger.warning("Please create these files with your account credentials.")
        logger.warning("See config/gft_account_1.py for an example template.")

        # In paper mode, we can continue with defaults
        if not args.paper:
            logger.error("Cannot run live trading without config files. Exiting.")
            sys.exit(1)
        else:
            logger.info("Continuing in paper mode with default configs...")

    # Create manager and start
    manager = MultiAccountManager(
        accounts=selected_accounts,
        paper_mode=args.paper,
        debug=args.debug
    )

    try:
        manager.start_all()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        manager.stop_all()


if __name__ == "__main__":
    # Required for multiprocessing on Windows
    multiprocessing.freeze_support()
    main()
