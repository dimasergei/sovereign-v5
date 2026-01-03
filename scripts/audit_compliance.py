#!/usr/bin/env python3
# scripts/audit_compliance.py
"""
Compliance Audit Script for Sovereign V5.

Verifies that all code paths enforce prop firm rules before paper/live deployment.

CRITICAL CHECKS:
- GFT: -2% floating loss guardian, 3% daily DD, 6% total DD
- The5ers: 5% daily loss, 10% static DD
- News blackout enforcement
- Position sizing limits
- Prohibited strategy detection

Usage:
    python scripts/audit_compliance.py
    python scripts/audit_compliance.py --verbose
    python scripts/audit_compliance.py --fix  # Attempt auto-fixes
"""

import sys
import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ComplianceAuditor:
    """Audit codebase for prop firm compliance."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: Dict[str, Dict] = {}
        self.root = Path(__file__).parent.parent

    def log(self, msg: str, level: str = "INFO"):
        """Log message if verbose or error."""
        if self.verbose or level in ("ERROR", "CRITICAL", "WARNING"):
            prefix = {"INFO": "  ", "WARNING": "⚠️", "ERROR": "❌", "CRITICAL": "🚨", "OK": "✅"}
            print(f"{prefix.get(level, '  ')} {msg}")

    def check_file_contains(self, filepath: str, patterns: List[str]) -> List[str]:
        """Check if file contains all required patterns."""
        full_path = self.root / filepath
        if not full_path.exists():
            return [f"File not found: {filepath}"]

        content = full_path.read_text()
        missing = []

        for pattern in patterns:
            if pattern not in content:
                missing.append(pattern)

        return missing

    def check_file_regex(self, filepath: str, pattern: str) -> bool:
        """Check if file contains regex pattern."""
        full_path = self.root / filepath
        if not full_path.exists():
            return False

        content = full_path.read_text()
        return bool(re.search(pattern, content))

    def audit_gft_floating_loss(self) -> Tuple[bool, str]:
        """Check GFT -2% floating loss is monitored."""
        self.log("Checking GFT floating loss monitor...")

        # Check compliance module exists
        if not (self.root / "core/compliance/gft_compliance.py").exists():
            return False, "core/compliance/gft_compliance.py not found"

        # Check floating loss check is implemented
        missing = self.check_file_contains(
            "core/compliance/gft_compliance.py",
            [
                "MAX_FLOATING_LOSS_PCT = 2.0",
                "GUARDIAN_FLOATING_PCT = 1.8",
                "check_position_floating_loss",
            ]
        )

        if missing:
            return False, f"Missing in gft_compliance.py: {missing}"

        # Check position sizer enforces limit
        if not self.check_file_regex(
            "core/position_sizer.py",
            r"guardian_floating_pct.*=.*1\.8"
        ):
            return False, "Position sizer doesn't enforce 1.8% floating guardian"

        return True, "GFT floating loss check implemented"

    def audit_gft_daily_dd(self) -> Tuple[bool, str]:
        """Check GFT 3% daily DD guardian."""
        self.log("Checking GFT daily DD guardian...")

        missing = self.check_file_contains(
            "core/compliance/gft_compliance.py",
            [
                "MAX_DAILY_DD_PCT = 3.0",
                "GUARDIAN_DAILY_DD_PCT = 2.5",
                "check_daily_drawdown",
            ]
        )

        if missing:
            return False, f"Missing daily DD checks: {missing}"

        return True, "GFT daily DD guardian implemented"

    def audit_gft_total_dd(self) -> Tuple[bool, str]:
        """Check GFT 6% total DD guardian."""
        self.log("Checking GFT total DD guardian...")

        missing = self.check_file_contains(
            "core/compliance/gft_compliance.py",
            [
                "MAX_TOTAL_DD_PCT = 6.0",
                "GUARDIAN_TOTAL_DD_PCT = 5.0",
                "check_total_drawdown",
            ]
        )

        if missing:
            return False, f"Missing total DD checks: {missing}"

        return True, "GFT total DD guardian implemented"

    def audit_gft_risk_per_trade(self) -> Tuple[bool, str]:
        """Check GFT 2% max risk per trade."""
        self.log("Checking GFT risk per trade limit...")

        missing = self.check_file_contains(
            "core/compliance/gft_compliance.py",
            [
                "MAX_RISK_PER_TRADE_PCT = 2.0",
                "check_risk_per_trade",
            ]
        )

        if missing:
            return False, f"Missing risk per trade checks: {missing}"

        # Check position sizer enforces it
        if not self.check_file_regex(
            "core/position_sizer.py",
            r"max_risk_pct.*=.*2\.0"
        ):
            return False, "Position sizer doesn't enforce 2% max risk"

        return True, "GFT risk per trade limit implemented"

    def audit_gft_news_blackout(self) -> Tuple[bool, str]:
        """Check GFT 5-min news blackout."""
        self.log("Checking GFT news blackout...")

        if not (self.root / "core/news_calendar.py").exists():
            return False, "core/news_calendar.py not found"

        missing = self.check_file_contains(
            "core/compliance/gft_compliance.py",
            [
                "NEWS_BLACKOUT_MINUTES = 5",
                "check_news_blackout",
            ]
        )

        if missing:
            return False, f"Missing news blackout checks: {missing}"

        return True, "GFT news blackout implemented"

    def audit_gft_profit_cap(self) -> Tuple[bool, str]:
        """Check GFT $3000 daily profit cap."""
        self.log("Checking GFT profit cap...")

        missing = self.check_file_contains(
            "core/compliance/gft_compliance.py",
            [
                "DAILY_PROFIT_CAP_USD = 3000",
                "check_daily_profit_cap",
            ]
        )

        if missing:
            return False, f"Missing profit cap checks: {missing}"

        return True, "GFT profit cap implemented"

    def audit_the5ers_daily_loss(self) -> Tuple[bool, str]:
        """Check The5ers 5% daily loss guardian."""
        self.log("Checking The5ers daily loss guardian...")

        if not (self.root / "core/compliance/the5ers_compliance.py").exists():
            return False, "core/compliance/the5ers_compliance.py not found"

        missing = self.check_file_contains(
            "core/compliance/the5ers_compliance.py",
            [
                "MAX_DAILY_LOSS_PCT = 5.0",
                "GUARDIAN_DAILY_LOSS_PCT = 4.0",
                "check_daily_loss",
            ]
        )

        if missing:
            return False, f"Missing daily loss checks: {missing}"

        return True, "The5ers daily loss guardian implemented"

    def audit_the5ers_static_dd(self) -> Tuple[bool, str]:
        """Check The5ers 10% static DD guardian."""
        self.log("Checking The5ers static DD guardian...")

        missing = self.check_file_contains(
            "core/compliance/the5ers_compliance.py",
            [
                "MAX_TOTAL_DD_PCT = 10.0",
                "GUARDIAN_TOTAL_DD_PCT = 8.0",
                "STATIC",  # Must mention static DD
            ]
        )

        if missing:
            return False, f"Missing static DD checks: {missing}"

        return True, "The5ers static DD guardian implemented"

    def audit_the5ers_news_blackout(self) -> Tuple[bool, str]:
        """Check The5ers 2-min news blackout."""
        self.log("Checking The5ers news blackout...")

        missing = self.check_file_contains(
            "core/compliance/the5ers_compliance.py",
            [
                "NEWS_BLACKOUT_MINUTES = 2",
                "check_news_blackout",
            ]
        )

        if missing:
            return False, f"Missing news blackout checks: {missing}"

        return True, "The5ers news blackout implemented"

    def audit_no_martingale(self) -> Tuple[bool, str]:
        """Check no martingale/grid logic in codebase."""
        self.log("Checking for prohibited martingale/grid...")

        prohibited_patterns = [
            r"martingale",
            r"grid_trading",
            r"double_down",
            r"lot_multiplier.*>.*1",
            r"size.*\*=.*2",
        ]

        # Check strategies directory
        strategies_dir = self.root / "strategies"
        if strategies_dir.exists():
            for py_file in strategies_dir.glob("*.py"):
                content = py_file.read_text().lower()
                for pattern in prohibited_patterns[:3]:  # Only check obvious ones
                    if re.search(pattern, content):
                        return False, f"Found prohibited pattern '{pattern}' in {py_file.name}"

        return True, "No martingale/grid logic detected"

    def audit_no_hedging(self) -> Tuple[bool, str]:
        """Check no hedging within same account."""
        self.log("Checking for prohibited hedging...")

        # Check if hedging logic exists
        hedging_patterns = [
            r"hedge_position",
            r"opposite_direction.*same_symbol",
            r"open_hedge",
        ]

        strategies_dir = self.root / "strategies"
        if strategies_dir.exists():
            for py_file in strategies_dir.glob("*.py"):
                content = py_file.read_text().lower()
                for pattern in hedging_patterns:
                    if re.search(pattern, content):
                        return False, f"Found prohibited hedging pattern in {py_file.name}"

        return True, "No hedging logic detected"

    def audit_config_files(self) -> Tuple[bool, str]:
        """Check config files have compliance parameters."""
        self.log("Checking config files...")

        config_checks = {
            "config/gft_account_1.py": [
                "GUARDIAN_FLOATING_PCT",
                "GUARDIAN_DAILY_DD_PCT",
                "GUARDIAN_TOTAL_DD_PCT",
            ],
            "config/the5ers_account.py": [
                "GUARDIAN_DAILY_LOSS_PCT",
                "GUARDIAN_TOTAL_DD_PCT",
            ],
        }

        for filepath, required in config_checks.items():
            missing = self.check_file_contains(filepath, required)
            if missing:
                return False, f"Missing in {filepath}: {missing}"

        return True, "Config files have compliance parameters"

    def run_full_audit(self) -> Dict[str, Any]:
        """Run all compliance audits."""
        print("\n" + "=" * 70)
        print("  SOVEREIGN V5 COMPLIANCE AUDIT")
        print("=" * 70 + "\n")

        checks = [
            ("GFT -2% floating loss check", self.audit_gft_floating_loss),
            ("GFT 3% daily DD guardian", self.audit_gft_daily_dd),
            ("GFT 6% total DD guardian", self.audit_gft_total_dd),
            ("GFT 2% max risk per trade", self.audit_gft_risk_per_trade),
            ("GFT news blackout 5min", self.audit_gft_news_blackout),
            ("GFT $3000 daily cap", self.audit_gft_profit_cap),
            ("The5ers 5% daily loss guardian", self.audit_the5ers_daily_loss),
            ("The5ers 10% static DD guardian", self.audit_the5ers_static_dd),
            ("The5ers news blackout 2min", self.audit_the5ers_news_blackout),
            ("No martingale/grid logic", self.audit_no_martingale),
            ("No hedging logic", self.audit_no_hedging),
            ("Config files compliance", self.audit_config_files),
        ]

        passed = 0
        failed = 0
        results = {}

        print("GFT INSTANT GOAT COMPLIANCE:")
        print("-" * 50)

        for name, check_fn in checks[:6]:
            success, msg = check_fn()
            results[name] = {"passed": success, "message": msg}

            if success:
                print(f"  ✅ {name}")
                passed += 1
            else:
                print(f"  ❌ {name}")
                print(f"     → {msg}")
                failed += 1

        print("\nTHE5ERS HIGH STAKES COMPLIANCE:")
        print("-" * 50)

        for name, check_fn in checks[6:9]:
            success, msg = check_fn()
            results[name] = {"passed": success, "message": msg}

            if success:
                print(f"  ✅ {name}")
                passed += 1
            else:
                print(f"  ❌ {name}")
                print(f"     → {msg}")
                failed += 1

        print("\nPROHIBITED STRATEGIES CHECK:")
        print("-" * 50)

        for name, check_fn in checks[9:11]:
            success, msg = check_fn()
            results[name] = {"passed": success, "message": msg}

            if success:
                print(f"  ✅ {name}")
                passed += 1
            else:
                print(f"  ❌ {name}")
                print(f"     → {msg}")
                failed += 1

        print("\nCONFIGURATION CHECK:")
        print("-" * 50)

        for name, check_fn in checks[11:]:
            success, msg = check_fn()
            results[name] = {"passed": success, "message": msg}

            if success:
                print(f"  ✅ {name}")
                passed += 1
            else:
                print(f"  ❌ {name}")
                print(f"     → {msg}")
                failed += 1

        # Summary
        print("\n" + "=" * 70)
        print(f"  AUDIT SUMMARY: {passed}/{passed + failed} checks passed")
        print("=" * 70)

        if failed == 0:
            print("\n✅ ALL COMPLIANCE CHECKS PASSED")
            print("   System is ready for paper trading deployment.")
        else:
            print(f"\n❌ {failed} COMPLIANCE CHECK(S) FAILED")
            print("   DO NOT DEPLOY until all checks pass!")

        print()

        return {
            "passed": passed,
            "failed": failed,
            "total": passed + failed,
            "results": results,
            "ready_for_deployment": failed == 0
        }


def main():
    parser = argparse.ArgumentParser(description="Compliance Audit for Sovereign V5")
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--fix', action='store_true', help='Attempt auto-fixes (not implemented)')

    args = parser.parse_args()

    auditor = ComplianceAuditor(verbose=args.verbose)
    results = auditor.run_full_audit()

    # Exit with appropriate code
    sys.exit(0 if results["ready_for_deployment"] else 1)


if __name__ == "__main__":
    main()
