"""
Diagnostics Module - System diagnostics and health reporting.

Provides comprehensive system health reports.
"""

import logging
import platform
import os
import socket
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """Component health status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class ComponentDiagnostic:
    """Diagnostic info for a single component."""
    name: str
    status: ComponentStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    last_check: datetime = field(default_factory=datetime.now)


@dataclass
class DiagnosticReport:
    """Complete diagnostic report."""
    timestamp: datetime
    overall_status: ComponentStatus
    components: Dict[str, ComponentDiagnostic]
    system_info: Dict[str, Any]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'overall_status': self.overall_status.value,
            'components': {
                k: {
                    'name': v.name,
                    'status': v.status.value,
                    'message': v.message,
                    'details': v.details,
                }
                for k, v in self.components.items()
            },
            'system_info': self.system_info,
            'recommendations': self.recommendations,
        }


class SystemDiagnostics:
    """
    System diagnostics for health monitoring.

    Provides:
    - System resource checks
    - Component health aggregation
    - Diagnostic recommendations
    - Health reporting

    Usage:
        diagnostics = SystemDiagnostics()

        # Register components
        diagnostics.register_component("mt5", mt5_check_func)

        # Generate report
        report = diagnostics.run_diagnostics()
    """

    def __init__(self):
        """Initialize diagnostics."""
        self._component_checks: Dict[str, callable] = {}

    def register_component(
        self,
        name: str,
        check_func: callable
    ) -> None:
        """
        Register a component for diagnostics.

        Args:
            name: Component name
            check_func: Function returning ComponentDiagnostic
        """
        self._component_checks[name] = check_func
        logger.info(f"Registered diagnostic component: {name}")

    def run_diagnostics(self) -> DiagnosticReport:
        """
        Run full system diagnostics.

        Returns:
            DiagnosticReport with all findings
        """
        components = {}
        recommendations = []

        # Run component checks
        for name, check_func in self._component_checks.items():
            try:
                diagnostic = check_func()
                components[name] = diagnostic

                if diagnostic.status == ComponentStatus.WARNING:
                    recommendations.append(f"[{name}] {diagnostic.message}")
                elif diagnostic.status == ComponentStatus.ERROR:
                    recommendations.append(f"[{name}] CRITICAL: {diagnostic.message}")

            except Exception as e:
                components[name] = ComponentDiagnostic(
                    name=name,
                    status=ComponentStatus.ERROR,
                    message=f"Diagnostic check failed: {str(e)}"
                )
                recommendations.append(f"[{name}] Fix diagnostic check: {str(e)}")

        # Add system checks
        system_diagnostic = self._check_system_resources()
        components['system'] = system_diagnostic

        if system_diagnostic.status != ComponentStatus.HEALTHY:
            recommendations.append(f"[system] {system_diagnostic.message}")

        # Determine overall status
        overall = self._determine_overall_status(components)

        # Get system info
        system_info = self._get_system_info()

        return DiagnosticReport(
            timestamp=datetime.now(),
            overall_status=overall,
            components=components,
            system_info=system_info,
            recommendations=recommendations
        )

    def _check_system_resources(self) -> ComponentDiagnostic:
        """Check system resources."""
        try:
            import psutil

            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent

            # Determine status
            status = ComponentStatus.HEALTHY
            messages = []

            if cpu_percent > 90:
                status = ComponentStatus.ERROR
                messages.append(f"CPU critical: {cpu_percent}%")
            elif cpu_percent > 75:
                status = ComponentStatus.WARNING
                messages.append(f"CPU high: {cpu_percent}%")

            if memory_percent > 90:
                status = ComponentStatus.ERROR
                messages.append(f"Memory critical: {memory_percent}%")
            elif memory_percent > 80:
                if status != ComponentStatus.ERROR:
                    status = ComponentStatus.WARNING
                messages.append(f"Memory high: {memory_percent}%")

            if disk_percent > 90:
                status = ComponentStatus.ERROR
                messages.append(f"Disk critical: {disk_percent}%")
            elif disk_percent > 80:
                if status != ComponentStatus.ERROR:
                    status = ComponentStatus.WARNING
                messages.append(f"Disk high: {disk_percent}%")

            message = "; ".join(messages) if messages else "All resources within limits"

            return ComponentDiagnostic(
                name="system",
                status=status,
                message=message,
                details={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'memory_available_gb': memory.available / 1024 / 1024 / 1024,
                    'disk_percent': disk_percent,
                    'disk_free_gb': disk.free / 1024 / 1024 / 1024,
                }
            )

        except ImportError:
            return ComponentDiagnostic(
                name="system",
                status=ComponentStatus.UNKNOWN,
                message="psutil not installed - cannot check system resources"
            )

        except Exception as e:
            return ComponentDiagnostic(
                name="system",
                status=ComponentStatus.ERROR,
                message=f"System check failed: {str(e)}"
            )

    def _determine_overall_status(
        self,
        components: Dict[str, ComponentDiagnostic]
    ) -> ComponentStatus:
        """Determine overall system status."""
        statuses = [c.status for c in components.values()]

        if ComponentStatus.ERROR in statuses:
            return ComponentStatus.ERROR
        elif ComponentStatus.WARNING in statuses:
            return ComponentStatus.WARNING
        elif ComponentStatus.UNKNOWN in statuses:
            return ComponentStatus.WARNING
        else:
            return ComponentStatus.HEALTHY

    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information."""
        info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'hostname': socket.gethostname(),
            'timestamp': datetime.now().isoformat(),
        }

        try:
            import psutil
            info['cpu_count'] = psutil.cpu_count()
            info['total_memory_gb'] = psutil.virtual_memory().total / 1024 / 1024 / 1024
        except:
            pass

        return info


# Pre-built diagnostic checks

def create_mt5_diagnostic(connector) -> callable:
    """Create MT5 diagnostic check."""
    def check():
        try:
            account = connector.get_account_info()
            if account:
                return ComponentDiagnostic(
                    name="mt5",
                    status=ComponentStatus.HEALTHY,
                    message="MT5 connected",
                    details={
                        'login': account.login,
                        'balance': account.balance,
                        'equity': account.equity,
                    }
                )
            else:
                return ComponentDiagnostic(
                    name="mt5",
                    status=ComponentStatus.ERROR,
                    message="MT5 not connected"
                )
        except Exception as e:
            return ComponentDiagnostic(
                name="mt5",
                status=ComponentStatus.ERROR,
                message=f"MT5 check failed: {str(e)}"
            )
    return check


def create_risk_diagnostic(risk_manager) -> callable:
    """Create risk engine diagnostic check."""
    def check():
        try:
            state = risk_manager.state

            if state.is_locked:
                return ComponentDiagnostic(
                    name="risk_engine",
                    status=ComponentStatus.ERROR,
                    message=f"Account locked: {state.lock_reason}",
                    details={'locked': True, 'reason': state.lock_reason}
                )

            dd_pct = risk_manager.get_current_drawdown_pct()

            if dd_pct >= 7.0:  # Guardian
                status = ComponentStatus.ERROR
                message = f"At guardian limit: {dd_pct:.2f}% drawdown"
            elif dd_pct >= 5.0:
                status = ComponentStatus.WARNING
                message = f"Approaching guardian: {dd_pct:.2f}% drawdown"
            else:
                status = ComponentStatus.HEALTHY
                message = f"Healthy: {dd_pct:.2f}% drawdown"

            return ComponentDiagnostic(
                name="risk_engine",
                status=status,
                message=message,
                details={
                    'drawdown_pct': dd_pct,
                    'daily_pnl': state.daily_pnl,
                    'total_trades': state.total_trades,
                }
            )
        except Exception as e:
            return ComponentDiagnostic(
                name="risk_engine",
                status=ComponentStatus.ERROR,
                message=f"Risk check failed: {str(e)}"
            )
    return check


def create_data_diagnostic(data_client) -> callable:
    """Create data feed diagnostic check."""
    def check():
        try:
            # Check if data is fresh
            last_update = data_client.get_last_update_time()
            if last_update:
                age_seconds = (datetime.now() - last_update).total_seconds()

                if age_seconds > 300:  # 5 minutes
                    return ComponentDiagnostic(
                        name="data_feed",
                        status=ComponentStatus.ERROR,
                        message=f"Data stale: {age_seconds:.0f}s old"
                    )
                elif age_seconds > 60:
                    return ComponentDiagnostic(
                        name="data_feed",
                        status=ComponentStatus.WARNING,
                        message=f"Data delayed: {age_seconds:.0f}s old"
                    )

            return ComponentDiagnostic(
                name="data_feed",
                status=ComponentStatus.HEALTHY,
                message="Data feed active"
            )
        except Exception as e:
            return ComponentDiagnostic(
                name="data_feed",
                status=ComponentStatus.ERROR,
                message=f"Data check failed: {str(e)}"
            )
    return check
