"""
Heartbeat Module - Periodic health checks.

Monitors system health and connectivity.
"""

import logging
import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class ComponentState(Enum):
    """Component health states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HeartbeatConfig:
    """Heartbeat configuration."""
    check_interval_seconds: int = 30
    timeout_seconds: int = 10
    max_consecutive_failures: int = 3
    alert_on_degraded: bool = True


@dataclass
class HeartbeatStatus:
    """Status of a heartbeat check."""
    component: str
    state: ComponentState
    last_check: datetime
    last_success: Optional[datetime]
    consecutive_failures: int
    latency_ms: float
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class Heartbeat:
    """
    Heartbeat monitor for system health.

    Features:
    - Periodic health checks
    - Component status tracking
    - Automatic alerting
    - Graceful degradation

    Usage:
        heartbeat = Heartbeat(config)

        # Register checks
        heartbeat.register_check("mt5", mt5_health_check)
        heartbeat.register_check("database", db_health_check)

        # Start monitoring
        heartbeat.start()

        # Get status
        status = heartbeat.get_all_status()
    """

    def __init__(
        self,
        config: HeartbeatConfig = None,
        on_alert_callback: Callable[[HeartbeatStatus], None] = None
    ):
        """
        Initialize heartbeat monitor.

        Args:
            config: Heartbeat configuration
            on_alert_callback: Callback for alerts
        """
        self.config = config or HeartbeatConfig()
        self.on_alert = on_alert_callback

        self._checks: Dict[str, Callable[[], bool]] = {}
        self._status: Dict[str, HeartbeatStatus] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def register_check(
        self,
        component: str,
        check_func: Callable[[], bool],
        timeout_seconds: int = None
    ) -> None:
        """
        Register a health check function.

        Args:
            component: Component name
            check_func: Function that returns True if healthy
            timeout_seconds: Override timeout for this check
        """
        self._checks[component] = check_func
        self._status[component] = HeartbeatStatus(
            component=component,
            state=ComponentState.UNKNOWN,
            last_check=datetime.now(),
            last_success=None,
            consecutive_failures=0,
            latency_ms=0,
            message="Not checked yet"
        )

        logger.info(f"Registered health check for: {component}")

    def start(self) -> None:
        """Start the heartbeat monitor."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        logger.info("Heartbeat monitor started")

    def stop(self) -> None:
        """Stop the heartbeat monitor."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

        logger.info("Heartbeat monitor stopped")

    def get_status(self, component: str) -> Optional[HeartbeatStatus]:
        """Get status for a specific component."""
        return self._status.get(component)

    def get_all_status(self) -> Dict[str, HeartbeatStatus]:
        """Get status for all components."""
        return self._status.copy()

    def is_healthy(self) -> bool:
        """Check if all components are healthy."""
        return all(
            s.state == ComponentState.HEALTHY
            for s in self._status.values()
        )

    def _run_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            for component, check_func in self._checks.items():
                self._run_check(component, check_func)

            time.sleep(self.config.check_interval_seconds)

    def _run_check(
        self,
        component: str,
        check_func: Callable[[], bool]
    ) -> None:
        """Run a single health check."""
        start_time = time.time()

        try:
            # Run check with timeout
            result = self._run_with_timeout(check_func, self.config.timeout_seconds)
            latency = (time.time() - start_time) * 1000

            if result:
                self._update_status(
                    component,
                    ComponentState.HEALTHY,
                    latency,
                    "Check passed"
                )
            else:
                self._update_status(
                    component,
                    ComponentState.UNHEALTHY,
                    latency,
                    "Check returned False"
                )

        except TimeoutError:
            latency = (time.time() - start_time) * 1000
            self._update_status(
                component,
                ComponentState.UNHEALTHY,
                latency,
                f"Check timed out after {self.config.timeout_seconds}s"
            )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            self._update_status(
                component,
                ComponentState.UNHEALTHY,
                latency,
                f"Check failed: {str(e)}"
            )

    def _update_status(
        self,
        component: str,
        state: ComponentState,
        latency: float,
        message: str
    ) -> None:
        """Update component status."""
        current = self._status.get(component)
        now = datetime.now()

        if state == ComponentState.HEALTHY:
            new_status = HeartbeatStatus(
                component=component,
                state=state,
                last_check=now,
                last_success=now,
                consecutive_failures=0,
                latency_ms=latency,
                message=message
            )
        else:
            consecutive = (current.consecutive_failures + 1) if current else 1
            last_success = current.last_success if current else None

            new_status = HeartbeatStatus(
                component=component,
                state=state,
                last_check=now,
                last_success=last_success,
                consecutive_failures=consecutive,
                latency_ms=latency,
                message=message
            )

            # Check if we need to alert
            if consecutive >= self.config.max_consecutive_failures:
                self._trigger_alert(new_status)

        self._status[component] = new_status

    def _trigger_alert(self, status: HeartbeatStatus) -> None:
        """Trigger an alert for unhealthy component."""
        logger.error(
            f"Health check alert: {status.component} - "
            f"{status.state.value} - {status.message}"
        )

        if self.on_alert:
            try:
                self.on_alert(status)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def _run_with_timeout(
        self,
        func: Callable,
        timeout: int
    ) -> bool:
        """Run function with timeout."""
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"Check timed out after {timeout}s")


# Pre-built health checks

def create_mt5_health_check(connector) -> Callable[[], bool]:
    """Create MT5 connectivity health check."""
    def check():
        try:
            account = connector.get_account_info()
            return account is not None
        except:
            return False
    return check


def create_database_health_check(connection) -> Callable[[], bool]:
    """Create database connectivity health check."""
    def check():
        try:
            connection.execute("SELECT 1")
            return True
        except:
            return False
    return check


def create_api_health_check(url: str) -> Callable[[], bool]:
    """Create API endpoint health check."""
    import requests

    def check():
        try:
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except:
            return False
    return check
