"""
Watchdog Module - Process monitoring and recovery.

Monitors critical processes and triggers recovery actions.
"""

import logging
import os
import signal
import subprocess
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class ProcessState(Enum):
    """Process states."""
    RUNNING = "running"
    STOPPED = "stopped"
    CRASHED = "crashed"
    RESTARTING = "restarting"
    UNKNOWN = "unknown"


@dataclass
class WatchdogConfig:
    """Watchdog configuration."""
    check_interval_seconds: int = 10
    restart_delay_seconds: int = 5
    max_restarts: int = 3
    restart_window_seconds: int = 300  # Reset restart count after this
    memory_limit_mb: Optional[int] = None
    cpu_limit_percent: Optional[float] = None


@dataclass
class WatchdogAlert:
    """Watchdog alert."""
    process_name: str
    alert_type: str
    message: str
    timestamp: datetime
    action_taken: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessInfo:
    """Information about a monitored process."""
    name: str
    pid: Optional[int]
    state: ProcessState
    start_command: Optional[str]
    restart_count: int
    last_restart: Optional[datetime]
    last_check: datetime
    memory_mb: float
    cpu_percent: float


class ProcessWatchdog:
    """
    Process watchdog for critical component monitoring.

    Features:
    - Automatic process restart
    - Resource usage monitoring
    - Crash detection
    - Alert generation

    Usage:
        watchdog = ProcessWatchdog(config)

        # Register process to monitor
        watchdog.register_process(
            "trading_bot",
            pid=12345,
            start_command="python main.py"
        )

        # Start monitoring
        watchdog.start()
    """

    def __init__(
        self,
        config: WatchdogConfig = None,
        on_alert_callback: Callable[[WatchdogAlert], None] = None
    ):
        """
        Initialize watchdog.

        Args:
            config: Watchdog configuration
            on_alert_callback: Callback for alerts
        """
        self.config = config or WatchdogConfig()
        self.on_alert = on_alert_callback

        self._processes: Dict[str, ProcessInfo] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def register_process(
        self,
        name: str,
        pid: int = None,
        start_command: str = None
    ) -> None:
        """
        Register a process to monitor.

        Args:
            name: Process name
            pid: Process ID
            start_command: Command to restart process
        """
        self._processes[name] = ProcessInfo(
            name=name,
            pid=pid,
            state=ProcessState.UNKNOWN,
            start_command=start_command,
            restart_count=0,
            last_restart=None,
            last_check=datetime.now(),
            memory_mb=0,
            cpu_percent=0
        )

        logger.info(f"Registered process for monitoring: {name} (PID: {pid})")

    def unregister_process(self, name: str) -> None:
        """Remove a process from monitoring."""
        if name in self._processes:
            del self._processes[name]
            logger.info(f"Unregistered process: {name}")

    def start(self) -> None:
        """Start the watchdog."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        logger.info("Process watchdog started")

    def stop(self) -> None:
        """Stop the watchdog."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

        logger.info("Process watchdog stopped")

    def get_process_info(self, name: str) -> Optional[ProcessInfo]:
        """Get info for a specific process."""
        return self._processes.get(name)

    def get_all_processes(self) -> Dict[str, ProcessInfo]:
        """Get info for all monitored processes."""
        return self._processes.copy()

    def _run_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            for name in list(self._processes.keys()):
                self._check_process(name)

            time.sleep(self.config.check_interval_seconds)

    def _check_process(self, name: str) -> None:
        """Check a single process."""
        info = self._processes.get(name)
        if not info:
            return

        # Check if process is running
        if info.pid:
            is_running = self._is_process_running(info.pid)

            if is_running:
                info.state = ProcessState.RUNNING

                # Update resource usage
                info.memory_mb, info.cpu_percent = self._get_process_resources(info.pid)

                # Check resource limits
                self._check_resource_limits(name, info)
            else:
                # Process died
                info.state = ProcessState.CRASHED
                self._handle_crash(name, info)
        else:
            info.state = ProcessState.UNKNOWN

        info.last_check = datetime.now()

    def _is_process_running(self, pid: int) -> bool:
        """Check if a process is running."""
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _get_process_resources(self, pid: int) -> tuple:
        """Get memory and CPU usage for process."""
        try:
            import psutil
            process = psutil.Process(pid)
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent(interval=0.1)
            return memory_mb, cpu_percent
        except:
            return 0.0, 0.0

    def _check_resource_limits(self, name: str, info: ProcessInfo) -> None:
        """Check if process exceeds resource limits."""
        if self.config.memory_limit_mb and info.memory_mb > self.config.memory_limit_mb:
            self._trigger_alert(
                name,
                "memory_exceeded",
                f"Memory usage ({info.memory_mb:.1f}MB) exceeds limit ({self.config.memory_limit_mb}MB)",
                "warning"
            )

        if self.config.cpu_limit_percent and info.cpu_percent > self.config.cpu_limit_percent:
            self._trigger_alert(
                name,
                "cpu_exceeded",
                f"CPU usage ({info.cpu_percent:.1f}%) exceeds limit ({self.config.cpu_limit_percent}%)",
                "warning"
            )

    def _handle_crash(self, name: str, info: ProcessInfo) -> None:
        """Handle a crashed process."""
        logger.error(f"Process crashed: {name}")

        # Reset restart count if outside window
        if info.last_restart:
            time_since_restart = (datetime.now() - info.last_restart).total_seconds()
            if time_since_restart > self.config.restart_window_seconds:
                info.restart_count = 0

        # Check if we can restart
        if info.restart_count >= self.config.max_restarts:
            self._trigger_alert(
                name,
                "max_restarts_exceeded",
                f"Process {name} exceeded max restart attempts ({self.config.max_restarts})",
                "critical"
            )
            return

        # Attempt restart
        if info.start_command:
            self._restart_process(name, info)
        else:
            self._trigger_alert(
                name,
                "crash_no_restart",
                f"Process {name} crashed but no restart command configured",
                "error"
            )

    def _restart_process(self, name: str, info: ProcessInfo) -> None:
        """Restart a crashed process."""
        logger.info(f"Restarting process: {name}")

        info.state = ProcessState.RESTARTING
        info.restart_count += 1
        info.last_restart = datetime.now()

        time.sleep(self.config.restart_delay_seconds)

        try:
            process = subprocess.Popen(
                info.start_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            info.pid = process.pid
            info.state = ProcessState.RUNNING

            self._trigger_alert(
                name,
                "process_restarted",
                f"Process {name} restarted successfully (PID: {process.pid})",
                "info"
            )

        except Exception as e:
            logger.error(f"Failed to restart process {name}: {e}")
            info.state = ProcessState.CRASHED

            self._trigger_alert(
                name,
                "restart_failed",
                f"Failed to restart process {name}: {str(e)}",
                "error"
            )

    def _trigger_alert(
        self,
        process_name: str,
        alert_type: str,
        message: str,
        action: str
    ) -> None:
        """Trigger an alert."""
        alert = WatchdogAlert(
            process_name=process_name,
            alert_type=alert_type,
            message=message,
            timestamp=datetime.now(),
            action_taken=action
        )

        logger.warning(f"Watchdog alert: {alert_type} - {message}")

        if self.on_alert:
            try:
                self.on_alert(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
