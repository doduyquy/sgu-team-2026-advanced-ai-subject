"""
Workbench Process Runner (Gate 7.5B Pass B3 Hardened).

Manages QProcess execution for the isolated Workbench worker:
- Uses sys.executable to launch src/workbench/worker.py in the active environment.
- Enforces maximum one active worker process.
- Feeds LaunchRequestV1 safely through stdin / temporary JSON.
- Parses stdout sentinel lines via parse_sentinel_line().
- Tracks semantic completion (WORKER_DONE received) and distinguishes:
  - normal semantic completion (success or blocked);
  - unexpected worker termination (crash, os._exit, signal kill);
  - FailedToStart (idempotent, single clean transition);
  - user FORCE TERMINATE.
- Guarantees temporary launch request JSON cleanup on all terminal paths.
"""

from enum import Enum
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, Optional

from PySide6.QtCore import QObject, QProcess, Signal

from src.launcher.models import LaunchRequestV1
from src.workbench.protocol import (
    WorkbenchMessageType,
    WorkbenchMessageV1,
    parse_sentinel_line,
    serialize_launch_request,
)


class WorkerTerminalReason(str, Enum):
    """Internal semantic classification of worker process termination."""
    WORKER_DONE = "WORKER_DONE"
    UNEXPECTED_EXIT = "UNEXPECTED_EXIT"
    FAILED_TO_START = "FAILED_TO_START"
    FORCE_TERMINATED = "FORCE_TERMINATED"


class WorkbenchProcessRunner(QObject):
    """
    Subprocess manager for Workbench worker operations using QProcess.
    All signal handlers and state transitions run strictly on Qt main thread.
    """

    # Signals for protocol messages
    workerReady = Signal(dict)
    planResolved = Signal(dict)
    preflightReport = Signal(dict)
    launcherEvent = Signal(dict)
    executionReport = Signal(dict)
    workerError = Signal(dict)
    workerDone = Signal(dict)

    # General lifecycle & output signals
    rawOutputReceived = Signal(str)
    processStarted = Signal()
    processFinished = Signal(int)  # exit code
    unexpectedWorkerExit = Signal(dict)  # {"exit_code": int, "operation": str}

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._process: Optional[QProcess] = None
        self._temp_request_file: Optional[Path] = None
        self._is_running: bool = False
        self._current_operation: Optional[str] = None
        self._stdout_buffer: str = ""
        self._worker_done_received: bool = False
        self._force_terminated: bool = False
        self._failed_to_start: bool = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def current_operation(self) -> Optional[str]:
        return self._current_operation

    @property
    def force_terminated(self) -> bool:
        return self._force_terminated

    def start_operation(self, operation: str, request: LaunchRequestV1) -> None:
        """Starts worker subprocess for PLAN or RUN operation."""
        if self._is_running:
            raise RuntimeError("A worker process is already running.")

        if operation not in ("PLAN", "RUN"):
            raise ValueError(f"Unsupported operation: {operation}")

        self._current_operation = operation
        self._is_running = True
        self._stdout_buffer = ""
        self._worker_done_received = False
        self._force_terminated = False
        self._failed_to_start = False

        # Write request payload to a temporary file
        serialized_req = serialize_launch_request(request)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".json") as tf:
            json.dump(serialized_req, tf, indent=2)
            self._temp_request_file = Path(tf.name)

        worker_script = Path(__file__).resolve().parent / "worker.py"
        project_root = worker_script.parent.parent.parent

        self._process = QProcess(self)
        self._process.setWorkingDirectory(str(project_root))
        self._process.setProgram(sys.executable)
        self._process.setArguments([
            str(worker_script),
            "--operation",
            operation,
            "--request-file",
            str(self._temp_request_file),
        ])

        self._process.started.connect(self._on_process_started)
        self._process.errorOccurred.connect(self._on_process_error)
        self._process.readyReadStandardOutput.connect(self._on_ready_read_stdout)
        self._process.readyReadStandardError.connect(self._on_ready_read_stderr)
        self._process.finished.connect(self._on_process_finished)

        self._process.start()

    def _on_process_started(self) -> None:
        self.processStarted.emit()

    def _on_process_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            if self._failed_to_start:
                return  # Idempotent terminal guard
            self._failed_to_start = True
            self._is_running = False
            self._current_operation = None
            self._cleanup_temp_file()
            err_msg = "Worker process failed to start (executable not found or permission denied)."
            self.workerError.emit({"message": err_msg, "error_type": "FailedToStart"})
            self.rawOutputReceived.emit(f"[ERROR] {err_msg}")
            self.processFinished.emit(-1)

    def _cleanup_temp_file(self) -> None:
        if self._temp_request_file and self._temp_request_file.exists():
            try:
                self._temp_request_file.unlink()
            except OSError:
                pass
            self._temp_request_file = None

    def force_terminate(self) -> None:
        """
        Emergency non-graceful process kill.
        Specification Section 21: Labeled FORCE TERMINATE.
        Does not rewrite local run_state or claim INTERRUPTED unless Gate 7 persisted it.
        """
        if self._process and self._is_running:
            self._force_terminated = True
            self._process.kill()

    def _on_ready_read_stdout(self) -> None:
        if not self._process:
            return
        data = self._process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        self._stdout_buffer += data
        lines = self._stdout_buffer.split("\n")
        # Keep incomplete last line in buffer
        self._stdout_buffer = lines[-1]

        for line in lines[:-1]:
            msg = parse_sentinel_line(line)
            if msg is not None:
                self._dispatch_protocol_message(msg)
            else:
                if line.strip():
                    self.rawOutputReceived.emit(line)

    def _on_ready_read_stderr(self) -> None:
        if not self._process:
            return
        data = self._process.readAllStandardError().data().decode("utf-8", errors="replace")
        for line in data.splitlines():
            if line.strip():
                self.rawOutputReceived.emit(f"[STDERR] {line}")

    def _dispatch_protocol_message(self, msg: WorkbenchMessageV1) -> None:
        mtype = msg.type
        payload = msg.payload
        if mtype == WorkbenchMessageType.WORKER_READY:
            self.workerReady.emit(payload)
        elif mtype == WorkbenchMessageType.PLAN_RESOLVED:
            self.planResolved.emit(payload)
        elif mtype == WorkbenchMessageType.PREFLIGHT_REPORT:
            self.preflightReport.emit(payload)
        elif mtype == WorkbenchMessageType.LAUNCHER_EVENT:
            self.launcherEvent.emit(payload)
        elif mtype == WorkbenchMessageType.EXECUTION_REPORT:
            self.executionReport.emit(payload)
        elif mtype == WorkbenchMessageType.WORKER_ERROR:
            self.workerError.emit(payload)
        elif mtype == WorkbenchMessageType.WORKER_DONE:
            self._worker_done_received = True
            self.workerDone.emit(payload)

    def _on_process_finished(self, exit_code: int) -> None:
        if self._failed_to_start:
            return  # FailedToStart already executed terminal cleanup

        # Flush remaining stdout buffer (handles trailing sentinel without newline)
        if self._stdout_buffer.strip():
            msg = parse_sentinel_line(self._stdout_buffer)
            if msg is not None:
                self._dispatch_protocol_message(msg)
            else:
                self.rawOutputReceived.emit(self._stdout_buffer.strip())
            self._stdout_buffer = ""

        # Semantic completion check (Specification Section 8 & 10)
        # If the process terminates without emitting WORKER_DONE, classify as unexpected exit
        if not self._worker_done_received and not self._force_terminated:
            err_data = {
                "operation": self._current_operation or "UNKNOWN",
                "exit_code": exit_code,
                "reason": "Process exited unexpectedly before emitting WORKER_DONE",
            }
            self.unexpectedWorkerExit.emit(err_data)
            self.workerError.emit({
                "message": f"Worker terminated unexpectedly (exit code {exit_code}) without completing protocol.",
                "error_type": "UnexpectedWorkerExit",
            })

        self._is_running = False
        self._current_operation = None
        self._cleanup_temp_file()
        self.processFinished.emit(exit_code)
