"""Sequential PLAN/RUN scheduling through the existing isolated worker."""

import json
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from PySide6.QtCore import QObject, QTimer, Signal

from src.workbench.benchmark_campaign import EvaluationCampaign, inspect_campaign_results
from src.workbench.process_runner import WorkbenchProcessRunner


class CampaignController(QObject):
    changed = Signal()
    busyChanged = Signal(bool)
    messageChanged = Signal(str)
    resultsReady = Signal(list)

    def __init__(self, runner: WorkbenchProcessRunner, project_root: Path, parent=None):
        super().__init__(parent)
        self.runner = runner
        self.project_root = project_root
        self.active = False
        self.campaign: Optional[EvaluationCampaign] = None
        self.operation = ""
        self.status = "IDLE"
        self.rows = []
        self.validated_fingerprint: Optional[str] = None
        self.current_index = 0
        self._done = None
        self._report = None
        self._preflight = None
        self._error = ""
        self._cancelled = False
        self._campaign_id = ""

    def invalidate(self):
        if not self.active:
            self.validated_fingerprint = None
            self.status = "IDLE"

    def start(self, operation: str, campaign: EvaluationCampaign):
        if self.active or self.runner.is_running:
            raise RuntimeError("Một tác vụ khác đang chạy.")
        if operation not in ("PLAN", "RUN"):
            raise ValueError("Unsupported campaign operation")
        if operation == "RUN" and (
            campaign.blocking_reasons or self.validated_fingerprint != campaign.fingerprint
        ):
            raise ValueError("Cần kiểm tra thành công toàn bộ kế hoạch hiện tại trước khi chạy.")
        self.campaign = campaign
        self.operation = operation
        self.active = True
        self.status = "CHECKING" if operation == "PLAN" else "RUNNING"
        self.current_index = 0
        self._cancelled = False
        self._campaign_id = f"campaign_{uuid4().hex[:12]}"
        self.rows = [dict(agent_id=j.request.agent_id, agent_version=j.agent_version,
                          seed=(getattr(j.request, "checkpoint_binding", None).training_run_seed
                                if getattr(j.request, "checkpoint_binding", None) is not None else j.request.agent_seed),
                          seed_kind="train" if getattr(j.request, "checkpoint_binding", None) is not None else "agent",
                          cases=j.case_count, status="PENDING", message="", report=None)
                     for j in campaign.jobs]
        if operation == "PLAN":
            self.validated_fingerprint = None
        self.busyChanged.emit(True)
        self.changed.emit()
        self._start_job()

    def _start_job(self):
        if not self.active or self._cancelled:
            return
        self._done = self._report = self._preflight = None
        self._error = ""
        self.rows[self.current_index]["status"] = self.status
        job = self.campaign.jobs[self.current_index]
        self.messageChanged.emit(
            f"{self.operation} {self.current_index + 1}/{len(self.rows)} · {job.request.agent_id} · seed {job.request.agent_seed}"
        )
        self.changed.emit()
        try:
            self.runner.start_operation(self.operation, job.request)
        except Exception as exc:
            self._error = str(exc)
            self.process_finished(-1)

    def preflight_received(self, payload: Dict[str, Any]):
        self._preflight = payload.get("report", payload)

    def report_received(self, payload: Dict[str, Any]):
        self._report = payload.get("execution_report", payload)

    def worker_done(self, payload: Dict[str, Any]):
        self._done = payload

    def worker_error(self, payload: Dict[str, Any]):
        self._error = payload.get("message", "Worker failed")

    def cancel(self):
        self._cancelled = True
        self.validated_fingerprint = None
        if not self.runner.is_running:
            self._finish("CANCELLED")

    def process_finished(self, exit_code: int):
        if not self.active:
            return
        row = self.rows[self.current_index]
        if self._cancelled:
            row.update(status="CANCELLED", message="Dừng theo yêu cầu; không tạo kết quả hoàn tất giả.")
            self._finish("CANCELLED")
            return
        if exit_code != 0 or self._error or not self._done or not self._done.get("success"):
            row.update(status="FAILED", message=self._error or "Worker không hoàn tất giao thức thành công.")
            self._finish("FAILED")
            return
        if self.operation == "PLAN":
            allowed = bool(self._preflight and self._preflight.get("can_execute"))
            failures = [c["message"] for c in (self._preflight or {}).get("checks", []) if c["status"] == "FAIL"]
            row.update(status="READY" if allowed else "BLOCKED", message="; ".join(failures))
        else:
            if not self._report or self._report.get("status") != "COMPLETE":
                row.update(status="FAILED", message="Thiếu báo cáo COMPLETE.")
                self._finish("FAILED")
                return
            row.update(status="COMPLETE", report=self._report, message=self._report["run_id"])
        self.changed.emit()
        self.current_index += 1
        if self.current_index < len(self.rows):
            # Wait until QProcess terminal cleanup has completed before the next start.
            campaign_id = self._campaign_id
            QTimer.singleShot(0, lambda: self._start_job() if self.active and self._campaign_id == campaign_id else None)
            return
        if self.operation == "PLAN":
            ready = not self.campaign.blocking_reasons and all(r["status"] == "READY" for r in self.rows)
            if ready:
                self.validated_fingerprint = self.campaign.fingerprint
            self._finish("READY" if ready else "BLOCKED")
        else:
            try:
                metrics = inspect_campaign_results(self.campaign, [r["report"] for r in self.rows], self.project_root)
            except Exception as exc:
                row["message"] = str(exc)
                self._finish("RESULTS_REJECTED")
                return
            self.resultsReady.emit(metrics)
            self._finish("COMPLETE")

    def _finish(self, status: str):
        self.status = status
        for row in self.rows:
            if row["status"] == "PENDING":
                row.update(status="NOT_RUN", message="Lịch dừng trước tác vụ này.")
        self.active = False
        self.changed.emit()
        self.busyChanged.emit(False)
        message = f"Lịch {self.operation}: {status}."
        if self.operation == "RUN":
            # App index only. Gate-7 run artifacts are never edited by this controller.
            try:
                root = self.project_root / "artifacts" / "benchmark_campaigns"
                root.mkdir(parents=True, exist_ok=True)
                path = root / f"{self._campaign_id}.json"
                path.write_text(json.dumps(dict(
                    campaign_id=self._campaign_id, profile_id=self.campaign.profile.profile_id,
                    fingerprint=self.campaign.fingerprint, status=status, jobs=self.rows,
                ), ensure_ascii=False, indent=2), encoding="utf-8")
                message += f" Chỉ mục: {path}"
            except OSError as exc:
                message += f" Không lưu được chỉ mục ứng dụng: {exc}"
        self.messageChanged.emit(message)
