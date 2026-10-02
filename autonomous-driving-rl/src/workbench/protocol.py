"""
Workbench Protocol Specification and Serialization (Gate 7.5B).

Defines the JSONL sentinel protocol connecting the Workbench GUI process to the isolated
worker child process:
- WORKBENCH_SENTINEL: "@@WORKBENCH@@" prefix identifying structured protocol messages.
- WorkbenchMessageType: Supported protocol message types.
- WorkbenchMessageV1: Standardized protocol envelope.
- Serialization and deserialization utilities for requests, plans, reports, events, and errors.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.launcher.events import LauncherEventType, LauncherEventV1
from src.launcher.models import LaunchRequestV1, LauncherMode, ResolvedExperimentPlanV1
from src.launcher.preflight import PreflightCheckV1, PreflightReportV1
from src.platform import WandbMode
from src.workbench.contracts import WORKBENCH_PROTOCOL_VERSION

WORKBENCH_SENTINEL = "@@WORKBENCH@@"


class WorkbenchMessageType(str, Enum):
    """Protocol message types emitted by worker process to GUI."""
    WORKER_READY = "WORKER_READY"
    PLAN_RESOLVED = "PLAN_RESOLVED"
    PREFLIGHT_REPORT = "PREFLIGHT_REPORT"
    LAUNCHER_EVENT = "LAUNCHER_EVENT"
    EXECUTION_REPORT = "EXECUTION_REPORT"
    WORKER_ERROR = "WORKER_ERROR"
    WORKER_DONE = "WORKER_DONE"


@dataclass(frozen=True)
class WorkbenchMessageV1:
    """Standardized envelope for worker stdout protocol lines."""
    protocol_version: str
    type: WorkbenchMessageType
    timestamp: str
    payload: Dict[str, Any]

    def to_line(self) -> str:
        """Encodes envelope into sentinel JSON stdout line."""
        body = {
            "protocol_version": self.protocol_version,
            "type": self.type.value,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }
        return f"{WORKBENCH_SENTINEL}{json.dumps(body, ensure_ascii=False)}"

    @classmethod
    def create(cls, msg_type: WorkbenchMessageType, payload: Dict[str, Any]) -> "WorkbenchMessageV1":
        ts = datetime.now(timezone.utc).isoformat()
        return cls(
            protocol_version=WORKBENCH_PROTOCOL_VERSION,
            type=msg_type,
            timestamp=ts,
            payload=payload,
        )


def parse_sentinel_line(line: str) -> Optional[WorkbenchMessageV1]:
    """
    Parses a single line of stdout.
    Returns WorkbenchMessageV1 if the line contains @@WORKBENCH@@ and valid JSON.
    Returns None if the line does not start with or contain the sentinel, or if JSON is invalid.
    Safe against non-sentinel MetaDrive / Panda3D / OS stdout.
    """
    clean_line = line.strip()
    idx = clean_line.find(WORKBENCH_SENTINEL)
    if idx < 0:
        return None

    raw_json = clean_line[idx + len(WORKBENCH_SENTINEL):].strip()
    try:
        data = json.loads(raw_json)
        if not isinstance(data, dict):
            return None
        proto = data.get("protocol_version")
        m_type_raw = data.get("type")
        payload = data.get("payload")
        if not proto or not m_type_raw or payload is None or not isinstance(payload, dict):
            return None
        try:
            m_type = WorkbenchMessageType(m_type_raw)
        except ValueError:
            return None
        ts = data.get("timestamp", datetime.now(timezone.utc).isoformat())
        return WorkbenchMessageV1(
            protocol_version=str(proto),
            type=m_type,
            timestamp=str(ts),
            payload=payload,
        )
    except Exception:
        return None


def serialize_launch_request(req: LaunchRequestV1) -> Dict[str, Any]:
    """Serializes LaunchRequestV1 into a clean JSON-safe dict, never including secret tokens."""
    data = req.to_dict()
    # Explicitly ensure no sensitive environment keys leak into dict
    for forbidden in ("WANDB_API_KEY", "api_key", "secret", "token"):
        data.pop(forbidden, None)
    return data


def deserialize_launch_request(data: Dict[str, Any]) -> LaunchRequestV1:
    """Reconstructs LaunchRequestV1 from JSON-safe payload without modifying scientific semantics."""
    mode = LauncherMode(data["mode"])
    agent_id = str(data["agent_id"])
    tier = data.get("tier")
    if tier == "" or tier == "None":
        tier = None
    sequence = data.get("sequence")
    if sequence == "" or sequence == "None":
        sequence = None
    geom_seed = data.get("geometry_generation_seed")
    if geom_seed is not None and geom_seed != "":
        geom_seed = int(geom_seed)
    else:
        geom_seed = None
    env_seed = data.get("environment_seed")
    if env_seed is not None and env_seed != "":
        env_seed = int(env_seed)
    else:
        env_seed = None
    agent_seed = data.get("agent_seed")
    if agent_seed is not None and agent_seed != "":
        agent_seed = int(agent_seed)
    else:
        agent_seed = None
    render_mode = str(data.get("render_mode", "OFF")).upper()
    wandb_mode_raw = data.get("wandb_mode")
    wandb_mode = WandbMode(wandb_mode_raw) if wandb_mode_raw is not None and wandb_mode_raw != "" else None
    runs_root_raw = data.get("runs_root")
    runs_root = Path(runs_root_raw) if runs_root_raw is not None and str(runs_root_raw).strip() != "" else None
    custom_run_id = data.get("custom_run_id")
    if custom_run_id == "" or custom_run_id == "None":
        custom_run_id = None

    return LaunchRequestV1(
        mode=mode,
        agent_id=agent_id,
        tier=tier,
        sequence=sequence,
        geometry_generation_seed=geom_seed,
        environment_seed=env_seed,
        agent_seed=agent_seed,
        render_mode=render_mode,
        wandb_mode=wandb_mode,
        runs_root=runs_root,
        custom_run_id=custom_run_id,
    )
