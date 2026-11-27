"""Version selection and provenance for the public TaskContext producer."""
import hashlib
import inspect
import json
from pathlib import Path

LEGACY_TASK_PRODUCER = "legacy_v1"
REFERENCE_TASK_PRODUCER = "reference_route_station_proposal_v1"
TASK_PRODUCERS = (LEGACY_TASK_PRODUCER, REFERENCE_TASK_PRODUCER)


def task_producer_identity(producer_id: str) -> dict:
    if producer_id not in TASK_PRODUCERS:
        raise ValueError(f"Unknown TaskContext producer: {producer_id!r}")
    configuration = {"lookahead_count": 20, "spacing_m": 2.5, "reference_lane": "first_on_each_route_edge"}
    if producer_id == LEGACY_TASK_PRODUCER:
        from src.platform.agent_input import _task_lane_width_and_goal, extract_task_context
        source = inspect.getsource(_task_lane_width_and_goal) + inspect.getsource(extract_task_context)
        configuration["implementation_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    else:
        from src.platform.reference_station import SEAM_TOLERANCE_M
        source = Path(__file__).with_name("reference_station.py").read_bytes().replace(b"\r\n", b"\n")
        configuration.update(
            checkpoint_window=1, seam_tolerance_m=SEAM_TOLERANCE_M,
            unlocalized_policy="abort_on_first_sample", implementation_sha256=hashlib.sha256(source).hexdigest(),
        )
    encoded = json.dumps({"id": producer_id, "configuration": configuration}, sort_keys=True, separators=(",", ":"))
    return {"id": producer_id, "fingerprint": hashlib.sha256(encoded.encode()).hexdigest(),
            "configuration": configuration}


def require_checkpoint_producer(metadata: dict, requested: str) -> dict:
    expected = task_producer_identity(requested)
    recorded = metadata.get("task_context_producer", task_producer_identity(LEGACY_TASK_PRODUCER))
    if recorded != expected:
        raise ValueError(f"TaskContext producer mismatch: requested {expected}, checkpoint records {recorded}")
    return expected


class TaskContextLocalizationError(RuntimeError):
    """The corrected producer cannot establish the designated route station."""
