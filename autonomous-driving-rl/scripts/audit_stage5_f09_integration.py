"""Exercise train and Launcher observation producer paths on matched TRAIN states."""
import gzip
import json
import subprocess
import sys
from functools import partial
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import audit_stage5_task_alignment as fixtures
from src.launcher.executor import build_launcher_agent_input
from src.platform.task_context_producer import (
    LEGACY_TASK_PRODUCER, REFERENCE_TASK_PRODUCER, TaskContextLocalizationError, task_producer_identity,
)
import numpy as np


def audit(geometry, actions, **options):
    frames = []
    base = fixtures.Stage5TrainingEnv

    def sample(env, observation, train_input, step):
        md = env._env
        lane = md.agent.navigation.current_lane
        launcher = build_launcher_agent_input(md, train_input.core_observation.features,
                                              step, train_input.profile_id, REFERENCE_TASK_PRODUCER)
        legacy = build_launcher_agent_input(md, train_input.core_observation.features,
                                            step, train_input.profile_id, LEGACY_TASK_PRODUCER)
        encoded = env.encoder.encode(launcher)
        assert np.array_equal(encoded, observation)
        assert np.array_equal(env.encoder.encode(legacy)[:325], encoded[:325])
        assert tuple(encoded.shape) == (409,)
        frame = {"step": step, "valid": int(launcher.task_context.validity_mask.sum()),
                 "near_end": launcher.task_context.route_end_within_lookahead,
                 "lane": list(lane.index), "lane_type": type(lane).__name__,
                 "checkpoints": list(md.agent.navigation._target_checkpoints_index),
                 "first_waypoint": encoded[325:328].tolist(),
                 "legacy_first_waypoint": legacy.task_context.waypoints[0].to_tuple()}
        assert np.all(launcher.task_context.waypoints_array[~launcher.task_context.validity_mask] == 0)
        frames.append(frame)
        return frame

    with patch.object(fixtures, "Stage5TrainingEnv", partial(base, task_context_producer_id=REFERENCE_TASK_PRODUCER)),          patch.object(fixtures, "sample", side_effect=sample):
        try:
            run = fixtures.rollout(actions, geometry=geometry, **options)
        except TaskContextLocalizationError as exc:
            return {"geometry": geometry, "parity_rows": len(frames), "localization_abort": str(exc)}
    assert run["physics_calls"] == run["decisions"] * 5 and run["physics_dt"] == .02
    assert run["action_max_error"] < 1e-7
    return {
        "geometry": geometry, "case": run["case_id"], "parity_rows": len(frames),
        "all_masked_rows": sum(not r["valid"] for r in frames),
        "invalid_waypoint_fraction": sum(20-r["valid"] for r in frames)/(20*len(frames)),
        "near_end_rows": sum(r["near_end"] for r in frames),
        "junction_t_rows": sum(any("T" in node for node in r["lane"][:2]) for r in frames),
        "circular_rows": sum(r["lane_type"] == "CircularLane" for r in frames),
        "checkpoint_transitions": sum(a["checkpoints"] != b["checkpoints"] for a,b in zip(frames,frames[1:])),
        "alternate_lane_rows": sum(r["lane"][-1] != 0 for r in frames),
        "last": frames[-1], "witness_614": next((r for r in frames if r["step"] == 614), None),
        "config": {"decision_repeat": run["decision_repeat"], "physics_dt": run["physics_dt"]},
    }


def main():
    with gzip.open(fixtures.RUN / "stage1_reference_decisions.jsonl.gz", "rt", encoding="utf-8") as stream:
        clean = [json.loads(line)["canonical_action"] for line in stream]
    traces = [
        audit(fixtures.GEOMETRY, clean),
        audit(fixtures.GEOMETRY, [[0, 0]]*900, offset_lane=True, diagnostic_follower=True),
        audit("geom_easy_SCCS_seed0", [[0, 0]]*900, diagnostic_follower=True),
        audit("geom_medium_SCTCS_seed11", [[0, 0]]*1200, diagnostic_follower=True),
        audit("geom_medium_SCTCS_seed8", [[0, 0]]*1200, diagnostic_follower=True),
    ]
    result = {"source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "producer": task_producer_identity(REFERENCE_TASK_PRODUCER), "traces": traces}
    path = Path(sys.argv[1])
    path.parent.mkdir(exist_ok=True, parents=True)
    payload = json.dumps(result, indent=2, default=fixtures.encode_scalar, allow_nan=False)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
