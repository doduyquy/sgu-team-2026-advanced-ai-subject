"""Export screening policies without changing the archived learner or its state identity."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from cloud.kaggle import runtime
from cloud.kaggle.common import sha, write


def save_screening_policy(model, directory, step):
    target = directory / f"step-{step:012d}"
    if target.exists():
        raise FileExistsError(target)
    stage = target.with_name(target.name + ".staging")
    stage.mkdir(parents=True)
    model.save(stage / "policy.zip")
    marker = stage / "POLICY.json"
    write(marker, {"schema": 1, "transitions": step, "updates": model._n_updates,
                   "model_sha256": sha(stage / "policy.zip"), "purpose": "SCREENING_ONLY"})
    stage.rename(target)
    return sha(target / "POLICY.json")


def train(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-dir", type=Path)
    parser.add_argument("--policy-every", type=int, choices=(2500, 5000, 50000))
    parser.add_argument("--state-every", type=int)
    parser.add_argument("--warm-start-cursor", type=int)
    hooks, learner_args = parser.parse_known_args(args)
    if bool(hooks.policy_dir) != bool(hooks.policy_every):
        parser.error("--policy-dir and --policy-every must be paired")
    if hooks.warm_start_cursor is not None and (hooks.warm_start_cursor < 0 or not
            any(flag in learner_args for flag in ("--source", "--warm-model"))):
        parser.error("A nonnegative cursor offset requires a fresh warm start")
    base_telemetry, make_vec = runtime.Telemetry, runtime.make_vec
    argv = sys.argv
    try:
        if hooks.policy_dir:
            class PolicyTelemetry(base_telemetry):
                def _on_step(self):
                    accepted = super()._on_step()
                    step = self.num_timesteps
                    if step % hooks.policy_every == 0 and (not hooks.state_every or step % hooks.state_every):
                        digest = save_screening_policy(self.model, hooks.policy_dir, step)
                        print(f"Screening policy step-{step:012d} marker_sha256={digest}", flush=True)
                    return accepted
            runtime.Telemetry = PolicyTelemetry
        if hooks.warm_start_cursor is not None:
            def shifted_vec(workers, phase, cursors):
                return make_vec(workers, phase, [cursor + hooks.warm_start_cursor for cursor in cursors])
            runtime.make_vec = shifted_vec
        sys.argv = [argv[0], "train", *learner_args]
        runtime.main()
    finally:
        sys.argv = argv
        runtime.Telemetry, runtime.make_vec = base_telemetry, make_vec


def screen(args):
    argv = sys.argv
    seeds = runtime.RECIPE["evaluation_seeds"]
    try:
        runtime.RECIPE["evaluation_seeds"] = seeds[:1]
        sys.argv = [argv[0], "evaluate", *args]
        runtime.main()
    finally:
        runtime.RECIPE["evaluation_seeds"] = seeds
        sys.argv = argv


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "train":
        train(sys.argv[2:])
    elif command == "screen":
        screen(sys.argv[2:])
    else:
        raise SystemExit("Expected train or screen")
