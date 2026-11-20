"""Run two explicit TRAIN checkpoint paths without SAC learning or holdout access."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # Initialize before the simulator on Windows.

from scripts.evaluate_stage5_pilot import evaluate_checkpoint
from src.launcher.executor import ExperimentExecutor, build_launcher_agent_input
from src.launcher.models import LauncherMode, LaunchRequestV1
from src.launcher.registry import build_default_agent_registry, make_sac_agent_factory
from src.launcher.resolver import resolve_experiment_plan
from src.platform.task_context_producer import (
    LEGACY_TASK_PRODUCER, REFERENCE_TASK_PRODUCER, TaskContextLocalizationError,
)
from src.training.stage5.sac_trainer import SacBaselineTrainer
from src.training.stage5.env_adapter import Stage5TrainingEnv
from src.training.stage5.train_sampler import TrainCaseSampler


def audit_localization_abort():
    geometry = "geom_medium_SCTCS_seed11"
    sampler = TrainCaseSampler(allowed_geometry_ids=[geometry])
    env = Stage5TrainingEnv(sampler=sampler, task_context_producer_id=REFERENCE_TASK_PRODUCER)
    try:
        case = sampler.get_case_for_geometry_id(geometry, environment_seed=0)
        observation, _ = env.reset(options={"case": case})
        md = env._env
        raw = observation[:259]
        calls = {
            "train": lambda: env._build_agent_input(raw, 0),
            "launcher": lambda: build_launcher_agent_input(md, raw, 0, "STATE_DECISION_V1", REFERENCE_TASK_PRODUCER),
        }
        errors = {}
        with patch.object(md.agent.navigation, "_current_lane", None):
            for name, call in calls.items():
                try:
                    call()
                except TaskContextLocalizationError as exc:
                    errors[name] = str(exc)
                else:
                    raise AssertionError(f"{name} delivered an unlocalized observation")
        return {"geometry": geometry, "seed": 0, "injected_fault": "current lane unset at reset", "errors": errors}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    baseline_meta = json.loads((args.legacy / "metadata.json").read_text(encoding="utf-8"))
    assert "task_context_producer" not in baseline_meta
    with torch.random.fork_rng():
        trainer = SacBaselineTrainer(config_overrides={"task_context_producer_id": REFERENCE_TASK_PRODUCER})
        try:
            assert trainer.model._n_updates == trainer.model.num_timesteps == 0
            corrected = trainer.save_checkpoint(args.output / "zero_update_checkpoint")
        finally:
            trainer.close()
    split = ROOT / "results/stage5/pilot_split_v1.json"
    registry = build_default_agent_registry()
    request = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="stage5_sac", tier="Easy",
                              sequence="SCS", geometry_generation_seed=0, environment_seed=0)
    plan = resolve_experiment_plan(request, registry=registry, project_root=ROOT)
    assert len(plan.resolved_cases) == 1 and plan.resolved_cases[0].split == "TRAIN"
    rows = []
    for checkpoint, producer in ((args.legacy, LEGACY_TASK_PRODUCER), (corrected, REFERENCE_TASK_PRODUCER)):
        diag = evaluate_checkpoint(checkpoint, split, args.output / producer, max_cases=1,
                                   max_steps_per_episode=12, task_context_producer_id=producer)
        factory = make_sac_agent_factory(checkpoint, task_context_producer_id=producer)
        # Sandbox execution seam only. Unmodified preflight hashes the official holdout files.
        with patch("src.launcher.executor.run_preflight", return_value=SimpleNamespace(can_execute=True)):
            report = ExperimentExecutor(plan, project_root=ROOT, registry=registry, agent_factory=factory,
                                        runs_root=args.output / "launcher").execute()
        manifest = json.loads((Path(report.run_dir) / "run_manifest.json").read_text(encoding="utf-8"))
        recorded = manifest["config"]["algorithm_hyperparameters"]["task_context_producer"]
        assert recorded == diag["task_context_producer"] == factory.experiment_parameters()["task_context_producer"]
        rows.append({"producer": recorded, "model_sha256": diag["checkpoint_sha256"],
                     "diagnostic_cases": diag["planned_cases"], "diagnostic_state_unchanged": diag["evaluation_state_unchanged"],
                     "launcher_status": report.status, "launcher_completed_episodes": report.completed_episodes,
                     "launcher_preflight": "stubbed in audit to avoid reading official holdout manifests"})
    (args.output / "checkpoint_runtime.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (args.output / "localization_abort.json").write_text(
        json.dumps(audit_localization_abort(), indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
