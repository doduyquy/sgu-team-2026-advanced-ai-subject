"""Versioned artifact adapter for trained heuristic checkpoints."""

import json
from pathlib import Path

from src.agents.heuristic_model import HeuristicMLP, file_sha256
from src.agents.planning_graph import load_astar_config, planner_fingerprint
from src.launcher.models import CheckpointBindingV1
from src.platform import canonical_json_sha256


def _inside_project(project_root, relative):
    root = Path(project_root).resolve()
    path = (root / relative).resolve()
    path.relative_to(root)
    return path


def validate_checkpoint_binding(binding, project_root):
    model_path = _inside_project(project_root, binding.model_path)
    config_path = _inside_project(project_root, binding.configuration_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if canonical_json_sha256(config) != binding.training_config_sha256:
        raise ValueError("Training configuration changed after checkpoint creation")
    if (planner_fingerprint(config["planner"]) != binding.planner_fingerprint
            or canonical_json_sha256(load_astar_config()) != binding.training_config_sha256):
        raise ValueError("Checkpoint does not match the active A* configuration")
    model = HeuristicMLP.load(model_path, binding.model_sha256, binding.planner_fingerprint)
    expected = dict(training_run_seed=binding.training_run_seed, training_config_sha256=binding.training_config_sha256,
                    data_sha256=binding.data_sha256, planner_fingerprint=binding.planner_fingerprint,
                    training_scope="TRAIN_ONLY")
    if any(model.metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint metadata does not match its declared training provenance")
    if not model.metadata.get("optimizer_steps", 0):
        raise ValueError("Checkpoint has no recorded optimization steps")
    if set(model.metadata.get("train_geometry_ids", [])) & set(model.metadata.get("internal_holdout_geometry_ids", [])):
        raise ValueError("Training geometry leaked into the internal holdout")
    from src.launcher.cases import load_manifest_csv, LOCKED_GEOMETRY_SPLIT_MANIFEST_SHA256
    rows = load_manifest_csv(Path(project_root) / "results/audits/evaluation_protocol/geometry_split_manifest.csv",
                             LOCKED_GEOMETRY_SPLIT_MANIFEST_SHA256)
    allowed = {row["geometry_id"] for row in rows if row["split"] == "TRAIN"}
    trained = set(model.metadata.get("train_geometry_ids", []))
    holdout = set(model.metadata.get("internal_holdout_geometry_ids", []))
    if not trained or not holdout or not (trained | holdout) <= allowed:
        raise ValueError("Checkpoint data must come exclusively from locked TRAIN geometries")
    return model_path


def bindings_from_training_manifest(project_root, manifest_path):
    path = _inside_project(project_root, manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != "astar_training_v1" or manifest.get("status") != "COMPLETE":
        raise ValueError("Training manifest is not COMPLETE / astar_training_v1")
    bindings = []
    for model in manifest["models"]:
        binding = CheckpointBindingV1(
            model_path=model["checkpoint_path"], model_sha256=model["checkpoint_sha256"],
            training_run_seed=model["training_run_seed"], training_config_sha256=manifest["training_config_sha256"],
            data_sha256=model["data_sha256"], planner_fingerprint=manifest["planner_fingerprint"],
        )
        validate_checkpoint_binding(binding, project_root)
        bindings.append(binding)
    if not bindings or len({b.training_run_seed for b in bindings}) != len(bindings):
        raise ValueError("Missing or duplicate training checkpoints")
    return tuple(sorted(bindings, key=lambda b: b.training_run_seed))


def latest_training_bindings(project_root):
    pointer = Path(project_root) / "artifacts/astar_learning/latest.json"
    if not pointer.exists():
        return ()
    data = json.loads(pointer.read_text(encoding="utf-8"))
    manifest = _inside_project(project_root, data["manifest"])
    if file_sha256(manifest) != data["manifest_sha256"]:
        raise ValueError("Training manifest digest differs from the selected checkpoint index")
    return bindings_from_training_manifest(project_root, data["manifest"])
