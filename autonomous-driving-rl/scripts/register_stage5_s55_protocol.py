"""Freeze the S5.5 protocol from the locked S5.4 internal TRAIN pools."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys_path_root = str(ROOT)
import sys
if sys_path_root not in sys.path:
    sys.path.insert(0, sys_path_root)

from src.launcher.cases import LOCKED_GEOMETRY_SPLIT_MANIFEST_SHA256, load_manifest_csv
from src.training.stage5.reward_experiments import RewardScale
OUT = ROOT / "results" / "stage5" / "s5_5_meter_reward_seed101_20261120"
S54 = ROOT / "results" / "stage5" / "s5_4_learnability_seed101_20261119" / "protocol.json"
SPLIT = ROOT / "results" / "stage5" / "pilot_split_v1.json"
MANIFEST = ROOT / "results" / "audits" / "evaluation_protocol" / "geometry_split_manifest.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    s54 = json.loads(S54.read_text(encoding="utf-8"))
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    records = load_manifest_csv(MANIFEST, expected_sha256=LOCKED_GEOMETRY_SPLIT_MANIFEST_SHA256)
    lookup = {r["geometry_id"]: r for r in records}
    train_ids = {r["geometry_id"] for r in split["pilot_train"]}
    diag_ids = {r["geometry_id"] for r in split["pilot_diagnostic"]}

    train = s54["training_geometry_pool"]
    diagnostic = s54["diagnostic_schedule"]
    if len(train) != 8 or len(diagnostic) != 6:
        raise RuntimeError("S5.5 requires the exact S5.4 pool sizes (8 train, 6 diagnostic)")
    if {r["geometry_id"] for r in train} - train_ids:
        raise RuntimeError("S5.4 training pool is not contained in PILOT_TRAIN")
    if {r["geometry_id"] for r in diagnostic} - diag_ids:
        raise RuntimeError("S5.4 diagnostic pool is not contained in PILOT_DIAGNOSTIC")
    if {r["geometry_id"] for r in train} & {r["geometry_id"] for r in diagnostic}:
        raise RuntimeError("Training and diagnostic geometry overlap")

    for case in [*train, *diagnostic]:
        row = lookup.get(case["geometry_id"])
        if row is None or row["split"].upper() != "TRAIN":
            raise RuntimeError(f"Geometry is not an authoritative TRAIN record: {case['geometry_id']}")
        if row["geometry_sha256"] != case["geometry_sha256"]:
            raise RuntimeError(f"Geometry hash mismatch: {case['geometry_id']}")
        if row["tier"] != "Easy":
            raise RuntimeError(f"S5.5 Easy-only pool contains non-Easy geometry: {case['geometry_id']}")

    lengths = [float(lookup[r["geometry_id"]]["route_length_m"]) for r in train]
    scale = RewardScale.from_route_lengths(lengths)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    protocol = {
        "protocol": "stage5_s5_5_meter_reward_v1",
        "status": "PRE_REGISTERED",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source,
        "source_geometry_manifest": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "source_geometry_manifest_file_sha256": sha256(MANIFEST),
        "source_geometry_manifest_canonical_sha256": LOCKED_GEOMETRY_SPLIT_MANIFEST_SHA256,
        "source_s5_4_protocol_sha256": sha256(S54),
        "pilot_split_sha256": split["split_sha256"],
        "training_root_seed": 101,
        "training_geometry_selection": "exact S5.4 eight Easy PILOT_TRAIN geometry IDs and hashes",
        "training_geometry_pool": [
            {"geometry_id": r["geometry_id"], "sequence": r["sequence"],
             "geometry_generation_seed": int(r["geometry_generation_seed"]),
             "geometry_sha256": r["geometry_sha256"],
             "manifest_route_length_m": float(lookup[r["geometry_id"]]["route_length_m"])}
            for r in train
        ],
        "training_pool_sha256": hashlib.sha256(json.dumps(
            [{"geometry_id": r["geometry_id"], "geometry_sha256": r["geometry_sha256"]} for r in train],
            sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "diagnostic_pool": [
            {"geometry_id": r["geometry_id"], "sequence": r["sequence"],
             "geometry_generation_seed": int(r["geometry_generation_seed"]),
             "geometry_sha256": r["geometry_sha256"],
             "environment_seed": int(r["environment_seed"])}
            for r in diagnostic
        ],
        "diagnostic_pool_sha256": s54["diagnostic_geometry_pool_sha256"],
        "sampler": {
            "strategy": "ROUND_ROBIN",
            "episode_index_starts_at": 1,
            "environment_seed_derivation": "SHA256(train_env_seed:{root_seed}:{episode_index}) first 32-bit word modulo 1,000,000",
            "pool_order": [r["geometry_id"] for r in train],
        },
        "observations": {
            "canonical_encoder": "stage5_flat409_v1",
            "dimension": 409,
            "post_encoder_transform": "pilot_groupwise_physical_v1",
            "transform_sha256": s54["observation_transform_sha256"],
            "same_for_all_arms": True,
            "policy_input_contract_changed": False,
        },
        "actions": {"adapter": "continuous_box2_v1", "shape": [2], "range": [-1.0, 1.0],
                    "order": ["steering", "throttle_brake"], "frequency_hz": 10},
        "reward_variants": {
            "version": "stage5_meter_reward_experiment_v1",
            "platform_reward_spec_unchanged": "RewardSpecV1",
            "clipping": "none",
            "route_progress_definition": "signed route_completion delta from Stage5TrainingEnv; initialized from reset progress",
            "formula_v1": "delta_p - 0.25/H + T_t",
            "formula_scale": "k * (delta_p - 0.25/H + T_t)",
            "formula_meter": "L_effective_m * delta_p / 10 + k * (T_t - 0.25/H)",
            "median_training_manifest_route_length_m": scale.median_route_length_m,
            "k": scale.scale_k,
            "progress_unit": "signed route metres",
            "time_penalty_budget": 0.25,
            "terminal_component_source": "RewardSpecV1 breakdown exactly once on terminal transition",
        },
        "arms": {
            "A": {"reward": "V1"},
            "B": {"reward": "SCALE"},
            "C": {"reward": "METER"},
            "only_arm_difference": "training reward signal",
        },
        "sac": {
            "algorithm": "SAC",
            "policy": "MlpPolicy",
            "net_arch": [256, 256],
            "activation": "ReLU",
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "tau": 0.005,
            "buffer_size": 100000,
            "batch_size": 256,
            "learning_starts": 1000,
            "ent_coef": "auto",
            "train_freq": 1,
            "gradient_steps": 1,
            "total_steps_max_per_arm": 30000,
            "checkpoint_steps": [0, 2000, 5000, 10000, 30000],
            "shared_fresh_initial_model": True,
            "initial_model_root_seed": 101,
        },
        "evaluation": {
            "fast_probe_cap_steps": 300,
            "probe_cases": diagnostic,
            "full_episode_checkpoints": [10000, 30000],
            "full_episode_cases": diagnostic,
            "outcome_classes": ["CLEAN_SUCCESS", "SAFETY_FAILURE", "TIMEOUT"],
            "censor_label": "PROBE_CENSORED",
            "holdouts_accessed": False,
        },
        "preflight_trace_schedule": {
            "geometry_id": "geom_easy_SCS_seed0",
            "geometry_sha256": lookup["geom_easy_SCS_seed0"]["geometry_sha256"],
            "environment_seed": 0,
            "stage1_reference": {"agent_version": "0.6.1", "source_worktree_commit": "c3714be7720c372fc5842159a4db426564928c28"},
            "stationary_reference": {"source": "S5.4 historical A 100k frozen checkpoint", "relative_path": "runs/stage5_learnability/s5_4_learnability_seed101_20261119/arm_A/checkpoints/step_100000"},
            "stage0_safety_stress_agent_seeds": [42, 101],
            "trace_use": "reward sanity only; never enters SAC replay or training",
        },
        "dependency_versions": {
            "python": "3.10.21", "metadrive": "0.4.3", "stable_baselines3": "2.9.2",
            "torch": "2.14.1+cpu", "gymnasium": "1.3.0",
        },
        "reproducibility_note": "One root seed and CPU simulator; deterministic protocol and hashes are recorded, but bitwise repeatability is not claimed.",
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    protocol["protocol_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "protocol.json"
    path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "protocol.sha256").write_text(protocol["protocol_sha256"] + "  protocol.json\n", encoding="ascii")
    print(json.dumps({"path": str(path), "protocol_sha256": protocol["protocol_sha256"],
                      "median_route_length_m": scale.median_route_length_m, "k": scale.scale_k,
                      "train_count": len(train), "diagnostic_count": len(diagnostic)}, indent=2))


if __name__ == "__main__":
    main()
