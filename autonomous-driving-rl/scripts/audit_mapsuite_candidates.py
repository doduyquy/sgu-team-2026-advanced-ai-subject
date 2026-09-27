"""
Audit and candidate generator for MapSuiteV1 difficulty tiers.
Gate 2 of Research Platform V1.

Authoritative source: MetaDrive 0.4.3 (commit 85e5dadc6c7436d324348f6e3d8f8e680c06b4db)
"""

import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.utils.draw_top_down_map import draw_top_down_map


EXPECTED_COMMIT = "85e5dadc6c7436d324348f6e3d8f8e680c06b4db"
EXPECTED_VERSION = "0.4.3"


def verify_metadrive_source():
    """
    Authoritative verification of the imported MetaDrive git repository.
    Verifies git commit, working tree cleanliness, and package version.
    """
    import metadrive
    import importlib.metadata

    file_path = Path(metadrive.__file__).resolve()
    print(f"[VERIFY] MetaDrive file: {file_path}")

    repo_root = file_path.parent
    while repo_root.parent != repo_root:
        if (repo_root / ".git").is_dir() or (repo_root / ".git").is_file():
            break
        repo_root = repo_root.parent

    if not (repo_root / ".git").exists():
        raise RuntimeError(f"Could not locate .git repository root starting from {file_path}")

    print(f"[VERIFY] MetaDrive repository root: {repo_root}")

    commit = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        text=True
    ).strip()
    print(f"[VERIFY] Commit: {commit}")
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(
            f"MetaDrive commit mismatch! Expected: {EXPECTED_COMMIT}, Actual: {commit}"
        )

    status = subprocess.check_output(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        text=True
    ).strip()
    if status:
        raise RuntimeError(
            f"MetaDrive repository working tree is dirty! Changes found:\n{status}"
        )
    print("[VERIFY] Git working tree is clean.")

    try:
        ver = importlib.metadata.version("metadrive-simulator")
    except Exception:
        ver = getattr(metadrive, "__version__", "unknown")

    print(f"[VERIFY] MetaDrive package version: {ver}")
    if ver != EXPECTED_VERSION:
        raise RuntimeError(
            f"MetaDrive version mismatch! Expected: {EXPECTED_VERSION}, Actual: {ver}"
        )

    return {
        "file": str(file_path),
        "repo_root": str(repo_root),
        "commit": commit,
        "version": ver,
    }


def make_jsonable(obj):
    """Recursively convert MetaDrive Config / NumPy data into JSON-serializable types."""
    if hasattr(obj, "get_serializable_dict"):
        obj = obj.get_serializable_dict()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_jsonable(v) for v in obj]
    return obj


def count_decision_points(block_ids):
    """Estimate decision points based on branching block types."""
    decisions = 0
    for bid in block_ids:
        if bid == "X":
            decisions += 3  # 4-way intersection has 3 branching turns
        elif bid == "T":
            decisions += 2  # 3-way intersection has 2 branching turns
        elif bid == "O":
            decisions += 3  # Roundabout has multiple exits
        elif bid in ("r", "R"):
            decisions += 1  # Ramp merge or divert
        elif bid in ("y", "Y"):
            decisions += 1  # Bottleneck merge or split
    return decisions


def run_mapsuite_audit(tier_configs, sweep_seeds, previews_dir):
    """
    Sweep scenario seeds across candidate sequences for all tiers.
    Picks 3 provisional canonical candidates per tier, renders top-down previews,
    and extracts exact serializable block configurations.
    """
    print(f"\n--- Sweeping {sweep_seeds} Seeds per Candidate Sequence ---")
    all_metrics = []
    canonical_candidates = {tier: [] for tier in tier_configs}

    for tier, data in tier_configs.items():
        sequences = data["candidate_sequences"]
        traffic_density = data["traffic_density"]
        print(f"\n============================================================")
        print(f"AUDITING TIER: {tier.upper()} (traffic_density={traffic_density})")
        print(f"============================================================")

        for seq_rank, seq in enumerate(sequences, 1):
            t0 = time.time()
            success_count = 0
            fail_count = 0
            seq_entries = []

            try:
                env = MetaDriveEnv(dict(
                    use_render=False,
                    num_scenarios=sweep_seeds,
                    start_seed=0,
                    map=seq,
                    traffic_density=traffic_density,
                    vehicle_config=dict(lidar=dict(num_others=0))
                ))
            except Exception as e:
                print(f"  {seq:12s}: FAILED TO INITIALIZE: {e}")
                continue

            print(f"  {seq:12s}: sweeping {sweep_seeds} seeds...", flush=True)
            for seed in range(sweep_seeds):
                try:
                    obs, info = env.reset(seed=seed)
                    m = env.current_map
                    nav = env.agent.navigation
                    route_len = float(nav.total_length)
                    b_box = m.road_network.get_bounding_box()
                    block_ids = [b.ID for b in m.blocks]

                    curve_count = block_ids.count("C")
                    x_count = block_ids.count("X")
                    t_count = block_ids.count("T")
                    roundabout_count = block_ids.count("O")
                    ramp_count = block_ids.count("r") + block_ids.count("R")
                    merge_split_count = block_ids.count("y") + block_ids.count("Y")
                    straight_count = block_ids.count("S") + block_ids.count("I")
                    decisions = count_decision_points(block_ids)

                    tm = env.engine.traffic_manager
                    total_planned_traffic = len(tm.traffic_vehicles) + sum(
                        len(bv.vehicles) for bv in tm.block_triggered_vehicles
                    )

                    # Horizon check: 1000 steps * 0.1s = 100s budget.
                    # At conservative 25 km/h (~6.94 m/s), max reach is ~694m.
                    # At 30 km/h (~8.33 m/s), max reach is ~833m.
                    horizon_risk = route_len > 750.0

                    metric_entry = {
                        "difficulty_tier": tier,
                        "sequence": seq,
                        "scenario_seed": seed,
                        "generation_success": True,
                        "block_count": len(block_ids),
                        "block_ids": "".join(block_ids),
                        "route_total_length_m": round(route_len, 2),
                        "bbox_width_m": round(float(b_box[1] - b_box[0]), 2),
                        "bbox_height_m": round(float(b_box[3] - b_box[2]), 2),
                        "lane_width": 3.5,
                        "base_lane_num": 2,
                        "straight_count": straight_count,
                        "curve_count": curve_count,
                        "intersection_count": x_count,
                        "t_intersection_count": t_count,
                        "roundabout_count": roundabout_count,
                        "ramp_count": ramp_count,
                        "merge_split_count": merge_split_count,
                        "decision_point_count": decisions,
                        "traffic_density": traffic_density,
                        "traffic_vehicle_count": total_planned_traffic,
                        "horizon_1000_risk": horizon_risk,
                    }
                    all_metrics.append(metric_entry)
                    seq_entries.append((seed, metric_entry))
                    success_count += 1
                except Exception as e:
                    fail_count += 1
                    all_metrics.append({
                        "difficulty_tier": tier,
                        "sequence": seq,
                        "scenario_seed": seed,
                        "generation_success": False,
                        "block_count": 0,
                        "block_ids": "",
                        "route_total_length_m": 0.0,
                        "bbox_width_m": 0.0,
                        "bbox_height_m": 0.0,
                        "lane_width": 3.5,
                        "base_lane_num": 2,
                        "straight_count": 0,
                        "curve_count": 0,
                        "intersection_count": 0,
                        "t_intersection_count": 0,
                        "roundabout_count": 0,
                        "ramp_count": 0,
                        "merge_split_count": 0,
                        "decision_point_count": 0,
                        "traffic_density": traffic_density,
                        "traffic_vehicle_count": 0,
                        "horizon_1000_risk": False,
                    })

            routes = [e[1]["route_total_length_m"] for e in seq_entries]
            min_r = min(routes) if routes else 0
            max_r = max(routes) if routes else 0
            mean_r = sum(routes) / len(routes) if routes else 0
            print(f"  {seq:12s}: success={success_count:2d}/{sweep_seeds} ({success_count/sweep_seeds*100:3.0f}%), fail={fail_count}, routes=[{min_r:.1f}m - {max_r:.1f}m], mean={mean_r:.1f}m, time={time.time()-t0:.1f}s")

            # Select the median candidate from this sequence as a provisional canonical candidate
            if seq_entries:
                sorted_entries = sorted(seq_entries, key=lambda x: x[1]["route_total_length_m"])
                median_seed, median_cand = sorted_entries[len(sorted_entries) // 2]

                # Reset to median seed while env is still active to extract map metadata and render
                env.reset(seed=median_seed)
                m = env.current_map
                nav = env.agent.navigation

                # Render top-down preview
                img = draw_top_down_map(m, resolution=(512, 512))
                preview_filename = f"{tier.lower()}_rank{seq_rank}_{seq}_seed{median_seed}.png"
                preview_path = previews_dir / preview_filename
                cv2.imwrite(str(preview_path), img)

                canonical_info = {
                    "tier": tier,
                    "provisional_rank": seq_rank,
                    "sequence": seq,
                    "scenario_seed": median_seed,
                    "route_length_m": median_cand["route_total_length_m"],
                    "block_count": median_cand["block_count"],
                    "block_ids": median_cand["block_ids"],
                    "decision_point_count": median_cand["decision_point_count"],
                    "traffic_density": median_cand["traffic_density"],
                    "traffic_vehicle_count": median_cand["traffic_vehicle_count"],
                    "horizon_1000_risk": median_cand["horizon_1000_risk"],
                    "preview_image": f"results/audits/mapsuite/previews/{preview_filename}",
                    "checkpoints": make_jsonable(nav.checkpoints),
                    "exact_block_sequence": make_jsonable(m.get_meta_data()["block_sequence"])
                }
                canonical_candidates[tier].append(canonical_info)
                print(f"    -> Canonical Candidate {seq_rank}: {seq} seed={median_seed} (len={median_cand['route_total_length_m']:.1f}m, decisions={median_cand['decision_point_count']})")

            env.close()

    return all_metrics, canonical_candidates


def write_manifest_and_schemas(canonical_candidates, all_metrics, configs_dir, results_dir):
    """
    Write machine-readable manifest (mapsuite_v1_candidates.json),
    full metrics CSV (candidate_metrics.csv), and canonical summary (canonical_candidates.json).
    """
    # 1. candidate_metrics.csv
    csv_path = results_dir / "candidate_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(all_metrics[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"[SAVED] Candidate sweep metrics saved to: {csv_path}")

    # 2. canonical_candidates.json
    canon_json_path = results_dir / "canonical_candidates.json"
    with open(canon_json_path, "w", encoding="utf-8") as f:
        json.dump(canonical_candidates, f, indent=2)
    print(f"[SAVED] Canonical candidates JSON saved to: {canon_json_path}")

    # 3. mapsuite_v1_candidates.json in configs/maps/
    manifest = {
        "metadata": {
            "suite_name": "MapSuiteV1",
            "status": "NON-FROZEN RESEARCH CANDIDATE POOL",
            "gate": "Gate 2 (Research Platform V1)",
            "pinned_metadrive_commit": EXPECTED_COMMIT,
            "pinned_metadrive_version": EXPECTED_VERSION,
            "lane_width": 3.5,
            "base_lane_num": 2,
            "decision_repeat": 5,
            "control_frequency_hz": 10
        },
        "tier_definitions": {
            "Easy": {
                "intent": "Basic steering and lane following on clean roads. Zero multi-exit intersections. Zero traffic.",
                "traffic_density": 0.0,
                "recommended_canonical_sequence": canonical_candidates["Easy"][0]["sequence"],
                "recommended_canonical_seed": canonical_candidates["Easy"][0]["scenario_seed"],
                "provisional_canonical_pool": canonical_candidates["Easy"]
            },
            "Medium": {
                "intent": "Moderate distance with single decision block (T or X intersection) and low traffic interaction.",
                "traffic_density": 0.08,
                "recommended_canonical_sequence": canonical_candidates["Medium"][0]["sequence"],
                "recommended_canonical_seed": canonical_candidates["Medium"][0]["scenario_seed"],
                "provisional_canonical_pool": canonical_candidates["Medium"]
            },
            "Hard": {
                "intent": "Longer distance featuring multiple decision blocks (Intersection, Roundabout, Ramp) with moderate traffic.",
                "traffic_density": 0.15,
                "recommended_canonical_sequence": canonical_candidates["Hard"][0]["sequence"],
                "recommended_canonical_seed": canonical_candidates["Hard"][0]["scenario_seed"],
                "provisional_canonical_pool": canonical_candidates["Hard"]
            },
            "Extreme": {
                "intent": "Heterogeneous multi-block composition with multiple decision nodes, complex geometries, and dense traffic.",
                "traffic_density": 0.25,
                "recommended_canonical_sequence": canonical_candidates["Extreme"][0]["sequence"],
                "recommended_canonical_seed": canonical_candidates["Extreme"][0]["scenario_seed"],
                "provisional_canonical_pool": canonical_candidates["Extreme"]
            }
        }
    }

    manifest_path = configs_dir / "mapsuite_v1_candidates.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[SAVED] MapSuiteV1 candidate manifest saved to: {manifest_path}")


def main():
    print("============================================================")
    print("STARTING GATE 2 MAPSUITE V1 CANDIDATE GENERATION & AUDIT")
    print("============================================================")

    project_root = Path(__file__).resolve().parent.parent
    configs_dir = project_root / "configs" / "maps"
    results_dir = project_root / "results" / "audits" / "mapsuite"
    previews_dir = results_dir / "previews"

    configs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PATH] Project Root: {project_root}")
    print(f"[PATH] Configs Dir:  {configs_dir}")
    print(f"[PATH] Results Dir:  {results_dir}")
    print(f"[PATH] Previews Dir: {previews_dir}")

    # 1. Authoritative MetaDrive source verification
    verify_metadrive_source()

    # 2. Tier configuration targets
    tier_configs = {
        "Easy": {
            "candidate_sequences": ["SCS", "SCSS", "SCCS"],
            "traffic_density": 0.0,
        },
        "Medium": {
            "candidate_sequences": ["SCXCS", "SCTCS", "SCXCCS"],
            "traffic_density": 0.08,
        },
        "Hard": {
            "candidate_sequences": ["SCXOCS", "SCTXrCS", "XTOCS"],
            "traffic_density": 0.15,
        },
        "Extreme": {
            "candidate_sequences": ["CrXROSTR", "SCXOCrTYCS", "SCTXORyCCS"],
            "traffic_density": 0.25,
        }
    }

    # 3. Sweep seeds, collect metrics, select canonical candidates, and render previews
    all_metrics, canonical_candidates = run_mapsuite_audit(tier_configs, sweep_seeds=20, previews_dir=previews_dir)

    # 4. Write outputs
    write_manifest_and_schemas(canonical_candidates, all_metrics, configs_dir, results_dir)

    print("\n============================================================")
    print("GATE 2 MAPSUITE V1 AUDIT COMPLETED SUCCESSFULLY!")
    print("============================================================")


if __name__ == "__main__":
    main()
