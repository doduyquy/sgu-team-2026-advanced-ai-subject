"""Collect public local graphs on TRAIN and fit a supervised A* heuristic.

Internal model selection groups by TRAIN geometry. It never consumes the frozen
VALIDATION or TEST geometries and is not an official driving benchmark result.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from src.agents.astar_agent import AStarAgent
from src.agents.heuristic_model import HeuristicMLP, file_sha256
from src.agents.planning_graph import (
    astar_search, build_planning_graph, exact_cost_to_goal, load_astar_config, planner_fingerprint,
)
from src.launcher.cases import (
    LOCKED_GEOMETRY_SPLIT_MANIFEST_SHA256, load_locked_geometry_block_sequence,
    load_manifest_csv, resolve_sandbox_case,
)
from src.launcher.executor import build_metadrive_case_config
from src.launcher.models import LaunchRequestV1, LauncherMode
from src.platform import (
    AgentPublicEpisodeContext, PINNED_METADRIVE_COMMIT, PINNED_METADRIVE_VERSION,
    build_agent_input, canonical_json_sha256, capture_environment_provenance, capture_git_provenance,
    classify_episode_outcome, get_action_adapter,
)
from src.platform.protocol import derive_seed
from src.training.runtime import local_training_runtime


def select_train_geometries(project_root, limit):
    source = Path(project_root) / "results/audits/evaluation_protocol/geometry_split_manifest.csv"
    rows = [r for r in load_manifest_csv(source, LOCKED_GEOMETRY_SPLIT_MANIFEST_SHA256) if r["split"] == "TRAIN"]
    # Interleave tiers and families before sampling seeds, preserving all difficulty levels in smoke runs.
    tiers = ("Easy", "Medium", "Hard", "Extreme")
    families = {tier: sorted({r["sequence"] for r in rows if r["tier"] == tier}) for tier in tiers}
    groups = {(tier, sequence): sorted([r for r in rows if r["tier"] == tier and r["sequence"] == sequence],
                                       key=lambda r: int(r["geometry_generation_seed"]))
              for tier in tiers for sequence in families[tier]}
    ordered = []
    for seed_rank in range(max(map(len, groups.values()))):
        for family_rank in range(max(map(len, families.values()))):
            for tier in tiers:
                if family_rank < len(families[tier]):
                    group = groups[tier, families[tier][family_rank]]
                    if seed_rank < len(group):
                        ordered.append(group[seed_rank])
    return ordered if limit is None else ordered[:limit]


def grouped_training_split(groups, seed):
    unique = np.unique(groups)
    if len(unique) < 3:
        raise ValueError("Need at least three TRAIN geometries for a disjoint internal holdout")
    order = np.random.default_rng(seed).permutation(unique)
    holdout_groups = set(order[:max(1, int(round(len(unique) * .25)))].tolist())
    holdout = np.array([group in holdout_groups for group in groups])
    return ~holdout, holdout, sorted(holdout_groups)


def _collect(project_root, settings, seed, output, progress):
    from metadrive import MetaDriveEnv
    geometries = select_train_geometries(project_root, settings["geometry_limit"])
    feature_rows, target_rows, group_rows = [], [], []
    geometry_stats, diagnostic_graphs = [], {}
    adapter = get_action_adapter("discrete9_lowbranch_v1")
    graph_config = load_astar_config()["planner"]
    for index, row in enumerate(geometries):
        geometry_id = row["geometry_id"]
        env_seed = derive_seed(seed, "environment", index, geometry_id)
        request = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="stage2_astar", tier=row["tier"],
                                  sequence=row["sequence"], geometry_generation_seed=int(row["geometry_generation_seed"]),
                                  environment_seed=env_seed)
        case = resolve_sandbox_case(request, project_root)
        if case.split != "TRAIN":
            raise ValueError("Training collector may only use TRAIN geometries")
        progress(dict(phase="COLLECT", training_seed=seed, geometry_index=index + 1,
                      geometry_count=len(geometries), geometry_id=geometry_id))
        blocks, _, _ = load_locked_geometry_block_sequence(case.to_dict(), project_root)
        env, agent = None, AStarAgent()
        samples, steps, outcome_name = 0, 0, "COLLECTION_BUDGET"
        flags = {}
        try:
            env = MetaDriveEnv(build_metadrive_case_config(case, blocks, render_mode="OFF"))
            obs, info = env.reset(seed=case.environment_seed)
            agent.reset(AgentPublicEpisodeContext(mode="TRAINING", horizon_steps=case.horizon_steps,
                                                  action_adapter_id="discrete9_lowbranch_v1"))
            for step in range(min(settings["steps_per_geometry"], case.horizon_steps)):
                manager = env.engine.traffic_manager
                agent_input = build_agent_input(
                    raw_obs=obs, ego_vehicle=env.agent,
                    traffic_vehicles=list(getattr(manager, "traffic_vehicles", [])),
                    navigation=env.agent.navigation, road_network=env.current_map.road_network, step_index=step,
                )
                if step % settings["sample_every"] == 0:
                    graph = build_planning_graph(agent_input, graph_config)
                    if graph is not None:
                        exact = exact_cost_to_goal(graph)
                        if np.isfinite(exact[0]):
                            mask = np.isfinite(exact) & (exact > 1e-8)
                            targets = np.maximum(0., exact[mask] - graph.euclidean_heuristic()[mask]) / graph.scale_m
                            feature_rows.extend(graph.features[mask])
                            target_rows.extend(targets)
                            group_rows.extend([geometry_id] * len(targets))
                            samples += len(targets)
                            if len(diagnostic_graphs.setdefault(geometry_id, [])) < 2:
                                diagnostic_graphs[geometry_id].append(graph)
                decision = agent.act(agent_input)
                obs, _, terminated, truncated, info = env.step(adapter.to_canonical(decision.action_payload).to_numpy())
                steps = step + 1
                for key in ("arrive_dest", "out_of_road", "crash_vehicle", "crash_object", "crash_building", "crash_human", "crash_sidewalk", "max_step"):
                    flags[key] = flags.get(key, False) or bool(info.get(key, False))
                outcome = classify_episode_outcome(raw_flags=flags, terminated=terminated, truncated=truncated)
                if outcome.is_done:
                    outcome_name = outcome.primary_reason.value
                    break
        finally:
            agent.close()
            if env is not None:
                env.close()
        geometry_stats.append(dict(geometry_id=geometry_id, split="TRAIN", case_id=case.case_id,
                                   geometry_seed=case.geometry_generation_seed, environment_seed=case.environment_seed,
                                   collected_steps=steps, node_samples=samples, collection_end=outcome_name))
    features = np.asarray(feature_rows, dtype=np.float64)
    targets = np.asarray(target_rows, dtype=np.float64)
    groups = np.asarray(group_rows, dtype=str)
    if not len(features):
        raise ValueError("No reachable graph samples collected from TRAIN")
    path = output / "dataset.npz"
    np.savez_compressed(path, features=features, residual_targets=targets, geometry_ids=groups)
    return features, targets, groups, geometry_stats, diagnostic_graphs, file_sha256(path)


def _graph_diagnostics(model, graph_groups, holdout_groups):
    records = []
    for group in holdout_groups:
        for graph in graph_groups.get(group, []):
            exact = exact_cost_to_goal(graph)
            learned_h = model.heuristic(graph)
            baseline = astar_search(graph)
            learned = astar_search(graph, learned_h)
            mask = np.isfinite(exact)
            records.append(dict(
                geometry_id=group, baseline_expansions=baseline.expansions, learned_expansions=learned.expansions,
                exact_local_cost=float(exact[0]), baseline_local_cost=baseline.cost if baseline.found else None,
                learned_local_cost=learned.cost if learned.found else None,
                learned_overestimate_fraction=float(np.mean(learned_h[mask] > exact[mask] + 1e-6)),
            ))
    return records


def train_astar_heuristic(project_root, preset="smoke", progress=None, *, publish_latest=True):
    project_root = Path(project_root).resolve()
    progress = progress or (lambda message: None)
    config = load_astar_config()
    if preset not in config["presets"]:
        raise ValueError("Unknown A* training preset")
    settings = config["presets"][preset]
    config_sha = canonical_json_sha256(config)
    run_id = f"train_astar_{uuid4().hex[:12]}"
    output = project_root / "artifacts/astar_learning" / run_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(format_version="astar_training_v1", run_id=run_id, status="RUNNING", preset=preset,
                    started_at_utc=datetime.now(timezone.utc).isoformat(), training_config=config,
                    training_config_sha256=config_sha, planner_fingerprint=planner_fingerprint(config["planner"]),
                    selection_scope="INTERNAL_HOLDOUT_OF_TRAIN_GEOMETRIES", models=[])
    manifest["source_sha256"] = {
        relative: file_sha256(project_root / relative) for relative in (
            "src/agents/planning_graph.py", "src/agents/astar_agent.py", "src/agents/heuristic_model.py",
            "src/agents/path_following.py", "src/training/astar_heuristic.py",
        )
    }
    manifest_path = output / "training_manifest.json"

    def save_manifest():
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(manifest_path)

    try:
        with local_training_runtime(project_root):
            environment = capture_environment_provenance(project_root)
            if (environment.get("metadrive_version"), environment.get("metadrive_commit")) != (
                PINNED_METADRIVE_VERSION, PINNED_METADRIVE_COMMIT,
            ):
                raise ValueError("Training requires the pinned MetaDrive version and commit")
            manifest.update(environment=environment, git=capture_git_provenance(project_root.parent))
            save_manifest()
            for seed in settings["training_seeds"]:
                seed_output = output / f"seed_{seed}"
                seed_output.mkdir()
                x, y, groups, geometries, graph_groups, data_sha = _collect(project_root, settings, seed, seed_output, progress)
                train_mask, holdout_mask, holdout_groups = grouped_training_split(groups, seed)
                model = HeuristicMLP(config["model"]["hidden_units"], seed)
                log_path = seed_output / "training_metrics.jsonl"
                with log_path.open("w", encoding="utf-8") as log:
                    def on_epoch(record):
                        log.write(json.dumps(record) + "\n")
                        log.flush()
                        progress(dict(phase="FIT", training_seed=seed, epochs=settings["epochs"], **record))
                    model.fit(x[train_mask], y[train_mask], x[holdout_mask], y[holdout_mask],
                              epochs=settings["epochs"], batch_size=config["model"]["batch_size"],
                              learning_rate=config["model"]["learning_rate"], seed=seed, on_epoch=on_epoch)
                checkpoint = seed_output / "heuristic.npz"
                metadata = dict(training_run_seed=seed, training_config_sha256=config_sha,
                                planner_fingerprint=manifest["planner_fingerprint"], data_sha256=data_sha,
                                train_samples=int(train_mask.sum()), internal_holdout_samples=int(holdout_mask.sum()),
                                train_geometry_ids=sorted(set(groups[train_mask].tolist())),
                                internal_holdout_geometry_ids=holdout_groups,
                                training_scope="TRAIN_ONLY", training_run_id=run_id)
                digest = model.save(checkpoint, metadata)
                model_record = dict(model.metadata)
                model_record.update(
                    checkpoint_path=str(checkpoint.relative_to(project_root)), checkpoint_sha256=digest,
                    euclidean_holdout_residual_mse=float(np.mean(y[holdout_mask] ** 2)),
                    collected_geometries=geometries,
                    internal_holdout_search_diagnostics=_graph_diagnostics(model, graph_groups, holdout_groups),
                )
                manifest["models"].append(model_record)
                save_manifest()
                progress(dict(phase="CHECKPOINT", training_seed=seed, checkpoint_sha256=digest,
                              train_samples=int(train_mask.sum()), holdout_samples=int(holdout_mask.sum())))
        manifest.update(status="COMPLETE", finished_at_utc=datetime.now(timezone.utc).isoformat())
        save_manifest()
        # A convenience pointer; the full manifest and each model digest are checked before loading.
        if publish_latest:
            pointer = output.parent / "latest.json"
            temporary = pointer.with_suffix(".tmp")
            temporary.write_text(json.dumps(dict(manifest=str(manifest_path.relative_to(project_root)),
                                                 manifest_sha256=file_sha256(manifest_path))), encoding="utf-8")
            temporary.replace(pointer)
        return dict(manifest_path=str(manifest_path), manifest_sha256=file_sha256(manifest_path), **manifest)
    except BaseException as exc:
        manifest.update(status="INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED", error=str(exc))
        save_manifest()
        raise
