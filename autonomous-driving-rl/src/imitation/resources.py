"""Thống kê tài nguyên và ngoại suy có ghi giả định; không huấn luyện chính sách."""

from collections import Counter, defaultdict
import math
from pathlib import Path

from src.imitation.audit import EvidenceError
from src.platform.specs import compute_route_aware_horizon


def disk_inventory(directory):
    directory = Path(directory)
    categories, episodes = Counter(), Counter()
    count = 0
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(directory)
        parts = relative.parts
        size = path.stat().st_size
        if parts[0] in ("source_snapshot", "simulator_source", "expert"):
            category = "shared_source_and_expert"
        elif len(parts) > 1 and parts[0] == "episodes" and "sources" in parts:
            category = "per_episode_source_copies"
        elif path.name == "agent_decisions.jsonl.gz":
            category = "decision_traces_gzip"
        elif path.name == "transitions.jsonl.gz":
            category = "transition_traces_gzip"
        elif path.name == "map.json.gz":
            category = "maps_gzip"
        elif path.name == "raw_sample_index.jsonl.gz":
            category = "sample_index_gzip"
        elif path.name == "console.log":
            category = "console_logs"
        else:
            category = "other_metadata_and_integrity"
        categories[category] += size
        if len(parts) > 1 and parts[0] == "episodes":
            episodes[parts[1]] += size
        count += 1
    return {"files": count, "total_bytes": sum(categories.values()), "bytes_by_category": dict(categories),
            "bytes_by_episode": dict(episodes), "size_kind": "logical file bytes, not filesystem allocation"}


def summarize_samples(rows):
    if not rows:
        raise EvidenceError("Không có mẫu đo RAM")
    previous = -1.0
    phase_peak = defaultdict(int)
    for row in rows:
        if row["elapsed_s"] < previous or row["tree_rss_bytes"] < 0:
            raise EvidenceError("Mốc thời gian hoặc RAM không hợp lệ")
        if row["tree_rss_bytes"] != sum(p["rss_bytes"] for p in row["processes"]):
            raise EvidenceError("Tổng RSS cây tiến trình không khớp các tiến trình")
        identities = [(p["pid"], p["create_time"]) for p in row["processes"]]
        if len(identities) != len(set(identities)):
            raise EvidenceError("Một tiến trình bị cộng RAM hai lần trong cùng mẫu")
        phase_peak[row.get("episode_id") or "setup_or_finalization"] = max(
            phase_peak[row.get("episode_id") or "setup_or_finalization"], row["tree_rss_bytes"])
        previous = row["elapsed_s"]
    return {"samples": len(rows), "peak_tree_rss_bytes": max(r["tree_rss_bytes"] for r in rows),
            "peak_profiler_rss_bytes": max(r["profiler_rss_bytes"] for r in rows),
            "minimum_system_available_bytes": min(r["system_available_bytes"] for r in rows),
            "maximum_live_processes": max(len(r["processes"]) for r in rows),
            "peak_tree_rss_by_episode": dict(phase_peak),
            "access_denied_observations": sum(r["access_denied"] for r in rows),
            "vanished_process_observations": sum(r["vanished"] for r in rows),
            "maximum_sample_gap_s": max((b["elapsed_s"] - a["elapsed_s"] for a, b in zip(rows, rows[1:])), default=0),
            "metric_limit": "Sampled working-set sum; shared pages may be counted multiple times; profiler excluded from tree"}


def summarize_phases(events, episodes, wall_seconds):
    starts, durations = {}, {}
    for event in events:
        if event.get("phase") == "collecting":
            name = event["episode_id"]
            if name in starts:
                raise EvidenceError("Lặp mốc bắt đầu lượt")
            starts[name] = event["observed_elapsed_s"]
        elif event.get("phase") == "verified":
            name = event["episode_id"]
            if name not in starts or name in durations or event["status"] != "COMPLETE":
                raise EvidenceError("Lượt chưa đầy đủ hoặc mốc kết thúc không hợp lệ")
            elapsed = event["observed_elapsed_s"] - starts[name]
            if elapsed <= 0:
                raise EvidenceError("Thời gian mỗi lượt không dương")
            durations[name] = elapsed
    if set(durations) != {e["episode_id"] for e in episodes}:
        raise EvidenceError("Thiếu đo thời gian của lượt")
    overhead = wall_seconds - sum(durations.values())
    if overhead < -1e-6:
        raise EvidenceError("Thời gian các lượt vượt thời gian toàn tiến trình")
    return {"case_wall_seconds": durations, "batch_setup_and_finalize_seconds": max(0, overhead),
            "boundary_limit": "Includes worker startup, simulation, recording and offline verification/replay; stdout observation adds sampling jitter"}


def project_costs(episodes, timing, disk, geometries, seed_counts):
    groups, train_groups = defaultdict(list), defaultdict(list)
    for e in episodes:
        if e["status"] != "COMPLETE" or e["case"]["split"] != "TRAIN" or e["steps"] <= 0:
            raise EvidenceError("Chỉ ngoại suy từ lượt TRAIN hoàn tất")
        groups[e["case"]["tier"]].append(e)
    for geometry in geometries:
        if geometry["split"] != "TRAIN":
            raise EvidenceError("Ngoại suy chỉ dùng hình học TRAIN")
        train_groups[geometry["tier"]].append(geometry)
    if not train_groups or set(groups) != set(train_groups):
        raise EvidenceError("Thiếu mức khó trong mẫu đo hoặc phân chia TRAIN")
    tiers = {}
    for tier, rows in groups.items():
        n = len(rows)
        mean_steps = sum(e["steps"] for e in rows) / n
        mean_time = sum(timing["case_wall_seconds"][e["episode_id"]] for e in rows) / n
        mean_bytes = sum(disk["bytes_by_episode"][e["episode_id"]] for e in rows) / n
        horizon_sum = sum(compute_route_aware_horizon(float(g["route_length_m"])) for g in train_groups[tier])
        tiers[tier] = {"measured_cases": n, "measured_families": sorted({e["case"]["sequence"] for e in rows}),
                       "target_train_geometries": len(train_groups[tier]), "mean_steps_per_measured_case": mean_steps,
                       "mean_case_wall_seconds": mean_time, "mean_episode_bytes": mean_bytes,
                       "total_target_horizon_steps": horizon_sum}
    index_per_step = disk["bytes_by_category"].get("sample_index_gzip", 0) / sum(e["steps"] for e in episodes)
    shared_bytes = disk["total_bytes"] - sum(disk["bytes_by_episode"].values()) - disk["bytes_by_category"].get("sample_index_gzip", 0)
    scenarios = []
    for seed_count in seed_counts:
        if type(seed_count) is not int or seed_count <= 0:
            raise EvidenceError("Số seed phải là số nguyên dương")
        for mode in ("observed_length_scenario", "all_cases_full_horizon_scenario"):
            wall = timing["batch_setup_and_finalize_seconds"]
            size, steps = shared_bytes, 0.0
            for t in tiers.values():
                target_steps = (t["mean_steps_per_measured_case"] * t["target_train_geometries"]
                                if mode == "observed_length_scenario" else t["total_target_horizon_steps"])
                scaled_steps = target_steps * seed_count
                steps += scaled_steps
                wall += scaled_steps * t["mean_case_wall_seconds"] / t["mean_steps_per_measured_case"]
                size += scaled_steps * (t["mean_episode_bytes"] / t["mean_steps_per_measured_case"] + index_per_step)
            if not all(math.isfinite(x) and x > 0 for x in (wall, size, steps)):
                raise EvidenceError("Kết quả ngoại suy không hợp lệ")
            scenarios.append({"scenario": mode, "environment_seeds_per_geometry": seed_count,
                              "episodes": sum(len(g) for g in train_groups.values()) * seed_count,
                              "estimated_steps": steps, "estimated_wall_seconds": wall, "estimated_total_bytes": size})
    return {"tier_assumptions": tiers, "scenarios": scenarios, "parallelism": 1,
            "limitations": ["Only four sampled families of twelve; no statistical confidence bounds",
                            "Same machine, collector, compression and sequential execution assumed",
                            "Full-horizon scenario scales amortized case cost; not a guaranteed upper bound",
                            "Unmeasured traffic interactions and safety outcomes can change cost",
                            "Per-episode source/map copies retained; no deduplication assumed",
                            "No Stage 4 training time or training VRAM estimate has been measured"]}
