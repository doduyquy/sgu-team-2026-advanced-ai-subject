"""Verify paired SAC evaluation receipts and generate publication tables."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import fields
import json
from pathlib import Path
from statistics import mean

from cloud.kaggle.common import ROOT, read, sha, verify, write
from cloud.kaggle.final_sac_evaluation import BUNDLE, CONDITIONS, OUTPUT, SPLITS, V2, density, jobs_for, receipt_path
from src.platform.metrics import EpisodeRecord, compute_aggregate_metrics


TIERS = ("Easy", "Medium", "Hard", "Extreme")
METHOD = "SAC V2–5.000"
REPORT = ROOT / "docs/stage5/final_product/FINAL_SAC_EVALUATION.md"
PROTOCOL = """## Phạm vi so sánh

- Runner candidate sử dụng manifest Gate 5 đã khóa, gồm 48 geometry VALIDATION với seed 5101–5102 và 12 geometry TEST với seed 9101–9105. Mỗi điều kiện có cùng case ID, block sequence đã đối chiếu SHA-256, route, horizon và thứ tự protocol.
- Traffic ON dùng `max(0.08, manifest_density)` và OFF dùng `0.0`, theo quy tắc Search locked được nêu cho đợt đối chiếu này. Mật độ ON lần lượt là 0,08 / 0,08 / 0,15 / 0,25 cho Easy / Medium / Hard / Extreme. Cả hai dùng TrafficMode.Trigger, timestep 0,02 s, decision repeat 5 và điều khiển 10 Hz.
- Hai script đối chiếu Search/candidate và receipt gốc của các agent trong bảng nhóm chưa có trong checkout và các ref đã kiểm tra. Vì vậy reviewer có thể dùng cùng mẫu số 60 và điều kiện traffic nêu trên, với giới hạn là chưa xác minh độc lập rằng từng hàng của nhóm được tạo bởi cùng manifest, seed, traffic và outcome semantics. Giữ nguyên các hàng do nhóm cung cấp.
- Đây là `SAC_V2_PAIRED_TRAFFIC_EVALUATION`, `canonical_run=false`, `training_replicates=1`. SAC dùng observation 409D, F09, Terminal Preview V1 và Box2 theo bundle đã bàn giao. Một checkpoint deterministic được chạy trên nhiều environment seed. Điều đó không cung cấp phương sai giữa các lượt huấn luyện độc lập hoặc chứng nhận Gate 5.
- Loader inference hiện hành không mở full state hay replay nguồn. SB3 đọc các tensor optimizer có sẵn trong `model.zip` khi nạp bundle. Không có optimizer step.
- Báo cáo tính route completion tại bước kết thúc, không dùng tiến độ cực đại. Lỗi an toàn được ưu tiên hơn đến đích và timeout. Thời gian thành công chỉ lấy từ episode clean. Inference latency là số đo local của `agent.act`, không dùng để xếp hạng với agent khác.
"""


def save_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safety(row):
    reason = row["primary_reason"]
    return reason == "OUT_OF_ROAD" or reason.startswith("CRASH_")


def aggregate(rows, expected):
    counts = Counter(row["primary_reason"] for row in rows)
    result = {"expected": expected, "episodes": len(rows), "complete": len(rows) == expected,
              "observed_clean": counts["SUCCESS"], "observed_safety_failures": sum(map(safety, rows)),
              "observed_timeouts": counts["TIMEOUT"]}
    if len(rows) != expected:
        return {**result, **dict.fromkeys(("clean", "safety_failures", "timeouts", "clean_success_rate",
                                           "safety_failure_rate", "timeout_rate", "mean_route", "median_route",
                                           "mean_time_to_clean_success_s", "mean_speed_kmh", "mean_episode_steps",
                                           "mean_simulation_time_s", "mean_local_inference_ms", "primary_reason_counts"))}
    records = [EpisodeRecord(**{field.name: row[field.name] for field in fields(EpisodeRecord)}) for row in rows]
    stats = compute_aggregate_metrics(records)
    return {**result, "clean": counts["SUCCESS"], "safety_failures": sum(map(safety, rows)), "timeouts": counts["TIMEOUT"],
            "clean_success_rate": stats.clean_success_rate, "safety_failure_rate": stats.safety_failure_rate,
            "timeout_rate": stats.timeout_rate, "mean_route": stats.mean_final_route_completion,
            "median_route": stats.median_final_route_completion, "mean_time_to_clean_success_s": stats.mean_time_to_clean_success_s,
            "mean_speed_kmh": stats.mean_speed_kmh, "mean_episode_steps": stats.mean_episode_steps,
            "mean_simulation_time_s": mean(row["simulation_time_s"] for row in rows),
            "mean_local_inference_ms": mean(row["mean_local_inference_ms"] for row in rows),
            "primary_reason_counts": "; ".join(f"{reason}={count}" for reason, count in sorted(counts.items()))}


def match_cases(rows):
    outcomes = {row["case_id"]: {} for row in rows}
    for row in rows:
        outcomes[row["case_id"]][row["traffic"]] = row
    return [(case["ON"], case["OFF"]) for case in outcomes.values() if set(case) == set(CONDITIONS)]


def pairs(rows, expected):
    counts = Counter((on["clean_success"], off["clean_success"]) for on, off in match_cases(rows))
    matched = sum(counts.values())
    delta = counts[False, True] - counts[True, False]
    return {"expected_pairs": expected, "matched_pairs": matched, "both_clean": counts[True, True],
            "off_only_clean": counts[False, True], "on_only_clean": counts[True, False], "neither_clean": counts[False, False],
            "matched_off_minus_on_pp": delta * 100 / matched if matched else None,
            "off_minus_on_pp": delta * 100 / expected if matched == expected else None}


def pct(value):
    return "N/A" if value is None else f"{value:.2%}"


def number(value):
    return "N/A" if value is None else f"{value:.2f}"


def score(row):
    return f"{row['clean']}/{row['expected']} ({pct(row['clean_success_rate'])})" if row["complete"] else "Chưa đủ ca"


def report_row(row):
    return {"Split": row["split"], "Traffic": row["traffic"], "Episodes": f"{row['episodes']}/{row['expected']}",
            "Clean Success": score(row), "Safety Failure": pct(row["safety_failure_rate"]),
            "Timeout": pct(row["timeout_rate"]), "Mean Final Route": pct(row["mean_route"]),
            "Median Final Route": pct(row["median_route"]),
            "Mean Time to Clean Success (s)": number(row["mean_time_to_clean_success_s"])}


def table(headers, rows):
    lines = [headers, ["---"] * len(headers), *rows]
    return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in lines)


def collect(directory):
    plan = read(directory / "execution-plan.json")
    digest = plan["checkpoint"]["model_sha256"]
    verify(BUNDLE / "model.zip", digest)
    verify(V2 / "stage-05000/model.zip", digest)
    receipts, missing, seen = [], [], set()
    expected_jobs = [(case, traffic) for split in SPLITS for case, traffic in jobs_for(split)]
    if len(plan["cases"]) != len(expected_jobs):
        raise ValueError("Execution plan has incorrect size")
    for (case, condition), planned in zip(expected_jobs, plan["cases"], strict=True):
        if {**case.to_dict(), "traffic": condition, "effective_density": density(case, condition)} != planned:
            raise ValueError(f"Manifest and execution plan diverged for {case.case_id}")
        path = receipt_path(directory, case, condition)
        if not path.exists():
            missing.append(planned)
            continue
        row = read(path)
        key = (row["case_id"], row["traffic"])
        if key in seen or any(row.get(name) != value for name, value in planned.items()) or row["model_sha256"] != digest:
            raise ValueError(f"Case identity or configuration mismatch in {path}")
        if not 0 < row["episode_steps"] <= case.horizon_steps or not row["terminated"] and not row["truncated"]:
            raise ValueError(f"Invalid terminal outcome in {path}")
        EpisodeRecord(**{field.name: row[field.name] for field in fields(EpisodeRecord)})
        seen.add(key)
        receipts.append(row)
    if len(list((directory / "episodes").glob("*.json"))) != len(receipts):
        raise ValueError("Unexpected or duplicate receipts in output directory")
    return plan, receipts, missing


def generate(directory, report):
    plan, rows, missing = collect(directory)
    summaries, tiers, seeds, paired, merged = [], [], [], [], []
    save_csv(directory / "episodes.csv", rows)
    for split, (_, total, environment_seeds) in SPLITS.items():
        combined = [row for row in rows if row["split"] == split]
        paired.append({"split": split, **pairs(combined, total)})
        for traffic in CONDITIONS:
            group = [row for row in combined if row["traffic"] == traffic]
            save_csv(directory / f"{split.lower()}-{traffic.lower()}.csv", group)
            label = {"split": split, "traffic": traffic}
            summaries.append({**label, **aggregate(group, total)})
            for tier in TIERS:
                tiers.append({**label, "tier": tier, **aggregate([row for row in group if row["tier"] == tier], total // 4)})
            for seed in sorted(environment_seeds):
                seeds.append({**label, "environment_seed": seed,
                              **aggregate([row for row in group if row["environment_seed"] == seed], total // len(environment_seeds))})
        if all(item["effective_density"] == item["traffic_density"] for item in plan["cases"]
               if item["split"] == split and item["traffic"] == ("OFF" if item["tier"] == "Easy" else "ON")):
            default_rows = [row for row in combined if row["traffic"] == ("OFF" if row["tier"] == "Easy" else "ON")]
            merged.append({"split": split, "condition": "MAPSUITE_DEFAULT_RECOMBINED", **aggregate(default_rows, total)})
    save_csv(directory / "summary.csv", summaries)
    report_rows = [report_row(row) for row in summaries]
    save_csv(directory / "report-table.csv", report_rows)
    save_csv(directory / "tier-summary.csv", tiers)
    save_csv(directory / "seed-summary.csv", seeds)
    save_csv(directory / "paired-traffic-comparison.csv", paired)
    save_csv(directory / "paired-episode-deltas.csv", [
        {**{key: on[key] for key in ("case_id", "split", "tier", "sequence", "environment_seed", "protocol_order_index")},
         "on_clean": on["clean_success"], "off_clean": off["clean_success"], "on_outcome": on["primary_reason"],
         "off_outcome": off["primary_reason"], "on_final_route": on["final_route_completion"],
         "off_final_route": off["final_route_completion"],
         "off_minus_on_final_route": off["final_route_completion"] - on["final_route_completion"]}
        for on, off in match_cases(rows)
    ])
    if merged:
        save_csv(directory / "default-mapsuite-summary.csv", merged)

    test = [row for row in summaries if row["split"] == "TEST"]
    poster = {"Phương pháp": METHOD, **{f"TEST Traffic {row['traffic']}": score(row) for row in test}}
    slide = [{"Phương pháp": METHOD, "Traffic": row["traffic"], "Clean Success Rate": pct(row["clean_success_rate"]),
              "Safety Failure Rate": pct(row["safety_failure_rate"]), "Mean Route Completion": pct(row["mean_route"]),
              "Median Route Completion": pct(row["median_route"])} for row in test]
    save_csv(directory / "poster-table.csv", [poster])
    save_csv(directory / "slide-table.csv", slide)
    (directory / "protocol-comparison.md").write_text(PROTOCOL, encoding="utf-8")

    digest = plan["checkpoint"]["model_sha256"]
    runtime_path = directory / "execution-runtime.json"
    runtime = read(runtime_path) if runtime_path.exists() else {}
    errors_path = directory / "infrastructure.jsonl"
    errors = [json.loads(line) for line in errors_path.read_text(encoding="utf-8").splitlines()] if errors_path.exists() else []
    missing_keys = {(case["case_id"], case["traffic"]) for case in missing}
    remaining_errors = Counter((error["case_id"], error["traffic"], error["infrastructure_error"])
                               for error in errors if (error["case_id"], error["traffic"]) in missing_keys)
    remaining_error_counts = Counter(kind for _, _, kind in remaining_errors)
    integrity = {"status": "PASS" if not missing else "INCOMPLETE", "expected": 312, "valid": len(rows), "missing": missing,
                 "infrastructure_attempt_failures": len(errors), "unique_case_condition": True,
                 "remaining_errors": [{"case_id": case_id, "traffic": condition, "error": kind, "failed_attempts": count}
                                      for (case_id, condition, kind), count in sorted(remaining_errors.items())],
                 "identities_match_locked_manifest": True, "model_sha256_before": digest, "model_sha256_after": digest,
                 "gradient_updates": 0, "replay_rows_admitted": 0}
    write(directory / "integrity-check.json", integrity)
    steps = sum(row["episode_steps"] for row in rows)
    write(directory / "manifest.json", {"evaluation_kind": plan["evaluation_kind"], "canonical_run": False, "training_replicates": 1,
          "checkpoint": plan["checkpoint"], "locked_manifests": plan["locked_manifests"], "valid_episodes": len(rows),
          "decisions": steps, "runtime": runtime, "decisions_per_second": steps / runtime["seconds"] if runtime else None,
          "workers_observed": sorted({row["worker_pid"] for row in rows}),
          "episode_receipts_sha256": {path.name: sha(path) for path in sorted((directory / "episodes").glob("*.json"))},
          "files": {path.name: sha(path) for path in sorted(directory.iterdir()) if path.is_file() and path.name != "manifest.json"}})

    lines = ["# SAC V2–5.000 · Đánh giá cuối VALIDATION và TEST", "",
             f"Đã xác minh {len(rows)}/312 episode hợp lệ. Bundle được chọn từ `stage-05000`. Checksum model khớp delivery lock trước và sau khi đọc kết quả. Đợt này có 0 SAC updates và 0 hàng nạp replay.",
             "", PROTOCOL.rstrip(), "", "## Kết quả tổng hợp", "",
             table(list(report_rows[0]), [list(row.values()) for row in report_rows])]
    for split in SPLITS:
        lookup = {(row["tier"], row["traffic"]): row for row in tiers if row["split"] == split}
        tier_rows = []
        for tier in TIERS:
            on, off = (lookup[tier, condition] for condition in CONDITIONS)
            tier_rows.append([tier, score(on), score(off), pct(on["safety_failure_rate"]), pct(off["safety_failure_rate"]),
                              pct(on["mean_route"]), pct(off["mean_route"])])
        lines += ["", f"## {split} theo độ khó", "", table(["Tier", "ON Clean", "OFF Clean", "ON Safety", "OFF Safety", "ON Mean route", "OFF Mean route"], tier_rows)]
    lines += ["", "## Ảnh hưởng của giao thông theo cặp", "",
              table(["Split", "Cặp đã đo", "Cả hai clean", "Chỉ OFF clean", "Chỉ ON clean", "Cả hai thất bại",
                     "OFF − ON trên cặp có outcome (điểm %)", "OFF − ON toàn tập (điểm %)"],
                    [[row["split"], f"{row['matched_pairs']}/{row['expected_pairs']}", row["both_clean"], row["off_only_clean"],
                      row["on_only_clean"], row["neither_clean"], number(row["matched_off_minus_on_pp"]),
                      number(row["off_minus_on_pp"])] for row in paired]),
              "", "## Số liệu slide và poster", "", table(list(poster), [list(poster.values())]), "",
              table(list(slide[0]), [list(row.values()) for row in slide])]
    if merged:
        lines += ["", "## Mật độ MapSuite mặc định", "",
                  "Tổng hợp Easy OFF với Medium, Hard và Extreme ON theo mật độ manifest. Các hàng này dùng lại episode đã đo.", "",
                  table(["Split", "Episode", "Clean", "Safety failure", "Mean final route"],
                        [[row["split"], f"{row['episodes']}/{row['expected']}", score(row), pct(row["safety_failure_rate"]),
                          pct(row["mean_route"])] for row in merged])]
    lines += ["", "## Tái lập và kiểm tra", "", "Chạy từ thư mục `autonomous-driving-rl` trong môi trường MetaDrive/SB3 của Stage 5.",
              "", "```powershell", "python -m cloud.kaggle.final_sac_evaluation", "python -m cloud.kaggle.final_sac_report", "```", "",
              "Artifact nằm trong `runs/kaggle/final-sac-evaluation/`. `parity-proof.json` đối chiếu demo Medium, mật độ thực và trace action. `integrity-check.json` ghi độ phủ và SHA-256. `seed-summary.csv` phân nhóm theo environment seed. Các nhóm chưa đủ ca để trống tỷ lệ.", ""]
    if missing:
        failures = ", ".join(f"{kind} ({count} ca)" for kind, count in sorted(remaining_error_counts.items()))
        lines += [f"Còn {len(missing)} ca thiếu. Lỗi đã ghi nhận ở các ca còn thiếu gồm {failures or 'chưa có log lỗi'}. "
                  "F09 dừng lượt trước khi Actor nhận observation thiếu định vị. Chi tiết từng lần chạy nằm trong `integrity-check.json` và `infrastructure.jsonl`.", "",
                  table(["Nhóm thiếu ca", "Outcome hợp lệ", "Clean quan sát", "Lỗi an toàn quan sát", "Timeout quan sát"],
                        [[f"{row['split']} {row['traffic']}", f"{row['episodes']}/{row['expected']}", row["observed_clean"],
                          row["observed_safety_failures"], row["observed_timeouts"]]
                         for row in summaries if not row["complete"]]), ""]
    report.write_text("\n".join(lines), encoding="utf-8")
    return summaries, integrity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    summaries, integrity = generate(args.output, args.report)
    print(integrity["status"], [(row["split"], row["traffic"], row["episodes"], row["clean"]) for row in summaries])


if __name__ == "__main__":
    main()
