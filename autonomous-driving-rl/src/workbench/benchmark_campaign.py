"""Fixed evaluation schedules above the existing launcher, without changing its rules.

Stochastic methods use inference seeds; Stage 6 consumes independently trained
checkpoints. Training happens separately on TRAIN, never inside evaluation.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from src.launcher.cases import resolve_sandbox_case, resolve_test_cases, resolve_validation_cases
from src.launcher.models import AgentRegistrationV1, CheckpointLaunchRequestV1, LaunchRequestV1, LauncherMode
from src.platform import PINNED_METADRIVE_COMMIT, PINNED_METADRIVE_VERSION, WandbMode, canonical_json_sha256
from src.workbench.results_repository import IntegrityDisplayStatus, RunArtifactRepository


@dataclass(frozen=True)
class EvaluationProfile:
    profile_id: str
    title: str
    mode: LauncherMode
    case_count: int
    replicate_seeds: Tuple[int, ...]
    description: str
    tier: str = ""
    sequence: str = ""
    geometry_generation_seed: int = 0
    environment_seed: int = 0

    @property
    def is_benchmark(self) -> bool:
        return self.mode in (LauncherMode.VALIDATION, LauncherMode.TEST)


def load_evaluation_profiles(project_root: Path) -> Tuple[EvaluationProfile, ...]:
    path = project_root / "configs" / "experiments" / "workbench_profiles_v1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != "1.0.0":
        raise ValueError("Unsupported Workbench profile version")
    profiles = []
    for raw in data["profiles"]:
        profile = EvaluationProfile(
            profile_id=raw["id"], title=raw["title"], mode=LauncherMode(raw["mode"]),
            case_count=raw["case_count"], replicate_seeds=tuple(raw["replicate_seeds"]),
            description=raw["description"], tier=raw.get("tier", ""),
            sequence=raw.get("sequence", ""),
            geometry_generation_seed=raw.get("geometry_generation_seed", 0),
            environment_seed=raw.get("environment_seed", 0),
        )
        if profile.is_benchmark:
            expected = 96 if profile.mode == LauncherMode.VALIDATION else 60
            if profile.case_count != expected or profile.replicate_seeds != (101, 202, 303):
                raise ValueError("Benchmark profiles must retain the locked case counts and replicate seeds")
            if any(key in raw for key in ("tier", "sequence", "geometry_generation_seed", "environment_seed")):
                raise ValueError("Benchmark profiles cannot select individual cases")
        elif profile.mode != LauncherMode.SANDBOX or profile.case_count != 1 or profile.replicate_seeds != (101,):
            raise ValueError("The development profile must be a single seeded TRAIN case")
        profiles.append(profile)
    if len({p.profile_id for p in profiles}) != len(profiles) or len(profiles) != 3:
        raise ValueError("Expected three uniquely identified evaluation profiles")
    return tuple(profiles)


@dataclass(frozen=True)
class CampaignJob:
    request: LaunchRequestV1
    agent_version: str
    input_profile_id: str
    action_adapter_id: str
    case_count: int


@dataclass(frozen=True)
class EvaluationCampaign:
    profile: EvaluationProfile
    jobs: Tuple[CampaignJob, ...]
    blocking_reasons: Tuple[str, ...]

    @property
    def fingerprint(self) -> str:
        return canonical_json_sha256({
            "profile_id": self.profile.profile_id,
            "jobs": [dict(request=j.request.to_dict(), version=j.agent_version,
                          input_profile=j.input_profile_id, adapter=j.action_adapter_id,
                          case_count=j.case_count) for j in self.jobs],
            "blocking_reasons": list(self.blocking_reasons),
        })


def build_evaluation_campaign(
    profile: EvaluationProfile, agents: Sequence[AgentRegistrationV1], *, show_demo: bool = True,
    checkpoint_bindings=(),
) -> EvaluationCampaign:
    if not agents:
        raise ValueError("Chọn ít nhất một thuật toán.")
    if len({a.agent_id for a in agents}) != len(agents):
        raise ValueError("Mỗi thuật toán chỉ được chọn một lần.")
    jobs: List[CampaignJob] = []
    reasons = []
    if len({(a.input_profile_id, a.action_adapter_id) for a in agents}) > 1:
        reasons.append("Nhóm so sánh này cần cùng input profile và action adapter; hãy tách các nhóm khác nhau.")
    for agent in sorted(agents, key=lambda a: a.agent_id):
        if profile.is_benchmark and not agent.benchmark_eligible:
            reasons.append(f"{agent.agent_id}: chưa được chứng nhận benchmark; chỉ chạy thử TRAIN/AUDIT.")
        if agent.requires_checkpoint:
            if agent.agent_id != "stage6_astar_learned":
                reasons.append(f"{agent.agent_id}: checkpoint adapter của luồng này hiện dành cho Stage 6 A*.")
            if not checkpoint_bindings:
                reasons.append(f"{agent.agent_id}: chưa có checkpoint; mở Train để học heuristic trước.")
            if profile.is_benchmark and {b.training_run_seed for b in checkpoint_bindings} != {101, 202, 303}:
                reasons.append(f"{agent.agent_id}: benchmark cần đủ ba mô hình train độc lập với seed 101, 202, 303.")
            if agent.inference_stochasticity != "deterministic":
                reasons.append("Chưa hỗ trợ thiết kế kết hợp training replicate và stochastic inference replicate.")
            model_bindings = tuple(checkpoint_bindings) if profile.is_benchmark else tuple(checkpoint_bindings[:1])
            schedules = [(None, binding) for binding in model_bindings] or [(None, None)]
        else:
            seeds = profile.replicate_seeds if agent.inference_stochasticity == "stochastic" else (None,)
            schedules = [(seed, None) for seed in seeds]
        for seed, binding in schedules:
            request_class = CheckpointLaunchRequestV1 if binding is not None else LaunchRequestV1
            request = request_class(
                mode=profile.mode, agent_id=agent.agent_id, agent_seed=seed,
                tier=profile.tier if not profile.is_benchmark else None,
                sequence=profile.sequence if not profile.is_benchmark else None,
                geometry_generation_seed=profile.geometry_generation_seed if not profile.is_benchmark else None,
                environment_seed=profile.environment_seed if not profile.is_benchmark else None,
                render_mode="NATIVE" if not profile.is_benchmark and show_demo else "OFF",
                wandb_mode=WandbMode.DISABLED,
                **({"checkpoint_binding": binding} if binding is not None else {}),
            )
            jobs.append(CampaignJob(request, agent.agent_version, agent.input_profile_id,
                                    agent.action_adapter_id, profile.case_count))
    return EvaluationCampaign(profile, tuple(jobs), tuple(reasons))


def inspect_campaign_results(
    campaign: EvaluationCampaign, reports: Sequence[Dict[str, Any]], project_root: Path,
) -> List[Dict[str, Any]]:
    """Read stored per-run metrics only after checking the entire cohort.

    No leaderboard, pooled episode statistics, or re-computation of official metrics.
    Demo results remain explicitly non-benchmark even if their files are VERIFIED.
    """
    if len(reports) != len(campaign.jobs):
        raise ValueError("Chưa đủ kết quả cho toàn bộ lịch chạy.")
    repo = RunArtifactRepository(project_root / "runs")
    snapshots = []
    run_ids = set()
    cases = None
    if campaign.profile.is_benchmark:
        cases = (resolve_validation_cases(project_root) if campaign.profile.mode == LauncherMode.VALIDATION
                 else resolve_test_cases(project_root))
    peer_signature = None
    for job, report in zip(campaign.jobs, reports):
        run_id = report.get("run_id")
        if not run_id or run_id in run_ids:
            raise ValueError("Run ID bị thiếu hoặc lặp lại trong nhóm kết quả.")
        run_ids.add(run_id)
        snap = repo.load_run_snapshot(run_id)
        if snap is None or snap.status != "COMPLETE" or snap.integrity_status != IntegrityDisplayStatus.VERIFIED:
            raise ValueError(f"{run_id}: dữ liệu chưa COMPLETE / VERIFIED.")
        if (snap.agent_id, snap.agent_version, snap.agent_seed, snap.input_profile_id, snap.action_adapter_id) != (
            job.request.agent_id, job.agent_version, job.request.agent_seed, job.input_profile_id, job.action_adapter_id,
        ):
            raise ValueError(f"{run_id}: cấu hình agent không khớp lịch chạy.")
        if len(snap.episode_rows) != job.case_count or snap.recorded_episode_count != job.case_count:
            raise ValueError(f"{run_id}: số episode không đầy đủ.")
        summary = snap.summary_payload or {}
        binding = getattr(job.request, "checkpoint_binding", None)
        if binding is not None:
            manifest = json.loads((Path(snap.run_dir) / "run_manifest.json").read_text(encoding="utf-8"))
            run_config = manifest.get("config", {})
            if (run_config.get("training_run_seed") != binding.training_run_seed
                    or run_config.get("algorithm_hyperparameters", {}).get("checkpoint") != binding.semantic_identity()):
                raise ValueError(f"{run_id}: checkpoint/training seed không khớp lịch chạy.")
        if summary.get("technical_failure_count") != 0 or snap.technical_failures_payload:
            raise ValueError(f"{run_id}: có lỗi kỹ thuật.")
        if snap.canonical_run != campaign.profile.is_benchmark:
            raise ValueError(f"{run_id}: phạm vi benchmark không khớp lịch chạy.")
        expected_cases = cases if cases is not None else [resolve_sandbox_case(job.request, project_root)]
        for index, (row, case) in enumerate(zip(snap.episode_rows, expected_cases), 1):
            expected = dict(case_id=case.case_id, split=case.split, episode_index=index,
                            protocol_order_index=case.protocol_order_index,
                            geometry_generation_seed=case.geometry_generation_seed,
                            environment_seed=case.environment_seed, horizon_steps=case.horizon_steps)
            if any(str(row.get(key)) != str(value) for key, value in expected.items()):
                raise ValueError(f"{run_id}: episode không khớp manifest cố định.")
        if cases is not None:
            if (snap.git_worktree_dirty is not False or snap.dirty_override or snap.unverified_env_override
                    or snap.environment_verification_status != "VERIFIED"
                    or snap.metadrive_version != PINNED_METADRIVE_VERSION
                    or snap.metadrive_commit != PINNED_METADRIVE_COMMIT):
                raise ValueError(f"{run_id}: provenance chưa đủ điều kiện benchmark.")
            signature = (snap.git_commit_sha, snap.input_profile_id, snap.action_adapter_id,
                         tuple(sorted(snap.platform_contracts.items())))
            if peer_signature is not None and signature != peer_signature:
                raise ValueError("Các run khác code, contract, input hoặc action; chưa thể so sánh trực tiếp.")
            peer_signature = signature
        snapshots.append(snap)
    metrics = ("clean_success_rate", "safety_failure_rate", "mean_final_route_completion",
               "median_final_route_completion", "mean_time_to_clean_success_s")
    return [dict(run_id=s.run_id, agent_id=s.agent_id, agent_seed=s.agent_seed,
                 training_run_seed=getattr(job.request, "checkpoint_binding", None).training_run_seed
                 if getattr(job.request, "checkpoint_binding", None) is not None else None,
                 canonical=s.canonical_run, **{key: s.summary_payload["overall_metrics"].get(key) for key in metrics})
            for job, s in zip(campaign.jobs, snapshots)]
