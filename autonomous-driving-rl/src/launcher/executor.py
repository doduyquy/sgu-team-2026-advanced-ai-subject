"""
Experiment Executor and Simulation Pipeline for Research Platform V1 Launcher (Gate 7.5A).

This module implements the execution engine that runs a ResolvedExperimentPlanV1:
- Consumes Gate-6 AgentPolicy via registered factory.
- Reconstructs exact road geometry from Gate-5 manifests using MapGenerateMethod.PG_MAP_FILE.
- Enforces strict Information Parity Principle: agent receives only AgentInputV1 and AgentPublicEpisodeContext.
- Executes 10 Hz control loop, measuring agent decision latency strictly around agent.act().
- Classifies outcomes via Gate-3 safety-first precedence; computes rewards via Gate-4 RewardSpecV1.
- Persists canonical Gate-7 local artifacts (LocalExperimentLogger) and downstream W&B mirror.
- Handles technical exceptions and SIGINT cleanly without fabricating driving outcomes.
"""

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.envs.metadrive_env import MetaDriveEnv
import numpy as np

from src.launcher.cases import (
    get_default_project_root,
    load_locked_geometry_block_sequence,
)
from src.launcher.events import (
    EventCallback,
    LauncherEventType,
    LauncherEventV1,
)
from src.launcher.models import (
    LauncherMode,
    PreflightBlockedError,
    ResolvedCaseV1,
    ResolvedExperimentPlanV1,
    validate_custom_run_id,
)
from src.launcher.preflight import run_preflight
from src.platform import (
    AgentPolicy,
    AgentPublicEpisodeContext,
    EpisodeOutcome,
    EpisodeRecord,
    EpisodeTimingRecord,
    ExperimentRunConfig,
    InputProfileId,
    LocalExperimentLogger,
    RewardBreakdown,
    RewardSpecV1,
    RunKind,
    RunStatus,
    TerminalReason,
    WandbBackend,
    WandbMode,
    build_agent_input,
    classify_episode_outcome,
    get_action_adapter,
    sanitize_error_message,
)


def compute_step_route_progress_and_reward(
    reward_spec: RewardSpecV1,
    current_route_completion: float,
    last_route_completion: float,
    horizon_steps: int,
    outcome: Optional[EpisodeOutcome] = None
) -> Tuple[float, float, RewardBreakdown]:
    """
    Authoritative production helper used by ExperimentExecutor for step progress and reward calculation.
    Preserves signed delta (route_t - route_{t-1}) without clamping.
    Returns (delta_route, new_last_route_completion, reward_breakdown).
    """
    delta_route = float(current_route_completion - last_route_completion)
    new_last_route = float(current_route_completion)
    breakdown = reward_spec.compute_step_reward(
        delta_route_completion=delta_route,
        horizon_steps=horizon_steps,
        outcome=outcome
    )
    return delta_route, new_last_route, breakdown


def resolve_manifest_name_for_mode(mode: LauncherMode) -> str:
    """Maps LauncherMode to expected Gate-7 manifest_name provenance string."""
    if mode in (LauncherMode.SANDBOX, LauncherMode.AUDIT):
        return "geometry_split_manifest.csv"
    elif mode == LauncherMode.VALIDATION:
        return "validation_case_manifest.csv"
    elif mode == LauncherMode.TEST:
        return "test_case_manifest.csv"
    else:
        raise ValueError(f"Unknown launcher mode: {mode}")


def build_metadrive_case_config(
    case: ResolvedCaseV1,
    blocks: List[Any],
    render_mode: str = "OFF"
) -> Dict[str, Any]:
    """
    Constructs the exact, pure, unit-testable MetaDrive environment configuration.
    Explicitly locks physics step, decision repeat, 10 Hz control frequency,
    trigger traffic mode, and horizon.
    """
    physics_world_step_size = 0.02
    decision_repeat = 5
    decision_dt = physics_world_step_size * decision_repeat  # 0.1s
    control_frequency_hz = 1.0 / decision_dt                 # 10.0 Hz

    assert abs(decision_dt - 0.1) < 1e-6, f"Invalid decision dt: {decision_dt}"
    assert abs(control_frequency_hz - 10.0) < 1e-6, f"Invalid control frequency: {control_frequency_hz}"

    from metadrive.manager.traffic_manager import TrafficMode

    return {
        "use_render": (render_mode.upper() == "NATIVE"),
        "num_scenarios": 1,
        "start_seed": int(case.environment_seed),
        "map_config": {"type": MapGenerateMethod.PG_MAP_FILE, "config": blocks},
        "traffic_density": float(case.traffic_density),
        "traffic_mode": TrafficMode.Trigger,
        "horizon": int(case.horizon_steps),
        "physics_world_step_size": physics_world_step_size,
        "decision_repeat": decision_repeat,
        "truncate_as_terminate": False
    }


@dataclass(frozen=True)
class ExecutionReportV1:
    """Summary record returned upon completion or interruption of an experiment execution."""
    run_id: str
    status: str
    total_cases: int
    completed_episodes: int
    summary_payload: Optional[Dict[str, Any]]
    run_integrity: Optional[Dict[str, Any]]
    wandb_sync: Optional[Dict[str, Any]]
    run_dir: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperimentExecutor:
    """
    Algorithm-neutral execution harness running ResolvedExperimentPlanV1.
    Ties together MetaDrive simulator, Gate-6 AgentPolicy, Gate-3/4 evaluation, and Gate-7 logging.
    """
    def __init__(
        self,
        plan: ResolvedExperimentPlanV1,
        agent_factory: Optional[Callable[[], AgentPolicy]] = None,
        registry: Optional[Any] = None,
        runs_root: Optional[Path] = None,
        custom_run_id: Optional[str] = None,
        event_callback: Optional[EventCallback] = None,
        project_root: Optional[Path] = None
    ):
        self.plan = plan
        self.project_root = project_root or get_default_project_root()
        self.runs_root = runs_root or (self.project_root / "runs")
        self.custom_run_id = validate_custom_run_id(custom_run_id, self.runs_root)
        self.event_callback = event_callback

        from src.launcher.registry import build_canonical_agent_registry
        canonical_reg = build_canonical_agent_registry()

        if plan.canonical_run:
            if registry is not None:
                raise ValueError("External registry is forbidden for canonical benchmark execution")
            if agent_factory is not None:
                raise ValueError(
                    "Explicit agent_factory override is strictly forbidden on canonical benchmark runs! "
                    "Agent must be instantiated via authoritative AgentRegistryV1."
                )
            self.registry = canonical_reg
            self.agent_factory = canonical_reg.get_factory(plan.agent_registration.agent_id)
        else:
            self.registry = registry or canonical_reg
            self.agent_factory = agent_factory or self.registry.get_factory(plan.agent_registration.agent_id)

    def _emit(self, event: LauncherEventV1) -> None:
        """Emits event to listener if registered."""
        if self.event_callback is not None:
            try:
                self.event_callback(event)
            except Exception:
                pass  # Observers must never disrupt execution engine

    def execute(self) -> ExecutionReportV1:
        """
        Executes all resolved cases in the plan sequentially under Gate-5 protocol ordering.
        Enforces preflight validation prior to any simulator, logger, or W&B side effects.
        Emits lifecycle events and produces durable Gate-7 scientific records.
        """
        # 0. Enforce preflight validation prior to ANY simulator / logger / W&B side effects
        preflight_registry = None if self.plan.canonical_run else self.registry
        preflight = run_preflight(self.plan, self.project_root, registry=preflight_registry)
        if not preflight.can_execute:
            fail_msgs = [f"[{c.category}] {c.message}" for c in preflight.checks if c.status == "FAIL"]
            raise PreflightBlockedError(
                f"Preflight validation failed with {preflight.fail_count} blocking error(s). "
                f"Execution is blocked to protect scientific integrity: {'; '.join(fail_msgs)}"
            )

        # 1. Embed launcher provenance into algorithm_hyperparameters (Section 18)
        launcher_prov = {
            "launcher_mode": self.plan.launcher_mode.value,
            "launcher_contract_sha256": self.plan.launcher_contract_sha256,
            "platform_execution_contract_sha256": self.plan.platform_execution_contract_sha256,
            "resolved_plan_sha256": self.plan.resolved_plan_sha256,
            "protocol_scope": self.plan.protocol_scope
        }

        # 2. Build Gate-7 ExperimentRunConfig
        run_config = ExperimentRunConfig(
            run_kind=self.plan.run_kind,
            benchmark_contract_sha256=self.plan.benchmark_contract_sha256,
            agent_contract_sha256=self.plan.agent_contract_sha256,
            platform_runtime_contract_sha256=self.plan.platform_runtime_contract_sha256,
            logging_contract_sha256=self.plan.logging_contract_sha256,
            platform_observability_contract_sha256=self.plan.platform_observability_contract_sha256,
            agent_id=self.plan.agent_registration.agent_id,
            agent_version=self.plan.agent_registration.agent_version,
            input_profile_id=self.plan.agent_registration.input_profile_id,
            action_adapter_id=self.plan.agent_registration.action_adapter_id,
            inference_stochasticity=self.plan.agent_registration.inference_stochasticity,
            stateful_within_episode=self.plan.agent_registration.stateful_within_episode,
            agent_seed=self.plan.agent_seed,
            expected_episode_count=len(self.plan.resolved_cases),
            manifest_name=resolve_manifest_name_for_mode(self.plan.launcher_mode),
            allow_dirty_worktree_override=(not self.plan.canonical_run),
            wandb_mode=self.plan.wandb_mode,
            algorithm_hyperparameters={"launcher": launcher_prov}
        )

        # 3. Instantiate authoritative Gate-7 logger and downstream W&B mirror
        logger = LocalExperimentLogger(
            config=run_config,
            runs_root=self.runs_root,
            custom_run_id=self.custom_run_id,
            repo_root=self.project_root.parent,
            force_noncanonical=(not self.plan.canonical_run)
        )

        backend = WandbBackend(mode=self.plan.wandb_mode)
        backend.start_run(logger.manifest, run_config)

        self._emit(LauncherEventV1.create(
            event_type=LauncherEventType.RUN_STARTED,
            run_id=logger.run_id,
            total_episodes=len(self.plan.resolved_cases),
            status="RUNNING",
            message=f"Started run {logger.run_id} ({self.plan.launcher_mode.value})"
        ))

        completed_count = 0
        total_cases = len(self.plan.resolved_cases)
        action_adapter = get_action_adapter(self.plan.agent_registration.action_adapter_id)
        reward_spec = RewardSpecV1()
        agent = None

        try:
            # Verify factory reference against registry on canonical runs
            expected_factory_ref = self.registry.get_factory_ref(self.plan.agent_registration.agent_id)
            actual_factory = self.agent_factory
            actual_factory_ref = f"{getattr(actual_factory, '__module__', 'unknown')}:{getattr(actual_factory, '__qualname__', getattr(actual_factory, '__name__', 'unknown'))}"
            if self.plan.canonical_run and actual_factory_ref != expected_factory_ref:
                raise RuntimeError(
                    f"Agent factory identity mismatch! Expected '{expected_factory_ref}', got '{actual_factory_ref}'. "
                    "Unauthorized factory substitution detected!"
                )

            # 4. Instantiate agent policy ONCE per ExperimentRun (Section 9)
            agent = self.agent_factory()
            if not hasattr(agent, "descriptor"):
                raise TypeError(f"Agent instance of type '{type(agent).__name__}' lacks required 'descriptor' property.")

            desc = agent.descriptor
            reg = self.plan.agent_registration

            # Validate runtime descriptor against registration (Section 8)
            if desc.agent_id != reg.agent_id:
                raise ValueError(f"Agent descriptor agent_id mismatch: descriptor='{desc.agent_id}' != registration='{reg.agent_id}'")
            if desc.agent_version != reg.agent_version:
                raise ValueError(f"Agent descriptor agent_version mismatch: descriptor='{desc.agent_version}' != registration='{reg.agent_version}'")
            if desc.input_profile_id != reg.input_profile_id:
                raise ValueError(f"Agent descriptor input_profile_id mismatch: descriptor='{desc.input_profile_id}' != registration='{reg.input_profile_id}'")
            if desc.action_adapter_id != reg.action_adapter_id:
                raise ValueError(f"Agent descriptor action_adapter_id mismatch: descriptor='{desc.action_adapter_id}' != registration='{reg.action_adapter_id}'")
            if desc.inference_stochasticity != reg.inference_stochasticity:
                raise ValueError(f"Agent descriptor inference_stochasticity mismatch: descriptor='{desc.inference_stochasticity}' != registration='{reg.inference_stochasticity}'")
            if desc.stateful_within_episode != reg.stateful_within_episode:
                raise ValueError(f"Agent descriptor stateful_within_episode mismatch: descriptor='{desc.stateful_within_episode}' != registration='{reg.stateful_within_episode}'")
            if desc.method_family != reg.method_family:
                raise ValueError(f"Agent descriptor method_family mismatch: descriptor='{desc.method_family}' != registration='{reg.method_family}'")

            # Verify runtime implementation class reference (Fix B)
            actual_impl_ref = f"{type(agent).__module__}:{type(agent).__qualname__}"
            if self.plan.canonical_run and actual_impl_ref != self.plan.agent_registration.implementation_ref:
                raise RuntimeError(
                    f"Agent runtime implementation mismatch! Expected class '{self.plan.agent_registration.implementation_ref}', "
                    f"got '{actual_impl_ref}'. Implementation impersonation detected!"
                )

            # Validate runtime descriptor against registration (Section 8)
            if desc.agent_id != reg.agent_id:
                raise ValueError(f"Agent descriptor agent_id mismatch: descriptor='{desc.agent_id}' != registration='{reg.agent_id}'")
            if desc.agent_version != reg.agent_version:
                raise ValueError(f"Agent descriptor agent_version mismatch: descriptor='{desc.agent_version}' != registration='{reg.agent_version}'")
            if desc.input_profile_id != reg.input_profile_id:
                raise ValueError(f"Agent descriptor input_profile_id mismatch: descriptor='{desc.input_profile_id}' != registration='{reg.input_profile_id}'")
            if desc.action_adapter_id != reg.action_adapter_id:
                raise ValueError(f"Agent descriptor action_adapter_id mismatch: descriptor='{desc.action_adapter_id}' != registration='{reg.action_adapter_id}'")
            if desc.inference_stochasticity != reg.inference_stochasticity:
                raise ValueError(f"Agent descriptor inference_stochasticity mismatch: descriptor='{desc.inference_stochasticity}' != registration='{reg.inference_stochasticity}'")
            if desc.stateful_within_episode != reg.stateful_within_episode:
                raise ValueError(f"Agent descriptor stateful_within_episode mismatch: descriptor='{desc.stateful_within_episode}' != registration='{reg.stateful_within_episode}'")
            if desc.method_family != reg.method_family:
                raise ValueError(f"Agent descriptor method_family mismatch: descriptor='{desc.method_family}' != registration='{reg.method_family}'")

            for case_idx, case in enumerate(self.plan.resolved_cases, 1):
                self._emit(LauncherEventV1.create(
                    event_type=LauncherEventType.EPISODE_STARTED,
                    run_id=logger.run_id,
                    episode_index=case_idx,
                    total_episodes=total_cases,
                    message=f"Starting case {case.case_id} (order={case.protocol_order_index})"
                ))

                env = None
                try:
                    # A. Reconstruct exact locked road blocks
                    blocks, _, _ = load_locked_geometry_block_sequence(case.to_dict(), self.project_root)

                    # B. Configure MetaDrive simulator with explicit 10 Hz control config
                    env_config = build_metadrive_case_config(case, blocks, render_mode=self.plan.render_mode)
                    env = MetaDriveEnv(env_config)
                    raw_obs, info = env.reset(seed=case.environment_seed)

                    # C. Reset agent with public context (Information Parity)
                    public_context = AgentPublicEpisodeContext(
                        control_frequency_hz=int(self.plan.control_frequency_hz),
                        control_dt_s=float(self.plan.control_dt_s),
                        horizon_steps=int(case.horizon_steps),
                        input_profile_id=self.plan.agent_registration.input_profile_id,
                        action_adapter_id=self.plan.agent_registration.action_adapter_id,
                        mode="INFERENCE"
                    )
                    agent.reset(public_context, agent_seed=self.plan.agent_seed)

                    # D. Step loop (10 Hz decision cycle)
                    latencies_ms: List[float] = []
                    speeds_kmh: List[float] = []
                    cumulative_reward = 0.0
                    step_idx = 0
                    last_route_completion = 0.0
                    max_route_completion = 0.0
                    terminated = False
                    truncated = False
                    outcome: Optional[EpisodeOutcome] = None
                    env_flags: Dict[str, Any] = {}

                    while not (terminated or truncated) and step_idx < case.horizon_steps:
                        # 1. Assemble clean AgentInputV1 (no evaluator metadata)
                        tm = env.engine.traffic_manager
                        traffic_v = list(getattr(tm, "traffic_vehicles", [])) if hasattr(tm, "traffic_vehicles") else []
                        agent_input = build_agent_input(
                            raw_obs=raw_obs,
                            ego_vehicle=env.agent,
                            traffic_vehicles=traffic_v,
                            navigation=env.agent.navigation,
                            road_network=env.current_map.road_network,
                            step_index=step_idx,
                            profile_id=InputProfileId(self.plan.agent_registration.input_profile_id)
                        )

                        # 2. High-resolution decision latency timer wrapping agent.act() strictly
                        t0 = time.perf_counter_ns()
                        decision = agent.act(agent_input)
                        t1 = time.perf_counter_ns()
                        latencies_ms.append((t1 - t0) / 1e6)

                        # 3. Action adaptation to physical actuator Box(2)
                        canonical_act = action_adapter.to_canonical(decision.action_payload)

                        # 4. Environment physics step
                        raw_obs, r_env, term_step, trunc_step, step_info = env.step(canonical_act.to_numpy())
                        step_idx += 1

                        # Update raw safety and task flags from step_info
                        for k in ("arrive_dest", "out_of_road", "crash_vehicle", "crash_object", "crash_building", "crash_human", "crash_sidewalk", "max_step"):
                            if bool(step_info.get(k, False)):
                                env_flags[k] = True

                        # 5. Gate-3 Safety-First Termination Classification
                        outcome = classify_episode_outcome(
                            raw_flags=env_flags,
                            terminated=term_step,
                            truncated=trunc_step or (step_idx >= case.horizon_steps)
                        )
                        terminated = outcome.terminated
                        truncated = outcome.truncated

                        # 6. Gate-4 Step Reward Computation (Signed Delta, No Clamping)
                        speed_kmh = float(env.agent.speed_kmh) if hasattr(env, "agent") and hasattr(env.agent, "speed_kmh") else 0.0
                        speeds_kmh.append(speed_kmh)

                        curr_route = float(step_info.get("route_completion", 0.0))
                        if curr_route > max_route_completion:
                            max_route_completion = curr_route

                        delta_route, last_route_completion, breakdown = compute_step_route_progress_and_reward(
                            reward_spec=reward_spec,
                            current_route_completion=curr_route,
                            last_route_completion=last_route_completion,
                            horizon_steps=case.horizon_steps,
                            outcome=outcome if (terminated or truncated) else None
                        )
                        cumulative_reward += breakdown.total_reward

                        # Emit progress event periodically (every 10 steps, preserving 10 Hz control)
                        if step_idx % 10 == 0:
                            self._emit(LauncherEventV1.create(
                                event_type=LauncherEventType.EPISODE_PROGRESS,
                                run_id=logger.run_id,
                                episode_index=case_idx,
                                total_episodes=total_cases,
                                step_index=step_idx,
                                route_completion=curr_route,
                                speed_kmh=speed_kmh,
                                status="RUNNING"
                            ))

                    # E. Finalize episode outcome
                    if outcome is None:
                        outcome = classify_episode_outcome(
                            raw_flags=env_flags,
                            terminated=True,
                            truncated=(step_idx >= case.horizon_steps)
                        )

                finally:
                    if env is not None:
                        env.close()

                # F. Construct Gate-4 EpisodeRecord
                sim_time_s = step_idx * float(self.plan.control_dt_s)
                mean_spd = float(np.mean(speeds_kmh)) if speeds_kmh else 0.0
                max_spd = float(np.max(speeds_kmh)) if speeds_kmh else 0.0
                clean_succ = bool(outcome.clean_success)
                raw_arr = bool(outcome.raw_flags.get("arrive_dest", False))
                time_to_succ = sim_time_s if clean_succ else None

                ep_record = EpisodeRecord(
                    tier=case.tier,
                    sequence=case.sequence,
                    scenario_seed=case.geometry_generation_seed,
                    terminated=outcome.terminated,
                    truncated=outcome.truncated,
                    primary_reason=outcome.primary_reason,
                    raw_arrival=raw_arr,
                    clean_success=clean_succ,
                    final_route_completion=float(last_route_completion),
                    max_route_completion=float(max_route_completion),
                    raw_crash_vehicle=bool(env_flags.get("crash_vehicle", False)),
                    raw_crash_object=bool(env_flags.get("crash_object", False)),
                    raw_crash_building=bool(env_flags.get("crash_building", False)),
                    raw_crash_human=bool(env_flags.get("crash_human", False)),
                    raw_crash_sidewalk=bool(env_flags.get("crash_sidewalk", False)),
                    raw_out_of_road=bool(env_flags.get("out_of_road", False)),
                    episode_steps=step_idx,
                    simulation_time_s=sim_time_s,
                    mean_speed_kmh=mean_spd,
                    max_speed_kmh=max_spd,
                    episode_return=float(cumulative_reward),
                    time_to_clean_success_s=time_to_succ,
                    agent_seed=self.plan.agent_seed
                )

                # G. Create EpisodeTimingRecord
                timing = EpisodeTimingRecord.from_latencies(case_idx, latencies_ms)

                # H. Append to authoritative local log & mirror to W&B
                log_row = logger.log_episode(
                    record=ep_record,
                    timing=timing,
                    episode_index=case_idx,
                    protocol_order_index=case.protocol_order_index,
                    case_id=case.case_id,
                    split=case.split,
                    environment_seed=case.environment_seed,
                    geometry_generation_seed=case.geometry_generation_seed,
                    horizon_steps=case.horizon_steps,
                    agent_seed=self.plan.agent_seed
                )

                backend.log_episode(log_row, timing=timing)
                completed_count += 1

                self._emit(LauncherEventV1.create(
                    event_type=LauncherEventType.EPISODE_FINISHED,
                    run_id=logger.run_id,
                    episode_index=case_idx,
                    total_episodes=total_cases,
                    route_completion=float(last_route_completion),
                    speed_kmh=mean_spd,
                    status="SUCCESS" if clean_succ else "FAILURE",
                    message=f"Case {case.case_id} finished ({outcome.primary_reason.value}, route={last_route_completion:.1%})"
                ))

            # 5. Finalize run cleanly
            summary_payload = logger.prepare_summary()
            backend.log_summary(summary_payload)
            sync_meta = backend.finish(RunStatus.COMPLETE)
            integrity = logger.finalize_run(summary_payload=summary_payload, wandb_sync_info=sync_meta)

            self._emit(LauncherEventV1.create(
                event_type=LauncherEventType.RUN_FINISHED,
                run_id=logger.run_id,
                total_episodes=total_cases,
                status="COMPLETE",
                message=f"Run {logger.run_id} completed successfully ({completed_count}/{total_cases} episodes)."
            ))

            with open(logger.wandb_sync_json_path, "r", encoding="utf-8") as f:
                sync_data = json.load(f)

            return ExecutionReportV1(
                run_id=logger.run_id,
                status="COMPLETE",
                total_cases=total_cases,
                completed_episodes=completed_count,
                summary_payload=summary_payload,
                run_integrity=integrity.to_dict(),
                wandb_sync=sync_data,
                run_dir=str(logger.run_dir)
            )

        except KeyboardInterrupt:
            # Graceful cancellation handling (Section 38)
            if logger.status != RunStatus.FAILED:
                logger.mark_interrupted("Execution interrupted by user (KeyboardInterrupt / SIGINT)")
            sync_meta = backend.fail("Execution interrupted by user")
            self._emit(LauncherEventV1.create(
                event_type=LauncherEventType.RUN_INTERRUPTED,
                run_id=logger.run_id,
                status="INTERRUPTED",
                message="Execution interrupted by user."
            ))
            return ExecutionReportV1(
                run_id=logger.run_id,
                status="INTERRUPTED",
                total_cases=total_cases,
                completed_episodes=completed_count,
                summary_payload=None,
                run_integrity=None,
                wandb_sync=sync_meta,
                run_dir=str(logger.run_dir)
            )

        except Exception as e:
            sanitized_e = sanitize_error_message(e)
            if logger.status != RunStatus.FAILED:
                logger.mark_failed(
                    failure_category="TECHNICAL_EXECUTION_ERROR",
                    failure_message=sanitized_e
                )
            sync_meta = backend.fail(sanitized_e)
            self._emit(LauncherEventV1.create(
                event_type=LauncherEventType.RUN_FAILED,
                run_id=logger.run_id,
                status="FAILED",
                message=f"Execution failed: {sanitized_e}"
            ))
            raise RuntimeError(f"Experiment execution failed in run '{logger.run_id}': {sanitized_e}") from e
        finally:
            if agent is not None and hasattr(agent, "close"):
                try:
                    agent.close()
                except Exception:
                    pass
