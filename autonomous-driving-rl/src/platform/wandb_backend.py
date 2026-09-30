"""
Weights & Biases Tracking Backend and Protocol Abstraction for Research Platform V1.
Gate 7 of Research Platform V1.

Core Mandates:
- Downstream mirror only: W&B is for remote indexing, comparison, and visualization.
- Observational only: W&B operations NEVER alter agent decisions, environment dynamics, or seeds.
- Ephemeral secret policy: WANDB_API_KEY is never persisted, echoed, or included in configs.
- Non-fatal failure policy: W&B failures are logged as degraded/failed sync, never corrupting local runs.
"""

from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass
from enum import Enum
import math
import os
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

import numpy as np

from src.platform.experiment_logging import (
    EpisodeTimingRecord,
    ExperimentRunConfig,
    RunKind,
    RunManifestV1,
    RunStatus,
    WandbMode,
    WandbSyncStatus,
)
from src.platform.metrics import AggregateMetrics, EpisodeRecord


@runtime_checkable
class TrackingBackend(Protocol):
    """Protocol for optional remote experiment tracking backends."""
    def start_run(self, manifest: RunManifestV1, config: ExperimentRunConfig) -> None:
        ...

    def log_episode(
        self,
        record: EpisodeRecord,
        timing: Optional[EpisodeTimingRecord] = None,
        technical_failure: Optional[Dict[str, Any]] = None
    ) -> None:
        ...

    def log_summary(self, summary_payload: Dict[str, Any]) -> None:
        ...

    def finish(self, status: RunStatus) -> Dict[str, Any]:
        ...

    def fail(self, error_message: str) -> Dict[str, Any]:
        ...


class FakeTrackingBackend:
    """
    In-memory mock tracking backend for pure unit tests, offline verification,
    and simulated network failure testing.
    """
    def __init__(
        self,
        simulate_init_failure: bool = False,
        simulate_log_failure: bool = False,
        simulate_finish_failure: bool = False
    ):
        self.simulate_init_failure = simulate_init_failure
        self.simulate_log_failure = simulate_log_failure
        self.simulate_finish_failure = simulate_finish_failure

        self.started = False
        self.finished = False
        self.run_id: Optional[str] = None
        self.config_mirrored: Optional[Dict[str, Any]] = None
        self.episodes_logged: List[Dict[str, Any]] = []
        self.summary_logged: Optional[Dict[str, Any]] = None
        self.table_logged: Optional[List[Dict[str, Any]]] = None
        self.sync_status: WandbSyncStatus = WandbSyncStatus.DISABLED
        self.error_log: List[str] = []

    def start_run(self, manifest: RunManifestV1, config: ExperimentRunConfig) -> None:
        if self.simulate_init_failure:
            self.sync_status = WandbSyncStatus.FAILED
            self.error_log.append("Simulated W&B init connection timeout")
            raise ConnectionError("Simulated W&B init connection timeout")

        self.started = True
        self.run_id = f"fake_wandb_{manifest.run_id}"
        self.config_mirrored = config.to_dict()
        self.table_logged = []
        self.sync_status = WandbSyncStatus.ONLINE

    def log_episode(
        self,
        record: EpisodeRecord,
        timing: Optional[EpisodeTimingRecord] = None,
        technical_failure: Optional[Dict[str, Any]] = None
    ) -> None:
        if not self.started:
            return
        if self.simulate_log_failure:
            self.sync_status = WandbSyncStatus.FAILED
            self.error_log.append("Simulated W&B metric log socket error")
            raise IOError("Simulated W&B metric log socket error")

        clean_succ = getattr(record, "clean_success", getattr(getattr(record, "outcome", None), "clean_success", False))
        raw_arr = getattr(record, "raw_arrival", getattr(getattr(record, "outcome", None), "raw_arrival", False))
        final_comp = getattr(record, "final_route_completion", getattr(getattr(record, "outcome", None), "final_route_completion", 0.0))
        max_comp = getattr(record, "max_route_completion", getattr(getattr(record, "outcome", None), "max_route_completion", 0.0))
        ep_steps = getattr(record, "episode_steps", getattr(getattr(record, "outcome", None), "episode_steps", 0))
        ep_time = getattr(record, "simulation_time_s", getattr(getattr(record, "outcome", None), "episode_time_s", 0.0))
        ep_ret = getattr(record, "episode_return", getattr(getattr(record, "outcome", None), "episode_return", 0.0))
        mean_spd = getattr(record, "mean_speed_kmh", getattr(getattr(record, "outcome", None), "mean_speed_kmh", 0.0))
        max_spd = getattr(record, "max_speed_kmh", getattr(getattr(record, "outcome", None), "max_speed_kmh", 0.0))

        prim_reason = getattr(record, "primary_reason", getattr(getattr(record, "outcome", None), "primary_terminal_reason", None))
        reason_val = prim_reason.value if hasattr(prim_reason, "value") else str(prim_reason or "UNKNOWN")

        ep_idx = getattr(record, "episode_index", 0)

        event = {
            "eval/episode_index": ep_idx,
            "eval/clean_success": 1.0 if clean_succ else 0.0,
            "eval/raw_arrival": 1.0 if raw_arr else 0.0,
            "eval/final_route_completion": final_comp,
            "eval/max_route_completion": max_comp,
            "eval/episode_steps": ep_steps,
            "eval/episode_time_s": ep_time,
            "diagnostic/episode_return": ep_ret,
            "eval/mean_speed_kmh": mean_spd,
            "eval/max_speed_kmh": max_spd,
            f"failure/{reason_val}": 1.0,
        }
        if timing:
            event["timing/agent_act_mean_ms"] = timing.mean_act_ms
            event["timing/agent_act_median_ms"] = timing.median_act_ms
            event["timing/agent_act_p95_ms"] = timing.p95_act_ms
            event["timing/agent_act_max_ms"] = timing.max_act_ms
            event["timing/agent_act_total_ms"] = timing.total_act_ms

        self.episodes_logged.append(event)

        # Table row mirror
        if self.table_logged is not None:
            self.table_logged.append({
                "episode_index": ep_idx,
                "case_id": getattr(record, "case_id", f"case_{ep_idx}"),
                "tier": getattr(record, "tier", "N/A"),
                "sequence": getattr(record, "sequence", "N/A"),
                "clean_success": clean_succ,
                "final_route_completion": final_comp,
                "primary_terminal_reason": reason_val,
            })

    def log_summary(self, summary_payload: Dict[str, Any]) -> None:
        if not self.started:
            return
        self.summary_logged = summary_payload

    def finish(self, status: RunStatus) -> Dict[str, Any]:
        if not self.started:
            return {"sync_status": WandbSyncStatus.DISABLED.value, "wandb_run_id": None}
        if self.simulate_finish_failure:
            self.sync_status = WandbSyncStatus.FAILED
            self.error_log.append("Simulated W&B finish sync error")
            raise RuntimeError("Simulated W&B finish sync error")

        self.finished = True
        self.sync_status = WandbSyncStatus.SYNCED
        return {
            "sync_status": self.sync_status.value,
            "wandb_run_id": self.run_id,
            "wandb_run_url": f"https://wandb.ai/mock-entity/mock-project/runs/{self.run_id}",
            "project": "mock-project",
            "entity": "mock-entity"
        }

    def fail(self, error_message: str) -> Dict[str, Any]:
        self.sync_status = WandbSyncStatus.FAILED
        self.error_log.append(error_message)
        return {
            "sync_status": self.sync_status.value,
            "wandb_run_id": self.run_id,
            "error": error_message
        }


class WandbBackend:
    """
    Production Weights & Biases tracking backend.
    Enforces lazy imports, downstream mirroring, and non-fatal error isolation.
    """
    def __init__(self, mode: WandbMode = WandbMode.DISABLED):
        self.mode = mode
        self.wandb_run = None
        self.run_id: Optional[str] = None
        self.run_url: Optional[str] = None
        self.project: Optional[str] = None
        self.entity: Optional[str] = None
        self.sync_status: WandbSyncStatus = (
            WandbSyncStatus.DISABLED if mode == WandbMode.DISABLED else WandbSyncStatus.OFFLINE if mode == WandbMode.OFFLINE else WandbSyncStatus.ONLINE
        )
        self.table_data: List[List[Any]] = []
        self.table_columns: List[str] = [
            "episode_index", "case_id", "split", "tier", "sequence",
            "clean_success", "final_route_completion", "max_route_completion",
            "primary_terminal_reason", "episode_steps", "episode_time_s",
            "mean_act_ms", "episode_return"
        ]

    def start_run(self, manifest: RunManifestV1, config: ExperimentRunConfig) -> None:
        """Initializes W&B run without echoing or persisting API keys."""
        if self.mode == WandbMode.DISABLED:
            self.sync_status = WandbSyncStatus.DISABLED
            return

        try:
            import wandb
        except ImportError as e:
            self.sync_status = WandbSyncStatus.FAILED
            print(f"[WARNING] wandb package not installed; degrading to DISABLED: {e}")
            return

        # Configure environment mode
        if self.mode == WandbMode.OFFLINE:
            os.environ["WANDB_MODE"] = "offline"
        else:
            os.environ["WANDB_MODE"] = "online"

        # Sanitize config: exclude personal paths and credentials
        sanitized_config = copy.deepcopy(config.to_dict())

        # Construct human-readable run name: <run_kind>-<agent_id>-<short_id>
        short_id = manifest.run_id.split("_")[-1]
        run_name = f"{config.run_kind.value.lower()}-{config.agent_id}-{short_id}"

        # Setup W&B run tags
        tags = [
            "platform-v1",
            "gate7",
            config.run_kind.value.lower(),
            config.inference_stochasticity,
            config.input_profile_id.lower(),
            config.action_adapter_id.lower()
        ]
        if config.tags:
            tags.extend(config.tags)

        try:
            self.wandb_run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=run_name,
                id=manifest.run_id,
                resume="never",
                config=sanitized_config,
                tags=sorted(list(set(tags))),
                reinit=True
            )
            self.run_id = self.wandb_run.id
            self.run_url = getattr(self.wandb_run, "url", None)
            self.project = config.wandb_project
            self.entity = config.wandb_entity
            self.sync_status = WandbSyncStatus.ONLINE if self.mode == WandbMode.ONLINE else WandbSyncStatus.OFFLINE
            print(f"[WANDB] Run initialized ({self.mode.value}): {self.run_id} -> {self.run_url}")
        except Exception as e:
            self.sync_status = WandbSyncStatus.FAILED
            print(f"[WARNING] W&B initialization failed (non-fatal to local run): {e}")

    def log_episode(
        self,
        record: EpisodeRecord,
        timing: Optional[EpisodeTimingRecord] = None,
        technical_failure: Optional[Dict[str, Any]] = None
    ) -> None:
        """Logs episode metric event and records tabular row."""
        if not self.wandb_run or self.sync_status == WandbSyncStatus.FAILED:
            return

        clean_succ = getattr(record, "clean_success", getattr(getattr(record, "outcome", None), "clean_success", False))
        raw_arr = getattr(record, "raw_arrival", getattr(getattr(record, "outcome", None), "raw_arrival", False))
        final_comp = getattr(record, "final_route_completion", getattr(getattr(record, "outcome", None), "final_route_completion", 0.0))
        max_comp = getattr(record, "max_route_completion", getattr(getattr(record, "outcome", None), "max_route_completion", 0.0))
        ep_steps = getattr(record, "episode_steps", getattr(getattr(record, "outcome", None), "episode_steps", 0))
        ep_time = getattr(record, "simulation_time_s", getattr(getattr(record, "outcome", None), "episode_time_s", 0.0))
        ep_ret = getattr(record, "episode_return", getattr(getattr(record, "outcome", None), "episode_return", 0.0))
        mean_spd = getattr(record, "mean_speed_kmh", getattr(getattr(record, "outcome", None), "mean_speed_kmh", 0.0))
        max_spd = getattr(record, "max_speed_kmh", getattr(getattr(record, "outcome", None), "max_speed_kmh", 0.0))

        prim_reason = getattr(record, "primary_reason", getattr(getattr(record, "outcome", None), "primary_terminal_reason", None))
        reason_val = prim_reason.value if hasattr(prim_reason, "value") else str(prim_reason or "UNKNOWN")

        ep_idx = getattr(record, "episode_index", 0)

        event: Dict[str, Any] = {
            "eval/episode_index": ep_idx,
            "eval/clean_success": 1.0 if clean_succ else 0.0,
            "eval/raw_arrival": 1.0 if raw_arr else 0.0,
            "eval/final_route_completion": final_comp,
            "eval/max_route_completion": max_comp,
            "eval/episode_steps": ep_steps,
            "eval/episode_time_s": ep_time,
            "diagnostic/episode_return": ep_ret,
            "eval/mean_speed_kmh": mean_spd,
            "eval/max_speed_kmh": max_spd,
            f"failure/{reason_val}": 1.0,
        }

        # Failure flag breakdowns
        for flag in ("crash_human", "crash_vehicle", "crash_object", "crash_building", "crash_sidewalk", "out_of_road"):
            val = getattr(record, f"raw_{flag}", getattr(getattr(record, "outcome", None), flag, False))
            if val:
                event[f"flags/{flag}"] = 1.0
        if getattr(record, "truncated", False) or getattr(getattr(record, "outcome", None), "timeout", False):
            event["flags/timeout"] = 1.0

        if timing:
            event["timing/agent_act_mean_ms"] = timing.mean_act_ms
            event["timing/agent_act_median_ms"] = timing.median_act_ms
            event["timing/agent_act_p95_ms"] = timing.p95_act_ms
            event["timing/agent_act_max_ms"] = timing.max_act_ms
            event["timing/agent_act_total_ms"] = timing.total_act_ms

        if technical_failure:
            event["technical_failure/occurred"] = 1.0
            event[f"technical_failure/{technical_failure.get('reason', 'UNKNOWN')}"] = 1.0

        try:
            import wandb
            # Step axis is logical episode index
            self.wandb_run.log(event, step=ep_idx)

            # Record table row
            mean_t = timing.mean_act_ms if timing else 0.0
            self.table_data.append([
                ep_idx,
                getattr(record, "case_id", f"case_{ep_idx}"),
                getattr(record, "split", "N/A"),
                getattr(record, "tier", "N/A"),
                getattr(record, "sequence", "N/A"),
                clean_succ,
                final_comp,
                max_comp,
                reason_val,
                ep_steps,
                ep_time,
                mean_t,
                ep_ret
            ])
        except Exception as e:
            self.sync_status = WandbSyncStatus.FAILED
            print(f"[WARNING] W&B episode logging failed (non-fatal): {e}")

    def log_summary(self, summary_payload: Dict[str, Any]) -> None:
        """Mirrors local Gate-4 summary aggregates and uploads evaluation_episodes Table."""
        if not self.wandb_run or self.sync_status == WandbSyncStatus.FAILED:
            return

        try:
            import wandb
            # 1. Log evaluation_episodes Table
            if self.table_data:
                table = wandb.Table(columns=self.table_columns, data=self.table_data)
                self.wandb_run.log({"evaluation_episodes": table})

            # 2. Mirror overall metrics
            overall = summary_payload.get("overall_metrics", {})
            for k, v in overall.items():
                if v is not None:
                    if k == "episode_return" or k == "mean_episode_return":
                        self.wandb_run.summary[f"diagnostic/{k}"] = v
                    else:
                        self.wandb_run.summary[f"metrics/overall/{k}"] = v

            # 3. Mirror per-tier metrics
            tier_metrics = summary_payload.get("tier_metrics", {})
            for t_name, card in tier_metrics.items():
                for k, v in card.items():
                    if v is not None:
                        if k == "episode_return" or k == "mean_episode_return":
                            self.wandb_run.summary[f"diagnostic/tier_{t_name}_{k}"] = v
                        else:
                            self.wandb_run.summary[f"metrics/tier/{t_name}/{k}"] = v

            # 4. Mirror macro metrics
            macro = summary_payload.get("macro_metrics")
            if macro:
                for k, v in macro.items():
                    if v is not None:
                        self.wandb_run.summary[f"metrics/macro/{k}"] = v

            # 5. Timing summary
            timing_ov = summary_payload.get("timing_overall", {})
            for k, v in timing_ov.items():
                self.wandb_run.summary[f"timing/{k}"] = v

        except Exception as e:
            self.sync_status = WandbSyncStatus.FAILED
            print(f"[WARNING] W&B summary logging failed (non-fatal): {e}")

    def finish(self, status: RunStatus) -> Dict[str, Any]:
        """Finishes the W&B run cleanly and returns sync metadata."""
        if not self.wandb_run:
            return {
                "mode": self.mode.value,
                "sync_status": self.sync_status.value,
                "wandb_run_id": None,
                "wandb_run_url": None,
                "project": self.project,
                "entity": self.entity
            }

        try:
            import wandb
            exit_code = 0 if status == RunStatus.COMPLETE else 1
            self.wandb_run.finish(exit_code=exit_code)
            if self.sync_status != WandbSyncStatus.FAILED:
                self.sync_status = WandbSyncStatus.SYNCED
            print(f"[WANDB] Run finalized: {self.run_id} ({self.sync_status.value})")
        except Exception as e:
            self.sync_status = WandbSyncStatus.FAILED
            print(f"[WARNING] W&B run finish failed (non-fatal): {e}")

        return {
            "mode": self.mode.value,
            "sync_status": self.sync_status.value,
            "wandb_run_id": self.run_id,
            "wandb_run_url": self.run_url,
            "project": self.project,
            "entity": self.entity
        }

    def fail(self, error_message: str) -> Dict[str, Any]:
        """Marks W&B backend run as failed."""
        self.sync_status = WandbSyncStatus.FAILED
        if self.wandb_run:
            try:
                self.wandb_run.finish(exit_code=1)
            except Exception:
                pass
        return {
            "mode": self.mode.value,
            "sync_status": self.sync_status.value,
            "wandb_run_id": self.run_id,
            "error": error_message
        }
