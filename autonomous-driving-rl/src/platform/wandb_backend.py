"""
Weights & Biases Tracking Backend and Protocol Abstraction for Research Platform V1.
Gate 7 of Research Platform V1.

Core Mandates:
- Downstream mirror only: W&B is for remote indexing, comparison, and visualization.
- Single-source-of-truth: Consumes canonical EpisodeLogRowV1 directly.
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

import numpy as np

from src.platform.experiment_logging import (
    EpisodeLogRowV1,
    EpisodeTimingRecord,
    ExperimentRunConfig,
    RunKind,
    RunManifestV1,
    RunStatus,
    WANDB_TABLE_COLUMNS,
    WandbMode,
    WandbSyncStatus,
    sanitize_error_message,
)


@runtime_checkable
class TrackingBackend(Protocol):
    """Protocol for optional remote experiment tracking backends."""
    def start_run(self, manifest: RunManifestV1, config: ExperimentRunConfig) -> None:
        ...

    def log_episode(
        self,
        record: EpisodeLogRowV1,
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
        mode: WandbMode = WandbMode.ONLINE,
        simulate_init_failure: bool = False,
        simulate_log_failure: bool = False,
        simulate_finish_failure: bool = False
    ):
        self.mode = mode
        self.simulate_init_failure = simulate_init_failure
        self.simulate_log_failure = simulate_log_failure
        self.simulate_finish_failure = simulate_finish_failure

        self.started = False
        self.finished = False
        self.run_id: Optional[str] = None
        self.config_mirrored: Optional[Dict[str, Any]] = None
        self.episodes_logged: List[Dict[str, Any]] = []
        self.summary_logged: Optional[Dict[str, Any]] = None
        self.table_logged: Optional[List[List[Any]]] = None
        self.sync_status: WandbSyncStatus = (
            WandbSyncStatus.DISABLED if mode == WandbMode.DISABLED
            else WandbSyncStatus.OFFLINE if mode == WandbMode.OFFLINE
            else WandbSyncStatus.ONLINE
        )
        self.error_log: List[str] = []
        self.configured_project: Optional[str] = None
        self.configured_entity: Optional[str] = None
        self.resolved_project: Optional[str] = None
        self.resolved_entity: Optional[str] = None

    def start_run(self, manifest: RunManifestV1, config: ExperimentRunConfig) -> None:
        if self.simulate_init_failure:
            self.sync_status = WandbSyncStatus.FAILED
            sanitized_err = sanitize_error_message("Simulated W&B init connection timeout")
            self.error_log.append(sanitized_err)
            raise ConnectionError("Simulated W&B init connection timeout")

        self.started = True
        self.run_id = f"fake_wandb_{manifest.run_id}"
        self.config_mirrored = config.to_dict()
        self.table_logged = []
        if self.mode == WandbMode.OFFLINE:
            self.sync_status = WandbSyncStatus.OFFLINE
        elif self.mode == WandbMode.DISABLED:
            self.sync_status = WandbSyncStatus.DISABLED
        else:
            self.sync_status = WandbSyncStatus.ONLINE
        self.configured_project = config.wandb_project
        self.configured_entity = config.wandb_entity
        self.resolved_project = config.wandb_project
        self.resolved_entity = config.wandb_entity or "mock_entity"

    def log_episode(
        self,
        record: EpisodeLogRowV1,
        timing: Optional[EpisodeTimingRecord] = None,
        technical_failure: Optional[Dict[str, Any]] = None
    ) -> None:
        if not isinstance(record, EpisodeLogRowV1):
            raise TypeError(
                f"TrackingBackend accepts EpisodeLogRowV1 only, got {type(record).__name__}."
            )
        if not self.started:
            return
        if self.finished:
            raise RuntimeError("Cannot log episode after tracking backend has finished!")
        if self.simulate_log_failure:
            self.sync_status = WandbSyncStatus.FAILED
            sanitized_err = sanitize_error_message("Simulated W&B metric log socket error")
            self.error_log.append(sanitized_err)
            raise IOError("Simulated W&B metric log socket error")

        event = record.to_wandb_event(timing)
        table_row = record.to_wandb_table_row(timing)

        self.episodes_logged.append(event)
        if self.table_logged is not None:
            self.table_logged.append(table_row)

    def log_summary(self, summary_payload: Dict[str, Any]) -> None:
        if not self.started:
            return
        if self.finished:
            raise RuntimeError("Cannot log summary after tracking backend has finished!")
        self.summary_logged = copy.deepcopy(summary_payload)

    def finish(self, status: RunStatus) -> Dict[str, Any]:
        if not self.started:
            return {
                "mode": self.mode.value,
                "sync_status": WandbSyncStatus.DISABLED.value,
                "wandb_run_id": None
            }
        if self.simulate_finish_failure:
            self.sync_status = WandbSyncStatus.FAILED
            sanitized_err = sanitize_error_message("Simulated W&B finish sync error")
            self.error_log.append(sanitized_err)
            raise RuntimeError("Simulated W&B finish sync error")

        self.finished = True
        if self.mode == WandbMode.OFFLINE:
            self.sync_status = WandbSyncStatus.OFFLINE
        elif self.mode == WandbMode.DISABLED:
            self.sync_status = WandbSyncStatus.DISABLED
        else:
            self.sync_status = WandbSyncStatus.SYNCED

        return {
            "mode": self.mode.value,
            "sync_status": self.sync_status.value,
            "wandb_run_id": self.run_id,
            "wandb_run_url": f"https://wandb.ai/{self.resolved_entity}/{self.resolved_project}/runs/{self.run_id}" if self.mode == WandbMode.ONLINE else None,
            "configured_project": self.configured_project,
            "configured_entity": self.configured_entity,
            "resolved_project": self.resolved_project,
            "resolved_entity": self.resolved_entity
        }

    def fail(self, error_message: str) -> Dict[str, Any]:
        self.sync_status = WandbSyncStatus.FAILED
        sanitized_msg = sanitize_error_message(error_message)
        self.error_log.append(sanitized_msg)
        return {
            "mode": self.mode.value,
            "sync_status": self.sync_status.value,
            "wandb_run_id": self.run_id,
            "error": sanitized_msg
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
        self.configured_project: Optional[str] = None
        self.configured_entity: Optional[str] = None
        self.resolved_project: Optional[str] = None
        self.resolved_entity: Optional[str] = None
        self._prev_wandb_mode: Optional[str] = None

        self.sync_status: WandbSyncStatus = (
            WandbSyncStatus.DISABLED if mode == WandbMode.DISABLED else WandbSyncStatus.OFFLINE if mode == WandbMode.OFFLINE else WandbSyncStatus.ONLINE
        )
        self.table_data: List[List[Any]] = []
        self.table_columns: List[str] = list(WANDB_TABLE_COLUMNS)

    def _restore_env(self) -> None:
        """Restores previous WANDB_MODE environment state safely."""
        if self._prev_wandb_mode is not None:
            os.environ["WANDB_MODE"] = self._prev_wandb_mode
        else:
            os.environ.pop("WANDB_MODE", None)

    def start_run(self, manifest: RunManifestV1, config: ExperimentRunConfig) -> None:
        """Initializes W&B run without echoing or persisting API keys."""
        if self.mode == WandbMode.DISABLED:
            self.sync_status = WandbSyncStatus.DISABLED
            return

        try:
            import wandb
        except ImportError as e:
            self.sync_status = WandbSyncStatus.FAILED
            sanitized_e = sanitize_error_message(e)
            print(f"[WARNING] wandb package not installed; degrading to DISABLED: {sanitized_e}")
            return

        # Save and configure environment mode safely
        self._prev_wandb_mode = os.environ.get("WANDB_MODE")
        if self.mode == WandbMode.OFFLINE:
            os.environ["WANDB_MODE"] = "offline"
        else:
            os.environ["WANDB_MODE"] = "online"

        self.configured_project = config.wandb_project
        self.configured_entity = config.wandb_entity

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

        # Privacy-preserving W&B settings
        privacy_settings = wandb.Settings(disable_git=True, save_code=False)
        mode_str = "offline" if self.mode == WandbMode.OFFLINE else "online"

        init_kwargs: Dict[str, Any] = {
            "project": config.wandb_project,
            "entity": config.wandb_entity,
            "name": run_name,
            "id": manifest.run_id,
            "mode": mode_str,
            "config": sanitized_config,
            "tags": sorted(list(set(tags))),
            "settings": privacy_settings,
            # Explicitly finish any lingering previous run cleanly without deprecated boolean reinit
            "reinit": "finish_previous"
        }
        # resume policy is applicable only to ONLINE mode; omitted for OFFLINE to avoid warning
        if self.mode == WandbMode.ONLINE:
            init_kwargs["resume"] = "never"

        try:
            self.wandb_run = wandb.init(**init_kwargs)
            self.run_id = self.wandb_run.id
            self.run_url = getattr(self.wandb_run, "url", None)
            self.resolved_project = getattr(self.wandb_run, "project", config.wandb_project)
            self.resolved_entity = getattr(self.wandb_run, "entity", config.wandb_entity)
            self.sync_status = WandbSyncStatus.ONLINE if self.mode == WandbMode.ONLINE else WandbSyncStatus.OFFLINE
            print(f"[WANDB] Run initialized ({self.mode.value}): {self.run_id} -> {self.run_url}")
        except Exception as e:
            self.sync_status = WandbSyncStatus.FAILED
            self._restore_env()
            sanitized_err = sanitize_error_message(e)
            print(f"[WARNING] W&B initialization failed (non-fatal to local run): {sanitized_err}")

    def log_episode(
        self,
        record: EpisodeLogRowV1,
        timing: Optional[EpisodeTimingRecord] = None,
        technical_failure: Optional[Dict[str, Any]] = None
    ) -> None:
        """Logs episode metric event and records tabular row."""
        if not isinstance(record, EpisodeLogRowV1):
            raise TypeError(
                f"TrackingBackend accepts EpisodeLogRowV1 only, got {type(record).__name__}."
            )
        if not self.wandb_run or self.sync_status == WandbSyncStatus.FAILED:
            return

        event = record.to_wandb_event(timing)
        table_row = record.to_wandb_table_row(timing)
        step_idx = record.episode_index

        try:
            import wandb
            # Step axis is logical episode index
            self.wandb_run.log(event, step=step_idx)
            self.table_data.append(table_row)
        except Exception as e:
            self.sync_status = WandbSyncStatus.FAILED
            sanitized_err = sanitize_error_message(e)
            print(f"[WARNING] W&B episode logging failed (non-fatal): {sanitized_err}")

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
                    if k in ("episode_return", "mean_episode_return"):
                        self.wandb_run.summary[f"diagnostic/{k}"] = v
                    else:
                        self.wandb_run.summary[f"metrics/overall/{k}"] = v

            # 3. Mirror per-tier metrics
            tier_metrics = summary_payload.get("tier_metrics", {})
            for t_name, card in tier_metrics.items():
                for k, v in card.items():
                    if v is not None:
                        if k in ("episode_return", "mean_episode_return"):
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
            sanitized_err = sanitize_error_message(e)
            print(f"[WARNING] W&B summary logging failed (non-fatal): {sanitized_err}")

    def finish(self, status: RunStatus) -> Dict[str, Any]:
        """Finishes the W&B run cleanly, restores environment, and returns sync metadata."""
        if not self.wandb_run:
            self._restore_env()
            return {
                "mode": self.mode.value,
                "sync_status": self.sync_status.value,
                "wandb_run_id": None,
                "wandb_run_url": None,
                "configured_project": self.configured_project,
                "configured_entity": self.configured_entity,
                "resolved_project": self.resolved_project,
                "resolved_entity": self.resolved_entity
            }

        try:
            import wandb
            exit_code = 0 if status == RunStatus.COMPLETE else 1
            self.wandb_run.finish(exit_code=exit_code)
            if self.sync_status != WandbSyncStatus.FAILED:
                if self.mode == WandbMode.OFFLINE:
                    self.sync_status = WandbSyncStatus.OFFLINE
                else:
                    self.sync_status = WandbSyncStatus.SYNCED
            print(f"[WANDB] Run finalized: {self.run_id} ({self.sync_status.value})")
        except Exception as e:
            self.sync_status = WandbSyncStatus.FAILED
            sanitized_err = sanitize_error_message(e)
            print(f"[WARNING] W&B run finish failed (non-fatal): {sanitized_err}")
        finally:
            self._restore_env()

        return {
            "mode": self.mode.value,
            "sync_status": self.sync_status.value,
            "wandb_run_id": self.run_id,
            "wandb_run_url": self.run_url,
            "configured_project": self.configured_project,
            "configured_entity": self.configured_entity,
            "resolved_project": self.resolved_project,
            "resolved_entity": self.resolved_entity
        }

    def fail(self, error_message: str) -> Dict[str, Any]:
        """Marks W&B backend run as failed and restores environment."""
        self.sync_status = WandbSyncStatus.FAILED
        if self.wandb_run:
            try:
                self.wandb_run.finish(exit_code=1)
            except Exception:
                pass
        self._restore_env()
        sanitized_msg = sanitize_error_message(error_message)
        return {
            "mode": self.mode.value,
            "sync_status": self.sync_status.value,
            "wandb_run_id": self.run_id,
            "error": sanitized_msg
        }
