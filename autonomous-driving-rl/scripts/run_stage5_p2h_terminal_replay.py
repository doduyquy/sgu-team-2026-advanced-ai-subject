"""Run the preregistered P2-H terminal-aware critic replay A/B, TRAIN only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import SACPolicy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_stage5_p2_experiment as p2
from src.platform.episode import TerminalReason
from src.platform.task_context_producer import REFERENCE_TASK_PRODUCER
from src.training.stage5.actor_activation_ablation import ActorOnlyLeakyReLUPolicy, activation_inventory
from src.training.stage5.checkpoint import save_sac_checkpoint, verify_checkpoint_integrity
from src.training.stage5.observation_encoder import Stage5ObservationEncoder
from src.training.stage5.terminal_aware_replay import TerminalAwareSampler

PROTOCOL_PATH = ROOT / "results/stage5/p2h_terminal_replay_seed101_20261121/protocol.json"
RUN_ID = "s5_p2h_terminal_replay_seed101_20261121"
TRAIN_STEPS = 10_000
CHECKPOINTS = (2_000, 5_000, 10_000)
KIND_NONE, KIND_SAFETY, KIND_CLEAN, KIND_TIMEOUT = 0, 1, 2, 3
SAFETY_REASONS = {reason.value for reason in (
    TerminalReason.CRASH_HUMAN, TerminalReason.CRASH_VEHICLE, TerminalReason.CRASH_OBJECT,
    TerminalReason.CRASH_BUILDING, TerminalReason.CRASH_SIDEWALK, TerminalReason.OUT_OF_ROAD,
)}


class P2HInstrumentedSAC(p2.P2InstrumentedSAC):
    """P2 SAC whose only B-arm difference is critic minibatch sampling."""

    def __init__(self, *args: Any, p2h_sampler_config: dict | None = None, **kwargs: Any):
        self.p2h_sampler_config = p2h_sampler_config
        self.terminal_sampler = TerminalAwareSampler(**p2h_sampler_config) if p2h_sampler_config else None
        self.terminal_kinds = np.zeros(100_000, dtype=np.int8)
        self.priority_values = np.ones(100_000, dtype=np.float32)
        self.priority_updates = 0
        self.last_sampling_stats: dict[str, Any] = {}
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + [
            "terminal_sampler", "terminal_kinds", "priority_values", "last_sampling_stats", "p2h_sampler_config",
        ]

    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos) -> None:
        old_pos = int(replay_buffer.pos)
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)
        info = infos[0]
        done = bool(np.asarray(dones).reshape(-1)[0])
        timeout = bool(info.get("TimeLimit.truncated", False))
        if not done:
            kind = KIND_NONE
        elif timeout:
            kind = KIND_TIMEOUT
        else:
            outcome = info.get("outcome") or {}
            reason = str(outcome.get("primary_reason", info.get("primary_reason", "")))
            if bool(outcome.get("clean_success", info.get("clean_success", False))) and reason == TerminalReason.SUCCESS.value:
                kind = KIND_CLEAN
            elif reason in SAFETY_REASONS:
                kind = KIND_SAFETY
            else:
                raise RuntimeError(f"Unclassified non-timeout terminal cannot enter P2-H terminal replay: {reason!r}")
        self.terminal_kinds[old_pos] = kind
        size = int(replay_buffer.size())
        if size:
            valid = self.priority_values[:size]
            current_max = float(np.max(valid)) if len(valid) else 1.0
            self.priority_values[old_pos] = max(1.0, min(current_max, 10.0))

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        if not hasattr(self, "_p2_gradient_sq"):
            self.install_gradient_hooks()
        self._p2_gradient_sq = {"actor": 0.0, "critic": 0.0, "entropy": 0.0}
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)
        ent_coef_losses, ent_coefs, actor_losses, critic_losses = [], [], [], []
        cumulative = {"critic_draws": 0, "terminal_draws": 0, "safety_terminal_draws": 0,
                      "clean_terminal_draws": 0, "timeout_draws": 0, "terminal_candidates_seen": 0,
                      "ess_sum": 0.0, "priority_updates": 0}

        for gradient_step in range(gradient_steps):
            rb = self.replay_buffer
            size = int(rb.size())
            if size < batch_size:
                continue
            if self.terminal_sampler is None:
                # Exact standard SB3 uniform batch is shared by critic, actor and alpha.
                actor_data = rb.sample(batch_size, env=self._vec_normalize_env)
                critic_data = actor_data
                weights = th.ones((batch_size, 1), device=self.device)
            else:
                true_terminal = np.asarray(rb.dones[:size, 0], dtype=bool)
                if getattr(rb, "handle_timeout_termination", False):
                    true_terminal &= ~np.asarray(rb.timeouts[:size, 0], dtype=bool)
                kinds = self.terminal_kinds[:size]
                draws = self.terminal_sampler.sample(
                    size=batch_size, priorities=self.priority_values[:size], true_terminal=true_terminal,
                    kinds=kinds, optimizer_update=int(self._n_updates),
                )
                critic_data = rb._get_samples(draws.indices, env=self._vec_normalize_env)
                actor_data = rb.sample(batch_size, env=self._vec_normalize_env)
                weights = th.as_tensor(draws.normalized_importance_weights, device=self.device).reshape(-1, 1)
                sampled_kinds = kinds[draws.indices]
                cumulative["critic_draws"] += batch_size
                cumulative["terminal_draws"] += int(draws.terminal_mask.sum())
                cumulative["safety_terminal_draws"] += int(np.sum(draws.terminal_mask & (sampled_kinds == KIND_SAFETY)))
                cumulative["clean_terminal_draws"] += int(np.sum(draws.terminal_mask & (sampled_kinds == KIND_CLEAN)))
                cumulative["timeout_draws"] += int(np.sum(sampled_kinds == KIND_TIMEOUT))
                cumulative["terminal_candidates_seen"] = max(cumulative["terminal_candidates_seen"], draws.terminal_candidates)
                cumulative["ess_sum"] += draws.effective_sample_size

            if self.use_sde:
                self.actor.reset_noise()
            actions_pi, log_prob = self.actor.action_log_prob(actor_data.observations)
            log_prob = log_prob.reshape(-1, 1)
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(float(ent_coef_loss.item()))
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(float(ent_coef.item()))
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad(); ent_coef_loss.backward(); self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(critic_data.next_observations)
                target_qs = th.cat(self.critic_target(critic_data.next_observations, next_actions), dim=1)
                target_qs, _ = th.min(target_qs, dim=1, keepdim=True)
                target_qs = target_qs - ent_coef * next_log_prob.reshape(-1, 1)
                discounts = critic_data.discounts if critic_data.discounts is not None else self.gamma
                target_q = critic_data.rewards + (1 - critic_data.dones) * discounts * target_qs
            current_qs = self.critic(critic_data.observations, critic_data.actions)
            td_errors = [target_q - current for current in current_qs]
            if self.terminal_sampler is None:
                critic_loss = 0.5 * sum(F.mse_loss(current, target_q) for current in current_qs)
            else:
                per_sample = sum(error.square() for error in td_errors)
                critic_loss = 0.5 * (weights * per_sample).mean()
                abs_td = th.stack([error.detach().abs().reshape(-1) for error in td_errors]).max(dim=0).values.cpu().numpy()
                indices = draws.indices
                capped = np.minimum(abs_td, self.terminal_sampler.td_error_cap).astype(np.float32)
                self.priority_values[indices] = np.maximum(capped, 0.0)
                self.priority_updates += len(indices)
                cumulative["priority_updates"] += len(indices)

            self.critic.optimizer.zero_grad(); critic_loss.backward(); self.critic.optimizer.step()
            q_pi = th.cat(self.critic(actor_data.observations, actions_pi), dim=1)
            min_q_pi, _ = th.min(q_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_q_pi).mean()
            self.actor.optimizer.zero_grad(); actor_loss.backward(); self.actor.optimizer.step()
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
            critic_losses.append(float(critic_loss.item())); actor_losses.append(float(actor_loss.item()))

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if ent_coefs:
            self.logger.record("train/ent_coef", np.mean(ent_coefs))
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/critic_loss", np.mean(critic_losses))
        if ent_coef_losses:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        self.last_sampling_stats = {
            **cumulative, "mean_ess_per_update": cumulative["ess_sum"] / max(gradient_steps, 1),
            "cumulative_sampler": self.terminal_sampler.state_dict() if self.terminal_sampler else None,
        }
        callback = p2._AFTER_OPTIMIZER.get(self)
        if callback is not None:
            callback.after_optimizer_update()


class P2HTrainingCallback(p2.P2TrainingCallback):
    def _on_training_start(self) -> None:
        super()._on_training_start()
        self.sampling_file = (self.arm_dir / "critic_sampling_telemetry.jsonl").open("x", encoding="utf-8", newline="\n")
        self.terminal_sampler_rows = 0

    def after_optimizer_update(self) -> None:
        super().after_optimizer_update()
        if self.arm == "B" and self.terminal_sampler_rows % 100 == 0:
            row = {"training_step": int(self.model.num_timesteps), "optimizer_updates": int(self.model._n_updates),
                   **self.model.last_sampling_stats}
            self.sampling_file.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            self.sampling_file.flush()
        self.terminal_sampler_rows += 1

    def save_checkpoint(self, step: int) -> None:
        super().save_checkpoint(step)
        if self.arm == "B":
            rb = self.model.replay_buffer
            size = int(rb.size())
            true_terminal = np.asarray(rb.dones[:size, 0], dtype=bool)
            if getattr(rb, "handle_timeout_termination", False):
                timeouts = np.asarray(rb.timeouts[:size, 0], dtype=bool)
                true_terminal &= ~timeouts
            else:
                timeouts = np.zeros(size, dtype=bool)
            kinds = self.model.terminal_kinds[:size]
            counts = {"none_or_nonterminal": int(np.sum(kinds == KIND_NONE)),
                      "safety_terminal": int(np.sum(kinds == KIND_SAFETY)),
                      "clean_terminal": int(np.sum(kinds == KIND_CLEAN)),
                      "timeout_truncation": int(np.sum(kinds == KIND_TIMEOUT)),
                      "true_terminal_mask_count": int(np.sum(true_terminal))}
            entry = {"step": step, "replay_size": size, "counts": counts,
                     "sampled": self.model.last_sampling_stats if self.arm == "B" else None,
                     "priority_min": float(self.model.priority_values[:size].min()) if size else None,
                     "priority_max": float(self.model.priority_values[:size].max()) if size else None}
            path = self.arm_dir / "terminal_inventory.json"
            rows = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
            rows.append(entry)
            path.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            if step == TRAIN_STEPS:
                self.checkpoints[-1]["terminal_inventory"] = entry
                (self.arm_dir / "checkpoint_telemetry.json").write_text(
                    json.dumps(self.checkpoints, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                np.savez_compressed(self.arm_dir / "terminal_replay_index.npz", kinds=kinds,
                                    true_terminal=true_terminal, timeouts=timeouts,
                                    priorities=self.model.priority_values[:size])

    def _on_training_end(self) -> None:
        super()._on_training_end()
        handle = getattr(self, "sampling_file", None)
        if handle is not None and not handle.closed:
            handle.flush(); handle.close()


def read_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    recorded = protocol.pop("protocol_sha256", None)
    if recorded != p2.canonical_hash(protocol) or protocol.get("status") != "PRE_REGISTERED":
        raise ValueError("P2-H protocol hash/status validation failed")
    protocol["protocol_sha256"] = recorded
    if protocol["training"]["max_environment_steps_per_arm"] != TRAIN_STEPS:
        raise ValueError("P2-H budget mismatch")
    if protocol["geometry"] != p2.read_protocol()["geometry"]:
        raise ValueError("P2-H geometry schedule differs from the locked P2 schedule")
    for relative, expected in protocol["source_files_sha256"].items():
        if p2.sha256(ROOT / relative) != expected:
            raise ValueError(f"P2-H source fingerprint mismatch: {relative}")
    return protocol


def sampler_kwargs(protocol: dict[str, Any]) -> dict[str, Any]:
    spec = protocol["terminal_replay"]
    beta = spec["importance_beta"]
    return {"mixture": float(spec["terminal_priority_mixture_probability"]),
            "priority_alpha": float(spec["priority_alpha"]),
            "priority_beta_start": float(beta["start"]),
            "priority_beta_end": float(beta["end"]),
            "priority_beta_steps": int(beta["optimizer_updates"]),
            "td_error_cap": float(spec["td_error_cap"]),
            "epsilon": float(spec["priority_epsilon"]),
            "seed": int(spec["rng_seed"])}


def run_arm(protocol: dict[str, Any], arm: str, initial: dict[str, Any], run_root: Path) -> dict[str, Any]:
    arm_dir = run_root / f"arm_{arm}"
    if arm_dir.exists():
        raise FileExistsError(f"Refusing to overwrite P2-H artifacts: {arm_dir}")
    env, _ = p2.make_env(protocol, arm, training=True)
    callback = None
    try:
        p2.seed_everything(101)
        kwargs = p2.model_kwargs(protocol, "standard")
        kwargs["policy"] = ActorOnlyLeakyReLUPolicy
        model = P2HInstrumentedSAC(env=env, p2h_sampler_config=sampler_kwargs(protocol) if arm == "B" else None, **kwargs)
        if not bool(getattr(model.replay_buffer, "handle_timeout_termination", False)):
            raise RuntimeError("P2-H requires SB3 replay timeout masking/bootstrap semantics")
        p2.load_initial_into(model, initial["state"])
        model.install_gradient_hooks()
        fingerprint = p2.state_fingerprint(model)
        if fingerprint != initial["manifest"]["policy_state_fingerprint"]:
            raise RuntimeError("P2-H arm did not receive identical fresh initial tensors and entropy state")
        if activation_inventory(model.policy)["actor_id"] != "actor_leaky_relu_001_v1" or not activation_inventory(model.policy)["critic_is_relu_only"]:
            raise RuntimeError("P2-H actor/critic activation contract mismatch")
        callback = P2HTrainingCallback(arm_dir, arm, protocol, fingerprint)
        callback.model = model
        p2._AFTER_OPTIMIZER[model] = callback
        callback.save_checkpoint(0)
        model.learn(total_timesteps=TRAIN_STEPS, callback=callback, reset_num_timesteps=True, log_interval=10)
        if int(model.num_timesteps) != TRAIN_STEPS or [row["step"] for row in callback.checkpoints] != [0, *CHECKPOINTS]:
            raise RuntimeError("P2-H arm did not reach its registered step/checkpoint schedule")
        if model.action_mismatch_count:
            raise RuntimeError(f"P2-H action/replay mismatch count is {model.action_mismatch_count}")
        summary = callback.summary()
        summary.update({"protocol_sha256": protocol["protocol_sha256"], "arm": arm,
                       "critic_sampling": "uniform" if arm == "A" else "terminal_aware_prioritized_v1",
                       "actor_activation_id": "actor_leaky_relu_001_v1", "critic_activation": "ReLU",
                       "initial_policy_fingerprint": fingerprint, "action_mismatch_count": model.action_mismatch_count,
                       "final_terminal_counts": callback.checkpoints[-1].get("terminal_inventory")})
        (arm_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"[{arm}] COMPLETE steps={summary['actual_environment_steps']} updates={summary['optimizer_updates']} episodes={summary['completed_episodes']}", flush=True)
        return summary
    finally:
        if callback is not None:
            callback._on_training_end()
        env.close()


def create_shared_initial(protocol: dict[str, Any], run_root: Path) -> dict[str, Any]:
    target = run_root / "shared_initial"; target.mkdir(parents=True, exist_ok=False)
    env, _ = p2.make_env(protocol, "INITIAL", training=True)
    try:
        p2.seed_everything(101)
        kwargs = p2.model_kwargs(protocol, "standard"); kwargs["policy"] = ActorOnlyLeakyReLUPolicy
        model = P2HInstrumentedSAC(env=env, **kwargs)
        if not bool(getattr(model.replay_buffer, "handle_timeout_termination", False)):
            raise RuntimeError("P2-H initial model replay buffer lacks timeout semantics")
        state = {"policy": {key: value.detach().cpu().clone() for key, value in model.policy.state_dict().items()},
                 "log_ent_coef": model.log_ent_coef.detach().cpu().clone() if model.log_ent_coef is not None else None}
        path = target / "initial_state.pt"; th.save(state, path)
        manifest = {"path": path.relative_to(ROOT).as_posix(), "sha256": p2.sha256(path),
                    "policy_state_fingerprint": p2.state_fingerprint(model), "source_commit": p2.git_head(),
                    "alpha_initial": float(model.log_ent_coef.detach().exp().cpu()),
                    "initialization": "fresh shared LeakyReLU actor-only / ReLU critics; optimizers and replay independent"}
        (target / "initialization.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return {"state": state, "manifest": manifest}
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("A", "B", "ALL"), default="ALL")
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args()
    protocol = read_protocol()
    run_root = ROOT / "runs/stage5_p2h" / args.run_id
    if run_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing P2-H run: {run_root}")
    run_root.mkdir(parents=True)
    initial = create_shared_initial(protocol, run_root)
    (run_root / "run_manifest.json").write_text(json.dumps({"run_id": args.run_id,
        "protocol_sha256": protocol["protocol_sha256"], "source_commit": p2.git_head(),
        "initialization": initial["manifest"], "arms_requested": ["A", "B"],
        "official_validation_test_accessed": False, "benchmark_eligible": False, "status": "RUNNING"}, indent=2) + "\n")
    completed = []
    try:
        for arm in (("A", "B") if args.arm == "ALL" else (args.arm,)):
            completed.append(run_arm(protocol, arm, initial, run_root))
            (run_root / "run_manifest.json").write_text(json.dumps({"run_id": args.run_id,
                "protocol_sha256": protocol["protocol_sha256"], "source_commit": p2.git_head(),
                "initialization": initial["manifest"], "arms_requested": ["A", "B"], "completed_arms": completed,
                "status": "RUNNING", "official_validation_test_accessed": False, "benchmark_eligible": False}, indent=2) + "\n")
    except BaseException as exc:
        (run_root / "run_manifest.json").write_text(json.dumps({"run_id": args.run_id,
            "protocol_sha256": protocol["protocol_sha256"], "completed_arms": completed,
            "status": "FAILED_OR_INTERRUPTED", "exception_type": type(exc).__name__, "exception": str(exc),
            "official_validation_test_accessed": False, "benchmark_eligible": False}, indent=2) + "\n")
        raise
    (run_root / "run_manifest.json").write_text(json.dumps({"run_id": args.run_id,
        "protocol_sha256": protocol["protocol_sha256"], "source_commit": p2.git_head(),
        "initialization": initial["manifest"], "completed_arms": completed, "status": "COMPLETE" if len(completed) == 2 else "PARTIAL",
        "official_validation_test_accessed": False, "benchmark_eligible": False}, indent=2) + "\n")


if __name__ == "__main__":
    main()
