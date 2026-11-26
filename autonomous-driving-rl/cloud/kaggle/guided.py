"""Champion-guided Easy replay prefill and bounded online exploration."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from functools import partial
from pathlib import Path
import time

import numpy as np

from cloud.kaggle import learning_hooks, runtime
from cloud.kaggle.common import read, sha
from cloud.kaggle.replay import DTYPES, admit, check_snapshot, validate
from src.platform.task_context_producer import TaskContextLocalizationError

NOISE_STD = 0.04
BEHAVIOR = "champion_guided_v1"
BEHAVIOR_CONFIG = {"kind": "learner deterministic plus clipped Gaussian Box2 noise", "std": NOISE_STD}


def progress_band(progress):
    return "early" if progress < 0.25 else "middle" if progress < 0.75 else "late"


def prefill(root, identity):
    manifest = read(root / "GUIDED.json")
    snapshot = manifest["snapshot_sha"]
    version = check_snapshot(root / "snapshot", snapshot, identity)
    if version != 0 or read(root / "snapshot/SNAPSHOT.json")["policy_sha"] != identity["warm_source"]:
        raise ValueError("Guided replay must come from the approved initial champion")
    ledger, arrays, counts, seen = {}, [], Counter(), set()
    families = {row["geometry_id"]: row["sequence"] for row in runtime.pool("easy")}
    outcomes = Counter()
    for index, digest in enumerate(manifest["chunks"]):
        path = root / f"chunk-{index:02}"
        data, key = validate(path, digest, identity, ledger, version, (snapshot, version), guided=True)
        meta = read(path / "CHUNK.json")["guided"]
        case = (meta["geometry"], meta["seed"])
        if data is None or case in seen:
            raise ValueError("Duplicate guided episode")
        seen.add(case)
        ledger[key] = digest
        arrays.append(data)
        counts.update(progress_band(p) for p in data["progress"])
        outcomes[meta["outcome"]] += 1
    if not arrays:
        raise ValueError("No approved guided episodes")
    data = {key: np.concatenate([row[key] for row in arrays]) for key in DTYPES}
    count = len(data["reward"])
    dropped = count % identity["workers"]
    if dropped:
        # Remove only nonterminal transitions so the complete episodes retain terminal support.
        indices = np.flatnonzero(~(data["terminated"] | data["truncated"]))[:dropped]
        if len(indices) != dropped:
            raise ValueError("Cannot align guided buffer without dropping terminal support")
        progress = np.concatenate([row["progress"] for row in arrays])
        for index in indices:
            counts[progress_band(progress[index])] -= 1
        data = {key: np.delete(values, indices, axis=0) for key, values in data.items()}
    if count - dropped < 1024 or len({families[g] for g, _ in seen}) != 3 or not counts["late"] or not outcomes["SUCCESS"]:
        raise ValueError("Guided prefill needs all three families and real late-route and success support")
    return data, {"manifest_sha256": sha(root / "GUIDED.json"), "snapshot_sha256": snapshot,
                  "rows": count - dropped, "alignment_dropped": dropped, "progress_bands": dict(counts),
                  "cases": sorted(seen), "outcomes": dict(outcomes), "ledger": ledger,
                  "behavior": "frozen deterministic champion"}


class CampaignScheduledEnv(runtime.ScheduledEnv):
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.last_valid_obs, self.last_valid_info = obs, info
        return obs, info

    def step(self, action):
        try:
            outcome = super().step(action)
        except TaskContextLocalizationError:
            # SB3 requires a shaped transition to reset this worker. The learner
            # discards the entire vector batch, including this placeholder.
            info = {**self.last_valid_info, "primary_reason": "TASK_CONTEXT_LOCALIZATION",
                    "clean_success": False, "route_completion": self.last_valid_info.get("route_completion", 0.0),
                    "localization_discard": True}
            return self.last_valid_obs, 0.0, False, True, info
        self.last_valid_obs, _, _, _, self.last_valid_info = outcome
        return outcome


def campaign_vec(workers, phase, cursors):
    factories = [partial(CampaignScheduledEnv, phase, i, cursor) for i, cursor in enumerate(cursors)]
    return runtime.SubprocVecEnv(factories, start_method="spawn")



class GuidedSAC(runtime.TimedSAC):
    actor_lr_scale = 1.0
    rollout_axis = None

    def _update_learning_rate(self, optimizers):
        super()._update_learning_rate(optimizers)
        for group in self.actor.optimizer.param_groups:
            group["lr"] *= self.actor_lr_scale

    def _store_transition(self, replay_buffer, *args, **kwargs):
        if getattr(self, "localization_guard", False):
            _, new_obs, _, _, infos = args
            if any(info.get("localization_discard") for info in infos):
                if self._vec_normalize_env is not None:
                    raise ValueError("Localization discard requires the unnormalized campaign observation contract")
                self.localization_invalid_rows += self.n_envs
                self._last_obs = new_obs
                return
        protected = getattr(self, "protected_replay_slots", 0)
        if protected and replay_buffer.full and replay_buffer.pos < protected:
            replay_buffer.pos = protected
        return super()._store_transition(replay_buffer, *args, **kwargs)

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        action, _ = self.predict(self._last_obs, deterministic=True)
        scaled = self.policy.scale_action(action)
        scaled = np.clip(scaled + np.random.normal(0, NOISE_STD, scaled.shape), -1, 1).astype(np.float32)
        if getattr(self, "rollout_axis", None) is not None:
            anchored, _ = self.rollout_anchor.predict(self._last_obs, deterministic=True)
            scaled[:, self.rollout_axis] = self.policy.scale_action(anchored)[:, self.rollout_axis]
        if getattr(self, "traffic_guidance", False):
            from cloud.kaggle.traffic_guidance import traffic_follow
            teacher = getattr(self, "traffic_teacher_policy", None)
            if teacher is None:
                guided = traffic_follow(self._last_obs, scaled)
                self.traffic_guided_decisions += int(np.count_nonzero(guided[:, 1] != scaled[:, 1]))
                scaled = guided
            else:
                mask = self.teacher_episode_mask()
                self.teacher_mask = mask
                if mask.any():
                    baseline, _ = teacher.predict(self._last_obs[mask], deterministic=True)
                    nominal = self.policy.scale_action(baseline)
                    guided = traffic_follow(self._last_obs[mask], nominal)
                    self.traffic_guided_decisions += int(np.count_nonzero(guided[:, 1] != nominal[:, 1]))
                    scaled[mask] = guided
        return self.policy.unscale_action(scaled), scaled

    def teacher_episode_mask(self):
        for worker, info in enumerate(self.env.reset_infos):
            episode = info["episode"]
            previous = self.teacher_episodes.get(worker)
            if previous is None or episode != previous[0]:
                cutoff = getattr(self, "teacher_stop_at", None)
                before_cutoff = time.monotonic() < cutoff if cutoff is not None else self.num_timesteps < self.teacher_cutoff
                guided = (info["tier"] == "Medium" and before_cutoff
                          and (episode + worker) % 2 == 0)
                self.teacher_episodes[worker] = (episode, guided)
        return np.asarray([self.teacher_episodes[w][1] for w in range(self.n_envs)], dtype=bool)


REPLAY_FIELDS = ("observations", "next_observations", "actions", "rewards", "dones", "timeouts")


def prefix_sha(buffer, slots):
    digest = hashlib.sha256()
    for field in REPLAY_FIELDS:
        digest.update(memoryview(getattr(buffer, field)[:slots]).cast("B"))
    return digest.hexdigest()


def grow_replay(model, capacity, live_rows):
    old = model.replay_buffer
    if (type(old) is not runtime.ReplayBuffer or old.full or old.optimize_memory_usage
            or not old.handle_timeout_termination or old.pos * old.n_envs != live_rows
            or capacity % old.n_envs or capacity < old.buffer_size * old.n_envs):
        raise ValueError("Guided resume requires intact replay slots and compatible capacity")
    if capacity == old.buffer_size * old.n_envs:
        return
    new = runtime.ReplayBuffer(capacity, old.observation_space, old.action_space, device=model.device,
                               n_envs=old.n_envs, handle_timeout_termination=True)
    for field in REPLAY_FIELDS:
        getattr(new, field)[:old.pos] = getattr(old, field)[:old.pos]
    new.pos = old.pos
    model.buffer_size, model.replay_buffer = capacity, new


def train(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--guided-prefill", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--parent-sha")
    parser.add_argument("--guided-gradient-steps", type=int, choices=(1,))
    parser.add_argument("--guided-actor-lr-scale", type=float, choices=(0.25,))
    parser.add_argument("--phase", default="easy", choices=("easy", "medium", "hard", "extreme", "mixed"))
    parser.add_argument("--guided-curriculum-from", choices=("easy", "medium", "hard"))
    parser.add_argument("--guided-rollout-axis", choices=("steering", "throttle_brake"))
    parser.add_argument("--traffic-guidance", action="store_true")
    parser.add_argument("--traffic-teacher", type=Path)
    parser.add_argument("--traffic-teacher-sha")
    parser.add_argument("--autonomous-traffic", action="store_true")
    parser.add_argument("--bounded-guided-replay", action="store_true")
    parser.add_argument("--timeboxed", action="store_true")
    parser.add_argument("--localization-guard", action="store_true")
    parser.add_argument("--teacher-stop-at", type=float)
    parser.add_argument("--capacity", type=int, default=runtime.RECIPE["preflight"]["capacity"])
    parser.add_argument("--transitions", required=True, type=int)
    config, learner_args = parser.parse_known_args(args)
    if bool(config.guided_prefill) == bool(config.resume) or bool(config.resume) != bool(config.parent_sha):
        parser.error("Specify guided prefill or a verified guided resume")
    if "--measure-optimizer" not in learner_args:
        parser.error("Guided training requires the metered GuidedSAC learner")
    if (config.guided_gradient_steps or config.guided_actor_lr_scale) and not config.resume:
        parser.error("Update profile changes require guided full-state resume")
    learner_args.extend(["--capacity", str(config.capacity), "--transitions", str(config.transitions), "--phase", config.phase])
    if (config.guided_curriculum_from or config.guided_rollout_axis) and not config.resume:
        parser.error("Curriculum transition and axis retention require a verified guided parent")
    if config.traffic_guidance and (not config.resume or config.phase != "medium" or config.guided_rollout_axis):
        parser.error("Traffic guidance requires continuation in the Medium phase")
    if (bool(config.traffic_teacher) != bool(config.traffic_teacher_sha)
            or (config.traffic_teacher and not config.traffic_guidance)
            or (config.autonomous_traffic and config.traffic_guidance)):
        parser.error("Use a verified teacher with traffic guidance, or retire guidance explicitly")
    if config.guided_curriculum_from and (
            {"easy": "medium", "medium": "hard", "hard": "extreme"}[config.guided_curriculum_from] != config.phase
            or config.guided_rollout_axis):
        parser.error("Curriculum transition requires the next phase and unassisted collection")
    if config.localization_guard and not config.timeboxed:
        parser.error("Localization discard is a timeboxed TRAIN-only behavior")
    if config.timeboxed != config.bounded_guided_replay:
        parser.error("Timeboxed training requires bounded guided replay")
    if config.bounded_guided_replay and not config.resume:
        parser.error("Bounded replay requires verified GuidedSAC full state")
    parent = None
    parent_gradient_steps = -1
    parent_actor_lr_scale = 1.0
    traffic_profile = "traffic_follow_v1"
    parent_traffic = False
    parent_teacher_sha = None
    if config.resume:
        learner_args.extend(["--resume", str(config.resume), "--parent-sha", config.parent_sha])
        parent = runtime.check_state(config.resume, config.parent_sha)
        if parent["identity"]["phase"] != (config.guided_curriculum_from or config.phase):
            raise ValueError("Phase must match the parent or use the approved curriculum transition")
        parent_gradient_steps = parent["identity"].get("guided_gradient_steps", -1)
        parent_actor_lr_scale = parent["identity"].get("guided_actor_lr_scale", 1.0)
        parent_traffic = parent["identity"].get("traffic_guidance") == traffic_profile
        parent_teacher_sha = parent["identity"].get("traffic_teacher_sha")
        parent_behavior = {**BEHAVIOR_CONFIG, **({"traffic_guidance": traffic_profile} if parent_traffic else {})}
        if parent_teacher_sha:
            parent_behavior["traffic_teacher_sha"] = parent_teacher_sha
            if not config.autonomous_traffic and config.traffic_teacher_sha != parent_teacher_sha:
                raise ValueError("Teacher continuation requires the same teacher or explicit autonomous collection")
        if (parent.get("learning_mode") != BEHAVIOR or parent["identity"].get("online_behavior") != BEHAVIOR
                or read(config.resume / "metrics.json")["online_behavior"] != parent_behavior
                or not parent["ledger"] or parent["external_admitted"] <= 0
                or parent_gradient_steps not in (-1, 1)
                or parent_actor_lr_scale not in (1.0, 0.25)
                or (not config.bounded_guided_replay and (parent["identity"].get("bounded_guided_replay")
                    or parent["external_admitted"] + parent["transitions"] + config.transitions > config.capacity))):
            raise ValueError("Resume must retain guided behavior and all prior replay throughout the budget")
    model_for, base_sac, telemetry, save = runtime.model_for, runtime.TimedSAC, runtime.Telemetry, runtime.save_state
    recipe_identity, make_vec = runtime.recipe_identity, runtime.make_vec
    online = Counter()
    tiers = Counter()
    teacher_counts = Counter()

    def parent_identity(phase, workers, capacity, learning_starts, warm):
        identity = recipe_identity(config.guided_curriculum_from or phase, workers, parent["identity"]["capacity"], learning_starts, warm)
        if parent_gradient_steps == 1:
            identity["guided_gradient_steps"] = 1
        if parent_actor_lr_scale != 1.0:
            identity["guided_actor_lr_scale"] = parent_actor_lr_scale
        if parent_traffic:
            identity["traffic_guidance"] = traffic_profile
        if parent_teacher_sha:
            identity["traffic_teacher_sha"] = parent_teacher_sha
        if parent["identity"].get("bounded_guided_replay"):
            identity["bounded_guided_replay"] = True
        if parent["identity"].get("localization_guard"):
            identity["localization_guard"] = True
        return {**identity, "online_behavior": BEHAVIOR}

    class ResumedGuidedSAC(GuidedSAC):
        def _excluded_save_params(self):
            return [*super()._excluded_save_params(), "rollout_anchor", "traffic_teacher_policy"]

        def train(self, gradient_steps, batch_size=64):
            super().train(gradient_steps, batch_size)
            if config.timeboxed:
                transitions = self.num_timesteps - parent["transitions"]
                if transitions == self.n_envs or (transitions > 0 and transitions % 50000 == 0):
                    print(f"TRAINING_ACTIVE transitions={transitions} updates={self._n_updates - parent['updates']}", flush=True)

        def load_replay_buffer(self, path, truncate_last_traj=True):
            super().load_replay_buffer(path, truncate_last_traj=truncate_last_traj)
            if self.device.type == "cuda":
                optimizers = (self.actor.optimizer, self.critic.optimizer, self.ent_coef_optimizer)
                actor_device = next(self.actor.parameters()).device
                if (self.replay_buffer.device.type != "cuda" or self.log_ent_coef.device != actor_device
                        or any(not optimizer.state for optimizer in optimizers)
                        or any(moment.device != actor_device for optimizer in optimizers
                               for state in optimizer.state.values() for key, moment in state.items()
                               if key in {"exp_avg", "exp_avg_sq"})):
                    raise ValueError("Guided GPU resume did not move replay sampling, alpha and Adam moments")
            evidence = self.guided_prefill
            if (self.buffer_size != parent["identity"]["capacity"]
                    or self.replay_buffer.buffer_size * self.replay_buffer.n_envs != self.buffer_size
                    or self.actor_lr_scale != parent_actor_lr_scale
                    or self.gradient_steps != parent_gradient_steps or evidence["rows"] != parent["external_admitted"]
                    or evidence["ledger"] != parent["ledger"]):
                raise ValueError("Guided state differs from parent identity, behavior or provenance")
            self.localization_guard = config.localization_guard
            if parent["identity"].get("localization_guard") and not self.localization_guard:
                raise ValueError("Resume must retain the parent's localization handling")
            self.localization_invalid_rows = read(config.resume / "metrics.json").get("localization_invalid_rows", 0)
            old_bounded = parent["identity"].get("bounded_guided_replay", False)
            if old_bounded:
                if config.capacity != parent["identity"]["capacity"] or not config.bounded_guided_replay:
                    raise ValueError("A bounded replay must resume with the same capacity and preservation policy")
            else:
                grow_replay(self, config.capacity, evidence["rows"] + parent["transitions"])
            if config.bounded_guided_replay:
                replay = self.replay_buffer
                if evidence["rows"] % replay.n_envs or evidence["rows"] >= config.capacity:
                    raise ValueError("Guided replay and workers are not aligned")
                self.protected_replay_slots = evidence["rows"] // replay.n_envs
                self.protected_replay_sha = prefix_sha(replay, self.protected_replay_slots)
                if old_bounded and read(config.resume / "metrics.json")["protected_replay_sha"] != self.protected_replay_sha:
                    raise ValueError("Protected replay prefix no longer matches the parent")
            if config.guided_gradient_steps:
                if self.n_envs != 2:
                    raise ValueError("The reduced update profile requires two environments")
                self.gradient_steps = config.guided_gradient_steps
            self.actor_lr_scale = config.guided_actor_lr_scale or parent_actor_lr_scale
            self.traffic_guidance = not config.autonomous_traffic and (config.traffic_guidance or parent_traffic)
            self.traffic_guided_decisions = 0
            if config.traffic_teacher:
                from cloud.kaggle.common import verify
                verify(config.traffic_teacher, config.traffic_teacher_sha)
                self.traffic_teacher_policy = runtime.load_policy(config.traffic_teacher, device="cpu")
                self.teacher_cutoff = parent["transitions"] + config.transitions // 2
                self.teacher_episodes = {}
                self.teacher_stop_at = config.teacher_stop_at
            if config.guided_rollout_axis:
                self.rollout_anchor = runtime.load_policy(config.resume / "model.zip", self.device)
                self.rollout_axis = ("steering", "throttle_brake").index(config.guided_rollout_axis)

    def seeded(env, device, capacity, learning_starts, source=None, *, warm_model=None, warm_sha=None, metered=False):
        if learning_starts != 0 or not warm_model or not metered or source:
            raise ValueError("Guided prefill requires a fresh metered champion warm-start with zero random steps")
        identity = runtime.recipe_identity("easy", env.num_envs, capacity, 0, True)
        identity["warm_source"] = warm_sha
        data, evidence = prefill(config.guided_prefill, identity)
        if evidence["rows"] + config.transitions > capacity:
            raise ValueError("Pilot capacity would overwrite verified guided replay")
        model = model_for(env, device, capacity, learning_starts, warm_model=warm_model, warm_sha=warm_sha, metered=True)
        admit(model.replay_buffer, data)
        model.guided_prefill = evidence
        return model

    class GuidedTelemetry(telemetry):
        def _on_step(self):
            online.update(progress_band(info["route_completion"]) for info in self.locals["infos"])
            tiers.update(info["tier"] for info in self.locals["infos"])
            if config.traffic_teacher:
                for guided, done, info in zip(self.model.teacher_mask, self.locals["dones"], self.locals["infos"], strict=True):
                    if info["tier"] == "Medium":
                        mode = "teacher" if guided else "autonomous"
                        teacher_counts[f"{mode}_decisions"] += 1
                        if done:
                            teacher_counts[f"{mode}_{info['primary_reason']}"] += 1
            return super()._on_step()

    def save_guided(out, model, manifest, metrics):
        evidence = model.guided_prefill
        live = model.replay_buffer.size() * model.n_envs
        dropped = getattr(model, "localization_invalid_rows", 0)
        expected_live = evidence["rows"] + model.num_timesteps - dropped
        if config.bounded_guided_replay:
            online_rows = model.num_timesteps - dropped
            expected_live = min(expected_live, config.capacity)
            if prefix_sha(model.replay_buffer, model.protected_replay_slots) != model.protected_replay_sha:
                raise ValueError("Bounded replay replaced a guided row")
            metrics["protected_replay_sha"] = model.protected_replay_sha
            metrics["online_evicted_rows"] = evidence["rows"] + online_rows - live
            if config.localization_guard:
                metrics["localization_invalid_rows"] = dropped
        if live != expected_live:
            raise ValueError("Guided replay was overwritten or replay accounting diverged")
        manifest.update(external_admitted=evidence["rows"], ledger=evidence["ledger"], learning_mode=BEHAVIOR)
        manifest["identity"] = {**manifest["identity"], "online_behavior": BEHAVIOR, "capacity": config.capacity}
        manifest["identity"]["phase"] = config.phase
        if config.bounded_guided_replay:
            manifest["identity"]["bounded_guided_replay"] = True
        if config.localization_guard:
            manifest["identity"]["localization_guard"] = True
        if config.autonomous_traffic:
            manifest["identity"].pop("traffic_guidance", None)
            manifest["identity"].pop("traffic_teacher_sha", None)
        if config.guided_rollout_axis:
            manifest["identity"]["guided_rollout_axis"] = config.guided_rollout_axis
            metrics["guided_rollout_anchor_sha"] = parent["files"]["model.zip"]
        if config.guided_curriculum_from:
            metrics["curriculum_from"] = config.guided_curriculum_from
        if model.gradient_steps == 1:
            manifest["identity"]["guided_gradient_steps"] = 1
        if model.actor_lr_scale != 1.0:
            manifest["identity"]["guided_actor_lr_scale"] = model.actor_lr_scale
            metrics["guided_actor_lr"] = model.actor.optimizer.param_groups[0]["lr"]
        behavior = dict(BEHAVIOR_CONFIG)
        if getattr(model, "traffic_guidance", False):
            manifest["identity"]["traffic_guidance"] = traffic_profile
            behavior["traffic_guidance"] = traffic_profile
            metrics["traffic_guided_decisions"] = model.traffic_guided_decisions
        if config.traffic_teacher:
            manifest["identity"]["traffic_teacher_sha"] = config.traffic_teacher_sha
            behavior["traffic_teacher_sha"] = config.traffic_teacher_sha
            metrics["traffic_mixture"] = dict(teacher_counts)
        if config.guided_rollout_axis:
            behavior.update(anchored_axis=config.guided_rollout_axis, anchor_model_sha=parent["files"]["model.zip"])
        if config.timeboxed and "device" in metrics and metrics["updates"] > 0:
            metrics["termination_budget"] = "transitions" if metrics["transitions"] == config.transitions else "time"
            manifest["result"] = "COMPLETE"
        metrics.update(guided_prefill=evidence, online_progress_bands=dict(online), online_tier_decisions=dict(tiers),
                       online_behavior=behavior,
                       guided_live_rows=evidence["rows"], online_live_rows=live - evidence["rows"],
                       updates_per_total_real_transition=model._n_updates / (evidence["rows"] + model.num_timesteps - dropped))
        save(out, model, manifest, metrics)

    try:
        runtime.TimedSAC = ResumedGuidedSAC if parent else GuidedSAC
        runtime.model_for, runtime.Telemetry, runtime.save_state = seeded, GuidedTelemetry, save_guided
        if parent:
            runtime.recipe_identity = parent_identity
        if config.localization_guard:
            runtime.make_vec = campaign_vec
        learning_hooks.train(learner_args)
    finally:
        runtime.TimedSAC, runtime.model_for, runtime.Telemetry, runtime.save_state = base_sac, model_for, telemetry, save
        runtime.recipe_identity, runtime.make_vec = recipe_identity, make_vec


def main(argv=None):
    import sys

    args = sys.argv[1:] if argv is None else argv
    train(args[1:] if args[:1] == ["train"] else args)


if __name__ == "__main__":
    main()
