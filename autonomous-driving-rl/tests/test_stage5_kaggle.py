"""Local correctness regressions for the Stage 5 Kaggle campaign."""
import ast
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from cloud.kaggle import control
from cloud.kaggle import credentials
from cloud.kaggle import kernel_version
from cloud.kaggle.capacity import log_times
from cloud.kaggle.checkpoint import PARTS, export_checkpoint, restore_checkpoint
from cloud.kaggle.common import SOURCE_SHA, sha, write
from cloud.kaggle.credentials import dpapi
from cloud.kaggle.pilot import compare, easy_coverage, screen_candidates
from cloud.kaggle import learning_hooks
from cloud.kaggle.guided import GuidedSAC, NOISE_STD, progress_band
from cloud.kaggle import guided
from cloud.kaggle.replay import admit, check_snapshot, validate
from cloud.kaggle.runtime import check_state, load_policy, model_for, pool, restore_rng, save_state, schedule

class TinyEnv(gym.Env):
    observation_space = gym.spaces.Box(-2, 2, shape=(409,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_index = 0
        return np.zeros(409, dtype=np.float32), {}

    def step(self, action):
        self.step_index += 1
        return np.full(409, self.step_index, dtype=np.float32), float(action[0]), False, self.step_index == 2, {}

class NonPortableReplay:
    def __init__(self, *args, storage_dir, **kwargs):
        Path(storage_dir).write_bytes(b"would overwrite")
        raise RuntimeError("Historical replay factory must never initialize")


def test_guided_behavior_uses_deterministic_actor_and_valid_box2(monkeypatch):
    def predict(obs, *, deterministic):
        assert deterministic
        return np.array([[0.99, -0.99]], dtype=np.float32), None
    policy = SimpleNamespace(scale_action=lambda x: x, unscale_action=lambda x: x)
    model = SimpleNamespace(predict=predict, _last_obs=None, policy=policy)
    def noise(mean, sigma, shape):
        assert sigma == NOISE_STD and shape == (1, 2)
        return np.array([[0.1, -0.1]])
    monkeypatch.setattr(np.random, "normal", noise)
    executed, stored = GuidedSAC._sample_action(model, 0)
    np.testing.assert_array_equal(executed, [[1, -1]])
    np.testing.assert_array_equal(stored, executed)
    assert [progress_band(x) for x in (0.0, 0.25, 0.75, 1.0)] == ["early", "middle", "late", "late"]


def test_traffic_guidance_uses_forward_relative_velocity_and_preserves_box2():
    from cloud.kaggle.traffic_guidance import traffic_follow

    obs = np.zeros((3, 409), dtype=np.float32)
    obs[:, 3] = 37. / 81.  # 36 km/h in the existing encoder
    obs[1:, 315] = 1.
    obs[1, 259:266] = [5. / 50., 0., -6. / 30., 0., 0., 4.5 / 5., 2. / 5.]
    obs[2, 259:266] = [-5. / 50., 0., -6. / 30., 0., 0., 4.5 / 5., 2. / 5.]
    original = np.array([[.5, .7], [-.4, .7], [.1, .7]], dtype=np.float32)
    action = traffic_follow(obs, original)
    np.testing.assert_array_equal(action[:, 0], original[:, 0])
    np.testing.assert_array_equal(action[0], original[0])
    assert -1. <= action[1, 1] < action[2, 1] < 0.
    np.testing.assert_array_equal(original[:, 1], np.full(3, .7, dtype=np.float32))


def test_traffic_teacher_mixture_retires_at_episode_boundary():
    model = SimpleNamespace(env=SimpleNamespace(reset_infos=[{"episode": 2, "tier": "Medium"},
                                                            {"episode": 3, "tier": "Easy"}]),
                            teacher_episodes={}, teacher_cutoff=100, num_timesteps=50, n_envs=2)
    assert GuidedSAC.teacher_episode_mask(model).tolist() == [True, False]
    model.num_timesteps = 102
    assert GuidedSAC.teacher_episode_mask(model).tolist() == [True, False]
    model.env.reset_infos[0] = {"episode": 4, "tier": "Medium"}
    assert GuidedSAC.teacher_episode_mask(model).tolist() == [False, False]


@pytest.mark.parametrize("capacity,accepted", [(None, 1), (2, 2)])
def test_campaign_admits_within_capacity_and_reports_other_identities(tmp_path, monkeypatch, capacity, accepted):
    from collections import Counter
    from contextlib import nullcontext
    from cloud.kaggle import campaign, parallel

    root = tmp_path / "queue"
    bundle = tmp_path / "payload.zip"
    bundle.write_bytes(b"test payload")
    campaign.init(root, bundle)
    doc = campaign.manifest(root)
    assert Counter(x["account"] for x in doc["sessions"]) == {slot: 2 for slot in control.SLOTS}
    assert len({x["job"] for x in doc["sessions"]}) == 16
    registry = {"jobs": {}}
    monkeypatch.setattr(campaign, "WORK", tmp_path)
    monkeypatch.setattr(campaign, "slot", lambda _: (None, {"owner": "own", "observed_at": control.now()["utc"],
                                                         "concurrency_observed": capacity}))
    monkeypatch.setattr(campaign, "verify_owner", lambda alias: alias == "account_01")
    monkeypatch.setattr(control, "locked", nullcontext)
    monkeypatch.setattr(control, "registry", lambda: registry)

    def planned(package, digest, workload, state):
        job = {"round": "test", "job": parallel.job_id(workload), "account": "account_01"}
        state["jobs"][job["job"]] = {"status": "PLANNED", "planned": job}
        write(tmp_path / "plan-stage5-test.json", {"fingerprint": "test-plan"})
        return job

    launched = []

    def launch(args, state):
        assert args.server_admission == (capacity is None)
        job = next(key for key, row in state["jobs"].items() if row["status"] == "PLANNED")
        launched.append(job)
        state["jobs"][job].update(status="QUEUED", version=1, submitted=control.now(), owner="own")

    monkeypatch.setattr(parallel, "plan", planned)
    monkeypatch.setattr(control, "launch", launch)
    monkeypatch.setattr(control, "status", lambda *_: None)
    with campaign.agent_lock(root / "receipts/account_01"):
        campaign.run_once(root, bundle, doc, "account_01")
        campaign.run_once(root, bundle, doc, "account_01")
    campaign.report(root, doc)
    r = control.read(root / "status.json")
    assert len(launched) == accepted and r["totals"]["queued"] == accepted and r["totals"]["running"] == 0
    if capacity is None:
        assert r["sessions"][1]["blocker"] == "LAUNCHER_UNKNOWN_CAPACITY_WITH_ACTIVE_RECEIPT"
    assert r["sessions"][2]["blocker"] == "OPERATOR_RECEIPT_MISSING"
    monkeypatch.setattr(campaign, "verify_owner", lambda _: False)
    with campaign.agent_lock(root / "receipts/account_01"):
        assert not campaign.run_once(root, bundle, doc, "account_01")
    campaign.report(root, doc)
    totals = control.read(root / "status.json")["totals"]
    assert totals["submitted"] == accepted and totals["queued"] == 0 and totals["stale_active"] == accepted


def test_guided_actor_learning_rate_does_not_change_critic_or_alpha():
    env = DummyVecEnv([TinyEnv])
    try:
        model = GuidedSAC("MlpPolicy", env, learning_rate=1e-4, buffer_size=8, device="cpu")
        model.set_logger(configure(format_strings=[]))
        model.actor_lr_scale = 0.25
        optimizers = [model.actor.optimizer, model.critic.optimizer, model.ent_coef_optimizer]
        for _ in range(2):
            model._update_learning_rate(optimizers)
            assert [o.param_groups[0]["lr"] for o in optimizers] == [2.5e-5, 1e-4, 1e-4]
    finally:
        env.close()


@pytest.mark.parametrize("axis,expected", [(0, [[0.3, -0.2]]), (1, [[0.8, -0.4]])])
def test_guided_retention_stores_the_executed_anchored_axis(monkeypatch, axis, expected):
    policy = SimpleNamespace(scale_action=lambda x: x, unscale_action=lambda x: x)
    anchor = SimpleNamespace(predict=lambda *_args, **_kwargs: (np.array([[0.3, -0.4]]), None))
    model = SimpleNamespace(predict=lambda *_args, **_kwargs: (np.array([[0.8, -0.2]]), None),
                            _last_obs=None, policy=policy, rollout_axis=axis, rollout_anchor=anchor)
    monkeypatch.setattr(np.random, "normal", lambda *_args: 0)
    executed, stored = GuidedSAC._sample_action(model, 0)
    np.testing.assert_allclose(executed, expected)
    np.testing.assert_array_equal(stored, executed)


def test_guided_notebook_command_passes_training_flags_once(monkeypatch):
    calls = []
    monkeypatch.setattr(guided, "train", calls.append)
    args = ["--guided-prefill", "dataset", "--transitions", "5000", "--measure-optimizer"]
    guided.main(["train", *args])
    guided.main(args)
    assert calls == [args, args]


def test_guided_resume_grows_replay_without_reordering_or_overwriting():
    replay = ReplayBuffer(8, TinyEnv.observation_space, TinyEnv.action_space, n_envs=2, device="cpu")
    replay.pos = 3
    replay.actions[:3] = np.arange(12).reshape(3, 2, 2) / 12
    replay.observations[:3] = 0.5
    replay.next_observations[:3] = 0.75
    replay.rewards[:3] = [[0, 1], [2, 3], [4, 5]]
    replay.dones[1, 0] = replay.dones[2, 1] = replay.timeouts[2, 1] = 1
    model = SimpleNamespace(replay_buffer=replay, buffer_size=8, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="intact replay"):
        guided.grow_replay(model, 12, 4)
    guided.grow_replay(model, 12, 6)
    assert model.buffer_size == 12 and model.replay_buffer.n_envs == 2
    assert model.replay_buffer.pos == 3 and not model.replay_buffer.full
    for field in ("observations", "next_observations", "actions", "rewards", "dones", "timeouts"):
        np.testing.assert_array_equal(getattr(model.replay_buffer, field)[:3], getattr(replay, field)[:3])


def test_cloud_guided_resume_checks_verified_parent_behavior_and_provenance(tmp_path, monkeypatch):
    parent_path = tmp_path / "parent"
    parent_path.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    anchors = [{"reward": 0.5}, {"reward": 1.0}]
    bundle = tmp_path / "runtime.zip"
    with zipfile.ZipFile(bundle, "x") as archive:
        archive.writestr("parity.json", json.dumps(anchors))
        archive.writestr("source_inventory.json", json.dumps({"metadrive_pin": "pinned"}))
    (output / "runtime_versions.txt").write_text("metadrive-commit=pinned\nstable_baselines3==2.9.2\n")
    write(output / "parity.json", anchors)

    ledger = {"chunk": "digest"}
    evidence = {"ledger": ledger, "snapshot_sha256": "snapshot"}
    parent_metrics = {"guided_prefill": evidence, "online_behavior": guided.BEHAVIOR_CONFIG}
    write(parent_path / "metrics.json", parent_metrics)
    parent = {"identity": {"capacity": 8}, "learning_mode": guided.BEHAVIOR, "ledger": ledger,
              "external_admitted": 4, "transitions": 2, "updates": 2,
              "files": {"metrics.json": sha(parent_path / "metrics.json")}}
    marker = parent_path / "COMPLETE.json"
    write(marker, parent)
    monkeypatch.setattr(control, "parent_checkpoint", lambda *_: (parent_path, sha(marker)))

    manifest = {**parent, "identity": {"capacity": 12}, "transitions": 4, "updates": 4}
    metrics = {**parent_metrics, "guided_live_rows": 4, "online_live_rows": 4, "replay_size": 8,
               "online_progress_bands": {"early": 2}, "transitions": 2, "updates": 2, "device": "cpu",
               "seconds": 1, "decisions_per_second": 2, "updates_per_second": 2,
               "post_warmup_seconds": 1, "rss_peak_sampled_bytes": 1}
    entry = {"execution": {"parent_job": "parent", "runtime_sha": sha(bundle), "device": "cpu",
                           "guided_resume": True, "capacity": 12}, "planned": {"runtime_path": tmp_path}}
    write(output / "metrics.json", metrics)
    assert control.verify_cloud_evidence(output, manifest, entry, {}) == {"G3": "PASS"}
    entry["execution"].update(guided_gradient_steps=1, workers=2)
    reduced = {**manifest, "updates": 3, "identity": {**manifest["identity"], "guided_gradient_steps": 1}}
    write(output / "metrics.json", {**metrics, "updates": 1})
    assert control.verify_cloud_evidence(output, reduced, entry, {}) == {"G3": "PASS"}
    write(output / "metrics.json", metrics)
    with pytest.raises(ValueError, match="provenance, behavior or accounting"):
        control.verify_cloud_evidence(output, {**reduced, "updates": 4}, entry, {})
    del entry["execution"]["guided_gradient_steps"]
    for override in ({"online_behavior": {**guided.BEHAVIOR_CONFIG, "std": 0.4}},
                     {"guided_prefill": {**evidence, "snapshot_sha256": "other"}}):
        write(output / "metrics.json", {**metrics, **override})
        with pytest.raises(ValueError, match="provenance, behavior or accounting"):
            control.verify_cloud_evidence(output, manifest, entry, {})
    write(output / "metrics.json", metrics)
    write(parent_path / "metrics.json", {**parent_metrics, "additional": True})
    with pytest.raises(ValueError, match="mismatched SHA-256"):
        control.verify_cloud_evidence(output, manifest, entry, {})


def test_operator_cannot_open_another_slot(monkeypatch):
    monkeypatch.setenv("STAGE5_KAGGLE_OPERATOR_SLOT", "account_02")
    monkeypatch.setattr(credentials, "slot", lambda *_: pytest.fail("Another operator's slot was opened"))
    assert not credentials.verify_owner("account_01")
    with pytest.raises(RuntimeError, match="configured Kaggle operator slot"):
        credentials.slot_token("account_01")


def test_sharded_checkpoint_round_trip_and_tamper_rejection(tmp_path):
    raw = tmp_path / "state"
    raw.mkdir()
    for i, name in enumerate(PARTS):
        (raw / name).write_bytes(bytes([i + 1]) * 17)
    write(raw / "COMPLETE.json", {"schema": 1, "files": {name: sha(raw / name) for name in PARTS}})
    state_sha = sha(raw / "COMPLETE.json")
    bundle = tmp_path / "package"
    index = export_checkpoint(raw, state_sha, bundle, shard_bytes=8)
    assert len(list(bundle.glob("replay.pkl.*.bin"))) == 3
    target = tmp_path / "restored"
    assert restore_checkpoint(bundle, index, target) == state_sha
    assert all(sha(target / name) == sha(raw / name) for name in (*PARTS, "COMPLETE.json"))
    (bundle / "replay.pkl.000000.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        restore_checkpoint(bundle, index, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_replay_requires_matching_authorized_snapshot_and_rejects_staleness(tmp_path):
    identity = {"workers": 2}
    (tmp_path / "policy.zip").write_bytes(b"actor")
    write(tmp_path / "SNAPSHOT.json", {"schema": 1, "lineage": control.RECIPE["lineage"],
                                      "identity": identity, "policy_version": 2,
                                      "policy_sha": sha(tmp_path / "policy.zip")})
    snapshot_sha = sha(tmp_path / "SNAPSHOT.json")
    assert check_snapshot(tmp_path, snapshot_sha, identity) == 2
    marker = tmp_path / "CHUNK.json"
    write(marker, {"schema": 1, "count": 2, "stream": 100, "sequence": 0, "identity": identity,
                   "policy_version": 2, "snapshot_sha": snapshot_sha})
    chunk_sha = sha(marker)
    with pytest.raises(ValueError, match="authorized actor snapshot"):
        validate(tmp_path, chunk_sha, identity, {}, 2, ("wrong-sha", 2))
    with pytest.raises(ValueError, match="stale actor"):
        validate(tmp_path, chunk_sha, identity, {}, 4, (snapshot_sha, 2))
    assert validate(tmp_path, chunk_sha, identity, {"100:0": chunk_sha}, 9, (snapshot_sha, 2)) == (None, "100:0")


def test_publish_b1_snapshot_requires_matching_runtime_inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "WORK", tmp_path)
    monkeypatch.setattr(control, "ROOT", tmp_path)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    with zipfile.ZipFile(snapshot / "policy.zip", "x") as inner:
        inner.writestr("data", "{}")
    policy_sha = sha(snapshot / "policy.zip")
    monkeypatch.setattr(control, "SOURCE_SHA", policy_sha)
    files = {"cloud/kaggle/runtime.py": "runtime-digest"}
    identity = {"phase": "smoke", "warm_source": policy_sha, "workers": 2, "recipe": control.RECIPE, "files": files}
    meta = {"schema": 1, "lineage": control.RECIPE["lineage"], "policy_version": 0,
            "policy_sha": policy_sha, "identity": identity}
    write(snapshot / "SNAPSHOT.json", meta)
    runtime = tmp_path / "runtime.zip"
    with zipfile.ZipFile(runtime, "x") as carrier:
        carrier.writestr("source_inventory.json", json.dumps({"files": files}))
    write(tmp_path / "bundle.json", {"path": ".", "runtime_sha": sha(runtime)})
    monkeypatch.setattr(control, "account_for", lambda _: ("account_01", {"owner": "example"}))
    def upload(state, alias, owner, label, name, path, digest):
        control.audit_zip(path, label)
        assert label == "snapshot" and name == "snapshot.zip" and digest == sha(path)
        return "example/snapshot"
    monkeypatch.setattr(control, "upload", upload)
    args = SimpleNamespace(snapshot=snapshot, snapshot_sha=sha(snapshot / "SNAPSHOT.json"))
    control.publish_initial_snapshot(args, {})
    receipt = control.read(tmp_path / "b1" / args.snapshot_sha / "receipt.json")
    assert receipt["learner_workers"] == 2 and receipt["policy_version"] == 0
    identity["files"] = {"cloud/kaggle/runtime.py": "different"}
    write(snapshot / "SNAPSHOT.json", meta)
    args.snapshot_sha = sha(snapshot / "SNAPSHOT.json")
    with pytest.raises(ValueError, match="recipe differs"):
        control.publish_initial_snapshot(args, {})


def test_publish_replay_chunk_requires_attested_learner_and_private_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "WORK", tmp_path)
    chunk, learner = tmp_path / "chunk", tmp_path / "learner"
    chunk.mkdir()
    learner.mkdir()
    (chunk / "rows.npz").write_bytes(b"replay")
    snapshot_sha = "f" * 64
    meta = {"stream": 100, "sequence": 0, "identity": {"workers": 2}, "count": 256,
            "payload_sha": sha(chunk / "rows.npz"), "snapshot_sha": snapshot_sha, "policy_version": 0}
    write(chunk / "CHUNK.json", meta)
    chunk_sha = sha(chunk / "CHUNK.json")
    write(learner / "metrics.json", {"snapshot_sha": snapshot_sha, "admitted": 256, "updates": 8})
    manifest = {"job": "external-import", "ledger": {"100:0": chunk_sha}, "identity": meta["identity"],
                "external_admitted": 256, "files": {"metrics.json": sha(learner / "metrics.json")}}
    write(learner / "COMPLETE.json", manifest)
    snapshot = tmp_path / "b1" / snapshot_sha
    snapshot.mkdir(parents=True)
    write(snapshot / "receipt.json", {"snapshot_sha256": snapshot_sha, "policy_version": 0, "dataset": "example/snapshot"})
    state = {"uploads": {"example/snapshot": {"private_verified_at": "today", "remote_sha_verified_at": "today"}}}
    monkeypatch.setattr(control, "account_for", lambda _: ("account_01", {"owner": "example"}))
    def upload(state, alias, owner, label, name, path, digest):
        assert (label, name, digest) == ("replay", "replay.zip", sha(path))
        control.audit_zip(path, label)
        return "example/replay"
    monkeypatch.setattr(control, "upload", upload)
    args = SimpleNamespace(chunk=chunk, learner=learner, chunk_sha=chunk_sha, learner_sha=sha(learner / "COMPLETE.json"))
    control.publish_verified_replay_chunk(args, state)
    assert control.read(tmp_path / "b1" / chunk_sha / "receipt.json")["admitted_rows"] == 256
    state["uploads"]["example/snapshot"].pop("remote_sha_verified_at")
    with pytest.raises(ValueError, match="private transfer"):
        control.publish_verified_replay_chunk(args, state)


def test_easy_pilot_compares_two_distinct_checkpoints_on_exact_panel():
    def evaluation(model, successes):
        rows = [{"geometry": str(i // 2), "family": "SCS", "seed": i % 2, "censored": False,
                 "clean": i < successes, "outcome": "SUCCESS" if i < successes else "OUT_OF_ROAD",
                 "progress": 1.0 if i < successes else 0.4, "speed_km_h": 35,
                 "success_seconds": 40 if i < successes else None}
                for i in range(12)]
        return {"model_sha": model, "tier": "Easy", "cases": rows}
    baseline = evaluation(SOURCE_SHA, 8)
    candidate = evaluation("checkpoint-1", 10)
    report = compare(baseline, [candidate, evaluation("checkpoint-2", 11)])
    assert report["two_milestones_meet_thresholds"]
    assert report["milestones"][0]["improved_cases"] == 2
    assert report["milestones"][0]["mean_time_delta_on_common_successes_seconds"] == 0
    assert len(report["milestones"][0]["paired_cases"]) == 12
    assert report["recommended_model_sha"] == "checkpoint-2"
    challenger = evaluation("challenger", 11)
    compared = compare(candidate, [challenger], reference_sha="checkpoint-1")
    assert compared["reference_model_sha"] == "checkpoint-1"
    challenger["cases"] = [row for row in challenger["cases"] if row["seed"] == 0]
    ranked = screen_candidates(candidate, [challenger], reference_sha="checkpoint-1")
    assert ranked["purpose"] == "SCREENING_ONLY" and ranked["seed"] == 0
    medium = {**evaluation("medium-baseline", 0), "tier": "Medium"}
    improved = {**evaluation("medium-candidate", 2), "tier": "Medium"}
    assert compare(medium, [improved], reference_sha="medium-baseline", tier="Medium")["milestones"][0]["improved_cases"] == 2
    with pytest.raises(ValueError, match="12-case Medium"):
        compare(medium, [candidate], reference_sha="medium-baseline", tier="Medium")
    candidate["cases"][0]["seed"] = 99
    with pytest.raises(ValueError, match="exact baseline case IDs"):
        compare(baseline, [candidate])


def test_easy_coverage_counts_all_train_geometries_and_rejects_diagnostic():
    split = {"pilot_train": [{"geometry_id": f"train-{i}", "tier": "Easy", "sequence": "SCS"} for i in range(39)],
             "pilot_diagnostic": [{"geometry_id": "diag-1"}]}
    report = easy_coverage(split, [{"decisions_by_geometry": {"train-0": 11}},
                                   {"decisions_by_geometry": {"train-0": 12, "train-2": 2}}])
    assert report["pool"] == 39 and report["visited"] == 2 and report["decisions"]["train-0"] == 23
    with pytest.raises(ValueError, match="outside"):
        easy_coverage(split, [{"decisions_by_geometry": {"diag-1": 3}}])


def test_policy_screening_hooks_export_only_between_full_states(monkeypatch, tmp_path):
    class Telemetry:
        def _on_step(self):
            return True
    monkeypatch.setattr(learning_hooks.runtime, "Telemetry", Telemetry)
    monkeypatch.setattr(learning_hooks.runtime, "make_vec", lambda _, __, cursors: cursors)
    def simulate():
        assert learning_hooks.runtime.make_vec(2, "easy", [0, 0]) == [8, 8]
        probe = learning_hooks.runtime.Telemetry()
        probe.model = SimpleNamespace(_n_updates=1400, save=lambda path: path.write_bytes(b"policy"))
        for step in (2500, 2501, 12500, 15000):
            probe.num_timesteps = step
            assert probe._on_step()
    monkeypatch.setattr(learning_hooks.runtime, "main", simulate)
    learning_hooks.train(["--policy-dir", str(tmp_path), "--policy-every", "2500", "--state-every",
                          "12500", "--warm-start-cursor", "8", "--source", "checkpoint.zip"])
    assert sorted(path.name for path in tmp_path.iterdir()) == ["step-000000002500", "step-000000015000"]
    marker = tmp_path / "step-000000002500" / "POLICY.json"
    assert control.read(marker)["purpose"] == "SCREENING_ONLY"
    assert learning_hooks.runtime.Telemetry is Telemetry


def test_checkpoint_warm_start_copies_weights_without_replay_or_optimizers(tmp_path):
    torch.set_num_threads(1)
    env = DummyVecEnv([TinyEnv])
    target = DummyVecEnv([TinyEnv])
    try:
        source = model_for(env, "cpu", 32, 0)
        source.learn(4)
        path = tmp_path / "learned.zip"
        source.save(path)
        warmed = model_for(target, "cpu", 32, 0, warm_model=path, warm_sha=sha(path), metered=True)
        for key in ("actor", "critic", "critic_target"):
            for previous, current in zip(getattr(source, key).parameters(), getattr(warmed, key).parameters(), strict=True):
                torch.testing.assert_close(previous, current, rtol=0, atol=0)
        assert source.actor.optimizer.state and not warmed.actor.optimizer.state
        assert warmed.num_timesteps == warmed._n_updates == warmed.replay_buffer.size() == 0
        warmed.learn(4)
        assert warmed._n_updates == 4 and warmed.optimizer_seconds > 0
    finally:
        env.close()
        target.close()


def test_capacity_timing_uses_kernel_elapsed_clock():
    log = ('[{"time":1.2,"data":"runtime ABI"}\n'
           ',{"time":8.0,"data":"COMPLETE transitions=1280 updates=256"}\n'
           ',{"time":9.0,"data":"artifact closed"}\n]')
    assert log_times(log) == (8.0, 9.0)


def test_pilot_plan_uses_verified_cpu_when_gpu_only_wins_after_warmup(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "WORK", tmp_path)
    monkeypatch.setattr(control, "ROOT", tmp_path)
    def bundle(label, production):
        path = tmp_path / label
        path.mkdir()
        files = {"parity.json": "locked", "cloud/kaggle/linux-lock.txt": "unchanged"}
        if production:
            files["cloud/kaggle/checkpoint.py"] = "new"
        with zipfile.ZipFile(path / "runtime.zip", "x") as archive:
            archive.writestr("source_inventory.json", json.dumps({"files": files}))
        return {"runtime_sha": sha(path / "runtime.zip"), "source_sha": SOURCE_SHA,
                "path": label, "template_sha": "template"}
    base = bundle("verified", False)
    candidate = bundle("candidate", True)
    write(tmp_path / "bundle.json", candidate)
    rows = {}
    samples = {}
    for label, workers, device, rate, warm in (("j02", 1, "cpu", 8, 20), ("j03", 2, "cpu", 9, 24),
                                              ("j04", 2, "cuda", 7, 99)):
        planned = {k: base[k] for k in ("runtime_sha", "source_sha", "template_sha")}
        planned.update(campaign="stage5", workers=workers, device=device, runtime_path=base["path"],
                       **control.RECIPE["preflight"])
        rows[label] = {"status": "HARVESTED_VERIFIED", "planned": planned, "execution": planned,
                       "output_version": 1, "gates": {"G2": "PASS"} if label == "j02" else {"capacity_sample": "PASS"}}
        samples[label] = {"job": label, "version": 1, "end_to_end_transitions_per_second": rate,
                          "post_warmup_transitions_per_second": warm}
    rows["j07"] = dict(rows["j02"], gates={"G3": "PASS"})
    write(tmp_path / "capacity-capacity-report.json", {"G4": "PASS", "samples": samples})
    args = SimpleNamespace(campaign="stage5", round="easy", capacity_round="capacity", account="account_01")
    control.plan_pilot(args, {"jobs": rows})
    job = control.read(tmp_path / "plan-stage5-easy.json")["jobs"]["j08"]
    assert (job["workers"], job["device"], job["phase"]) == (2, "cpu", "easy")
    assert job["checkpoint_every"] == 4096 and job["runtime_sha"] == candidate["runtime_sha"]

    pilot_id = job["job"]
    receipts = {"status": "HARVESTED_VERIFIED", "planned": job, "periodic": {}}
    milestones = []
    for step, path in ((4096, tmp_path / "periodic" / pilot_id / "step-000000004096"),
                       (8192, tmp_path / "harvest" / pilot_id)):
        path.mkdir(parents=True)
        (path / "model.zip").write_bytes(str(step).encode())
        model_digest = sha(path / "model.zip")
        write(path / "COMPLETE.json", {"transitions": step, "job": pilot_id, "files": {"model.zip": model_digest}})
        checkpoint_digest = sha(path / "COMPLETE.json")
        milestones.append({"model_sha": model_digest})
        if step == 4096:
            receipts["periodic"]["4096"] = {"state_sha": checkpoint_digest}
        else:
            receipts["state_sha"] = checkpoint_digest
    rows[pilot_id] = receipts
    args.round, args.reference_job, args.checkpoint, args.transitions = "longer", pilot_id, "4096", 25000
    args.evaluation_report = tmp_path / "report.json"
    write(args.evaluation_report, {"baseline_clean_successes": 8, "baseline_cases": 12, "milestones": milestones})
    control.plan_pilot(args, {"jobs": rows})
    job = control.read(tmp_path / "plan-stage5-longer.json")["jobs"]["j09"]
    assert (job["workers"], job["device"], job["transitions"]) == (2, "cuda", 25000)
    assert job["warm_model_sha"] == milestones[0]["model_sha"]
    assert job["sharded_output"] and "parent_job" not in job

    parent = job["job"]
    marker = tmp_path / "harvest" / parent / "COMPLETE.json"
    marker.parent.mkdir()
    write(marker, {"transitions": 25000})
    rows[parent] = {"status": "HARVESTED_VERIFIED", "planned": job, "state_sha": sha(marker)}
    control.plan_learning_resume(SimpleNamespace(
        campaign="stage5", round="resume", account="account_01", parent=parent, transitions=12500,
    ), {"jobs": rows})
    continued = control.read(tmp_path / "plan-stage5-resume.json")["jobs"]["j10"]
    assert (continued["device"], continued["workers"], continued["start_transition"]) == ("cuda", 2, 25000)
    assert continued["parent_sha"] == sha(marker)
    assert continued["parent_archive_limit"] > 120_000_000 and "warm_step" not in continued
    periodic = tmp_path / "periodic" / parent / "step-000000012500" / "COMPLETE.json"
    periodic.parent.mkdir(parents=True)
    write(periodic, {"transitions": 12500})
    rows[parent].update(version=1, periodic={"12500": {"version": 1, "state_sha": sha(periodic)}})
    control.plan_learning_resume(SimpleNamespace(
        campaign="stage5", round="mid-resume", account="account_01", parent=parent, parent_step=12500,
        transitions=12500,
    ), {"jobs": rows})
    mid = control.read(tmp_path / "plan-stage5-mid-resume.json")["jobs"]["j10"]
    assert (mid["parent_step"], mid["parent_sha"], mid["start_transition"]) == (12500, sha(periodic), 12500)
    rows[parent]["periodic"]["12500"]["version"] = 2
    with pytest.raises(ValueError, match="recorded version"):
        control.parent_checkpoint({"jobs": rows}, parent, 12500)

    guided_parent = "guided-parent"
    marker = tmp_path / "harvest" / guided_parent / "COMPLETE.json"
    marker.parent.mkdir()
    write(marker, {"transitions": 5000, "external_admitted": 5300, "ledger": {"episode": "hash"},
                   "learning_mode": guided.BEHAVIOR, "identity": {"capacity": 16384, "online_behavior": guided.BEHAVIOR}})
    rows[guided_parent] = {"status": "HARVESTED_VERIFIED", "state_sha": sha(marker),
                           "planned": dict(job, role="easy_guided", device="cpu", guided_path="prefill", guided_sha="hash")}
    control.plan_learning_resume(SimpleNamespace(
        campaign="stage5", round="guided-resume", account="account_01", parent=guided_parent, transitions=10000,
    ), {"jobs": rows})
    continuation = control.read(tmp_path / "plan-stage5-guided-resume.json")["jobs"]["j10"]
    assert continuation["guided_resume"] and continuation["capacity"] == 20480
    assert continuation["start_transition"] == 5000 and continuation["checkpoint_every"] == 10000
    assert not {"guided_path", "guided_sha", "warm_step"} & continuation.keys()
    production = SimpleNamespace(campaign="stage5", round="guided-prod", account="account_01", parent=guided_parent,
                                 transitions=100000, device="cuda", max_seconds=5400, phase="medium")
    control.plan_learning_resume(production, {"jobs": rows})
    large = control.read(tmp_path / "plan-stage5-guided-prod.json")["jobs"]["j10"]
    assert (large["phase"], large["guided_curriculum_from"], large["accelerator"]) == ("medium", "easy", "NvidiaTeslaT4")
    assert large["capacity"] >= 110300 and large["checkpoint_every"] == 25000 and large["production_continuation"]
    control_id = continuation["job"]
    rows[control_id].update(status="HARVESTED_VERIFIED", execution=continuation)
    args = SimpleNamespace(campaign="stage5", round="guided-utd05", account="account_01", parent=guided_parent,
                           transitions=5000, reduce_guided_utd_after=control_id)
    control.plan_learning_resume(args, {"jobs": rows})
    comparison = control.read(tmp_path / "plan-stage5-guided-utd05.json")["jobs"]["j10"]
    assert (comparison["utd_control"], comparison["guided_gradient_steps"]) == (control_id, 1)
    assert comparison["parent_sha"] == continuation["parent_sha"] and comparison["capacity"] == continuation["capacity"]
    rows[comparison["job"]].update(status="HARVESTED_VERIFIED", execution=comparison)
    actor = SimpleNamespace(campaign="stage5", round="guided-actor25", account="account_01", parent=guided_parent,
                            transitions=5000, dampen_guided_actor_after=comparison["job"])
    control.plan_learning_resume(actor, {"jobs": rows})
    planned_actor = control.read(tmp_path / "plan-stage5-guided-actor25.json")["jobs"]["j10"]
    assert planned_actor["actor_lr_control"] == comparison["job"] and planned_actor["guided_actor_lr_scale"] == 0.25
    for key in ("parent_sha", "device", "workers", "capacity", "transitions", "guided_gradient_steps"):
        assert planned_actor[key] == comparison[key]
    rows[control_id]["status"] = "RUNNING"
    with pytest.raises(ValueError, match="verified two-environment CPU control"):
        control.plan_learning_resume(args, {"jobs": rows})


def test_large_parent_audit_accepts_expected_members_over_preflight_limit(tmp_path):
    from io import BytesIO
    model = BytesIO()
    with zipfile.ZipFile(model, "w") as inner:
        inner.writestr("data", "{}")
    state = tmp_path / "parent.zip"
    with zipfile.ZipFile(state, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("model.zip", model.getvalue())
        for name in ("COMPLETE.json", "rng.pt", "metrics.json"):
            archive.writestr(name, "{}")
        with archive.open("replay.pkl", "w") as replay:
            block = b"\0" * 1_000_000
            for _ in range(121):
                replay.write(block)
    with pytest.raises(ValueError, match="size"):
        control.audit_zip(state, "parent")
    control.audit_zip(state, "long-parent")


@pytest.mark.parametrize("workers", [2, 4])
def test_external_replay_vector_rows_preserve_termination_and_reject_partial_batch(workers):
    env = TinyEnv()
    replay = ReplayBuffer(2 * workers, env.observation_space, env.action_space, device="cpu", n_envs=workers)
    data = {"obs": np.zeros((workers, 409), dtype=np.float32), "next_obs": np.ones((workers, 409), dtype=np.float32),
            "action": np.zeros((workers, 2), dtype=np.float32), "reward": np.arange(workers, dtype=np.float32),
            "terminated": np.arange(workers) == 0, "truncated": np.arange(workers) < 2}
    with pytest.raises(ValueError, match="complete vector replay rows"):
        admit(replay, {key: values[:-1] for key, values in data.items()})
    assert replay.size() == 0
    admit(replay, data)
    assert replay.size() == 1
    np.testing.assert_array_equal(replay.rewards[0], data["reward"])
    np.testing.assert_array_equal(replay.dones[0], data["terminated"] | data["truncated"])
    np.testing.assert_array_equal(replay.timeouts[0], data["truncated"] & ~data["terminated"])


def test_policy_loader_never_initializes_historical_replay(tmp_path):
    source = tmp_path / "historical.zip"
    sentinel = tmp_path / "preserved.npy"
    sentinel.write_bytes(b"original")
    env = DummyVecEnv([TinyEnv])
    try:
        model = SAC("MlpPolicy", env, buffer_size=4, device="cpu", policy_kwargs={"net_arch": [8]})
        model.replay_buffer_class = NonPortableReplay
        model.replay_buffer_kwargs = {"storage_dir": str(sentinel)}
        model.save(source)
        loaded = load_policy(source)
        assert sentinel.read_bytes() == b"original"
        assert type(loaded.replay_buffer) is ReplayBuffer
        assert loaded.replay_buffer.buffer_size == 1
    finally:
        env.close()

@pytest.mark.parametrize("workers", [1, 2, 4])
def test_vector_collection_and_timeout_bootstrap(workers):
    torch.set_num_threads(1)
    env = DummyVecEnv([TinyEnv] * workers)
    try:
        model = SAC("MlpPolicy", env, learning_starts=0, batch_size=4, buffer_size=64,
                    gradient_steps=-1, policy_kwargs={"net_arch": [8]}, device="cpu", seed=101)
        model.learn(4 * workers)
        replay = model.replay_buffer
        assert model._n_updates == model.num_timesteps == 4 * workers
        assert np.all(replay.next_observations[1] == 2)
        assert np.all(replay.dones[1] == 1)
        assert np.all(replay.timeouts[1] == 1)
        sample = replay._get_samples(np.asarray([1]))
        assert sample.dones.item() == 0
    finally:
        env.close()

def test_save_load_preserves_optimizer_replay_and_next_gradients(tmp_path):
    torch.set_num_threads(1)
    env = DummyVecEnv([TinyEnv])
    restored_env = DummyVecEnv([TinyEnv])
    try:
        original = SAC("MlpPolicy", env, learning_starts=0, batch_size=4, buffer_size=32,
                       gradient_steps=-1, policy_kwargs={"net_arch": [8]}, device="cpu", seed=101)
        original.learn(4)
        original._last_obs = None
        target = tmp_path / "state"
        save_state(target, original, {"transitions": 4, "updates": 4}, {})
        digest = sha(target / "COMPLETE.json")
        check_state(target, digest)
        original.learn(4, reset_num_timesteps=False)
        restored = SAC.load(target / "model.zip", env=restored_env, device="cpu")
        restored.load_replay_buffer(target / "replay.pkl")
        restore_rng(torch.load(target / "rng.pt", weights_only=False), "cpu")
        restored.learn(4, reset_num_timesteps=False)
        assert restored.num_timesteps == original.num_timesteps == 8
        assert restored._n_updates == original._n_updates == 8
        for p, q in zip(original.policy.parameters(), restored.policy.parameters(), strict=True):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        torch.testing.assert_close(original.log_ent_coef, restored.log_ent_coef, rtol=0, atol=0)
        np.testing.assert_array_equal(original.replay_buffer.observations, restored.replay_buffer.observations)
        (target / "replay.pkl").write_bytes(b"damaged")
        with pytest.raises(ValueError, match="SHA-256"):
            check_state(target, digest)
    finally:
        env.close()
        restored_env.close()

def test_external_replay_wrap_and_terminal_precedence():
    env = TinyEnv()
    replay = ReplayBuffer(2, env.observation_space, env.action_space, device="cpu")
    data = {"obs": np.zeros((3, 409), dtype=np.float32), "next_obs": np.ones((3, 409), dtype=np.float32),
            "action": np.zeros((3, 2), dtype=np.float32), "reward": np.asarray([1, 2, 3], dtype=np.float32),
            "terminated": np.asarray([False, False, True]), "truncated": np.asarray([False, True, True])}
    admit(replay, data)
    assert replay.full and replay.pos == 1
    assert replay.rewards[:, 0].tolist() == [3, 2]
    assert replay.timeouts[:, 0].tolist() == [0, 1]
    assert replay._get_samples(np.asarray([0, 1])).dones.flatten().tolist() == [1, 0]

def test_locked_train_schedule():
    rows = pool("smoke")
    keys = [schedule(rows, "smoke", worker, episode) for worker in range(4) for episode in range(32)]
    assert len(set(keys)) == len(keys)
    assert set(r["geometry_id"] for r in pool("eval")).isdisjoint(r["geometry_id"] for r in pool("mixed"))
    assert keys[0] == schedule(rows, "smoke", 0, 0)

@pytest.mark.skipif(__import__("os").name != "nt", reason="Windows DPAPI")
def test_dpapi_round_trip():
    sample = b"test data only"
    encrypted = dpapi(sample, True)
    assert sample not in encrypted
    assert dpapi(encrypted, False) == sample

@pytest.mark.parametrize("admission_probe, sessions, prior, expected", [
    (False, 1, None, "SUBMIT_UNKNOWN"),
    (True, None, None, "SUBMIT_UNKNOWN"),
    (True, 0, None, "PLANNED"),
    (False, None, None, "PLANNED"),
    (True, None, "SUBMIT_UNKNOWN", "PLANNED"),
    (True, None, "CAPACITY_REJECTED", "PLANNED"),
])
def test_admission_and_timeout_receipts_prevent_duplicate_submit(
    tmp_path, monkeypatch, admission_probe, sessions, prior, expected
):
    wrapper = SimpleNamespace(campaign="stage5", round="test", account="account_01", job="j01", parent=None)
    wrapper.admission_probe = admission_probe
    monkeypatch.setattr(control, "WORK", tmp_path)
    monkeypatch.setattr(control, "ROOT", tmp_path)
    monkeypatch.setattr(control, "slot", lambda _: (tmp_path, {"owner": "example", "observed_at": control.now()["utc"],
                                                               "concurrency_observed": sessions, "gpu_minutes_observed": 0}))
    monkeypatch.setattr(control, "verify_owner", lambda _: True)
    monkeypatch.setattr(control, "upload", lambda *a: "example/private-test")
    monkeypatch.setattr(control, "probe_inventory", lambda *a: {"notebooks_listed": 0})
    pushes = []
    def fake_cli(alias, arguments, **kwargs):
        pushes.append((alias, arguments))
        return -1, ""
    monkeypatch.setattr(control, "cli", fake_cli)
    template = Path(control.__file__).with_name("kernel_template.py.txt")
    archive = tmp_path / "runtime.zip"
    with zipfile.ZipFile(archive, "x") as bundle:
        bundle.write(template, "cloud/kaggle/kernel_template.py.txt")
    control.write(tmp_path / "bundle.json", {"runtime_sha": sha(archive), "source_sha": "source", "path": ".",
                                            "template_sha": sha(template)})
    state = control.registry()
    control.plan(wrapper, state)
    plan = control.read(tmp_path / "plan-stage5-test.json")
    wrapper.fingerprint = plan["fingerprint"]
    job_id = plan["jobs"]["j01"]["job"]
    if prior:
        state["jobs"]["old"] = {"status": prior, "owner": "example", "admission_probe": {"capacity": "UNKNOWN"},
                                "planned": {**plan["jobs"]["j01"], "role": "cpu1"}}
    if expected == "PLANNED":
        with pytest.raises(RuntimeError, match="concurrency|[Aa]dmission probe|active receipt"):
            control.launch(wrapper, state)
        assert not pushes
        assert state["jobs"][job_id]["status"] == expected
        return
    control.launch(wrapper, state)
    control.launch(wrapper, control.registry())
    assert len(pushes) == 1
    job = control.registry()["jobs"][job_id]
    assert job["status"] == expected
    assert ("admission_probe" in job) is admission_probe
    script = tmp_path / "kernels" / job_id / "run.py"
    compile(script.read_text(encoding="utf-8"), str(script), "exec")


@pytest.mark.parametrize("variant, error", [
    ("fixed", None), ("same_runtime", "new runtime fingerprint"),
    ("unrelated", "change for the diagnosed failure"), ("undiagnosed", "diagnosed"),
    ("running", "terminal FAILED"), ("different_remote_version", "confirm the failed notebook version"),
    ("unknown", "confirmed failed version"), ("capacity_rejected", "confirmed failed version"),
    ("later_rejection", "latest admission probe"),
])
def test_probe_retry_requires_new_repaired_runtime_and_terminal_receipt(tmp_path, monkeypatch, variant, error):
    monkeypatch.setattr(control, "ROOT", tmp_path)
    monkeypatch.setattr(control, "WORK", tmp_path)
    template = "cloud/kaggle/kernel_template.py.txt"
    lock = "cloud/kaggle/linux-lock.txt"

    def runtime(dir_name, files):
        path = tmp_path / dir_name
        path.mkdir()
        archive = path / "runtime.zip"
        with zipfile.ZipFile(archive, "x") as bundle:
            bundle.writestr("source_inventory.json", json.dumps({"files": files}))
        return {"campaign": "stage5", "account": "account_01", "runtime_path": dir_name, "runtime_sha": sha(archive)}

    old = runtime("old", {template: "old", lock: "unchanged"})
    new = old if variant == "same_runtime" else runtime(
        "new", {template: "old" if variant == "unrelated" else "fixed", lock: "new"}
    )
    terminal = {"unknown": "SUBMIT_UNKNOWN", "capacity_rejected": "CAPACITY_REJECTED"}.get(variant, "FAILED")
    previous = {"planned": old, "ref": "example/old", "owner": "example", "status": terminal, "version": 1,
                "admission_probe": {"capacity": "UNKNOWN"}, "submitted": {"utc": "2026-01-01"},
                "log_diagnostic": {"categories": [] if variant == "undiagnosed" else ["PYTHON_ABI"]}}
    jobs = {"old": previous}
    if variant == "later_rejection":
        jobs["later"] = dict(previous, status="CAPACITY_REJECTED", submitted={"utc": "2026-01-02"})
    remote_status = "RUNNING" if variant == "running" else "ERROR"
    monkeypatch.setattr(control, "current_private_kernel_version", lambda *_: 2 if variant == "different_remote_version" else 1)
    monkeypatch.setattr(control, "cli", lambda *a, **kw: (0, f'example/old/1 has status "KernelWorkerStatus.{remote_status}"'))
    if error:
        with pytest.raises(RuntimeError, match=error):
            control.probe_retry({"jobs": jobs}, new, "old", "example")
    else:
        repair = control.probe_retry({"jobs": jobs}, new, "old", "example")
        assert repair["retry_of"] == "old"
        assert repair["changed_runtime_files"] == [template]


def test_submit_response_requires_exact_ref_or_explicit_capacity_error():
    ref = "example/preflight"
    receipt = "Kernel version 2 successfully pushed.  Please check progress at https://www.kaggle.com/code/"
    assert control.push_result(0, receipt + ref, ref) == ("SUBMITTED", 2)
    assert control.push_result(0, receipt + ref + "-other", ref) == ("SUBMIT_UNKNOWN", None)
    assert control.push_result(0, "Kernel push error: Too many concurrent sessions", ref) == ("CAPACITY_REJECTED", None)
    assert control.push_result(1, "409 Client Error: Maximum batch CPU session count of 1 reached", ref) == (
        "CAPACITY_REJECTED", None
    )
    assert control.push_result(1, "429 Client Error: Too Many Requests", ref) == ("SUBMIT_UNKNOWN", None)
    assert control.kernel_state(0, 'example/preflight has status "KernelWorkerStatus.QUEUED"') == "QUEUED"


@pytest.mark.parametrize("run_state", ["COMPLETE", "RUNNING"])
def test_probe_checks_pagination_slug_and_recent_run(monkeypatch, run_state):
    def fake_cli(alias, args, **kwargs):
        if "--search" in args:
            assert args[args.index("--search") + 1] == "preflight"
            return 0, "Not found\n"
        if args[1] == "status":
            return 0, f'{args[2]} has status "KernelWorkerStatus.{run_state}"'
        if "--page-token" in args:
            assert args[-1] == "next"
            return 0, '[{"ref":"example/other","lastRunTime":""}]'
        return 0, 'Next Page Token = next\n[{"ref":"","lastRunTime":""}]'
    monkeypatch.setattr(control, "cli", fake_cli)
    if run_state == "RUNNING":
        with pytest.raises(RuntimeError, match="queued or running"):
            control.probe_inventory("account_01", "example", "example/preflight")
        result = control.probe_inventory("account_01", "example", "example/preflight", ("example/other",))
        assert result["latest_runs_checked"] == 1
    else:
        assert control.probe_inventory("account_01", "example", "example/preflight") == {
            "notebooks_listed": 1, "unaddressable_or_unowned": 1, "latest_runs_checked": 1,
            "latest_run_status_unknown": 0,
        }


@pytest.mark.parametrize("machine, version, supported", [
    ("x86_64", (3, 13, 8), True), ("AMD64", (3, 13, 8), True),
    ("aarch64", (3, 13, 8), False), ("x86_64", (3, 14, 0), False),
])
def test_kernel_runtime_abi(machine, version, supported):
    template = Path(control.__file__).with_name("kernel_template.py.txt")
    tree = ast.parse(template.read_text(encoding="utf-8"))
    guard = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "check_runtime_abi")
    version_label = ".".join(map(str, version))
    scope = {"platform": SimpleNamespace(system=lambda: "Linux", machine=lambda: machine,
                                         python_version=lambda: version_label),
             "sys": SimpleNamespace(implementation=SimpleNamespace(name="cpython"), version_info=version)}
    exec(compile(ast.Module(body=[guard], type_ignores=[]), str(template), "exec"), scope)
    if supported:
        scope["check_runtime_abi"]()
    else:
        with pytest.raises(RuntimeError, match=f"Linux {machine} Python {version_label}"):
            scope["check_runtime_abi"]()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_kernel_bootstrap_uses_isolated_pinned_runtime(tmp_path, device):
    template = Path(control.__file__).with_name("kernel_template.py.txt")
    tree = ast.parse(template.read_text(encoding="utf-8"))
    bootstrap = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "prepare_runtime")
    lock = tmp_path / "cloud/kaggle/linux-lock.txt"
    lock.parent.mkdir(parents=True)
    lock.write_text("torch==2.14.1\n", encoding="utf-8")
    calls = []
    scope = {"workspace": tmp_path, "JOB": {"device": device}, "sys": SimpleNamespace(executable="launcher-python"),
             "run": lambda command, deadline: calls.append((command, deadline))}
    exec(compile(ast.Module(body=[bootstrap], type_ignores=[]), str(template), "exec"), scope)
    interpreter, pip = scope["prepare_runtime"](10)
    assert "--without-pip" in calls[0][0]
    assert interpreter == [str(tmp_path / "venv/bin/python"), "-X", "faulthandler", "-u"]
    assert pip[pip.index("--python") + 1] == interpreter[0]
    assert any(cmd[-1] == "torch==2.14.1+cpu" for cmd, _ in calls) == (device == "cpu")
    assert calls[-2][0][-2:] == ["-r", str(lock)]
    assert calls[-1][0] == [*pip, "check"]
    assert all(deadline == 10 for _, deadline in calls)


def test_quota_uses_verified_api_minutes(monkeypatch):
    monkeypatch.setattr(credentials, "cli", lambda *a, **kw: (
        0, "informational prefix\nresource,used,remaining,total,refreshAt\n"
           "GPU,2.50h,27.50h,30.00h,2026-11-30\nTPU,0.00h,0.00h,0.00h,\n"))
    assert credentials.accelerator_quota("account_01") == {"GPU": 1649, "TPU": 0}
    assert credentials.authenticated_owner(0, "- username: person\n- auth_method: OAUTH") is None
    assert credentials.authenticated_owner(0, "- username: Person\n- auth_method: ACCESS_TOKEN") == "person"


def test_gate_requires_verified_matching_cloud_artifact():
    job = {"role": "cpu1", "campaign": "stage5", "runtime_sha": "current"}
    reference = {"status": "CLOUD_SUCCEEDED", "execution": {"campaign": "stage5", "runtime_sha": "current"}}
    state = {"jobs": {"j01": reference}}
    with pytest.raises(RuntimeError, match="verified J01"):
        control.require_gate(state, job)
    reference.update(status="HARVESTED_VERIFIED", gates={"G2": "PASS"})
    control.require_gate(state, job)
    with pytest.raises(RuntimeError, match="verified J01"):
        control.require_gate(state, {**job, "runtime_sha": "different"})


def test_private_input_requires_ready_version_and_acl(tmp_path, monkeypatch):
    ref = "example/private"
    state = {"uploads": {ref: {"status": "ACCEPTED"}}}
    monkeypatch.setattr(control, "WORK", tmp_path)
    monkeypatch.setattr(control, "cli", lambda *a: (
        0, 'informational prefix\n{"status":"ready","current_version_number":1}'))
    monkeypatch.setattr(control, "private_input", lambda *a: False)
    with pytest.raises(RuntimeError, match="private ACL"):
        control.dataset_ready(state, "account_01", ref)
    assert "version" not in state["uploads"][ref]
    monkeypatch.setattr(control, "private_input", lambda *a: True)
    assert control.dataset_ready(state, "account_01", ref) == ref
    assert state["uploads"][ref]["version"] == 1


def test_remote_input_requires_original_bytes_and_opaque_filename(tmp_path, monkeypatch):
    payload = tmp_path / "model.zip"
    payload.write_bytes(b"artifact bytes")
    remote_name = "model.bin"

    def fake_cli(alias, args, **kwargs):
        assert args[2] == "example/private/1"
        directory = Path(args[args.index("-p") + 1])
        with zipfile.ZipFile(directory / "download.zip", "w") as output:
            output.writestr(remote_name, payload.read_bytes())
        return 0, ""

    monkeypatch.setattr(control, "cli", fake_cli)
    control.verify_remote_input("account_01", "example/private", "model.bin", sha(payload))
    remote_name = "data"
    with pytest.raises(RuntimeError, match="layout differs"):
        control.verify_remote_input("account_01", "example/private", "model.bin", sha(payload))


def test_job_assignments_are_immutable(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "WORK", tmp_path)
    monkeypatch.setattr(control, "ROOT", tmp_path)
    args = SimpleNamespace(campaign="stage5", round="slots", account="account_01", parent=None,
                           assign=["j03=account_02", "j04=account_08"])
    control.write(tmp_path / "bundle.json", {"runtime_sha": "runtime", "source_sha": "source", "path": ".",
                                            "template_sha": "template"})
    state = control.registry()
    control.plan(args, state)
    jobs = control.read(tmp_path / "plan-stage5-slots.json")["jobs"]
    assert jobs["j01"]["account"] == "account_01"
    assert jobs["j03"]["account"] == "account_02"
    assert jobs["j04"]["account"] == "account_08"
    args.assign = ["j03=account_03"]
    with pytest.raises(ValueError, match="Immutable plan"):
        control.plan(args, state)


def test_followups_pin_harvested_runtime_even_after_repackaging(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "WORK", tmp_path)
    monkeypatch.setattr(control, "ROOT", tmp_path)
    planned = {"campaign": "stage5", "runtime_sha": "old-runtime", "runtime_path": "old-package",
               "source_sha": "source", "template_sha": "old-template", **control.RECIPE["preflight"]}
    execution = {"campaign": "stage5", "workers": 1, "device": "cpu"}
    anchor = {"status": "HARVESTED_VERIFIED", "planned": planned, "execution": execution,
              "state_sha": "parent-digest", "gates": {"G2": "PASS"}}
    state = {"jobs": {"anchor": anchor}, "uploads": {}}
    args = SimpleNamespace(campaign="stage5", round="capacity", account="account_01", parent=None,
                           capacity_after="anchor", assign=["j03=account_02", "j04=account_03"])
    control.write(tmp_path / "bundle.json", {"runtime_sha": "new-runtime"})
    control.plan(args, state)
    jobs = control.read(tmp_path / "plan-stage5-capacity.json")["jobs"]
    assert set(jobs) == {"j02", "j03", "j04"}
    assert {job["runtime_sha"] for job in jobs.values()} == {"old-runtime"}
    assert jobs["j03"]["account"] != jobs["j04"]["account"]
    args.round, args.parent, args.capacity_after, args.assign = "resume", "anchor", None, []
    control.plan(args, state)
    job = control.read(tmp_path / "plan-stage5-resume.json")["jobs"]["j07"]
    assert job["runtime_sha"] == "old-runtime"
    assert job["parent_sha"] == "parent-digest"


def test_download_output_sends_explicit_version_and_filename(tmp_path, monkeypatch):
    def fetch(request):
        assert (request.owner_slug, request.kernel_slug) == ("example", "preflight")
        assert (request.version_number, request.file_path) == (3, "preflight.zip")
        return nullcontext(SimpleNamespace(url="https://example.org/preflight.zip",
                                           iter_content=lambda chunk_size: [b"archived output"]))
    api = SimpleNamespace(build_kaggle_client=lambda: nullcontext(
        SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=SimpleNamespace(download_kernel_output=fetch)))))
    monkeypatch.setitem(sys.modules, "kaggle", SimpleNamespace(api=api))
    monkeypatch.setitem(sys.modules, "kagglesdk.kernels.types.kernels_api_service",
                        SimpleNamespace(ApiDownloadKernelOutputRequest=SimpleNamespace))
    target = tmp_path / "preflight.zip"
    assert kernel_version.download("example/preflight", "3", target) == 0
    assert target.read_bytes() == b"archived output"


@pytest.mark.parametrize("previous, permitted", [("RUNNING", False), ("CAPACITY_REJECTED", False), ("HARVESTED_VERIFIED", True)])
def test_followup_server_admission_respects_account_and_receipt(tmp_path, monkeypatch, previous, permitted):
    monkeypatch.setattr(control, "WORK", tmp_path)
    monkeypatch.setattr(control, "ROOT", tmp_path)
    template = Path(control.__file__).with_name("kernel_template.py.txt")
    runtime = tmp_path / "runtime.zip"
    with zipfile.ZipFile(runtime, "x") as bundle:
        bundle.write(template, "cloud/kaggle/kernel_template.py.txt")
    job = {"job": "test-cpu2", "campaign": "stage5", "role": "cpu2", "account": "account_01",
           "runtime_path": ".", "runtime_sha": sha(runtime), "template_sha": sha(template),
           "max_seconds": control.RECIPE["preflight"]["max_seconds"], "device": "cpu"}
    entry = {"status": "PLANNED", "planned": job}
    state = {"jobs": {"test-cpu2": entry,
                      "previous": {"planned": {**job, "role": "cpu1"}, "status": previous,
                                   "server_admission": {"capacity": "UNKNOWN"}, "owner": "example"}}, "uploads": {}}
    monkeypatch.setattr(control, "planned_job", lambda *_: (job, entry))
    monkeypatch.setattr(control, "require_gate", lambda *_: None)
    monkeypatch.setattr(control, "account_for", lambda _: ("account_01", {
        "owner": "example", "observed_at": control.now()["utc"], "concurrency_observed": None,
    }))
    monkeypatch.setattr(control, "inputs_for", lambda *_: {"runtime_ref": "example/runtime", "source_ref": "example/source"})
    monkeypatch.setattr(control, "probe_inventory", lambda *_: {})
    submissions = []
    monkeypatch.setattr(control, "cli", lambda *args, **kw: (submissions.append(args) or -1, ""))
    args = SimpleNamespace(job="j03", server_admission=True)
    if not permitted:
        with pytest.raises(RuntimeError, match="prior follow-up admission"):
            control.launch(args, state)
        assert not submissions
    else:
        control.launch(args, state)
        control.launch(args, state)
        assert len(submissions) == 1
        assert entry["status"] == "SUBMIT_UNKNOWN"




def test_bounded_replay_wrap_protects_guided_prefix(monkeypatch):
    from types import SimpleNamespace
    from cloud.kaggle import guided, runtime

    writes = []

    def store(model, buffer, value):
        writes.append(buffer.pos)
        buffer.observations[buffer.pos, 0] = value
        buffer.pos += 1
        if buffer.pos == buffer.buffer_size:
            buffer.pos, buffer.full = 0, True

    monkeypatch.setattr(runtime.TimedSAC, "_store_transition", store)
    rows = SimpleNamespace(pos=3, full=False, buffer_size=4)
    for field in guided.REPLAY_FIELDS:
        setattr(rows, field, np.arange(4, dtype=np.float32).reshape(4, 1).copy())
    model = object.__new__(guided.GuidedSAC)
    model.protected_replay_slots = 2
    before = guided.prefix_sha(rows, 2)
    for value in (40, 50, 60, 70):
        model._store_transition(rows, value)
    assert writes == [3, 2, 3, 2]
    assert rows.full and guided.prefix_sha(rows, 2) == before


def test_window_active_evidence_excludes_notebook_initialization(monkeypatch):
    from cloud.kaggle import campaign

    entry = {"status": "RUNNING", "version": 1, "ref": "private/job",
             "planned": {"start_transition": 30000}}
    monkeypatch.setattr(campaign.c, "current_private_kernel_version", lambda *args: 1)
    monkeypatch.setattr(campaign.c, "cli", lambda *args, **kwargs: (0, "Installing dependencies"))
    initial = campaign.training_evidence("account_01", entry, {})
    assert (initial["transitions"], initial["updates"]) == (0, 0)
    monkeypatch.setattr(campaign.c, "cli", lambda *args, **kwargs: (0,
        "TRAINING_ACTIVE transitions=64 updates=32\nPeriodic checkpoint step-000000250000 index_sha256=0123"))
    learned = campaign.training_evidence("account_01", entry, {})
    assert (learned["transitions"], learned["updates"]) == (220000, 32)



def test_unlocalized_vector_does_not_write_any_replay_row(monkeypatch):
    from cloud.kaggle import guided, runtime

    writes = []
    monkeypatch.setattr(runtime.TimedSAC, "_store_transition", lambda *args: writes.append(args))
    model = object.__new__(guided.GuidedSAC)
    model.localization_guard = True
    model.localization_invalid_rows = 0
    model._vec_normalize_env = None
    model.n_envs = 2
    model._last_obs = np.zeros((2, 409), dtype=np.float32)
    next_reset_obs = np.ones_like(model._last_obs)
    model._store_transition(object(), None, next_reset_obs, None, None, [{}, {"localization_discard": True}])
    assert writes == []
    assert model.localization_invalid_rows == 2
    assert model._last_obs is next_reset_obs
    model._store_transition(object(), None, next_reset_obs, None, None, [{}, {}])
    assert len(writes) == 1
