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
from stable_baselines3.common.vec_env import DummyVecEnv

from cloud.kaggle import control
from cloud.kaggle import credentials
from cloud.kaggle import kernel_version
from cloud.kaggle.capacity import log_times
from cloud.kaggle.checkpoint import PARTS, export_checkpoint, restore_checkpoint
from cloud.kaggle.common import SOURCE_SHA, sha, write
from cloud.kaggle.credentials import dpapi
from cloud.kaggle.pilot import compare
from cloud.kaggle.replay import admit, check_snapshot, validate
from cloud.kaggle.runtime import check_state, load_policy, pool, restore_rng, save_state, schedule

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


def test_easy_pilot_compares_two_distinct_checkpoints_on_exact_panel():
    def evaluation(model, successes):
        rows = [{"geometry": str(i // 2), "family": "SCS", "seed": i % 2, "censored": False,
                 "clean": i < successes, "outcome": "SUCCESS" if i < successes else "OUT_OF_ROAD"}
                for i in range(12)]
        return {"model_sha": model, "tier": "Easy", "cases": rows}
    baseline = evaluation(SOURCE_SHA, 8)
    candidate = evaluation("checkpoint-1", 10)
    report = compare(baseline, [candidate, evaluation("checkpoint-2", 11)])
    assert report["two_milestones_meet_thresholds"]
    assert report["milestones"][0]["improved_cases"] == 2
    candidate["cases"][0]["seed"] = 99
    with pytest.raises(ValueError, match="exact baseline case IDs"):
        compare(baseline, [candidate])


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

