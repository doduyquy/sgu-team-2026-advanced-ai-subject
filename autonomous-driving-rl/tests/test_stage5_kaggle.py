"""Local correctness regressions for the Stage 5 Kaggle campaign."""
from pathlib import Path
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
from cloud.kaggle.common import sha
from cloud.kaggle.credentials import dpapi
from cloud.kaggle.replay import admit
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

def test_timeout_receipt_prevents_duplicate_submit(tmp_path, monkeypatch):
    wrapper = SimpleNamespace(campaign="stage5", round="test", account="account_01", job="j01", parent=None)
    monkeypatch.setattr(control, "WORK", tmp_path)
    monkeypatch.setattr(control, "ROOT", tmp_path)
    monkeypatch.setattr(control, "slot", lambda _: (tmp_path, {"owner": "example", "observed_at": control.now()["utc"],
                                                               "concurrency_observed": 1, "gpu_minutes_observed": 0}))
    monkeypatch.setattr(control, "verify_owner", lambda _: True)
    monkeypatch.setattr(control, "upload", lambda *a: "example/private-test")
    pushes = []
    def fake_cli(alias, arguments, **kwargs):
        pushes.append((alias, arguments))
        return -1, ""
    monkeypatch.setattr(control, "cli", fake_cli)
    template = Path(control.__file__).with_name("kernel_template.py.txt")
    control.write(tmp_path / "bundle.json", {"runtime_sha": "runtime", "source_sha": "source", "path": ".",
                                            "template_sha": sha(template)})
    state = control.registry()
    control.plan(wrapper, state)
    wrapper.fingerprint = control.read(tmp_path / "plan-stage5-test.json")["fingerprint"]
    control.launch(wrapper, state)
    control.launch(wrapper, control.registry())
    assert len(pushes) == 1
    job = next(x for x in state["jobs"].values() if x["planned"]["role"] == "parity")
    assert job["status"] == "SUBMIT_UNKNOWN"
    script = tmp_path / "kernels" / job["planned"]["job"] / "run.py"
    compile(script.read_text(encoding="utf-8"), str(script), "exec")


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

