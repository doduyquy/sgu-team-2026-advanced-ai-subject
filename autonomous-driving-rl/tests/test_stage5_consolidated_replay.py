from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from cloud.kaggle.consolidated_replay import MixedReplay, cloud_update


class SentinelReplay:
    def __init__(self, value):
        self.value = value

    def sample(self, count, env=None):
        return ReplayBufferSamples(*(torch.full((count, width), self.value) for width in (409, 2, 409, 1, 1)))


def test_mixed_replay_passes_parent_and_balanced_source_samples_to_same_batch():
    mix = MixedReplay(SentinelReplay(0), {
        "S01": {"Easy": SentinelReplay(1), "Medium": SentinelReplay(2)},
        "S09": {"Easy": SentinelReplay(3), "Medium": SentinelReplay(4)},
    })
    batch = mix.sample(256)
    assert batch.observations.shape == (256, 409)
    assert batch.actions.shape == (256, 2)
    assert torch.bincount(batch.observations[:, 0]).tolist() == [192, 16, 16, 16, 16]
    assert dict(mix.draws) == {"parent": 192, "S01/Easy": 16, "S01/Medium": 16,
                              "S09/Easy": 16, "S09/Medium": 16}


def test_consolidation_rejects_updates_outside_kaggle(tmp_path):
    with pytest.raises(RuntimeError, match="only in the Kaggle"):
        cloud_update(SimpleNamespace(out=tmp_path / "state"))


def test_local_consolidation_rechecks_all_batches_against_one_anchor(tmp_path, monkeypatch):
    from cloud.kaggle import consolidated_replay as consolidation
    from cloud.kaggle.common import write

    datasets = [(tmp_path / name, name) for name in ("first", "second")]
    for path, _ in datasets:
        path.mkdir()
        write(path / "CONSOLIDATED.json", {"sources": [{"label": "S01", "model_sha": "pinned"}]})

    def validate(path, digest, parent, anchor, *, seen, previous_ledger, row_limit):
        assert anchor == "frozen-s01" and row_limit == 75000
        if digest == "second":
            assert seen == {"first"} and previous_ledger["first"] == "accepted"
        seen.add(digest)
        return {"S01": {"Hard": {"reward": np.array([1.0], dtype=np.float32)}}}, [digest], {
            **previous_ledger, digest: "accepted"}

    monkeypatch.setattr(consolidation, "validate_dataset", validate)
    replay, audit, ledger = consolidation.training_replay(
        datasets, {"ledger": {}}, "frozen-s01", row_limit=75000)
    assert replay["S01"]["Hard"]["reward"].tolist() == [1.0, 1.0]
    assert audit == ["first", "second"]
    assert ledger == {"first": "accepted", "second": "accepted"}
    with pytest.raises(ValueError, match="included twice"):
        consolidation.training_replay([datasets[0], datasets[0]], {"ledger": {}}, "frozen-s01", row_limit=75000)


def test_private_launch_stops_before_kaggle_when_operator_has_an_active_receipt(tmp_path, monkeypatch):
    from contextlib import nullcontext

    from cloud.kaggle import consolidation_job as job
    from cloud.kaggle.common import sha, write

    (tmp_path / "kernel.template").write_text("unused", encoding="utf-8")
    path = tmp_path / "PLAN.json"
    write(path, {"script_sha256": sha(tmp_path / "kernel.template"), "account": "account_01"})
    monkeypatch.setattr(job, "operator_slot", lambda: "account_01")
    monkeypatch.setattr(job, "slot", lambda alias: (None, {"authorized": True, "owner": "owner"}))
    monkeypatch.setattr(job.control, "locked", nullcontext)
    monkeypatch.setattr(job.control, "registry", lambda: {"jobs": {"existing": {
        "owner": "owner", "status": "RUNNING", "planned": {"account": "account_01"},
    }}})
    def unexpected(*args, **kwargs):
        raise AssertionError("Kaggle must not be queried when the operator has an active job")
    monkeypatch.setattr(job, "verify_owner", unexpected)
    monkeypatch.setattr(job.control, "probe_inventory", unexpected)
    with pytest.raises(RuntimeError, match="No unused execution slot"):
        job.submit(tmp_path, sha(path))
    assert not (tmp_path / "RECEIPT.json").exists()
