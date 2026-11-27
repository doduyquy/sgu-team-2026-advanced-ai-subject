from pathlib import Path
import shutil

import pytest

from cloud.kaggle import checkpoint, second_window_harvest
from cloud.kaggle.common import sha, write


def remote_checkpoint(tmp_path):
    root = tmp_path / "remote"
    root.mkdir()
    files = {}
    for name in checkpoint.PARTS:
        path = root / name
        path.write_text(name, encoding="utf-8")
        files[name] = sha(path)
    write(root / "COMPLETE.json", {"schema": 1, "job": "sample-job", "result": "COMPLETE", "files": files})
    package = tmp_path / "package"
    index = checkpoint.export_checkpoint(root, sha(root / "COMPLETE.json"), package, shard_bytes=100)
    return package, index


@pytest.mark.parametrize("full", [False, True])
def test_selective_harvest_verifies_versioned_bytes_and_limits_downloads(tmp_path, monkeypatch, full):
    package, index = remote_checkpoint(tmp_path)
    fetched = []

    def transport(entry, prefix, destination, name):
        assert prefix == "sample-job/final"
        fetched.append(name)
        shutil.copyfile(package / name, Path(destination) / name)

    monkeypatch.setattr(second_window_harvest.c, "fetch_output", transport)
    result = second_window_harvest.fetch_checkpoint(
        {"planned": {"job": "sample-job"}}, "sample-job/final", tmp_path / "local", index, full=full)
    assert result["policy_only"] is not full
    assert any(name.startswith("replay.pkl") for name in fetched) is full
    assert (tmp_path / "local/model.zip").is_file()
    assert (tmp_path / "local/rng.pt").is_file() is full


def test_checksum_mismatch_leaves_no_candidate(tmp_path, monkeypatch):
    package, index = remote_checkpoint(tmp_path)
    (package / "model.zip.000000.bin").write_bytes(b"changed")
    monkeypatch.setattr(second_window_harvest.c, "fetch_output",
                        lambda entry, prefix, folder, name: shutil.copyfile(package / name, folder / name))
    target = tmp_path / "local"
    with pytest.raises(ValueError):
        second_window_harvest.fetch_checkpoint({"planned": {"job": "sample-job"}},
                                               "sample-job/final", target, index)
    assert not target.exists()
