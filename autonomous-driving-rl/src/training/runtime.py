"""Process-scoped Git provenance access for the two explicit local checkouts."""

from contextlib import contextmanager
import os
from pathlib import Path


@contextmanager
def local_training_runtime(project_root: Path):
    repository = project_root.parent.resolve()
    additions = {}
    count = int(os.environ.get("GIT_CONFIG_COUNT", "0"))
    paths = (repository, repository / ".venv/metadrive-src")
    for index, path in enumerate(paths, count):
        additions[f"GIT_CONFIG_KEY_{index}"] = "safe.directory"
        additions[f"GIT_CONFIG_VALUE_{index}"] = path.as_posix()
    additions["GIT_CONFIG_COUNT"] = str(count + len(paths))
    previous = {key: os.environ.get(key) for key in additions}
    os.environ.update(additions)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
