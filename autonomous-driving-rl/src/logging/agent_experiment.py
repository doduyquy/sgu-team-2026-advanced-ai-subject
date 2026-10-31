"""Durable, algorithm-neutral source/configuration and decision artifacts."""

import gzip
import hashlib
import inspect
import json
from pathlib import Path


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload, exclusive=False):
    with Path(path).open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def public_configuration(agent):
    if callable(getattr(agent, "export_config", None)):
        return agent.export_config()
    # Legacy policies expose numeric/public configuration directly.
    return {key: value for key, value in vars(agent).items()
            if not key.startswith("_") and isinstance(value, (str, int, float, bool, type(None)))}


def save_agent_provenance(run_dir, agent, source_files=None):
    root = Path(__file__).resolve().parents[2]
    sources = dict(source_files or {
        "policy.py": Path(inspect.getsourcefile(type(agent))),
        "agent_input.py": root / "src/platform/agent_input.py",
    })
    # Include helper modules as well as the entry-point policy.
    if source_files is None and type(agent).__module__.startswith("src.agents"):
        for path in (root / "src/agents").rglob("*.py"):
            sources["agents/" + path.relative_to(root / "src/agents").as_posix()] = path
    sources["action_adapter.py"] = root / "src/platform/action_adapter.py"
    sources["agent_experiment.py"] = Path(__file__)
    destination = Path(run_dir) / "sources"
    destination.mkdir()
    hashes = {}
    for name, source in sources.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        content = Path(source).read_bytes()
        with target.open("xb") as stream:
            stream.write(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    configuration = public_configuration(agent)
    configuration_hash = hashlib.sha256(json.dumps(
        configuration, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    payload = {"descriptor": agent.descriptor.to_dict(), "configuration": configuration,
               "configuration_sha256": configuration_hash, "source_sha256": hashes}
    write_json(Path(run_dir) / "agent_provenance.json", payload, exclusive=True)
    return payload


class DecisionTrace:
    """Compressed public input and action trace; existing files are never reused."""

    def __init__(self, path):
        self.stream = gzip.open(path, "xt", encoding="utf-8", compresslevel=3)
        self.count = 0

    def write(self, case_id, agent_input, decision, canonical_action, latency_ms=None):
        self.stream.write(json.dumps({
            "case_id": case_id, "step_index": agent_input.step_index,
            "agent_input": agent_input.to_dict(), "decision": decision.to_dict(),
            "canonical_action": [float(x) for x in canonical_action],
            "act_latency_ms": latency_ms,
        }, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
        self.count += 1
        if self.count % 50 == 0:
            self.stream.flush()

    def close(self):
        self.stream.close()
