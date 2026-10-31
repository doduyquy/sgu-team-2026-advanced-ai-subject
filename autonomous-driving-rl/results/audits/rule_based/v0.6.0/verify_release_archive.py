"""Read-only archive integrity and flat-package P4 replay; new proof only."""
import gzip
import json
from pathlib import Path
import sys
import types
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.replay_rule_based_trace import load_module, restore_input
from src.logging.agent_experiment import sha256_file, write_json
from src.platform.agent import AgentPublicEpisodeContext
from src.platform.action_adapter import ContinuousBox2Adapter

out = Path(__file__).parent / "release_verification.json"
if out.exists():
    raise FileExistsError(out)
archives = {}
for label in ("0.5.0", "0.6.0"):
    folder = ROOT / "experiments/rule_based/versions" / label
    record = json.loads((folder / "snapshot.json").read_text(encoding="utf8"))
    checked = {name: sha256_file(folder / name) == expected for name, expected in record["source_sha256"].items()}
    if not all(checked.values()):
        raise ValueError("Archive source mismatch: " + label)
    archives[label] = {"snapshot_sha256": sha256_file(folder / "snapshot.json"), "files_match": checked}
folder = ROOT / "experiments/rule_based/versions/0.6.0"
package = types.ModuleType("_archive_060_verification")
package.__path__ = [str(folder)]
sys.modules[package.__name__] = package
policy = load_module(package.__name__ + ".policy", folder / "policy.py")
inputs = load_module(package.__name__ + ".inputs", folder / "agent_input.py")
rows = []
for scenario in ("safe_overtake", "fast_rear", "mandatory_route", "mandatory_left", "precommit_abort", "postcommit_front_block"):
    path = ROOT / "runs/rule_based_v060_probe10_20261030" / scenario / "decisions.jsonl.gz"
    agent = policy.Stage1RuleBasedAgent()
    agent.reset(AgentPublicEpisodeContext())
    count = 0
    with gzip.open(path, "rt", encoding="utf8") as stream:
        for count, line in enumerate(stream, 1):
            row = json.loads(line)
            inp = restore_input(row["agent_input"], inputs)
            if inp.to_dict() != row["agent_input"]:
                raise ValueError("Archived input round-trip mismatch")
            decision = agent.act(inp)
            action = ContinuousBox2Adapter().to_canonical(decision.action_payload).to_numpy().tolist()
            if decision.to_dict() != row["decision"] or action != row["canonical_action"]:
                raise ValueError(f"Archived decision mismatch: {scenario}/{count-1}")
    agent.close()
    rows.append({"scenario": scenario, "steps": count, "trace_sha256": sha256_file(path), "exact_match": True})
proof = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "archives": archives,
         "archive_flat_package_replays": rows, "total_exact_steps": sum(row["steps"] for row in rows),
         "scope": "Verbatim archives plus actual import of flat archived policy/helpers/input; live shared API/adapter hashes recorded; latency excluded",
         "shared_api_sha256": sha256_file(ROOT / "src/platform/agent.py"),
         "adapter_sha256": sha256_file(ROOT / "src/platform/action_adapter.py"),
         "verification_script_sha256": sha256_file(__file__)}
write_json(out, proof, exclusive=True)
print(json.dumps({"verified_archives": list(archives), "exact_archive_replay_steps": proof["total_exact_steps"]}))
