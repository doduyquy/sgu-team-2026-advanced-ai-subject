"""Experiment records preserve inputs, parameters and immutable source evidence."""

import gzip
import json
from pathlib import Path
import tempfile
import unittest

from src.agents.rule_based_agent import Stage1RuleBasedAgent
from src.logging.agent_experiment import DecisionTrace, save_agent_provenance, sha256_file
from tests.test_rule_based_agent import make_input


class TestAgentExperimentLogging(unittest.TestCase):
    def test_trace_contains_public_input_and_actual_action_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl.gz"
            inp = make_input()
            decision = Stage1RuleBasedAgent().act(inp)
            trace = DecisionTrace(path)
            trace.write("TRAIN/test", inp, decision, decision.action_payload, 0.2)
            trace.close()
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                row = json.loads(stream.readline())
            self.assertEqual(row["agent_input"], inp.to_dict())
            self.assertEqual(row["decision"], decision.to_dict())
            self.assertEqual(row["canonical_action"], decision.action_payload)
            with self.assertRaises(FileExistsError):
                DecisionTrace(path)

    def test_provenance_archives_sources_and_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = Stage1RuleBasedAgent(cruise_speed_kmh=35)
            provenance = save_agent_provenance(directory, agent)
            root = Path(directory)
            self.assertEqual(provenance["configuration"]["cruise_speed_kmh"], 35)
            self.assertEqual(provenance["descriptor"]["agent_version"], agent.descriptor.agent_version)
            for name, expected in provenance["source_sha256"].items():
                self.assertEqual(sha256_file(root / "sources" / name), expected)
            with self.assertRaises(FileExistsError):
                save_agent_provenance(directory, agent)


if __name__ == "__main__":
    unittest.main()
