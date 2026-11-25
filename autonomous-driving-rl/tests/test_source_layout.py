"""Import safety for the retained legacy compatibility entrypoint."""
import importlib
import unittest
from unittest.mock import patch


class TestLegacyImportSafety(unittest.TestCase):
    def test_import_and_reload_do_not_start_legacy_simulations(self):
        with patch("src.environments.course_env_v1.CourseEnvV1") as environment:
            module = importlib.import_module("src.evaluation.evaluate_random")
            importlib.reload(module)
            environment.assert_not_called()
            self.assertTrue(callable(module.main))
