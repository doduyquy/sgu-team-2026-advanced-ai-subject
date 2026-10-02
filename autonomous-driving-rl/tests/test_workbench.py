"""
Unit and Integration Tests for Research Workbench (Gate 7.5B Pass B1 Correction 1).

Tests must execute offscreen without requiring a visible display (QT_QPA_PLATFORM=offscreen).
Verifies:
1. QApplication / MainWindow smoke construction.
2. Request widgets map exactly to LaunchRequestV1-compatible payload.
3. Mode switching disables/enables the expected UX controls.
4. Editing any request field invalidates prior preflight state and disables Run.
5. Run stays disabled when preflight.can_execute=False.
6. Run enables only when preflight.can_execute=True.
7. Worker protocol parser accepts sentinel JSON.
8. Worker protocol parser ignores ordinary MetaDrive stdout.
9. Malformed sentinel JSON is handled safely.
10. PLAN operation returns plan + preflight without simulator execution.
11. Worker LAUNCHER_EVENT maps to UI run-monitor state (using production timestamp_utc).
12. GUI never exposes partial VALIDATION/TEST execution.
13. W&B API key is never serialized into request/protocol messages.
14. Workbench imports do not modify Gate 7.5A registries/contracts.
15. Explicit verification of all frozen contract hashes across Gates 5, 6, 7, and 7.5A.
16. Successful RUN emits WORKER_DONE.success == True and matching run_id.
17. CoreAdapter loads exact manifest counts (TRAIN=180, VALIDATION=96, TEST=60).
18. Extreme tier is available in Experiment Setup and Case Explorer.
19. Case Explorer manifest load errors are observable, not silently converted to valid empty state.
20. Worker WORKER_READY does not expose private absolute interpreter path.
21. Embedded sentinel text in stdout is ignored (prefix-strict).
22. Wrong protocol version is rejected.
23. QProcess start failure cleanly resets WorkbenchProcessRunner state.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

# Ensure offscreen Qt platform before importing PySide6
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QApplication

from src.launcher.cases import (
    LOCKED_GEOMETRY_SPLIT_MANIFEST_SHA256,
    LOCKED_TEST_CASE_MANIFEST_SHA256,
    LOCKED_VALIDATION_CASE_MANIFEST_SHA256,
    get_default_project_root,
)
from src.launcher.contracts import (
    GATE7_LOCKED_OBSERVABILITY_HASH,
    build_launcher_contract_core,
    compute_launcher_contract_sha256,
    compute_platform_execution_contract_sha256,
)
from src.launcher.events import LauncherEventType, LauncherEventV1
from src.launcher.models import LaunchRequestV1, LauncherMode
from src.launcher.registry import build_canonical_agent_registry, compute_canonical_registry_sha256
from src.launcher.resolver import (
    GATE5_LOCKED_BENCHMARK_HASH,
    GATE6_LOCKED_AGENT_HASH,
    GATE6_LOCKED_RUNTIME_HASH,
    GATE7_LOCKED_LOGGING_HASH,
)
from src.platform import RunStatus, WandbMode
from src.workbench.contracts import (
    GATE7_5A_BASELINE_MERGE_SHA,
    GATE7_5A_LAUNCHER_CONTRACT_HASH,
    GATE7_5A_PLATFORM_EXECUTION_CONTRACT_HASH,
    PINNED_PYSIDE6_VERSION,
    WORKBENCH_PROTOCOL_VERSION,
    build_workbench_contract_core,
    compute_platform_workbench_contract_sha256,
    compute_workbench_contract_sha256,
)
from src.workbench.core_adapter import CoreAdapter
from src.workbench.main_window import MainWindow
from src.workbench.process_runner import WorkbenchProcessRunner
from src.workbench.protocol import (
    WORKBENCH_SENTINEL,
    WorkbenchMessageType,
    WorkbenchMessageV1,
    deserialize_launch_request,
    parse_sentinel_line,
    serialize_launch_request,
)


class TestWorkbenchSuite(unittest.TestCase):
    """Comprehensive Gate 7.5B Workbench test suite."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication([])

    def test_01_app_and_main_window_smoke_construction(self):
        """MainWindow instantiates cleanly with all tabs in offscreen mode."""
        window = MainWindow()
        self.assertIsNotNone(window)
        self.assertEqual(window.tab_widget.count(), 5)
        self.assertEqual(window.tab_widget.tabText(0), "1. Experiment Setup")
        self.assertEqual(window.tab_widget.tabText(1), "2. Plan & Preflight")
        self.assertEqual(window.tab_widget.tabText(2), "3. Run Monitor")
        self.assertEqual(window.tab_widget.tabText(3), "4. Agents Explorer")
        self.assertEqual(window.tab_widget.tabText(4), "5. Case Explorer")
        window.close()

    def test_02_request_widgets_map_to_launch_request(self):
        """ExperimentSetupWidget correctly constructs LaunchRequestV1."""
        adapter = CoreAdapter()
        window = MainWindow(core_adapter=adapter)
        setup = window.setup_widget

        setup.combo_mode.setCurrentText("SANDBOX")
        idx = setup.combo_agent.findData("fixture_constant_continuous")
        self.assertGreaterEqual(idx, 0)
        setup.combo_agent.setCurrentIndex(idx)

        setup.combo_tier.setCurrentText("Easy")
        setup.edit_sequence.setText("SCCS")
        setup.edit_geom_seed.setText("0")
        setup.edit_env_seed.setText("0")
        setup.combo_render.setCurrentText("OFF")
        setup.combo_wandb.setCurrentText("DISABLED")

        req = setup.build_launch_request()
        self.assertIsInstance(req, LaunchRequestV1)
        self.assertEqual(req.mode, LauncherMode.SANDBOX)
        self.assertEqual(req.agent_id, "fixture_constant_continuous")
        self.assertEqual(req.tier, "Easy")
        self.assertEqual(req.sequence, "SCCS")
        self.assertEqual(req.geometry_generation_seed, 0)
        self.assertEqual(req.environment_seed, 0)
        self.assertEqual(req.render_mode, "OFF")
        self.assertEqual(req.wandb_mode, WandbMode.DISABLED)
        window.close()

    def test_03_mode_switching_disables_enables_ux_controls(self):
        """Benchmark modes disable case selection and lock render=OFF."""
        window = MainWindow()
        setup = window.setup_widget

        setup.combo_mode.setCurrentText("VALIDATION")
        self.assertFalse(setup.combo_tier.isEnabled())
        self.assertFalse(setup.edit_sequence.isEnabled())
        self.assertFalse(setup.edit_geom_seed.isEnabled())
        self.assertFalse(setup.edit_env_seed.isEnabled())
        self.assertFalse(setup.combo_render.isEnabled())
        self.assertEqual(setup.combo_render.currentText(), "OFF")

        setup.combo_mode.setCurrentText("SANDBOX")
        self.assertTrue(setup.combo_tier.isEnabled())
        self.assertTrue(setup.edit_sequence.isEnabled())
        self.assertTrue(setup.edit_geom_seed.isEnabled())
        self.assertTrue(setup.edit_env_seed.isEnabled())
        self.assertTrue(setup.combo_render.isEnabled())
        window.close()

    def test_04_editing_field_invalidates_preflight_and_disables_run(self):
        """Any field edit resets preflight state and disables the Run button."""
        window = MainWindow()
        setup = window.setup_widget

        setup.set_preflight_verdict(True)
        self.assertTrue(setup.btn_run.isEnabled())

        setup.edit_sequence.setText("NEW_SEQ")
        self.assertFalse(setup.btn_run.isEnabled())

        setup.set_preflight_verdict(True)
        self.assertTrue(setup.btn_run.isEnabled())
        setup.combo_render.setCurrentText("NATIVE")
        self.assertFalse(setup.btn_run.isEnabled())
        window.close()

    def test_05_run_stays_disabled_when_preflight_fails(self):
        """Run button remains disabled when preflight fails."""
        window = MainWindow()
        setup = window.setup_widget

        setup.set_preflight_verdict(False)
        self.assertFalse(setup.btn_run.isEnabled())
        window.close()

    def test_06_run_enables_only_when_preflight_succeeds(self):
        """Run button enables when preflight passes and worker is idle."""
        window = MainWindow()
        setup = window.setup_widget

        setup.set_preflight_verdict(True)
        self.assertTrue(setup.btn_run.isEnabled())

        setup.set_worker_running(True)
        self.assertFalse(setup.btn_run.isEnabled())
        self.assertFalse(setup.btn_resolve.isEnabled())
        self.assertTrue(setup.btn_terminate.isEnabled())

        setup.set_worker_running(False)
        self.assertTrue(setup.btn_run.isEnabled())
        self.assertTrue(setup.btn_resolve.isEnabled())
        self.assertFalse(setup.btn_terminate.isEnabled())
        window.close()

    def test_07_protocol_parser_accepts_sentinel_json(self):
        """Protocol parser parses valid sentinel lines into WorkbenchMessageV1."""
        payload = {"agent_id": "test_agent", "count": 42}
        msg = WorkbenchMessageV1.create(WorkbenchMessageType.WORKER_READY, payload)
        line = msg.to_line()

        parsed = parse_sentinel_line(line)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.type, WorkbenchMessageType.WORKER_READY)
        self.assertEqual(parsed.protocol_version, WORKBENCH_PROTOCOL_VERSION)
        self.assertEqual(parsed.payload, payload)

    def test_08_protocol_parser_ignores_ordinary_stdout(self):
        """Protocol parser safely returns None for non-sentinel stdout lines."""
        lines = [
            ":metadrive: MetaDrive version 0.4.3",
            "Known Pipes: wglGraphicsPipe",
            "Start Scenario Index: 0, Num Scenarios: 1",
            "Episode 1 finished. Reward: 12.3",
            "wandb: Currently logged in as: dev",
            "",
            "   ",
        ]
        for line in lines:
            self.assertIsNone(parse_sentinel_line(line))

    def test_09_malformed_sentinel_json_handled_safely(self):
        """Malformed JSON after sentinel does not crash parser."""
        bad_lines = [
            f"{WORKBENCH_SENTINEL}",
            f"{WORKBENCH_SENTINEL} not valid json",
            f"{WORKBENCH_SENTINEL} [1, 2, 3]",
            f"{WORKBENCH_SENTINEL} {{\"protocol_version\": \"WORKBENCH_PROTOCOL_V1\"}}",
            f"{WORKBENCH_SENTINEL} {{\"protocol_version\": \"WORKBENCH_PROTOCOL_V1\", \"type\": \"NONEXISTENT_TYPE\", \"payload\": {{}}}}",
        ]
        for line in bad_lines:
            self.assertIsNone(parse_sentinel_line(line))

    def test_10_plan_operation_returns_plan_and_preflight_without_execution(self):
        """Worker subprocess PLAN operation resolves plan and preflight without simulation."""
        project_root = get_default_project_root()
        worker_script = project_root / "src" / "workbench" / "worker.py"
        req = '{"mode": "SANDBOX", "agent_id": "fixture_constant_continuous", "tier": "Easy", "render_mode": "OFF", "wandb_mode": "DISABLED"}'

        proc = subprocess.run(
            [sys.executable, str(worker_script), "--operation", "PLAN", "--request-json", req],
            capture_output=True,
            text=True,
            cwd=str(project_root),
        )
        self.assertEqual(proc.returncode, 0)

        received_types = []
        for line in proc.stdout.splitlines():
            msg = parse_sentinel_line(line)
            if msg:
                received_types.append(msg.type)

        self.assertIn(WorkbenchMessageType.WORKER_READY, received_types)
        self.assertIn(WorkbenchMessageType.PLAN_RESOLVED, received_types)
        self.assertIn(WorkbenchMessageType.PREFLIGHT_REPORT, received_types)
        self.assertIn(WorkbenchMessageType.WORKER_DONE, received_types)
        self.assertNotIn(WorkbenchMessageType.LAUNCHER_EVENT, received_types)
        self.assertNotIn(WorkbenchMessageType.EXECUTION_REPORT, received_types)

    def test_11_worker_launcher_event_maps_to_run_monitor_state(self):
        """RunMonitorWidget updates UI elements when receiving LauncherEvent payloads with timestamp_utc."""
        window = MainWindow()
        monitor = window.run_monitor_widget

        event_payload = {
            "event": {
                "event_type": "EPISODE_STARTED",
                "run_id": "test_run_123",
                "episode_index": 0,
                "total_episodes": 1,
                "step_index": 0,
                "route_completion": 0.0,
                "speed_kmh": 0.0,
                "status": "RUNNING",
                "message": "Episode 1 started",
                "timestamp_utc": "2026-10-02T12:00:00Z",
            }
        }
        monitor.handle_launcher_event(event_payload)
        self.assertIn("test_run_123", monitor.lbl_run_id.text())
        self.assertIn("1 / 1", monitor.lbl_episode.text())
        self.assertIn("2026-10-02T12:00:00Z", monitor.text_event_log.toPlainText())

        step_payload = {
            "event": {
                "event_type": "STEP_COMPLETED",
                "run_id": "test_run_123",
                "episode_index": 0,
                "total_episodes": 1,
                "step_index": 50,
                "route_completion": 0.25,
                "speed_kmh": 32.5,
                "status": "RUNNING",
                "message": "Step 50",
                "timestamp_utc": "2026-10-02T12:00:05Z",
            }
        }
        monitor.handle_launcher_event(step_payload)
        self.assertIn("50", monitor.lbl_step.text())
        self.assertIn("25.0%", monitor.lbl_route.text())
        self.assertIn("32.5", monitor.lbl_speed.text())
        window.close()

    def test_12_gui_never_exposes_partial_validation_test_execution(self):
        """CaseExplorerWidget has no run actions and cannot trigger subset execution."""
        window = MainWindow()
        case_explorer = window.case_explorer_widget

        buttons = case_explorer.findChildren(type(window.setup_widget.btn_run))
        for btn in buttons:
            self.assertNotIn("run", btn.text().lower())
            self.assertNotIn("execute", btn.text().lower())
        window.close()

    def test_13_wandb_api_key_is_never_serialized(self):
        """Launch request serialization strictly strips secrets."""
        req = LaunchRequestV1(
            mode=LauncherMode.SANDBOX,
            agent_id="fixture_constant_continuous",
            render_mode="OFF",
        )
        serialized = serialize_launch_request(req)
        self.assertNotIn("WANDB_API_KEY", serialized)
        self.assertNotIn("api_key", serialized)
        self.assertNotIn("secret", serialized)

    def test_14_workbench_imports_do_not_modify_launcher_registry_or_contracts(self):
        """Importing Workbench does not alter Gate 7.5A registry or contract hashes."""
        reg = build_canonical_agent_registry()
        self.assertEqual(len(reg.list_all()), 4)
        launcher_core = build_launcher_contract_core()
        launcher_hash = compute_launcher_contract_sha256(launcher_core)
        self.assertEqual(launcher_hash, GATE7_5A_LAUNCHER_CONTRACT_HASH)

    def test_15_explicit_frozen_contract_hashes_across_all_gates(self):
        """Explicitly verifies all prior locked contract hashes from Gates 5, 6, 7, and 7.5A."""
        # Gate 5
        self.assertEqual(GATE5_LOCKED_BENCHMARK_HASH, "9ddd889b84d8705fae618879e5035556c80d0276a3dc2a58a7963937ebb59f77")
        # Gate 6
        self.assertEqual(GATE6_LOCKED_AGENT_HASH, "53aa37079ff44afa75d9a3f921b0c1f98c4600d882fced51fc8d9197795058eb")
        self.assertEqual(GATE6_LOCKED_RUNTIME_HASH, "c7698768539a769c7b2bc6b90771ff234276a974b6e0353bbb03119faf2f79ad")
        # Gate 7
        self.assertEqual(GATE7_LOCKED_LOGGING_HASH, "0d17915556d83f49d9519576e95920c0853700f33a91593ef85b3aeef6cbb9f2")
        self.assertEqual(GATE7_LOCKED_OBSERVABILITY_HASH, "f00c27fca8abf8145596576cd0dd2540181ff8ae7eccf7806c864921933d3581")
        # Gate 7.5A
        self.assertEqual(
            compute_canonical_registry_sha256(build_canonical_agent_registry()),
            "9fac42d07949796d00b6dc869b98c945eae7a59355d7928f429e457f1ee7f26f",
        )
        self.assertEqual(
            compute_launcher_contract_sha256(build_launcher_contract_core()),
            "5e6269b6e32349bd3ee34ca52583c00f832bd0a737330c894e9a8d314653db04",
        )
        self.assertEqual(
            compute_platform_execution_contract_sha256(GATE7_5A_LAUNCHER_CONTRACT_HASH),
            "442beafd86212419dc7b3edfa60e53715bf33877d3dd72e61c54091520b0563d",
        )

    def test_16_successful_run_emits_worker_done_success_and_matching_run_id(self):
        """Successful run reports status COMPLETE and WORKER_DONE success=True with matching run_id."""
        project_root = get_default_project_root()
        worker_script = project_root / "src" / "workbench" / "worker.py"
        req = {
            "mode": "SANDBOX",
            "agent_id": "fixture_constant_continuous",
            "tier": "Easy",
            "render_mode": "OFF",
            "wandb_mode": "DISABLED",
            "custom_run_id": "test_e2e_success_run",
        }

        proc = subprocess.run(
            [sys.executable, str(worker_script), "--operation", "RUN", "--request-json", json.dumps(req)],
            capture_output=True,
            text=True,
            cwd=str(project_root),
        )
        self.assertEqual(proc.returncode, 0)

        exec_report_payload = None
        worker_done_payload = None

        for line in proc.stdout.splitlines():
            msg = parse_sentinel_line(line)
            if msg:
                if msg.type == WorkbenchMessageType.EXECUTION_REPORT:
                    exec_report_payload = msg.payload["execution_report"]
                elif msg.type == WorkbenchMessageType.WORKER_DONE:
                    worker_done_payload = msg.payload

        self.assertIsNotNone(exec_report_payload)
        self.assertIsNotNone(worker_done_payload)

        self.assertEqual(exec_report_payload["status"], "COMPLETE")
        self.assertEqual(worker_done_payload["operation"], "RUN")
        self.assertTrue(worker_done_payload["success"])
        self.assertFalse(worker_done_payload["blocked"])
        self.assertEqual(worker_done_payload["run_id"], exec_report_payload["run_id"])

        # Clean up temporary run directory created in runs/
        import shutil
        run_dir = project_root / "runs" / "test_e2e_success_run"
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_17_core_adapter_manifest_exact_counts(self):
        """CoreAdapter loads authoritative Gate-5 manifests with exact counts: TRAIN 180, VAL 96, TEST 60."""
        adapter = CoreAdapter()
        train_cases = adapter.load_case_manifest("TRAIN")
        val_cases = adapter.load_case_manifest("VALIDATION")
        test_cases = adapter.load_case_manifest("TEST")

        self.assertEqual(len(train_cases), 180)
        self.assertEqual(len(val_cases), 96)
        self.assertEqual(len(test_cases), 60)

        # Verify known fields present in each split
        self.assertIn("geometry_generation_seed", train_cases[0])
        self.assertIn("traffic_density", val_cases[0])
        self.assertIn("horizon_steps", test_cases[0])

    def test_18_extreme_tier_available_in_setup_and_explorer(self):
        """Extreme tier is selectable in Experiment Setup and filterable in Case Explorer."""
        window = MainWindow()
        setup = window.setup_widget
        explorer = window.case_explorer_widget

        # Check Setup tier combobox
        setup_tiers = [setup.combo_tier.itemText(i) for i in range(setup.combo_tier.count())]
        self.assertIn("Extreme", setup_tiers)

        # Select Extreme tier and check request construction
        idx = setup.combo_tier.findText("Extreme")
        setup.combo_tier.setCurrentIndex(idx)
        req = setup.build_launch_request()
        self.assertEqual(req.tier, "Extreme")

        # Check Case Explorer tier filter combobox
        explorer_tiers = [explorer.combo_tier.itemText(i) for i in range(explorer.combo_tier.count())]
        self.assertIn("Extreme", explorer_tiers)
        window.close()

    def test_19_case_explorer_manifest_load_errors_observable(self):
        """Case Explorer shows visible error when manifest loading fails, rather than silently pretending valid empty."""
        from unittest.mock import patch
        window = MainWindow()
        window.show()
        explorer = window.case_explorer_widget

        with patch.object(explorer._core_adapter, "load_case_manifest", side_effect=FileNotFoundError("Mock manifest missing")):
            explorer._load_cases()
            self.assertFalse(explorer.lbl_status.isHidden())
            self.assertIn("ERROR", explorer.lbl_status.text())
            self.assertEqual(explorer.table_cases.rowCount(), 0)
        window.close()

    def test_20_worker_ready_sanitized_basename_only(self):
        """WORKER_READY payload does not contain private absolute interpreter path."""
        project_root = get_default_project_root()
        worker_script = project_root / "src" / "workbench" / "worker.py"
        req = '{"mode": "SANDBOX", "agent_id": "fixture_constant_continuous", "tier": "Easy", "render_mode": "OFF", "wandb_mode": "DISABLED"}'

        proc = subprocess.run(
            [sys.executable, str(worker_script), "--operation", "PLAN", "--request-json", req],
            capture_output=True,
            text=True,
            cwd=str(project_root),
        )
        self.assertEqual(proc.returncode, 0)

        for line in proc.stdout.splitlines():
            msg = parse_sentinel_line(line)
            if msg and msg.type == WorkbenchMessageType.WORKER_READY:
                py_exec = msg.payload.get("python_executable", "")
                self.assertNotIn("\\", py_exec)
                self.assertNotIn("/", py_exec)
                self.assertEqual(py_exec, Path(sys.executable).name)
                break

    def test_21_embedded_sentinel_is_ignored_prefix_strict(self):
        """Sentinel that appears in the middle of a line is strictly rejected."""
        embedded_line = f"MetaDrive log text {WORKBENCH_SENTINEL}{{\"protocol_version\": \"WORKBENCH_PROTOCOL_V1\", \"type\": \"WORKER_READY\", \"payload\": {{}}}}"
        self.assertIsNone(parse_sentinel_line(embedded_line))

        leading_space_line = f"   {WORKBENCH_SENTINEL}{{\"protocol_version\": \"WORKBENCH_PROTOCOL_V1\", \"type\": \"WORKER_READY\", \"payload\": {{}}}}"
        parsed = parse_sentinel_line(leading_space_line)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.type, WorkbenchMessageType.WORKER_READY)

    def test_22_wrong_protocol_version_is_rejected(self):
        """Valid JSON declaring wrong protocol version is rejected."""
        wrong_proto_line = f"{WORKBENCH_SENTINEL}{{\"protocol_version\": \"WORKBENCH_PROTOCOL_V2\", \"type\": \"WORKER_READY\", \"payload\": {{}}}}"
        self.assertIsNone(parse_sentinel_line(wrong_proto_line))

    def test_23_qprocess_failed_to_start_resets_runner_state(self):
        """When QProcess fails to start, WorkbenchProcessRunner resets state cleanly without hanging."""
        runner = WorkbenchProcessRunner()
        req = LaunchRequestV1(
            mode=LauncherMode.SANDBOX,
            agent_id="fixture_constant_continuous",
            render_mode="OFF",
        )

        error_received = []
        runner.workerError.connect(lambda err: error_received.append(err))

        runner.start_operation("PLAN", req)
        # Force simulate FailedToStart error
        runner._on_process_error(QProcess.ProcessError.FailedToStart)

        self.assertFalse(runner.is_running)
        self.assertIsNone(runner.current_operation)
        self.assertEqual(len(error_received), 1)
        self.assertEqual(error_received[0]["error_type"], "FailedToStart")


if __name__ == "__main__":
    unittest.main()
