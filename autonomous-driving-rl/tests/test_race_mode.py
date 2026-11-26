"""Simultaneous race timing, per-car outcomes, selection and registry isolation."""

from collections import deque
from io import StringIO
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from src.agents.MPPI import CONFIG_PATH as MPPI_CONFIG_PATH, MPPIAgent, load_mppi_config
from src.launcher.registry import build_default_agent_registry, compute_canonical_registry_sha256
from src.visualization.race_mode import RaceApp, Racer
from src.visualization.race_session import RACE_AGENTS, RaceComparison, RaceSelection, build_race_registry
from src.visualization.race_environment import build_grid, racer_route_network


class TestRaceMode(unittest.TestCase):
    def test_race_reuses_canonical_agents_and_their_action_adapters(self):
        canonical = build_default_agent_registry()
        before = compute_canonical_registry_sha256(canonical)
        registry = build_race_registry()
        for agent_id, _, _ in RACE_AGENTS:
            registration = registry.get(agent_id)
            agent = registry.get_factory(agent_id)()
            self.assertIs(registry.get_factory(agent_id), canonical.get_factory(agent_id))
            self.assertEqual(registration.action_adapter_id, agent.descriptor.action_adapter_id)
            self.assertEqual(registration.benchmark_eligible, canonical.get(agent_id).benchmark_eligible)
            agent.close()
        self.assertEqual(RACE_AGENTS[4][0], MPPIAgent.agent_id)
        self.assertIsInstance(registry.get_factory(RACE_AGENTS[4][0])(), MPPIAgent)
        self.assertFalse(registry.has_agent("race_mppi"))
        self.assertEqual(before, compute_canonical_registry_sha256(build_default_agent_registry()))

    def test_race_mppi_reads_json_instead_of_the_old_prototype_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "mppi.json"
            config_path.write_text(json.dumps({
                "version": "mppi_planner_v2",
                "parameters": {"horizon": 7, "num_samples": 32, "target_speed_m_s": 6.5},
            }), encoding="utf-8")
            with patch("src.agents.MPPI.CONFIG_PATH", config_path):
                factory = build_race_registry().get_factory(RACE_AGENTS[4][0])
                agent = factory()
                self.assertEqual(agent.config, load_mppi_config())
                self.assertEqual(agent.config.horizon, 7)
                self.assertEqual(agent.config.num_samples, 32)
                self.assertEqual(agent.config.target_speed_m_s, 6.5)
                self.assertEqual(factory.experiment_parameters()["parameters"]["horizon"], 7)
                agent.close()

    def test_session_manifest_records_the_connected_mppi_and_its_config(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            app = RaceApp(root, Path(directory) / "race")
            try:
                manifest = json.loads((app.output / "session.json").read_text(encoding="utf-8"))
                self.assertIn(MPPIAgent.agent_id, manifest["agents"])
                self.assertEqual(manifest["agents"][MPPIAgent.agent_id]["implementation_ref"],
                                 "src.agents.MPPI:MPPIAgent")
                self.assertEqual(manifest["mppi_parameters"]["parameters"]["horizon"],
                                 load_mppi_config().horizon)
                self.assertEqual(manifest["source_sha256"][manifest["mppi_config"]],
                                 hashlib.sha256(MPPI_CONFIG_PATH.read_bytes()).hexdigest())
                self.assertIn("src/agents/MPPI.py", manifest["source_sha256"])
            finally:
                app.decisions.close()
                app.results.close()

    def app(self, state="garage"):
        app = RaceApp.__new__(RaceApp)
        app.state, app.commands = state, deque()
        app.selection = RaceSelection()
        app.ui, app.env = MagicMock(), MagicMock()
        app.roads = [None] * 3
        app.hud = None
        app.racers = {}
        app.episode_open = False
        app.comparison = None
        app.quitting = False
        app._run_started = None
        app._active_wall_s = 0.
        app.save_manifest = MagicMock()
        app.save_comparison = MagicMock()
        app._start = MagicMock()
        return app

    def test_default_is_five_simultaneous_cars(self):
        self.assertEqual(RaceSelection().planned_indices(), tuple(range(5)))

    def test_callbacks_only_enqueue_until_outer_loop_handles_them(self):
        app = self.app()
        app.enqueue("start")
        app._start.assert_not_called()
        app.handle(*app.commands.popleft())
        app._start.assert_called_once()

    def test_pause_resume_never_steps_or_resets_the_simulator(self):
        app = self.app("running")
        app.handle("pause")
        self.assertEqual(app.state, "paused")
        app.handle("pause")
        self.assertEqual(app.state, "running")
        app.env.step.assert_not_called()
        app.env.reset.assert_not_called()

    def test_single_and_multi_selection_preserve_choices(self):
        selection = RaceSelection()
        selection.set_mode("single")
        selection.choose(4)
        self.assertEqual(selection.planned_indices(), (4,))
        selection.set_mode("compare")
        self.assertEqual(selection.planned_indices(), tuple(range(5)))
        selection.choose(2)
        self.assertEqual(selection.planned_indices(), (0, 1, 3, 4))
        selection.set_mode("single")
        self.assertEqual(selection.planned_indices(), (4,))

    def test_invalid_selection_cannot_launch_and_controls_ignored_mid_race(self):
        app = self.app()
        app.handle("clear_agents")
        app.handle("start")
        app._start.assert_not_called()
        self.assertIn("ít nhất 2", app.ui.status.__setitem__.call_args.args[1])
        app.handle("all_agents")
        self.assertEqual(app.selection.planned_indices(), tuple(range(5)))
        app.state = "running"
        app.handle("road", 2)
        app.handle("traffic")
        app.handle("agent", 4)
        self.assertEqual(app.selection.road_index, 0)
        self.assertFalse(app.selection.traffic)

    def test_grid_has_one_nonoverlapping_slot_per_selected_policy(self):
        original = {"nested": {"test": True}}
        grid = build_grid(original, (0, 1, 2, 3, 4), 3)
        positions = [(c["spawn_lane_index"], c["spawn_longitude"]) for c in grid.values()]
        self.assertEqual(len(set(positions)), 5)
        for lane in range(3):
            longs = sorted(pos for index, pos in positions if index[2] == lane)
            self.assertTrue(all(b - a >= 8 for a, b in zip(longs, longs[1:])))
        grid["racer_0"]["nested"]["test"] = False
        self.assertTrue(original["nested"]["test"])
        self.assertTrue(grid["racer_1"]["nested"]["test"])

    def test_results_can_arrive_out_of_order_and_reject_duplicates(self):
        case = {"case_id": "same-road"}
        race = RaceComparison((0, 2, 4), case, 1)
        row = {"agent_id": RACE_AGENTS[4][0], "case": case, "reason": "OUT_OF_ROAD"}
        race.record(row)
        self.assertEqual(race.to_dict()["pending_agent_ids"], ["stage2_astar", "stage2_dijkstra"])
        with self.assertRaises(ValueError):
            race.record(row)
        with self.assertRaises(ValueError):
            race.record({**row, "agent_id": "stage2_astar", "case": {"case_id": "other-road"}})
        race.record({**row, "agent_id": "stage2_dijkstra"})
        race.record({**row, "agent_id": "stage2_astar"})
        self.assertEqual(race.status, "COMPLETE")

    def test_reference_follows_each_cars_lane_without_mutating_map(self):
        lanes = [object(), object(), object()]
        road = SimpleNamespace(graph={"u": {"v": lanes}, "v": {"w": lanes[:1]}})
        car = SimpleNamespace(navigation=SimpleNamespace(current_lane=SimpleNamespace(index=("u", "v", 2))))
        view = racer_route_network(road, car)
        self.assertIs(view.graph["u"]["v"][0], lanes[2])
        self.assertIs(view.graph["v"]["w"][0], lanes[0])
        self.assertEqual(road.graph["u"]["v"], lanes)

    def running_app(self):
        app = self.app("running")
        app.case = SimpleNamespace(horizon_steps=1000, case_id="case", split="TRAIN", to_dict=lambda: {"case_id": "case"})
        app.steps, app.episode, app.smoke_steps = 0, 1, 0
        app.decisions, app.results = StringIO(), StringIO()
        app.manifest = {"completed_runs": 0}
        app.episode_open = True
        app._update_hud = MagicMock()
        app._update_standings = MagicMock()
        app._end_race = MagicMock()
        app.env.agents = {}
        app.obs, app.info = {}, {}
        app.comparison = RaceComparison((0, 1), app.case.to_dict(), 1)
        for i in (0, 1):
            policy, adapter = MagicMock(), MagicMock()
            from src.platform import CanonicalActionV1
            adapter.to_canonical.return_value = CanonicalActionV1(0., .6)
            policy.act.return_value.diagnostics = {}
            racer = Racer(i, policy, SimpleNamespace(action_adapter_id="adapter"), adapter)
            app.racers[racer.vehicle_id] = racer
            app.env.agents[racer.vehicle_id] = SimpleNamespace(
                speed_km_h=10., navigation=SimpleNamespace(current_lane=SimpleNamespace(index=("u", "v", i))))
            app.obs[racer.vehicle_id] = object()
        keys = tuple(app.racers)
        app.env.step.return_value = (
            dict(app.obs), {}, dict.fromkeys(keys, False), dict.fromkeys(keys, False),
            {key: {"route_completion": .1} for key in keys},
        )
        return app

    @patch("src.platform.build_agent_input")
    def test_all_cars_decide_before_one_shared_physics_step(self, build_input):
        app = self.running_app()
        events = []
        build_input.side_effect = lambda **kwargs: SimpleNamespace(step_index=kwargs["step_index"])
        for key, racer in app.racers.items():
            decision = racer.policy.act.return_value
            racer.policy.act.side_effect = lambda _, key=key, decision=decision: (events.append(key), decision)[1]
        step_result = app.env.step.return_value
        app.env.step.side_effect = lambda actions: (events.append("physics"), step_result)[1]
        app._step()
        self.assertEqual(events, ["racer_0", "racer_1", "physics"])
        self.assertEqual(app.env.step.call_count, 1)
        self.assertEqual(set(app.env.step.call_args.args[0]), set(app.racers))
        for call, (key, racer) in zip(build_input.call_args_list, app.racers.items()):
            self.assertIs(call.kwargs["raw_obs"], app.obs[key])
            self.assertIs(call.kwargs["ego_vehicle"], app.env.agents[key])
            self.assertEqual(call.kwargs["step_index"], 0)
            self.assertTrue(all(v in call.kwargs["traffic_vehicles"] for v in app.env.agents.values()))
            self.assertEqual(racer.steps, 1)

    @patch("src.platform.build_agent_input", return_value=SimpleNamespace(step_index=0))
    def test_one_car_crashing_does_not_stop_the_other(self, _):
        app = self.running_app()
        app.env.step.return_value[-1]["racer_0"]["crash_vehicle"] = True
        app._step()
        self.assertEqual(app.racers["racer_0"].summary["reason"], "CRASH_VEHICLE")
        self.assertIsNone(app.racers["racer_1"].summary)
        app.env.park.assert_called_once_with("racer_0")
        app._end_race.assert_not_called()
        app._step()
        self.assertEqual(app.racers["racer_0"].steps, 1)
        self.assertEqual(app.racers["racer_1"].steps, 2)
        self.assertEqual(app.env.step.call_args.args[0]["racer_0"], [0., 0.])

    @patch("src.platform.build_agent_input", return_value=SimpleNamespace(step_index=0))
    def test_policy_failure_is_recorded_per_car_and_others_keep_driving(self, _):
        app = self.running_app()
        app.racers["racer_0"].policy.act.side_effect = RuntimeError("bad policy")
        with patch("src.visualization.race_mode.traceback.print_exc"):
            app._step()
        self.assertEqual(app.racers["racer_0"].summary["reason"], "TECHNICAL_ERROR")
        self.assertIsNone(app.racers["racer_1"].summary)
        self.assertEqual(app.racers["racer_1"].steps, 1)
        app._end_race.assert_not_called()

    def test_restart_restarts_all_cars_and_garage_records_remaining_as_aborted(self):
        app = self.running_app()
        app.handle("restart")
        app._start.assert_called_once()
        app._finish("ABORTED", show_results=False)
        self.assertTrue(all(r.summary["reason"] == "ABORTED" for r in app.racers.values()))
        self.assertTrue(all(not r.summary["clean_success"] for r in app.racers.values()))
        app._end_race.assert_called_once_with(False, interrupted=True)


if __name__ == "__main__":
    unittest.main()
