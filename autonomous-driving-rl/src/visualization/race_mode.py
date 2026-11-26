"""Persistent MetaDrive garage and smooth, noncanonical interactive races.

AI workers never block interactive physics/rendering. The legacy lockstep
path remains available for deterministic diagnostics and smoke checks.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
import traceback

from src.agents.MPPI import CONFIG_PATH as MPPI_CONFIG_PATH, MPPIAgent
from src.visualization.race_session import (
    OUTCOME_LABELS, RACE_AGENTS, ROAD_NAMES, RaceComparison, RaceSelection,
    build_race_registry, load_race_roads, prepare_race_reset,
    HUMAN_INDEX, race_participant,
)
from src.visualization.phone_race_support import PhoneRaceSupport
from src.visualization.race_rules import RACE_FLAGS, classify_race_outcome, rules_manifest, limit_race_action, RACE_SPEED_LIMIT_KMH
from src.visualization.race_rules import race_planner_options, apply_race_planner_options


def configure_race_graphics(profile='lite'):
    from metadrive.constants import TerrainProperty
    from metadrive.engine.core import engine_core
    from src.visualization.race_graphics import configure_graphics, get_graphics_profile
    configure_graphics(profile)
    # Keep the full 2048 m simulator terrain data. race_landscape supplies
    # a separate bounded RGB material map and continuous lane-paint geometry.
    pixels_per_meter = 8 if profile == 'balanced' else 6
    TerrainProperty.get_semantic_map_pixel_per_meter = classmethod(lambda cls: pixels_per_meter)
    if not getattr(engine_core.init, "_race_mode", False):
        original = engine_core.init

        def initialize(*args, **kwargs):
            kwargs["msaa_samples"] = get_graphics_profile().msaa_samples
            kwargs['use_normal_maps'] = True
            return original(*args, **kwargs)

        initialize._race_mode = True
        engine_core.init = initialize


@dataclass
class Racer:
    index: int
    policy: object
    registration: object
    adapter: object
    steps: int = 0
    flags: dict = field(default_factory=dict)
    latencies: list = field(default_factory=list)
    route: float = None
    summary: dict = None
    last_control: tuple = None
    planner_options: dict = None

    @property
    def agent_id(self):
        return race_participant(self.index)[0]

    @property
    def vehicle_id(self):
        return f"racer_{self.index}"


class RaceApp(PhoneRaceSupport):
    def __init__(self, project_root, output=None, smoke_steps=0):
        self.project_root = Path(project_root)
        self.output = Path(output) if output else self.project_root / "artifacts/race_mode" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.output.mkdir(parents=True, exist_ok=False)
        self.selection = RaceSelection()
        self.map_set = 'original'
        self.test_track_catalog = None
        self.selection.road_index = 1
        self.registry = build_race_registry()
        self.commands = deque()
        self._init_phone()
        self.state = "garage"
        self.env = self.ui = self.hud = self.events = None
        self.graphics_runtime = None
        self.garage_pointer = None
        self.camera_input = None
        self.racers = {}
        self.steps = 0
        self.details_visible = self.camera_top = False
        self.summary = None
        self.episode = 0
        self.episode_open = False
        self.comparison = None
        self.comparison_counter = 0
        self.quitting = False
        self.smoke_steps = smoke_steps
        self.smoke_index = 0
        # First exercise the requested five-car race, then dynamic subset/single
        # resets, all three roads, and traffic in the SAME native window.
        self.smoke_jobs = [(tuple(range(5)), 0, False), ((1, 4), 0, False),
                           ((0,), 1, False), ((2, 3, 4), 2, True)]
        self.pause_verified = False
        self._pause_frames = 0
        self._window_identity = None
        self._run_started = None
        self._active_wall_s = 0.
        self.decisions = (self.output / "decisions.jsonl").open("w", encoding="utf-8", buffering=1)
        self.results = (self.output / "episodes.jsonl").open("w", encoding="utf-8", buffering=1)
        self.manifest = {
            "kind": "NATIVE_RACE_PLAYGROUND", "canonical_run": False, "split": None,
            "execution": "shared_world_lockstep", "inter_vehicle_collisions": True,
            "route_reference": "current_lane_center_for_each_car",
            "control_dt_s": .1, "agent_seed": 42, "stop_on_first_safety_event": "per_car",
            "status": "STARTING", "mppi_status": "BENCHMARK_INELIGIBLE_CANDIDATE",
            "completed_runs": 0, "smoke_steps": smoke_steps,
        }
        mppi_config_path = MPPI_CONFIG_PATH.relative_to(self.project_root.resolve()).as_posix()
        sources = ["src/agents/MPPI.py", "src/agents/astar_agent.py",
                   "src/agents/graph_search_agents.py", "src/agents/hybrid_astar_agent.py",
                   "src/agents/path_following.py", "src/agents/planning_graph.py",
                   "src/agents/hybrid_astar.py", "src/visualization/race_mode.py",
                   "src/visualization/race_environment.py", "src/visualization/race_session.py",
                   "src/visualization/race_menu.py", "configs/experiments/astar_learning_v1.json",
                   "src/visualization/race_style.py", "src/visualization/race_showroom.py",
                   "src/visualization/race_graphics.py", "src/visualization/test_track_catalog.py",
                   "src/visualization/track_picker.py",
                   "src/visualization/race_car_mesh.py",
                   "src/visualization/vf3_mesh.py",
                   "src/visualization/garage_orbit.py",
                   "src/visualization/race_camera_input.py", "src/visualization/race_rules.py",
                   "src/visualization/race_imported_assets.py",
                   "src/visualization/race_landscape.py",
                   "src/visualization/race_terrain_shader.py",
                   "src/visualization/race_road_markings.py",
                   "src/visualization/vietnam_track.py", "src/visualization/vietnam_map.py",
                   "src/visualization/vietnam_scenery.py", "src/visualization/vietnam_speed.py",
                   "src/visualization/vietnam_assets.py", "assets/scenery/sources.json",
                   "assets/maps/vietnam/vo_van_kiet_route.json",
                   "src/visualization/mobile_control.py", "src/visualization/mobile_server.py",
                   "src/visualization/phone_camera.py", "src/visualization/phone_race_support.py",
                   "src/visualization/phone_race_menu.py", "src/visualization/race_vehicle_models.py",
                   "src/visualization/mobile/index.html",
                   "src/visualization/async_policy.py", "src/visualization/realtime_race.py",
                   "src/visualization/phone_encoder.py", "src/visualization/nvenc_process.py",
                   "configs/experiments/search_planners_v1.json", mppi_config_path]
        self.manifest["source_sha256"] = {p: hashlib.sha256((self.project_root / p).read_bytes()).hexdigest() for p in sources}
        self.manifest["agents"] = {aid: self.registry.get(aid).to_dict() for aid, _, _ in RACE_AGENTS}
        self.manifest["mppi_config"] = mppi_config_path
        asset_manifest=self.project_root/'assets'/'vehicles'/'manifest.json'
        if asset_manifest.is_file():
            self.manifest['vehicle_assets']=json.loads(asset_manifest.read_text(encoding='utf-8'))['models']
        self.manifest["mppi_parameters"] = self.registry.get_factory(MPPIAgent.agent_id).experiment_parameters()
        self.save_manifest()

    def save_manifest(self):
        self.manifest.update(screen=self.state, selected_agent=self.selection.agent_id,
                             selected_test_case=self.selection.test_case.to_dict() if self.selection.test_case else None,
                             map_set=self.map_set,
                             selected_road=self.selection.road_index, playback_speed=self.selection.speed,
                             mode=self.selection.mode,
                             vehicle_models={str(i):self.selection.model_for(i) for i in range(6)},
                             selected_agent_ids=[RACE_AGENTS[i][0] for i in self.selection.chosen_indices])
        temporary = self.output / "session.tmp"
        temporary.write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.output / "session.json")

    def enqueue(self, command, argument=None):
        self.commands.append((command, argument))

    def save_comparison(self):
        if self.comparison is None:
            return
        name = f"comparison-{self.comparison.comparison_id:03d}.json"
        temporary = self.output / (name + ".tmp")
        temporary.write_text(json.dumps(self.comparison.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.output / name)
        self.manifest.update(last_comparison=name, comparison_status=self.comparison.status)
        self.save_manifest()

    def open(self):
        from panda3d.core import WindowProperties
        from direct.showbase.DirectObject import DirectObject
        from src.launcher.executor import build_metadrive_case_config
        from src.visualization.race_environment import RaceEnvironment, build_grid
        from src.visualization.phone_race_menu import PhoneRaceMenu as RaceMenu

        print(f"Preparing 3 verified {self.map_set} roads and Vietnam OSM corridor. Please wait...", flush=True)
        self.roads = load_race_roads(self.project_root, self.map_set)
        case = self.selection.case(self.roads)
        config = build_metadrive_case_config(case, self.selection.blocks(self.roads), "NATIVE")
        defaults = RaceEnvironment.default_config()
        config.update(window_size=(1280, 800), store_map=False, log_level=40,
                      camera_height=2.2, camera_dist=7.5, camera_pitch=None,
                      use_chase_camera_follow_lane=False,
                      show_skybox=True, show_terrain=True, show_sidewalk=True,
                      top_down_camera_initial_z=40,
                      is_multi_agent=True, num_agents=1, allow_respawn=False,
                      agent_configs=build_grid(defaults["vehicle_config"], (0,), defaults["map_config"]["lane_num"]))
        self.env = RaceEnvironment(config)
        self.obs, self.info = self.env.reset(seed=case.environment_seed)
        from src.visualization.race_camera_input import RaceCameraInput
        self.camera_input = RaceCameraInput(self.env.engine)
        self.manifest['camera_input'] = {'mouse_look':False, 'cursor_capture':False,
                                         'click_to_pan':False, 'garage_orbit':True,
                                         'scene_drag':'hold_left_button'}
        self.env.engine.force_fps.disable()
        # Our own HUD owns the display. MetaDrive's hidden debug message
        # otherwise destroys and rebuilds a TextNode after every physics tick.
        message=self.env.engine.on_screen_message
        if message is not None:
            message.onScreenText.cleanup()
            self.env.engine.on_screen_message=None
        if self.env.engine.pssm is not None:
            self.env.engine.pssm.buffer.setActive(False)
        props = WindowProperties()
        props.setTitle("XETUHANH | Race Mode - Dua nhieu xe dong thoi")
        self.env.engine.win.requestProperties(props)
        self._window_identity = int(self.env.engine.win.this)
        self.ui = RaceMenu(self.env.engine, self.enqueue)
        self.env.decorate("racer_0", 0, RACE_AGENTS[0][1], self.ui.font)
        self.ui.attach_preview(self.env.agents["racer_0"])
        from src.visualization.race_landscape import configure_landscape
        configure_landscape(self.env,self.ui.preview)
        from src.visualization.race_graphics import RaceGraphicsRuntime
        self.graphics_runtime = RaceGraphicsRuntime(self)
        from src.visualization.garage_orbit import GaragePointer
        self.garage_pointer = GaragePointer(self)
        self.ui.update_selection(self.selection, self.roads)
        self.ui.show(self.state)
        for key in ("p", "r", "c", "escape", "[", "]"):
            self.env.engine.ignore(key)
        self.events = DirectObject()
        for key, command in (("enter", "enter"), ("p", "pause"), ("escape", "pause"),
                             ("r", "restart"), ("g", "garage"), ("i", "details"),
                             ("c", "camera"), ("v", "focus")):
            self.events.accept(key, self.enqueue, [command])
        for i in range(5):
            self.events.accept(str(i + 1), self.enqueue, ["number", i])
        self.events.accept('6',self.enqueue,['number',HUMAN_INDEX])
        self.events.accept('h',self.enqueue,['recover'])
        self.manifest.update(status="RUNNING", split=case.split, roads=[r[0].to_dict() for r in self.roads],
                             graphics=self.graphics_runtime.profile.to_dict(), window_identity=self._window_identity)
        self.save_manifest()
        print(f"Race Mode ready. Logs: {self.output}", flush=True)
        if getattr(self,'startup_phone',False):
            self._open_phone()

    def _set_state(self, state):
        if getattr(self,'realtime',None):
            self.realtime.set_render_enabled(True)
            if state=='running': self.realtime.reset_clock()
        if getattr(self,'phone_control',None):
            self.phone_control.clear()
        if self.state == "running" and self._run_started is not None:
            self._active_wall_s += time.perf_counter() - self._run_started
            self._run_started = None
        self.state = state
        if state == "running":
            self._run_started = time.perf_counter()
        self.ui.show(state)
        if self.hud is not None:
            self.hud.agent_io.root.show() if self.details_visible and state in ("running", "paused") else self.hud.agent_io.root.hide()
        self.save_manifest()

    def _destroy_hud(self):
        if self.hud is not None:
            self.hud.destroy()
            self.hud = None

    def _close_agents(self):
        performance=self.realtime.stats() if getattr(self,'realtime',None) else None
        for racer in self.racers.values():
            if racer.policy is not None:
                racer.policy.close()
                racer.policy = None
        if getattr(self,'realtime',None):
            self.manifest['last_realtime_performance']=performance
            self.realtime.close()
            self.realtime=None

    def _begin_session(self, countdown=True):
        try:
            self.selection.planned_indices()
        except ValueError as error:
            self.ui.status["text"] = str(error)
            return False
        if self.selection.manual and not self._phone_ready():
            return False
        self._start(countdown=countdown)
        return True

    def _start(self, countdown=True):
        from src.platform import AgentPublicEpisodeContext, canonical_json_sha256, get_action_adapter
        from src.launcher.cases import make_jsonable
        from src.visualization.hud_manager import RacingHudManager

        indices = self.selection.planned_indices()
        if self.episode_open:
            self._finish("ABORTED", show_results=False)
        self._destroy_hud()
        self._close_agents()
        self.racers = {}
        self.comparison = None
        self.ui.status["text"] = "Đang nạp đường và các xe..."
        self.env.engine.taskMgr.step()
        self._phone_before_reset()
        self.case = self.selection.case(self.roads)
        from src.visualization.vietnam_track import is_vietnam, route_data
        vietnam=is_vietnam(self.case)
        self.race_speed_limit_kmh=self.env.race_speed_limit_kmh=80. if vietnam else 20.
        self.race_cruise_kmh=60. if vietnam else 15.
        self.manifest['custom_track']=route_data()['source'] if vietnam else None
        self.manifest['race_controller']='vn_continuous_path_pursuit_v1' if vietnam else 'canonical_discrete9_follower'
        prepare_race_reset(self.env, self.case, self.selection.blocks(self.roads))
        realtime=self.realtime_enabled and not self.smoke_steps
        self.env.decision_observations=realtime
        self.env.race_practice = self.selection.mode == 'practice'
        self.env.configure_grid(indices)
        self.control_dt_s=.02 if realtime else .1
        self.execution_mode='async_ai_shared_world' if realtime else 'shared_world_lockstep'
        self.env.config['decision_repeat']=1 if realtime else 5
        self.env.config['horizon']=self.case.horizon_steps*(5 if realtime else 1)
        self.manifest.update(execution=self.execution_mode,control_dt_s=self.control_dt_s,split=self.case.split,
                             race_rules=rules_manifest(self.env.race_practice,self.race_speed_limit_kmh,self.race_cruise_kmh))
        self.obs, self.info = self.env.reset(seed=self.case.environment_seed)
        if int(self.env.engine.win.this) != self._window_identity:
            raise RuntimeError("Native window changed during reset")
        blocks = make_jsonable(self.env.current_map.get_meta_data()["block_sequence"])
        if canonical_json_sha256(blocks) != self.case.geometry_sha256:
            raise RuntimeError("Road geometry differs from the verified map preset")
        self.steps = 0
        self.episode += 1
        self.episode_open = True
        self._active_wall_s = 0.
        self._run_started = None
        if len(indices) > 1:
            self.comparison_counter += 1
            self.comparison = RaceComparison(indices, self.case.to_dict(), self.comparison_counter,
                                             execution=self.execution_mode,control_dt_s=self.control_dt_s)
        for index in indices:
            aid = race_participant(index)[0]
            if index == HUMAN_INDEX:
                from types import SimpleNamespace
                from src.platform import InputProfileId
                from src.visualization.mobile_control import HumanPolicy
                registration = SimpleNamespace(input_profile_id=InputProfileId.STATE_DECISION_V1,
                                               action_adapter_id='continuous_box2_v1')
                policy = HumanPolicy(self.phone_control)
            else:
                registration = self.registry.get(aid)
                policy = self.registry.get_factory(aid)()
            racer = Racer(index, policy, registration,
                          get_action_adapter(registration.action_adapter_id))
            self.racers[racer.vehicle_id] = racer
            if index < 4:
                racer.planner_options = race_planner_options(self.env.agents[racer.vehicle_id].max_speed_km_h,self.race_cruise_kmh)
                if vietnam:
                    racer.planner_options['control_lookahead_m']=8.
                    racer.planner_options['vn_dynamic_lookahead']=True
                apply_race_planner_options(policy,racer.planner_options)
            elif index == 4 and vietnam:
                racer.planner_options={'mppi_target_speed_m_s':60./3.6}
                apply_race_planner_options(policy,racer.planner_options)
            self.env.decorate(racer.vehicle_id, index, race_participant(index)[1], self.ui.font,
                              self.selection.model_for(index))
            racer.route = self.info[racer.vehicle_id].get("route_completion")
            try:
                racer.policy.reset(AgentPublicEpisodeContext(
                    horizon_steps=self.case.horizon_steps, input_profile_id=registration.input_profile_id,
                    action_adapter_id=registration.action_adapter_id,
                ), agent_seed=42)
            except Exception as error:
                self._finish_racer(racer, "TECHNICAL_ERROR", error=str(error))
        self._phone_after_reset()
        if realtime:
            from src.visualization.realtime_race import RealtimeRace
            self.realtime=RealtimeRace(self)
        self.hud = RacingHudManager(self.env)
        self.hud.agent_io._events.ignore("i")
        self.hud.agent_io.root.setBin("fixed", 600)
        self.hud.agent_io.root.hide()
        self.camera_top = False
        self.env.focus(next(iter(self.racers)), self.camera_top)
        self._next_decision = time.perf_counter()
        self._update_hud()
        self.countdown_end = time.perf_counter() + 3.
        self._set_state("countdown" if countdown else "running")
        self._update_standings()
        self.manifest["lineup"] = {
            key: {"agent_id": racer.agent_id, "spawn_lane_index": list(self.env.agents[key].config["spawn_lane_index"]),
                  "spawn_longitude": self.env.agents[key].config["spawn_longitude"],
                  "runtime_planner_options": racer.planner_options}
            for key, racer in self.racers.items()
        }
        self.save_comparison()
        self.save_manifest()
        print(f"RACE {self.episode}: {len(indices)} simultaneous car(s) | {self.case.case_id}", flush=True)
        if all(r.summary is not None for r in self.racers.values()):
            self._end_race()

    def _step(self):
        if getattr(self,'realtime',None):
            return self.realtime.step()
        from src.platform import InputProfileId, build_agent_input
        from src.launcher.cases import make_jsonable
        from src.visualization.race_environment import racer_route_network

        started = time.perf_counter()
        traffic = list(getattr(self.env.engine.traffic_manager, "traffic_vehicles", []))
        actors = traffic + list(self.env.agents.values())
        controls = {}
        requested_actions = {}
        errors = []
        # No physics advances until EVERY active policy has read this frame.
        for key, racer in self.racers.items():
            if racer.summary is not None:
                continue
            vehicle = self.env.agents[key]
            try:
                inputs = build_agent_input(
                    raw_obs=self.obs[key], ego_vehicle=vehicle, traffic_vehicles=actors,
                    navigation=vehicle.navigation,
                    road_network=racer_route_network(self.env.current_map.road_network, vehicle),
                    step_index=self.steps, profile_id=InputProfileId.STATE_DECISION_V1,
                    **({'task_lookahead_count':40,'task_lookahead_spacing_m':3.,'traffic_radius_m':120.}
                       if getattr(self,'race_speed_limit_kmh',20.)>20. else {}),
                )
                t0 = time.perf_counter()
                from src.visualization.race_rules import prepare_race_policy
                prepare_race_policy(racer.policy,inputs)
                decision = racer.policy.act(inputs)
                latency = (time.perf_counter() - t0) * 1000
                requested_actions[key] = racer.adapter.to_canonical(decision.action_payload)
                from src.visualization.vietnam_speed import decision_action,vehicle_speed_limit
                requested=decision_action(decision,requested_actions[key],vehicle,
                                          getattr(self,'race_speed_limit_kmh',20.)>20.)
                action = limit_race_action(requested, self.selection.mode == 'practice', vehicle.speed_km_h,
                                           vehicle_speed_limit(self.env,vehicle))
                controls[key] = (inputs, decision, action, latency)
            except Exception as error:
                traceback.print_exc()
                errors.append((racer, str(error)))
        # Park failures only after every policy has read the unchanged frame.
        for racer, error in errors:
            self._finish_racer(racer, "TECHNICAL_ERROR", error=error)
        if not controls:
            self._end_race()
            return
        actions = {key: controls[key][2].to_numpy() if key in controls else [0., 0.]
                   for key in self.env.agents}
        if self.selection.manual and getattr(self,'phone_control',None):
            self.phone_control.mark_applied()
        self.obs, _, terminated, truncated, self.info = self.env.step(actions)
        self.steps += 1
        for key, (inputs, decision, action, latency) in controls.items():
            racer = self.racers[key]
            racer.steps += 1
            racer.latencies.append(latency)
            racer.last_control = (inputs, decision, action, latency)
            info = self.info[key]
            for name in RACE_FLAGS:
                if info.get(name, False):
                    racer.flags[name] = True
            outcome = classify_race_outcome(
                racer.flags, terminated=terminated[key], practice=self.selection.mode == 'practice',
                truncated=truncated[key] or self.steps >= self.case.horizon_steps,
            )
            racer.route = info.get("route_completion")
            record = {
                "episode": self.episode, "race_tick": self.steps, "case_id": self.case.case_id,
                "split": self.case.split, "agent_id": racer.agent_id, "vehicle_id": key,
                "input_step": inputs.step_index, "post_step": racer.steps,
                "action_adapter_id": racer.registration.action_adapter_id,
                "diagnostics": decision.diagnostics, "action": action.to_dict(),
                "requested_action": requested_actions[key].to_dict(),
                "act_latency_ms": latency, "speed_kmh": float(self.env.agents[key].speed_km_h),
                "route_completion": racer.route, "flags": dict(racer.flags),
            }
            self.decisions.write(json.dumps(make_jsonable(record), ensure_ascii=False) + "\n")
            if outcome.is_done:
                self._finish_racer(racer, outcome.reason, outcome=outcome,
                                   termination_source="race_rules" if not terminated[key] else "simulator")
            elif self.smoke_steps and self.steps >= self.smoke_steps:
                self._finish_racer(racer, "PREVIEW_LIMIT")
        self._update_hud()
        self._update_standings()
        self._next_decision = started + .1 / self.selection.speed
        if all(r.summary is not None for r in self.racers.values()):
            self._end_race()
        elif self.smoke_steps and self.smoke_index == 0 and self.steps == 2 and not self.pause_verified:
            self.screenshot("simultaneous-five-cars.png")
            self._paused_episode_step = self.env.episode_step
            self._paused_positions = {key: tuple(v.position) for key, v in self.env.agents.items()}
            self._set_state("paused")

    def _update_hud(self):
        if self.hud is None or not self.racers:
            return
        racer = self.racers[self.env.focus_id]
        kwargs = {}
        if racer.last_control is not None:
            inputs, decision, action, latency = racer.last_control
            kwargs.update(canonical_steering=action.steering, canonical_throttle_brake=action.throttle_brake,
                          agent_input=inputs, agent_decision=decision, agent_id=racer.agent_id, act_latency_ms=latency)
        self.hud.update(step_idx=racer.steps, control_dt_s=getattr(self,'control_dt_s',.1),
                        step_info=self.info[racer.vehicle_id], **kwargs)

    def _update_standings(self):
        focused = self.racers[self.env.focus_id]
        active = sum(r.summary is None for r in self.racers.values())
        speed_limit=getattr(self,'race_speed_limit_kmh',RACE_SPEED_LIMIT_KMH)
        self.ui.run_title["text"] = f"{active}/{len(self.racers)} xe  |  {race_participant(focused.index)[1]}  |  Tối đa {speed_limit:g} km/h  |  {self.selection.speed:g}×"
        if speed_limit>20. and hasattr(self.ui,'vn_route_status'):
            route=max(0.,min(1.,focused.route or 0.))
            total=float(self.env.agent.navigation.total_length)/1000
            self.ui.vn_route_status['text']=f'VIỆT NAM  ·  {route*total:.2f} / {total:.2f} km  ·  Còn {(1-route)*total:.2f} km'
        self.ui.update_standings(self.racers, self.env)
        if self.details_visible:
            self.ui.standings.hide()

    def _finish_racer(self, racer, reason, outcome=None, error=None, termination_source=None):
        if racer.summary is not None:
            return
        active_time = self._active_wall_s
        if self._run_started is not None:
            active_time += time.perf_counter() - self._run_started
        racer.summary = {
            "episode": self.episode, "agent_id": racer.agent_id, "vehicle_id": racer.vehicle_id,
            "case": self.case.to_dict(), "reason": reason, "steps": racer.steps,
            "race_tick": self.steps, "simulation_time_s": racer.steps * getattr(self,'control_dt_s',.1),
            "active_wall_time_s": active_time, "route_completion": racer.route, "flags": dict(racer.flags),
            "clean_success": bool(outcome and outcome.clean_success),
            "race_rules": rules_manifest(self.selection.mode == 'practice',
                                         getattr(self,'race_speed_limit_kmh',20.),getattr(self,'race_cruise_kmh',15.)),
            "ranked": self.selection.mode != 'practice',
            "controller": 'phone' if racer.index==HUMAN_INDEX else 'ai',
            "vehicle_model": self.selection.model_for(racer.index),
            "runtime_planner_options": racer.planner_options,
            "recoveries": getattr(self,'recoveries',0) if racer.index==HUMAN_INDEX else 0,
            "mean_act_latency_ms": sum(racer.latencies) / len(racer.latencies) if racer.latencies else None,
            "canonical_run": False, "error": error, "termination_source": termination_source,
            "comparison_id": self.comparison.comparison_id if self.comparison is not None else None,
            "execution": getattr(self,'execution_mode','shared_world_lockstep'),
        }
        self.results.write(json.dumps(racer.summary, ensure_ascii=False) + "\n")
        self.manifest["completed_runs"] += 1
        if racer.policy is not None:
            racer.policy.close()
            racer.policy = None
        self.env.park(racer.vehicle_id)
        if self.comparison is not None:
            self.comparison.record(racer.summary)
            self.save_comparison()
        print(f"END {self.episode}/{racer.agent_id}: {reason} | steps={racer.steps} | route={racer.route}", flush=True)

    def _finish(self, reason, show_results=True, error=None):
        if not self.episode_open:
            return
        for racer in self.racers.values():
            self._finish_racer(racer, reason, error=error)
        self._end_race(show_results, interrupted=reason in ("ABORTED", "TECHNICAL_ERROR"))

    def _end_race(self, show_results=True, interrupted=False):
        self.episode_open = False
        if self._run_started is not None:
            self._active_wall_s += time.perf_counter() - self._run_started
            self._run_started = None
        if self.comparison is not None:
            if interrupted:
                self.comparison.status = "STOPPED"
            self.save_comparison()
        self._close_agents()
        if not show_results:
            return
        self._destroy_hud()
        if self.comparison is not None:
            self.ui.update_comparison(self.comparison)
            self._set_state("comparison")
        else:
            self.summary = next(iter(self.racers.values())).summary
            row = self.summary
            self.ui.result_title["text"] = OUTCOME_LABELS.get(row["reason"], row["reason"])
            route = f"{row['route_completion'] * 100:.1f}%" if row["route_completion"] is not None else "—"
            crashes = any(value for key, value in row["flags"].items() if key.startswith("crash_"))
            latency = row["mean_act_latency_ms"]
            self.ui.result_body["text"] = (
                f"Người lái: {race_participant(next(iter(self.racers.values())).index)[1]}\n"
                f"Hoàn thành đường: {route}\nThời gian mô phỏng: {row['simulation_time_s']:.1f} giây\n"
                f"Va chạm: {'Có' if crashes else 'Không'}   |   Ra khỏi đường: {'Có' if row['flags'].get('out_of_road') else 'Không'}\n"
                + (f"Quyết định trung bình: {latency:.1f} ms" if latency is not None else "Quyết định trung bình: —")
            )
            self.ui.result_error["text"] = str(row["error"])[:150] if row["error"] else "Kết quả xem thử · log đã được lưu."
            self._set_state("results")

    def _garage(self):
        if self.episode_open:
            self._finish("ABORTED", show_results=False)
        self._destroy_hud()
        self._close_agents()
        self.ui.update_selection(self.selection, self.roads)
        self._set_state("garage")

    def _show_error(self, error):
        traceback.print_exc()
        self.manifest["last_error"] = str(error)
        if self.episode_open:
            self._finish("TECHNICAL_ERROR", error=str(error))
        else:
            self._destroy_hud()
            self._close_agents()
            self.ui.result_title["text"] = "Không bắt đầu được lượt chạy"
            self.ui.result_body["text"] = "Chưa có kết quả mô phỏng.\nBấm Đổi thuật toán để trở về garage."
            self.ui.result_error["text"] = str(error)[:150]
            self._set_state("results")

    def handle(self, command, arg=None):
        picker=getattr(self.ui,'track_picker',None)
        if self.state=='garage' and command=='track_open':
            if self.test_track_catalog is None:
                from src.visualization.test_track_catalog import TestTrackCatalog
                self.test_track_catalog=TestTrackCatalog(self.project_root)
            self.ui.open_tracks(self.test_track_catalog,self.selection.case(self.roads))
            return
        if picker is not None and picker.is_open is True:
            if command in ('track_close','pause'):
                picker.hide()
            elif command in ('track_apply','enter'):
                case,blocks=self.test_track_catalog.resolve(picker.draft.case.case_id)
                self.selection.select_test(case,blocks)
                picker.hide()
                self.ui.update_selection(self.selection,self.roads)
                self.save_manifest()
            elif command=='track_original':
                self.selection.clear_test()
                picker.hide()
                self.ui.update_selection(self.selection,self.roads)
                self.save_manifest()
            elif command=='quit':
                self.quitting=True
            else:
                picker.handle(command,arg)
            return
        if getattr(self,'phone_control',None) and self._phone_command(command,arg):
            return
        if command == "quit":
            self.quitting = True
        elif command in ("enter", "result_action") and self.state in ("garage", "results", "comparison"):
            self._begin_session()
        elif command == "start" and self.state == "garage":
            self._begin_session()
        elif command == "restart" and self.state in ("running", "paused", "results", "countdown", "comparison"):
            self._begin_session()
        elif command == "garage":
            self._garage()
        elif command == "pause":
            if self.state == "running":
                self._set_state("paused")
            elif self.state == "paused":
                self._next_decision = time.perf_counter()
                self._set_state("running")
            elif self.state == "countdown":
                self._garage()
        elif command == "details" and self.hud is not None:
            self.details_visible = not self.details_visible
            self.hud.agent_io.root.show() if self.details_visible else self.hud.agent_io.root.hide()
            self._update_hud()
            self._update_standings()
        elif command == "camera" and self.state in ("running", "paused", "countdown"):
            self.camera_top = not self.camera_top
            self.env.focus(self.env.focus_id, self.camera_top)
        elif command in ("focus", "number") and self.state in ("running", "paused", "countdown"):
            keys = list(self.racers)
            key = f"racer_{arg}" if command == "number" else keys[(keys.index(self.env.focus_id) + 1) % len(keys)]
            if key in self.racers:
                self.env.focus(key, self.camera_top)
                self.hud.reset()
                self._update_hud()
                self._update_standings()
        elif self.state == "garage":
            if command in ("agent", "number") and type(arg) is int and 0 <= arg < len(RACE_AGENTS):
                self.selection.choose(arg)
            elif command == "mode" and arg in ("single", "compare", "human", "practice"):
                self.selection.set_mode(arg)
            elif command == "all_agents" and self.selection.mode in ("compare", "human"):
                self.selection.comparison_indices = tuple(range(len(RACE_AGENTS)))
            elif command == "clear_agents" and self.selection.mode in ("compare", "human"):
                self.selection.comparison_indices = ()
            elif command == "road" and type(arg) is int and 0 <= arg < len(self.roads):
                self.selection.clear_test()
                self.selection.road_index = arg
            elif command == "traffic" and self.selection.test_case is None:
                self.selection.traffic = not self.selection.traffic
            elif command == "speed" and not self.selection.manual:
                self.selection.speed = {1.: 2., 2.: 5., 5.: 1.}[self.selection.speed]
            self.ui.update_selection(self.selection, self.roads)
            self.save_manifest()

    def screenshot(self, name):
        from panda3d.core import Filename
        self.env.update_race_camera()
        runtime = self.graphics_runtime
        uncapped = runtime.uncapped if runtime else False
        if runtime:
            runtime.uncapped = True
        try:
            self.env.engine.taskMgr.step()
            self.env.engine.graphicsEngine.renderFrame()
            self.env.engine.win.saveScreenshot(Filename.fromOsSpecific(str((self.output / name).resolve())))
        finally:
            if runtime:
                runtime.uncapped = uncapped

    def _smoke_tick(self):
        if not self.smoke_steps:
            return
        if self.state == "garage":
            indices, road, traffic = self.smoke_jobs[self.smoke_index]
            self.selection.set_mode("single" if len(indices) == 1 else "compare")
            self.selection.agent_index = indices[0]
            self.selection.comparison_indices = indices
            self.selection.road_index, self.selection.traffic = road, traffic
            self.ui.update_selection(self.selection, self.roads)
            self.screenshot(f"garage-{self.smoke_index}.png")
            self._start(countdown=False)
        elif self.state == "paused" and not self.pause_verified:
            self._pause_frames += 1
            if self.env.episode_step != self._paused_episode_step:
                raise RuntimeError("Physics advanced while paused")
            if any(tuple(v.position) != self._paused_positions[key] for key, v in self.env.agents.items()):
                raise RuntimeError("A car moved while paused")
            if self._pause_frames >= 4:
                self.screenshot("paused.png")
                self.handle("focus")
                self.handle("camera")
                self.handle("details")
                self.screenshot("focused-car-details.png")
                self.handle("details")
                self.handle("camera")
                self.pause_verified = True
                self.handle("pause")
        elif self.state in ("results", "comparison"):
            if any(r.summary["reason"] == "TECHNICAL_ERROR" for r in self.racers.values()):
                raise RuntimeError("Race encountered a technical error")
            self.screenshot(f"results-{self.smoke_index}.png")
            self.smoke_index += 1
            self._garage()
            if self.smoke_index == len(self.smoke_jobs):
                self.manifest.update(smoke_passed=True, pause_verified=self.pause_verified,
                                     persistent_window_verified=True, smoke_races=self.smoke_index,
                                     simultaneous_verified=True)
                self.screenshot("garage-final.png")
                self.quitting = True

    def run(self):
        try:
            self.open()
            while not self.quitting:
                self._phone_tick()
                while self.commands and not self.quitting:
                    command, arg = self.commands.popleft()
                    try:
                        self.handle(command, arg)
                    except Exception as error:
                        self._show_error(error)
                if self.quitting:
                    break
                self.ui.layout()
                if self.state in ("running", "paused", "countdown"):
                    self.env.update_race_camera()
                self._smoke_tick()
                if self.quitting:
                    break
                if self.state == "countdown":
                    if getattr(self,'realtime',None):
                        self.realtime._poll()
                    remaining = self.countdown_end - time.perf_counter()
                    self.ui.countdown["text"] = str(max(1, int(remaining) + 1))
                    if remaining <= 0:
                        runtime=getattr(self,'realtime',None)
                        if runtime and not runtime.workers_ready():
                            self.ui.countdown['text']='Đang chuẩn bị AI…'
                        else:
                            self._next_decision = time.perf_counter()
                            self._set_state("running")
                if self.state == "running" and time.perf_counter() >= self._next_decision:
                    try:
                        self._step()
                    except Exception as error:
                        self._show_error(error)
                else:
                    if self.state=='running' and getattr(self,'realtime',None):
                        # env.step renders once per 20 ms physics tick. Avoid
                        # extra unbounded GPU renders while waiting for it.
                        time.sleep(.002)
                    else:
                        self.env.engine.taskMgr.step()
                        time.sleep(.012)
            self.manifest["status"] = "CLOSED"
        except (KeyboardInterrupt, SystemExit):
            self.manifest["status"] = "CLOSED"
        except BaseException:
            self.manifest["status"] = "ERROR"
            raise
        finally:
            if self.episode_open:
                self._finish("ABORTED", show_results=False)
            self._destroy_hud()
            self._close_agents()
            self._close_phone()
            if self.events is not None:
                self.events.ignoreAll()
            if self.garage_pointer is not None:
                self.garage_pointer.close()
            if self.camera_input is not None:
                self.camera_input.close()
            if self.ui is not None:
                self.ui.destroy()
            if self.graphics_runtime is not None:
                self.graphics_runtime.close()
            if self.env is not None:
                self.env.close()
            self.save_manifest()
            self.decisions.close()
            self.results.close()
