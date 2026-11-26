"""Shared-world cars on the existing TRAIN maps, using the pinned MetaDrive API."""

from copy import deepcopy
from types import SimpleNamespace

from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.envs.metadrive_env import MetaDriveEnv
from src.visualization.race_style import RACE_COLORS, style_vehicle


class DecisionObservation:
    """Refresh expensive LiDAR at decision time, not every physical substep."""
    def __init__(self, observation):
        self.observation=observation
        self.cached=None

    def __getattr__(self,name):
        return getattr(self.observation,name)

    def reset(self,*args,**kwargs):
        self.cached=None
        return self.observation.reset(*args,**kwargs)

    def observe(self,vehicle):
        if self.cached is None:
            self.cached=self.observation.observe(vehicle)
        return self.cached

    def refresh(self,vehicle):
        self.cached=self.observation.observe(vehicle)
        return self.cached


def build_grid(vehicle_config, indices, lane_count):
    """Two staggered rows for five cars; no overlapping bodies at reset."""
    configs = {}
    for slot, index in enumerate(indices):
        config = deepcopy(dict(vehicle_config))
        config.update(
            spawn_lane_index=(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, slot % lane_count),
            spawn_longitude=13. - 8. * (slot // lane_count),
            spawn_lateral=0., use_special_color=False, random_color=True,
        )
        configs[f"racer_{index}"] = config
    return configs


def racer_route_network(road_network, vehicle):
    """Select each car's current lane as its public route reference.

    The shared extractor normally chooses lanes[0]. In a race that would tell
    every side-by-side car to merge into lane 0 at the starting grid. This
    read-only view changes only reference lane selection, never map geometry,
    physics, the policy implementation, or the canonical input builder.
    """
    lane_index = int(vehicle.navigation.current_lane.index[-1])
    return SimpleNamespace(graph={
        u: {v: [lanes[min(lane_index, len(lanes) - 1)], *lanes]
            if lanes else lanes for v, lanes in outgoing.items()}
        for u, outgoing in road_network.graph.items()
    })


class RaceEnvironment(MetaDriveEnv):
    """Always uses dictionary I/O, including a one-car practice session.

    Finished cars stay parked as physical obstacles until the race is reset.
    No respawning, engine replacement, or separate episode per algorithm.
    """

    focus_id = None
    race_practice = False

    def setup_engine(self):
        super().setup_engine()
        from src.visualization.vietnam_map import RaceMapManager, RaceTrafficManager
        self.engine.update_manager('map_manager', RaceMapManager())
        self.engine.update_manager('traffic_manager', RaceTrafficManager())

    def _step_simulator(self, actions):
        info = super()._step_simulator(actions)
        from src.visualization.race_rules import cap_race_velocity
        for key,vehicle in self.agents.items():
            cap_race_velocity(vehicle,getattr(self,'race_speed_limit_kmh',20.))
            if key in info and isinstance(info[key],dict) and 'velocity' in info[key]:
                info[key]['velocity'] = float(vehicle.speed)
        return info

    def _is_out_of_road(self, vehicle):
        # Preserve separate evidence: touching a solid line is not the same
        # as leaving the lane surface. Both end a competitive race below.
        return not vehicle.on_lane

    def done_function(self, vehicle_id):
        done,info = super().done_function(vehicle_id)
        vehicle = self.agents[vehicle_id] if hasattr(self, 'agent_manager') else None
        from src.visualization.race_rules import SOLID_LINE_FLAGS
        for name in (*SOLID_LINE_FLAGS, 'on_broken_line'):
            info[name] = bool(getattr(vehicle, name, False)) if vehicle else False
        if getattr(self, 'race_practice', False) and vehicle_id == 'racer_5':
            done = bool(info.get('arrive_dest') or info.get('max_step'))
        else:
            done = done or any(info[name] for name in SOLID_LINE_FLAGS)
        return done,info

    def _get_observations(self):
        observations={key:self.get_single_observation() for key in self.config['agent_configs']}
        return {key:DecisionObservation(value) for key,value in observations.items()} if getattr(self,'decision_observations',False) else observations

    def refresh_observation(self,key):
        observation=self.observations[key]
        return observation.refresh(self.agents[key]) if isinstance(observation,DecisionObservation) else observation.observe(self.agents[key])

    @property
    def agent(self):
        # The existing read-only HUD needs one focused vehicle.
        return self.agents.get(self.focus_id) or next(iter(self.agents.values()))

    def configure_grid(self, indices):
        configs = build_grid(self.config["vehicle_config"], indices, self.config["map_config"]["lane_num"])
        self.config["agent_configs"] = configs
        self.config["num_agents"] = self.num_agents = len(configs)
        self.focus_id = next(iter(configs))
        self.config["prefer_track_agent"] = self.focus_id
        # VehicleAgentManager fixes its observation/space templates at creation.
        # Refresh only those templates; reset() owns old-object teardown.
        manager = self.agent_manager
        manager._init_observations = self._get_observations()
        manager._init_observation_spaces = {
            key: obs.observation_space for key, obs in manager._init_observations.items()
        }
        manager._init_action_spaces = manager._get_action_space()

    def focus(self, vehicle_id, top_down=False):
        self.focus_id = vehicle_id
        self.config["prefer_track_agent"] = vehicle_id
        for key,vehicle in self.agents.items():
            label=vehicle.origin.find('race_identity')
            if not label.isEmpty():
                label.show() if top_down or key!=vehicle_id else label.hide()
        if self.main_camera is not None:
            self.main_camera.track(self.agent)
            if top_down:
                self.switch_to_top_down_view()
                self.main_camera.set_bird_view_pos_hpr(self.agent.position)

    def update_race_camera(self):
        if self.main_camera is not None and self.main_camera.is_bird_view_camera():
            # Keep overhead viewing centered as the focused car drives.
            self.main_camera.camera_x = float(self.agent.position[0])
            self.main_camera.camera_y = float(self.agent.position[1])

    def decorate(self, vehicle_id, index, name, font, model='lamborghini'):
        from src.visualization.race_vehicle_models import install_car
        label = style_vehicle(self.agents[vehicle_id], index, name, font)
        install_car(self.agents[vehicle_id], model, index)
        if getattr(self,'_race_surface_shader',None) is not None:
            from src.visualization.race_landscape import shade_vehicle
            shade_vehicle(self.agents[vehicle_id].origin,self._race_surface_shader)
        return label

    def recover(self, vehicle_id):
        """Move only the player to its current route lane; preserve other cars."""
        if not self.race_practice or vehicle_id != 'racer_5':
            return False
        vehicle = self.agents[vehicle_id]
        lane = vehicle.navigation.current_lane
        longitudinal, _ = lane.local_coordinates(vehicle.position)
        longitudinal = max(1., min(float(lane.length)-1., float(longitudinal)))
        # Avoid teleporting into another car. Search backwards along this lane.
        import numpy as np
        for offset in (0,-8,8,-16,16,-24,24,-32,32):
            target_long = max(1., min(float(lane.length)-1.,longitudinal-offset))
            position = lane.position(target_long, 0.)
            others = [v for key,v in self.agents.items() if key != vehicle_id]
            others += list(getattr(self.engine.traffic_manager, 'traffic_vehicles', []))
            if all(np.linalg.norm(np.asarray(v.position)-position) > 6. for v in others):
                break
        else:
            return False
        vehicle.set_static(False)
        vehicle.set_position(position, height=.6)
        vehicle.set_heading_theta(lane.heading_theta_at(target_long))
        vehicle.set_velocity([0.,0.])
        vehicle.set_angular_velocity(0.)
        vehicle.set_roll(0.)
        vehicle.set_pitch(0.)
        vehicle._init_step_info()
        return True

    def park(self, vehicle_id):
        vehicle = self.agents[vehicle_id]
        vehicle.set_velocity([0., 0.])
        vehicle.set_angular_velocity(0.)
        vehicle.set_static(True)
