"""CPU MPPI prototype for the native TRAIN playground, not a benchmark policy.

Receding-horizon importance-weighted control sampling with a local bicycle
model. Only AgentInputV1 is consumed. Model dynamics and obstacle/corridor costs
are approximations; bounded controls use projected Gaussian perturbations.
Reference: Williams et al., https://arxiv.org/abs/1707.02342.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np

from src.agents.mcts_model import MCTSConfig, PlanningScene
from src.platform.agent import AgentDecision, AgentDescriptor


@dataclass(frozen=True)
class MPPIConfig:
    samples: int = 192
    horizon: int = 20
    temperature: float = 8.0
    steering_noise: float = 0.18
    throttle_noise: float = 0.30
    target_speed_m_s: float = 5.0
    control_cost: float = 0.05

    def __post_init__(self):
        for name in ("samples", "horizon"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 2:
                raise ValueError(f"{name} must be an integer >= 2")
        for name in ("temperature", "steering_noise", "throttle_noise", "target_speed_m_s", "control_cost"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


class MPPIAgent:
    agent_id = "race_mppi"
    agent_version = "0.1.0"

    def __init__(self, config=None):
        self.config = config or MPPIConfig()
        self._rng = None
        self._controls = None
        self._dt = 0.1

    @property
    def descriptor(self):
        return AgentDescriptor(
            agent_id=self.agent_id, agent_version=self.agent_version,
            action_adapter_id="continuous_box2_v1", inference_stochasticity="stochastic",
            stateful_within_episode=True, method_family="MPPI_PROTOTYPE",
        )

    @classmethod
    def experiment_parameters(cls):
        return {"model": "local_bicycle_v1", "prototype": True, **asdict(MPPIConfig())}

    def reset(self, public_context, agent_seed=None):
        if public_context.input_profile_id != "STATE_DECISION_V1":
            raise ValueError("MPPI requires STATE_DECISION_V1")
        if public_context.action_adapter_id != "continuous_box2_v1":
            raise ValueError("MPPI requires continuous_box2_v1")
        self._dt = float(public_context.control_dt_s)
        self._rng = np.random.default_rng(42 if agent_seed is None else agent_seed)
        self._controls = np.zeros((self.config.horizon, 2), dtype=float)
        self._controls[:, 1] = 0.45
        self._fresh = True

    def close(self):
        self._rng = self._controls = None

    def _scene(self, inputs):
        # Pinned simulator normalization: z = (speed_kmh + 1) / (80 + 1).
        speed = max(0., float(inputs.core_observation.ego_state[3]) * 81. - 1.) / 3.6
        task, traffic = inputs.task_context, inputs.traffic_context
        planning = dict(speed_m_s=speed, lane_width=float(task.current_lane_width),
                        waypoints=task.waypoints_array[task.validity_mask],
                        actors=traffic.actors_array[traffic.validity_mask],
                        lidar=inputs.core_observation.lidar_points)
        return speed, PlanningScene(planning, MCTSConfig(max_steering_deg=60.))

    def _rollouts(self, scene, initial_speed, controls):
        count, horizon, _ = controls.shape
        x, y, yaw = np.zeros((3, count))
        speed = np.full(count, initial_speed)
        cost = np.zeros(count)
        trace = []
        for t in range(horizon):
            steer, pedal = controls[:, t, 0], controls[:, t, 1]
            for _ in range(2):
                dt = self._dt / 2
                acceleration = np.where(pedal >= 0, 2.8 * pedal, 5. * pedal) - .05 * speed
                speed = np.clip(speed + acceleration * dt, 0., 80. / 3.6)
                yaw += speed / 2.5 * np.tan(steer * math.radians(60)) * dt
                x += speed * np.cos(yaw) * dt
                y += speed * np.sin(yaw) * dt
            positions = np.column_stack((x, y))
            progress, lateral, heading = scene.project(positions)
            error = (yaw - heading + np.pi) % (2 * np.pi) - np.pi
            target_speed = self.config.target_speed_m_s * np.clip(1. - np.abs(error), .4, 1.)
            cost += 8 * lateral ** 2 + 3 * error ** 2 + .5 * (speed - target_speed) ** 2
            body_lateral = lateral + .95 + 2.3 * np.abs(np.sin(error))
            outside = np.maximum(0., body_lateral - scene.half_lane)
            cost += 120 * outside ** 2
            if len(scene.actors):
                centers = scene.actors[:, :2] + (t + 1) * self._dt * scene.actor_velocities
                delta = positions[:, None, :] - centers[None, :, :]
                # Conservative axis-aligned local ellipse, not exact simulator boxes.
                radii = np.column_stack((scene.actors[:, 5] / 2 + 2.5,
                                         scene.actors[:, 6] / 2 + 1.2))
                overlap = np.sum((delta / radii) ** 2, axis=2)
                cost += 350 * np.maximum(0., 1. - overlap).max(axis=1)
            if len(scene.static_hits):
                nearest = np.sort(np.sum((positions[:, None, :] - scene.static_hits) ** 2, axis=2), axis=1)[:, 0]
                cost += 180 * np.maximum(0., 1.5 ** 2 - nearest)
            cost += .2 * steer ** 2
            if t:
                cost += 2 * (steer - controls[:, t - 1, 0]) ** 2
            trace.append(positions)
        cost -= 3 * progress
        return cost, np.stack(trace, axis=1)

    def act(self, agent_input):
        if self._rng is None:
            raise RuntimeError("Reset MPPI before inference")
        if agent_input.profile_id != "STATE_DECISION_V1":
            raise ValueError("MPPI requires STATE_DECISION_V1")
        route = agent_input.task_context.waypoints_array[agent_input.task_context.validity_mask]
        if not len(route) or not np.any(np.linalg.norm(route[:, :2], axis=1) > 1e-6):
            self._controls.fill(0.)
            return AgentDecision([0., -0.7], {"planner_status": "NO_PUBLIC_ROUTE", "prototype": True})
        speed, scene = self._scene(agent_input)
        if self._fresh:
            distances = np.linalg.norm(route[:, :2], axis=1)
            ahead = np.flatnonzero(distances >= 5.)
            target = route[ahead[0] if len(ahead) else -1, :2]
            angle = math.atan2(float(target[1]), float(target[0]))
            nominal = math.atan2(5. * math.sin(angle), max(1., float(np.linalg.norm(target)))) / math.radians(60)
            self._controls[:, 0] = np.clip(nominal, -.65, .65)
            self._fresh = False
        sigma = np.array([self.config.steering_noise, self.config.throttle_noise])
        noise = self._rng.normal(size=(self.config.samples, self.config.horizon, 2)) * sigma
        noise[0] = 0.  # Include the unperturbed warm-start sequence.
        candidates = np.clip(self._controls + noise, [-.65, -1.], [.65, 1.])
        effective_noise = candidates - self._controls
        costs, traces = self._rollouts(scene, speed, candidates)
        costs += self.config.control_cost * np.sum(self._controls * effective_noise / sigma ** 2, axis=(1, 2))
        weights = np.exp(-(costs - np.min(costs)) / self.config.temperature)
        weights /= weights.sum()
        updated = np.clip(self._controls + np.einsum("k,khi->hi", weights, effective_noise),
                          [-.65, -1.], [.65, 1.])
        action = updated[0].copy()
        self._controls[:-1] = updated[1:]
        self._controls[-1] = updated[-1]
        best = int(np.argmin(costs))
        return AgentDecision(action.tolist(), {
            "planner_status": "CONTROL_SAMPLED", "prototype": True,
            "samples": self.config.samples, "prediction_horizon_s": self.config.horizon * self._dt,
            "minimum_sample_cost": float(costs[best]),
            "effective_samples": float(1. / np.sum(weights ** 2)),
            "best_sample_path": traces[best].round(3).tolist(),
        })
