import gymnasium as gym
import numpy as np

from gymnasium import spaces
from metadrive import MetaDriveEnv


class CourseEnvV1(gym.Wrapper):
    """
    Course environment abstraction for the development line.

    Raw MetaDrive observation:
        9 ego + 10 navigation + 240 lidar = 259D

    Course observation:
        9 ego + 10 navigation + 16 lidar sectors = 35D

    Course action:
        Discrete(5)
    """

    ACTION_MAP = {
        0: np.array([-0.35,  0.35], dtype=np.float32),  # LEFT
        1: np.array([ 0.00,  0.40], dtype=np.float32),  # STRAIGHT
        2: np.array([ 0.35,  0.35], dtype=np.float32),  # RIGHT
        3: np.array([ 0.00,  0.80], dtype=np.float32),  # ACCELERATE
        4: np.array([ 0.00, -0.80], dtype=np.float32),  # BRAKE
    }

    ACTION_NAMES = {
        0: "LEFT",
        1: "STRAIGHT",
        2: "RIGHT",
        3: "ACCELERATE",
        4: "BRAKE",
    }

    def __init__(self, render=False, traffic_density=0.0):
        env = MetaDriveEnv({
            "use_render": render,
            "traffic_density": traffic_density,
            "num_scenarios": 1,
            "start_seed": 0,
        })

        super().__init__(env)

        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(35,),
            dtype=np.float32,
        )

    @staticmethod
    def _compress_observation(obs):
        ego_navigation = obs[:19]

        lidar = obs[19:]

        # 240 rays -> 16 sectors
        sectors = np.array(
            [
                np.min(chunk)
                for chunk in np.array_split(lidar, 16)
            ],
            dtype=np.float32,
        )

        return np.concatenate(
            [ego_navigation, sectors]
        ).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        return self._compress_observation(obs), info

    def step(self, action):
        continuous_action = self.ACTION_MAP[int(action)]

        obs, reward, terminated, truncated, info = self.env.step(
            continuous_action
        )

        obs = self._compress_observation(obs)

        info["course_action"] = self.ACTION_NAMES[int(action)]

        return obs, reward, terminated, truncated, info