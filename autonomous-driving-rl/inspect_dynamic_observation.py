from metadrive import MetaDriveEnv
import numpy as np


env = MetaDriveEnv({
    "use_render": True,

    # Có traffic để LiDAR có thứ để nhìn
    "traffic_density": 0.2,
})

obs, info = env.reset()

print("Observation:", env.observation_space)
print("Action:", env.action_space)

for step in range(500):

    # steering = 0
    # throttle = 0.5
    action = np.array([0.0, 0.5], dtype=np.float32)

    obs, reward, terminated, truncated, info = env.step(action)

    ego = obs[:9]
    navigation = obs[9:19]
    lidar = obs[19:]

    if step % 20 == 0:
        print(
            f"step={step:4d} | "
            f"velocity={info['velocity']:7.3f} | "
            f"route={info['route_completion']:.3f} | "
            f"lidar_min={lidar.min():.3f} | "
            f"lidar_mean={lidar.mean():.3f} | "
            f"crash={info['crash']}"
        )

    if terminated or truncated:
        print("\nEpisode ended")
        print("arrive_dest:", info["arrive_dest"])
        print("crash:", info["crash"])
        print("out_of_road:", info["out_of_road"])
        print("route_completion:", info["route_completion"])
        break

env.close()