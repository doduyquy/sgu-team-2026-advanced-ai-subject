from src.environments.course_env_v1 import CourseEnvV1


env = CourseEnvV1(
    render=True,
    traffic_density=0.0,
)

obs, info = env.reset()

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
print("Observation shape:", obs.shape)

for step in range(300):

    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    if step % 20 == 0:
        print(
            f"step={step:3d} | "
            f"action={info['course_action']:10s} | "
            f"reward={reward:7.3f} | "
            f"route={info['route_completion']:.3f} | "
            f"crash={info['crash']} | "
            f"out={info['out_of_road']}"
        )

    if terminated or truncated:
        print("\nEpisode finished")
        print("arrive_dest:", info["arrive_dest"])
        print("crash:", info["crash"])
        print("out_of_road:", info["out_of_road"])
        print("route_completion:", info["route_completion"])
        break

env.close()