import numpy as np

from src.environments.course_env_v1 import CourseEnvV1


NUM_EPISODES = 20

results = []

env = CourseEnvV1(
    render=False,
    traffic_density=0.0,
)

for episode in range(NUM_EPISODES):

    obs, info = env.reset()

    total_reward = 0.0
    steps = 0

    while True:

        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    result = {
        "episode": episode,
        "reward": total_reward,
        "steps": steps,
        "route_completion": float(info["route_completion"]),
        "arrive_dest": bool(info["arrive_dest"]),
        "crash": bool(info["crash"]),
        "out_of_road": bool(info["out_of_road"]),
    }

    results.append(result)

    print(
        f"Episode {episode:02d} | "
        f"reward={total_reward:8.3f} | "
        f"steps={steps:4d} | "
        f"route={result['route_completion']:.3f} | "
        f"success={result['arrive_dest']} | "
        f"crash={result['crash']} | "
        f"out={result['out_of_road']}"
    )


success_rate = np.mean([x["arrive_dest"] for x in results])
crash_rate = np.mean([x["crash"] for x in results])
out_rate = np.mean([x["out_of_road"] for x in results])

avg_reward = np.mean([x["reward"] for x in results])
avg_route = np.mean([x["route_completion"] for x in results])
avg_steps = np.mean([x["steps"] for x in results])

print("\n===== RANDOM BASELINE =====")

print(f"Episodes:            {NUM_EPISODES}")
print(f"Success rate:        {success_rate:.3f}")
print(f"Crash rate:          {crash_rate:.3f}")
print(f"Out-of-road rate:    {out_rate:.3f}")
print(f"Average reward:      {avg_reward:.3f}")
print(f"Average route:       {avg_route:.3f}")
print(f"Average steps:       {avg_steps:.1f}")

env.close()