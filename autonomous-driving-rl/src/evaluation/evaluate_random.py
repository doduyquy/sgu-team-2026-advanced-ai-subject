"""Legacy 35D/Discrete(5) demo; run explicitly with python -m, not on import.

Current experiments use src.launcher.cli and its full provenance logging.
"""
import numpy as np

from src.environments.course_env_v1 import CourseEnvV1

NUM_EPISODES = 20


def main():
    results = []
    env = CourseEnvV1(render=False, traffic_density=0.0)
    try:
        for episode in range(NUM_EPISODES):
            env.reset()
            total_reward, steps = 0.0, 0
            while True:
                action = env.action_space.sample()
                _, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                if terminated or truncated:
                    break
            result = {
                "episode": episode, "reward": total_reward, "steps": steps,
                "route_completion": float(info["route_completion"]),
                "arrive_dest": bool(info["arrive_dest"]), "crash": bool(info["crash"]),
                "out_of_road": bool(info["out_of_road"]),
            }
            results.append(result)
            print(
                f"Episode {episode:02d} | reward={total_reward:8.3f} | steps={steps:4d} | "
                f"route={result['route_completion']:.3f} | success={result['arrive_dest']} | "
                f"crash={result['crash']} | out={result['out_of_road']}"
            )
        print("\n===== RANDOM BASELINE =====")
        print(f"Episodes:            {NUM_EPISODES}")
        for label, key in (("Success rate", "arrive_dest"), ("Crash rate", "crash"),
                           ("Out-of-road rate", "out_of_road"), ("Average reward", "reward"),
                           ("Average route", "route_completion"), ("Average steps", "steps")):
            value = np.mean([result[key] for result in results])
            precision = 1 if key == "steps" else 3
            print(f"{label + ':':20s}{value:.{precision}f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
