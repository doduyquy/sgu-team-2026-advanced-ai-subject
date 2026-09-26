from metadrive import MetaDriveEnv

env = MetaDriveEnv({
    "use_render": False
})

obs, info = env.reset()

ego = obs[:9]
navigation = obs[9:19]
lidar = obs[19:]

ego_labels = [
    "dist_left_boundary",
    "dist_right_boundary",
    "heading_diff",
    "speed",
    "current_steering",
    "last_action_steering",
    "last_action_throttle_brake",
    "yaw_rate",
    "lateral_position",
]

print("\n=== EGO STATE (9D) ===")
for i, (name, value) in enumerate(zip(ego_labels, ego)):
    print(f"{i:3d} {name:30s}: {value:.6f}")

print("\n=== NAVIGATION (10D) ===")
for checkpoint in range(2):
    base = checkpoint * 5
    values = navigation[base:base + 5]

    print(f"\nCheckpoint {checkpoint + 1}")
    print(" forward_projection :", values[0])
    print(" lateral_projection :", values[1])
    print(" lane_radius        :", values[2])
    print(" clockwise          :", values[3])
    print(" lane_angle         :", values[4])

print("\n=== LIDAR ===")
print("Shape:", lidar.shape)
print("Min:", lidar.min())
print("Max:", lidar.max())
print("Mean:", lidar.mean())

nearest_ray = lidar.argmin()

print("Nearest ray index:", nearest_ray)
print("Nearest ray normalized distance:", lidar[nearest_ray])

env.close()