from metadrive import MetaDriveEnv

env = MetaDriveEnv({
    "use_render": False
})

obs, info = env.reset()

print("=== OBSERVATION SPACE ===")
print(env.observation_space)

print("\n=== ACTION SPACE ===")
print(env.action_space)

print("\n=== OBS SHAPE ===")
print(obs.shape)

print("\n=== FIRST 20 OBS VALUES ===")
print(obs[:20])

print("\n=== INFO ===")
print(info)

env.close()