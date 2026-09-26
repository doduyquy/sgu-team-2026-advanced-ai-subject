from metadrive import MetaDriveEnv

env = MetaDriveEnv({
    "use_render": False
})

obs, info = env.reset()

vehicle = env.agent

print("Observation length:", len(obs))
print("Vehicle:", type(vehicle))
print("Navigation:", type(vehicle.navigation))

print("\nVehicle config lidar:")
print(vehicle.config["lidar"])

print("\nNavigation state:")
print(vehicle.navigation.get_state())

print("\nVehicle physical state:")
state = vehicle.get_state()

for key, value in state.items():
    print(key, "=", value)

env.close()