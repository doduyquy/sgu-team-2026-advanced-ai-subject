from metadrive import MetaDriveEnv


config = MetaDriveEnv.default_config()


keywords = [
    "reward",
    "penalty",
    "cost",
    "success",
    "collision",
    "out_of_road",
    "driving",
    "speed",
]


print("===== REWARD-RELATED CONFIG =====")

for key, value in config.items():

    key_lower = key.lower()

    if any(word in key_lower for word in keywords):
        print(f"{key:35s} = {value}")