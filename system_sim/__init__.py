from gymnasium.envs.registration import register

register(
    id="system_sim/SimpleDriveEnv-v0",
    entry_point="system_sim.envs:SimpleDriveEnv",
)
