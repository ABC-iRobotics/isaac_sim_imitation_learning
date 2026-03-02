import gymnasium as gym

gym.envs.register(
    id="IL_Gym_Env",
    entry_point="IL_Gym_Env:IL_Gym_Env",
    max_episode_steps=10000,
)
