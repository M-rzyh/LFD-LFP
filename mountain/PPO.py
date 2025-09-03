from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Vectorized environment
env = make_vec_env("MountainCar-v0", n_envs=1)

# Train PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

# Save the model
model.save("ppo_mountaincar")