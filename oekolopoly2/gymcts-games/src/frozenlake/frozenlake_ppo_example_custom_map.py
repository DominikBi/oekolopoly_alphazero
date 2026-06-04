import gymnasium as gym
import numpy as np

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env



custom_map = [
    "SFFFFF",
    "FHFFHF",
    "FFFHFF",
    "HFHFHF",
    "FFFHFF",
    "FHFHFG"
]

class NegativeHoleRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, negative_reward=-1.0):
        super().__init__(env)
        self.negative_reward = negative_reward

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Hole erkannt, wenn terminated und reward == 0 (Standard in FrozenLake)
        if terminated and reward == 0:
            reward = self.negative_reward
        return obs, reward, terminated, truncated, info



def make_env():
    env = gym.make('FrozenLake-v1', desc=custom_map, is_slippery=False, render_mode="ansi")
    # env = NegativeHoleRewardWrapper(env, negative_reward=-1.0)
    env = TimeLimit(env, max_episode_steps=50)
    return env

# Vektorisiertes Environment für stabileres Training
env = make_vec_env(make_env, n_envs=8)

model = PPO(
    "MlpPolicy",
    env,
    n_steps=128,
    batch_size=64,
    n_epochs=10,
    learning_rate=2.5e-4,
    gamma=0.99,
    gae_lambda=0.98,
    ent_coef=0.01,
    policy_kwargs=dict(net_arch=[32, 32]),
    verbose=1,
    seed=42
)

if __name__ == '__main__':
    model.learn(total_timesteps=100_000)
    model.save("ppo_frozenlake_custom")

    # Testen (einzelnes Environment, keine Vektorisierung)
    env = make_env()
    obs, info = env.reset()
    done, truncated = False, False
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        if isinstance(action, np.ndarray):
            action = action.item()
        obs, reward, done, truncated, info = env.step(action)
        print(env.render())
    print(f"Reward: {reward}")
