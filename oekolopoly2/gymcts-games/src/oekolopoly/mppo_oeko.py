import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env


import gymnasium
from stable_baselines3.common.noise import NormalActionNoise

from oekolopoly.env.oeko_env import OekoEnv, OekoActionBuilderWrapper
from oekolopoly.env.oeko_wrappers import OekoAuxRewardWrapper

gymnasium.register(
    id="OekoEnv-v0",
    entry_point="oekolopoly.env.oeko_env:OekoEnv",
)


def make_env():
    env = gym.make('OekoEnv-v0')
    env = OekoAuxRewardWrapper(env)
    env = OekoActionBuilderWrapper(env, auxilary_reward=True)
    def action_mask_fn(env: OekoActionBuilderWrapper):
        return env.valid_action_mask()
    env = ActionMasker(env=env, action_mask_fn=action_mask_fn)
    return env

env = make_vec_env(make_env, n_envs=8)

model = MaskablePPO(
    "MlpPolicy",
    env,
    #n_steps=128,
    #batch_size=64,
    #n_epochs=10,
    #learning_rate=2.5e-4,
    #gamma=0.99,
    #gae_lambda=0.98,
    #ent_coef=0.01,
    #policy_kwargs=dict(net_arch=[32, 32]),
    verbose=1,
)

if __name__ == '__main__':
    # n_actions = 9
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model.learn(total_timesteps=800_000)

    # Testen (einzelnes Environment, keine Vektorisierung)
    OekoEnv.PRINT_STEP_TRANSITIONS = True
    OekoEnv.PRINT_DONE_REASONS = True
    env = make_env()
    obs, info = env.reset()
    done, truncated = False, False
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True, action_masks=env.env.valid_action_mask())
        if isinstance(action, np.ndarray):
            action = action.item()
        obs, reward, done, truncated, info = env.step(action)
    env.render()

