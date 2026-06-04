import torch
import numpy as np
import gymnasium as gym
from gymcts.gymcts_neural_agent import GymctsNeuralAgent
from gymnasium.wrappers import TimeLimit

from gymcts.gymcts_agent import GymctsAgent
from gymcts.gymcts_deepcopy_wrapper import DeepCopyMCTSGymEnvWrapper
from gymcts.logger import log
from sb3_contrib.common.wrappers import ActionMasker

log.setLevel(20)

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


if __name__ == '__main__':
    custom_map = [
        "SFFFFF",
        "FHFFHF",
        "FFFHFF",
        "HFHFHF",
        "FFFHFF",
        "FHFHFG"
    ]


    # Action mapping: 0=Links, 1=Unten, 2=Rechts, 3=Oben
    def action_mask_fn(env):
        pos = env.unwrapped.s
        nrow, ncol = env.unwrapped.nrow, env.unwrapped.ncol
        row, col = pos // ncol, pos % ncol
        mask = np.ones(4, dtype=bool)
        if row == 0: mask[3] = False  # Oben
        if row == nrow - 1: mask[1] = False  # Unten
        if col == 0: mask[0] = False  # Links
        if col == ncol - 1: mask[2] = False  # Rechts
        return mask

    env = gym.make('FrozenLake-v1', desc=custom_map, is_slippery=False, render_mode="ansi")
    env = TimeLimit(env, max_episode_steps=50)
    env = ActionMasker(env=env, action_mask_fn=action_mask_fn)
    env.reset()

    env = DeepCopyMCTSGymEnvWrapper(env, action_mask_fn=action_mask_fn)


    model_kwargs = {
        "gamma": 0.99,
        "gae_lambda": 0.98,
        "normalize_advantage": True,
        "n_epochs": 10,
        "n_steps": 128,
        "batch_size": 64,
        "ent_coef": 0.01,
        "max_grad_norm": 0.5,
        "learning_rate": 2.5e-4,
        "seed": 42,
        "policy_kwargs": {
            "net_arch": {
                "pi": [32, 32],
                "vf": [32, 32],
            },
            "ortho_init": True,
            "activation_fn": torch.nn.ELU,
            "optimizer_kwargs": {
                "eps": 1e-7
            }
        }
    }

    agent_kwargs = {
        "clear_mcts_tree_after_step": True,
        "render_tree_after_step": True,
        "render_tree_max_depth": 2,
        "number_of_simulations_per_step": 750,
        "exclude_unvisited_nodes_from_render": True,
        "keep_whole_tree_till_initial_root": False,
        "calc_number_of_simulations_per_step": None,
        "score_variate": "MuZero_v0",
        "best_action_weight": 0.05,
        "model_kwargs": model_kwargs,
    }

    agent = GymctsNeuralAgent(
        env=env,
        clear_mcts_tree_after_step=True,
        render_tree_after_step=True,
        number_of_simulations_per_step=150,
        exclude_unvisited_nodes_from_render=True
    )

    agent.learn(total_timesteps=100_000)

    env.reset()

    actions = agent.solve()

    print(env.render())
    for a in actions:
        obs, rew, term, trun, info = env.step(a)
        print(env.render())

    episode_length = info["episode"]["l"]
    episode_return = info["episode"]["r"]

    if episode_return == 1.0:
        print(f"Environment solved in {episode_length} steps.")
    else:
        print(f"Environment not solved in {episode_length} steps.")
