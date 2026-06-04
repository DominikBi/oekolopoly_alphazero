import pprint

import gymnasium as gym
import numpy as np
import wandb
import wandb as wb
from gymcts.gymcts_deepcopy_wrapper import DeepCopyMCTSGymEnvWrapper
from gymcts.gymcts_neural_agent import GymctsNeuralAgent
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env

from gymcts.logger import log
from wandb.integration.sb3 import WandbCallback

from experiments.frozenlake.frozenlake_wrappers import ShorterEpisodeBonusWrapper

# the config also includes unimportant parameters that are not varied in the sweep,
# but are still logged for better reproducibility and analysis of the results.
experiment_sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'solved_in_steps',
        'goal': 'minimize'
    },
    'parameters': {
        "mcts_simulation_budget_per_step": {
            'values': [150]
        },
        "instance_size": {
            'values': ["4x4"]
        },

        "score_variate": {
            'values': ["MuZero_v1"]
        },

        "clear_mcts_tree_after_step": {
            'values': [True]
        },

        "keep_whole_tree_till_initial_root": {
            'values': [False]
        },

        # unimportant parameter
        # added to have more information logged for better reproducibility
        "exclude_unvisited_nodes_from_render": {
            'values': [True]
        },

        # unimportant parameter
        # added to have more information logged for better reproducibility
        "render_tree_max_depth": {
            'values': [2]
        },

        # unimportant parameter
        # added to have more information logged for better reproducibility
        "calc_number_of_simulations_per_step": {
            'values': [None]
        },

        # unimportant parameter
        # added to have more information logged for better reproducibility
        "render_tree_after_step": {
            'values': [True]
        },

        "best_action_weight": {
            'values': [0.0]
        },

        "time_limit_max_episode_steps": {
            'values': [50]
        },

        "n_steps": {
            'values': [128]
        },

        "batch_size": {
            'values': [64]
        },

        "n_epochs": {
            'values': [10]
        },

        "learning_rate": {
            'values': [2.5e-4]
        },

        "gamma": {
            'values': [0.99]
        },

        "gae_lambda": {
            'values': [0.98]
        },

        "ent_coef": {
            'values': [0.01]
        },

        "normalize_advantage": {
            'values': [True]
        },

        "net_arch_layer": {
            'values': [32]
        },

        "verbose": {
            'values': [1]
        },

        # used as seed
        "run_no": {
            'values': [1]
        },

        "num_timesteps": {
            'values': [2_500]
        },

        # used as seed
        "map_no": {
            'values': [0, 1, 2, 3, 4]
        },
    }
}


def perform_run():
    with wb.init(
            sync_tensorboard=False,
            monitor_gym=False,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
            # dir=f"{PATHS.WAND_OUTPUT_PATH}/"
    ) as run:
        log.info(f"run name: {run.name}, run id: {run.id}")

        experiment_params = wb.config
        log.info(f"experiment params: {pprint.pformat(experiment_params)}")

        custom_map_n0 = [
            "SFFF",
            "FHFH",
            "FFFH",
            "HFFG"
        ]
        custom_map_n1 = [
            "SHHF",
            "FFFF",
            "FFFF",
            "FFHG",
        ]
        custom_map_n2 = [
            "SFHH",
            "FHFF",
            "FFFF",
            "FHFG",
        ]
        custom_map_n3 = [
            "SFFF",
            "FHHF",
            "HHFF",
            "HFFG",
        ]
        custom_map_n4 = [
            "SFFF",
            "FHFH",
            "FFFF",
            "FFFG",
        ]

        available_maps = [custom_map_n0, custom_map_n1, custom_map_n2, custom_map_n3, custom_map_n4]

        map_idx = experiment_params["map_no"]
        map = available_maps[map_idx]

        wandb.log({"map": map})

        # Action mapping: 0=Left, 1=Down, 2=Right, 3=Up
        def action_mask_fn(env):
            pos = env.unwrapped.s
            nrow, ncol = env.unwrapped.nrow, env.unwrapped.ncol
            row, col = pos // ncol, pos % ncol
            mask = np.ones(4, dtype=bool)
            if row == 0: mask[3] = False  # Up
            if row == nrow - 1: mask[1] = False  # Down
            if col == 0: mask[0] = False  # Left
            if col == ncol - 1: mask[2] = False  # Right
            return mask

        env = gym.make('FrozenLake-v1', desc=map, is_slippery=False, render_mode="ansi")
        env = ShorterEpisodeBonusWrapper(env)
        env = TimeLimit(env,
                        max_episode_steps=50)  # limit the episode length to prevent infinite loops and too long MCTS simulations
        env = ActionMasker(env=env, action_mask_fn=action_mask_fn)
        env.reset()

        env = DeepCopyMCTSGymEnvWrapper(env, action_mask_fn=action_mask_fn)

        layers_dim = experiment_params["net_arch_layer"]
        masked_ppo_model_kwargs = {
            "n_steps": experiment_params["n_steps"],
            "batch_size": experiment_params["batch_size"],
            "n_epochs": experiment_params["n_epochs"],
            "learning_rate": experiment_params["learning_rate"],
            "gamma": experiment_params["gamma"],
            "gae_lambda": experiment_params["gae_lambda"],
            "ent_coef": experiment_params["ent_coef"],
            "policy_kwargs": dict(net_arch=[layers_dim, layers_dim]),
            "normalize_advantage": experiment_params["normalize_advantage"],
            "verbose": experiment_params["verbose"],
            "seed": experiment_params["run_no"]  # using run_no as seed for better
        }

        neural_mcts_agent_kwargs = {
            "clear_mcts_tree_after_step": experiment_params["clear_mcts_tree_after_step"],
            "render_tree_after_step": experiment_params["render_tree_after_step"],
            "render_tree_max_depth": experiment_params["render_tree_max_depth"],
            "number_of_simulations_per_step": experiment_params["mcts_simulation_budget_per_step"],
            "exclude_unvisited_nodes_from_render": experiment_params["exclude_unvisited_nodes_from_render"],
            "keep_whole_tree_till_initial_root": experiment_params["keep_whole_tree_till_initial_root"],
            "calc_number_of_simulations_per_step": experiment_params["calc_number_of_simulations_per_step"],
            "score_variate": experiment_params["score_variate"],
            "best_action_weight": experiment_params["best_action_weight"],
        }

        timesteps = experiment_params["num_timesteps"]
        log.info(f"training for {timesteps} timesteps...")

        agent = GymctsNeuralAgent(
            env=env,
            model_kwargs=masked_ppo_model_kwargs,
            **neural_mcts_agent_kwargs,
        )


        env.reset()

        agent.learn(
            total_timesteps=timesteps,
            callback=WandbCallback(verbose=2)
        )

        actions = agent.solve()
        env.reset()

        #### evaluation without mcts tree search, to see the performance of the learned policy alone
        print("##### Evaluating the learned policy without MCTS tree search #####")
        obs, info = env.reset()
        model = agent._model
        done, truncated = False, False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True, action_masks=action_mask_fn(env.unwrapped))
            if isinstance(action, np.ndarray):
                action = action.item()
            obs, reward, done, truncated, info = env.step(action)
            print(env.render())
        log.info(f"without mcts Reward: {reward}")

        episode_length = info["episode"]["l"]
        episode_return = info["episode"]["r"]

        wb.log(
            {
                "pre_solved_in_steps": episode_length,
                "pre_last_reward": reward,
                "pre_episode_return": episode_return,
                "pre_terminated": done,
                "pre_truncated": truncated
            }
        )

        if episode_return >= 1.0:
            log.info(f"Environment solved in {episode_length} steps. (without search)")
        else:
            log.warning(f"Environment not solved in {episode_length} steps. (without search)")

        print("##### Evaluating the learned policy with MCTS tree search #####")
        env.reset()

        print(env.render())
        for a in actions:
            obs, rew, term, trun, info = env.step(a)
            print(env.render())

        episode_length = info["episode"]["l"]
        episode_return = info["episode"]["r"]

        log.info(f"agent trajectory: {actions}")

        wb.log(
            {
                "solved_in_steps": episode_length,
                "last_reward": rew,
                "episode_return": episode_return,
                "terminated": term,
                "truncated": trun
            }
        )

        if episode_return >= 1.0:
            log.info(f"Environment solved in {episode_length} steps.")
        else:
            log.warning(f"Environment not solved in {episode_length} steps.")


if __name__ == '__main__':
    sweep_id = wb.sweep(experiment_sweep_config, project="test")
    wb.agent(sweep_id, function=perform_run, count=1, project="test")

