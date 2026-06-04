import pprint

import gymnasium as gym
import numpy as np
import wandb
import wandb as wb
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env

from gymcts.logger import log
from wandb.integration.sb3 import WandbCallback

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
            'values': [250]
        },
        "instance_size": {
            'values': ["6x6"]
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
           'values': [1, 2, 3, 4, 5]
        },

        "num_timesteps":{
            'values': [10_000, 100_000, 1_000_000]
        },

        "map_no": {
           'values': [0,1,2,3,4]
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
            "SFHFFF",
            "FFFFFF",
            "FHFFHF",
            "FFHHFF",
            "HFFFFH",
            "FFFFFG",
        ]


        custom_map_n1 = [
            "SHFFFF",
            "FFFFHF",
            "FFFFFF",
            "FFHHFF",
            "HFFFFF",
            "FFFFFG",
        ]

        custom_map_n2 = [
            "SFFFFHHH",
            "FFFFFFFF",
            "FFHFFFFF",
            "FFHFHFFF",
            "FFHHFFFF",
            "FFFFFFHH",
            "FHFFFHHF",
            "HFHHFFFG",
        ]

        custom_map_n3 = [
            "SFHFFF",
            "FFFFFF",
            "FFFFFF",
            "FFFFFF",
            "FFFFFF",
            "HFFHFG",
        ]

        custom_map_n4 = [
            "SFHFFF",
            "FFFFFF",
            "FFHFHF",
            "HHFFFF",
            "FFHHFF",
            "FFFHFG",
        ]

        available_maps = [custom_map_n0, custom_map_n1, custom_map_n2, custom_map_n3, custom_map_n4]

        map_idx = experiment_params["map_no"]
        map = available_maps[map_idx]

        wandb.log({"map": map})

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

        def make_env():
            env = gym.make('FrozenLake-v1', desc=map, is_slippery=False, render_mode="ansi")
            env = TimeLimit(env, max_episode_steps=50)
            env = ActionMasker(env=env, action_mask_fn=action_mask_fn)
            env = RecordEpisodeStatistics(env, buffer_length=100) # that's the default value used in the gymcts deepcopy wrapper
            return env

        env = make_vec_env(make_env, n_envs=8)

        layers_dim= experiment_params["net_arch_layer"]
        model_kwargs = {
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

        model = MaskablePPO(
            "MlpPolicy",
            env,
            **model_kwargs
        )

        timesteps = experiment_params["num_timesteps"]
        log.info(f"training for {timesteps} timesteps...")
        model.learn(
            total_timesteps=timesteps,
            callback=WandbCallback(verbose=2)
        )
        # model.save("ppo_frozenlake_custom_masked")

        env = make_env()
        obs, info = env.reset()
        done, truncated = False, False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True, action_masks=action_mask_fn(env.unwrapped))
            if isinstance(action, np.ndarray):
                action = action.item()
            obs, reward, done, truncated, info = env.step(action)
            print(env.render())
        log.info(f"Reward: {reward}")

        episode_length = info["episode"]["l"]
        episode_return = info["episode"]["r"]

        if episode_return == 1.0:
            log.info(f"Environment solved in {episode_length} steps.")
        else:
            log.warning(f"Environment not solved in {episode_length} steps.")

if __name__ == '__main__':
    sweep_id = wb.sweep(experiment_sweep_config, project="test")
    wb.agent(sweep_id, function=perform_run, count=1, project="test")

