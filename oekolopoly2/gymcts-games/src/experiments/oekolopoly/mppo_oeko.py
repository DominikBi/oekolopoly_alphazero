import pprint
from itertools import accumulate

import wandb as wb

import gymnasium as gym
import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env

from gymcts.logger import log


import gymnasium
from wandb.integration.sb3 import WandbCallback

from oekolopoly.env.oeko_env import OekoEnv, OekoActionBuilderWrapper
from oekolopoly.env.oeko_wrappers import OekoAuxRewardWrapper

gymnasium.register(
    id="OekoEnv-v0",
    entry_point="oekolopoly.env.oeko_env:OekoEnv",
)

experiment_sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'eval_accumulated_reward',
        'goal': 'maximize'
    },
    'parameters': {
        "approach": {
            'values': ["PPO"]
        },
        "environment": {
            'values': ["OekoEnv-v0"]
        },
        "n_steps": {
            'values': [2048]
        },

        "batch_size": {
            'values': [64]
        },

        "n_epochs": {
            'values': [10]
        },

        "learning_rate": {
            'values': [3e-4]
        },

        "gamma": {
            'values': [0.99]
        },

        "gae_lambda": {
            'values': [0.95]
        },

        "ent_coef": {
            'values': [0.0]
        },

        "vf_coef": {
            'values': [0.5]
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

        "run_no": {
           'values': [1, 2, 3, 4, 5]
        },

        "num_timesteps":{
            'values': [10_000, 100_000, 1_000_000]
        },
    }
}

class LoggingWrapper(gym.Wrapper):

    logging_enabled = True
    num_timestep = 0

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        LoggingWrapper.num_timestep += 8 # increment by number of environments if vectorized

        # log.info(f"Step info: \n{pprint.pformat(log_dict)}")
        if terminated and LoggingWrapper.logging_enabled:
            log.debug(f"Episode done. Reason: {info['done_reason']}. Final balance: {self.env.unwrapped.balance}")
            done_logs = {
                "final_balance": self.env.unwrapped.balance,
                "balance_always": self.env.unwrapped.balance_always,
                "balance_numerator": self.env.unwrapped.balance_numerator,

                "round": self.env.unwrapped.V[self.env.unwrapped.ROUND],
                "sanitation": self.env.unwrapped.V[self.env.unwrapped.SANITATION],
                "production": self.env.unwrapped.V[self.env.unwrapped.PRODUCTION],
                "education": self.env.unwrapped.V[self.env.unwrapped.EDUCATION],
                "quality_of_life": self.env.unwrapped.V[self.env.unwrapped.QUALITY_OF_LIFE],
                "population_growth": self.env.unwrapped.V[self.env.unwrapped.POPULATION_GROWTH],
                "environment": self.env.unwrapped.V[self.env.unwrapped.ENVIRONMENT],
                "population": self.env.unwrapped.V[self.env.unwrapped.POPULATION],
                "politics": self.env.unwrapped.V[self.env.unwrapped.POLITICS],

                "done_reason": info['done_reason'],

                "num_timestep": LoggingWrapper.num_timestep,
            }
            wb.log(done_logs)

        return obs, reward, terminated, truncated, info


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

        def make_env():
            env = gym.make('OekoEnv-v0')
            env = LoggingWrapper(env)
            env = OekoAuxRewardWrapper(env, scaling=0.1)
            env = OekoActionBuilderWrapper(env, auxilary_reward=True)

            def action_mask_fn(env: OekoActionBuilderWrapper):
                return env.valid_action_mask()

            env = ActionMasker(env=env, action_mask_fn=action_mask_fn)
            return env

        env = make_vec_env(make_env, n_envs=8)

        layers_dim = experiment_params["net_arch_layer"]
        model_kwargs = {
            "n_steps": experiment_params["n_steps"],
            "batch_size": experiment_params["batch_size"],
            "n_epochs": experiment_params["n_epochs"],
            "learning_rate": experiment_params["learning_rate"],
            "gamma": experiment_params["gamma"],
            "gae_lambda": experiment_params["gae_lambda"],
            "ent_coef": experiment_params["ent_coef"],
            "vf_coef": experiment_params["vf_coef"],
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

        num_timesteps = experiment_params["num_timesteps"]
        model.learn(
            total_timesteps=num_timesteps,
            callback=WandbCallback(verbose=2)
        )

        OekoEnv.PRINT_STEP_TRANSITIONS = True
        OekoEnv.PRINT_DONE_REASONS = True

        LoggingWrapper.logging_enabled = False # disable logging of intermediate steps during evaluation

        env = make_env()
        obs, info = env.reset()
        done, truncated = False, False
        accumulated_reward = 0
        action_history = []
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True, action_masks=env.env.valid_action_mask())
            if isinstance(action, np.ndarray):
                action = action.item()
            action_history.append(action)
            obs, reward, done, truncated, info = env.step(action)
            accumulated_reward += reward

        env.render()

        # log final evaluation metrics
        eval_logs = {
            "eval_final_balance": env.unwrapped.balance,
            "eval_balance_always": env.unwrapped.balance_always,
            "eval_balance_numerator": env.unwrapped.balance_numerator,

            "eval_round": env.unwrapped.V[env.unwrapped.ROUND],

            "eval_sanitation": env.unwrapped.V[env.unwrapped.SANITATION],
            "eval_production": env.unwrapped.V[env.unwrapped.PRODUCTION],
            "eval_education": env.unwrapped.V[env.unwrapped.EDUCATION],
            "eval_quality_of_life": env.unwrapped.V[env.unwrapped.QUALITY_OF_LIFE],
            "eval_population_growth": env.unwrapped.V[env.unwrapped.POPULATION_GROWTH],
            "eval_environment": env.unwrapped.V[env.unwrapped.ENVIRONMENT],
            "eval_population": env.unwrapped.V[env.unwrapped.POPULATION],
            "eval_politics": env.unwrapped.V[env.unwrapped.POLITICS],

            "eval_done_reason": info['done_reason'],

            "eval_accumulated_reward": accumulated_reward,
        }
        wb.log(eval_logs)
        wb.log({"eval_action_history": action_history})
        log.info(f"Eval results: \n{pprint.pformat(eval_logs)}")
        log.info(f"Final balance: {env.unwrapped.balance}, Reason: {info['done_reason']}. ")


if __name__ == '__main__':
    sweep_id = wb.sweep(experiment_sweep_config, project="gymcts-oekolopoly", entity="querry")
    print(sweep_id)
    #sweep_id = "es1qfpla"
    #wb.agent(sweep_id, function=perform_run, count=1, project="gymcts-oekolopoly", entity="querry")

