import copy
import pprint
import random
from itertools import accumulate
from typing import SupportsFloat, Any

import numpy as np
from collections import deque

import wandb as wb
import gymnasium as gym
from gymcts.gymcts_neural_agent import GymctsNeuralAgent
from gymcts.logger import log

from gymcts.gymcts_agent import GymctsAgent
from gymcts.gymcts_deepcopy_wrapper import DeepCopyMCTSGymEnvWrapper
from gymnasium.core import WrapperObsType, WrapperActType
from wandb.integration.sb3 import WandbCallback

from oekolopoly.env.oeko_env import OekoEnv, OekoActionBuilderWrapper
from oekolopoly.env.oeko_wrappers import OekoAuxRewardWrapper
experiment_sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'eval_150_accumulated_reward',
        'goal': 'maximize'
    },
    'parameters': {
        "approach": {
            'values': ["NMCTS"]
        },

        "score_variate": {
            'values': ["MuZero_v1"]
        },

        "clear_mcts_tree_after_step":{
            'values': [True]
        },

        "keep_whole_tree_till_initial_root":{
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
            'values': [0.95]
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
class ExperimentDeepCopyMCTSGymEnvWrapper(DeepCopyMCTSGymEnvWrapper):
    wrapper_logging_enabled = False

    final_balance_buffer = deque(maxlen=100)
    balance_always_buffer = deque(maxlen=100)
    balance_numerator_buffer = deque(maxlen=100)

    round_buffer = deque(maxlen=100)
    sanitation_buffer = deque(maxlen=100)
    production_buffer = deque(maxlen=100)
    education_buffer = deque(maxlen=100)
    quality_of_life_buffer = deque(maxlen=100)
    population_growth_buffer = deque(maxlen=100)
    environment_buffer = deque(maxlen=100)
    population_buffer = deque(maxlen=100)
    politics_buffer = deque(maxlen=100)

    return_buffer = deque(maxlen=100)

    model = None


    def __init__(self, env, action_mask_fn, model=None):
        super().__init__(env)
        self._action_mask_fn = action_mask_fn
        ExperimentDeepCopyMCTSGymEnvWrapper.model = model

    def step(
            self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if action == 0:
            log_dict ={
                f"round": self.env.unwrapped.V[self.env.unwrapped.ROUND],
                f"{ExperimentMctsAgent._experiment_prefix}unwrapped_sanitation": self.env.unwrapped.V[self.env.unwrapped.SANITATION],
                f"{ExperimentMctsAgent._experiment_prefix}unwrapped_production": self.env.unwrapped.V[self.env.unwrapped.PRODUCTION],
                f"{ExperimentMctsAgent._experiment_prefix}unwrapped_education": self.env.unwrapped.V[self.env.unwrapped.EDUCATION],
                f"{ExperimentMctsAgent._experiment_prefix}unwrapped_quality_of_life": self.env.unwrapped.V[self.env.unwrapped.QUALITY_OF_LIFE],
                f"{ExperimentMctsAgent._experiment_prefix}unwrapped_population_growth": self.env.unwrapped.V[self.env.unwrapped.POPULATION_GROWTH],
                f"{ExperimentMctsAgent._experiment_prefix}unwrapped_environment": self.env.unwrapped.V[self.env.unwrapped.ENVIRONMENT],
                f"{ExperimentMctsAgent._experiment_prefix}unwrapped_population": self.env.unwrapped.V[self.env.unwrapped.POPULATION],
                f"{ExperimentMctsAgent._experiment_prefix}unwrapped_politics": self.env.unwrapped.V[self.env.unwrapped.POLITICS],
            }
            wb.log(log_dict)
        return super().step(action)

    def rollout(self) -> float:
        if ExperimentDeepCopyMCTSGymEnvWrapper.model is None:
            res = super().rollout()
        else:
            model = ExperimentDeepCopyMCTSGymEnvWrapper.model
            if self._step_tuple:
                obs, reward, done, truncated, info = copy.deepcopy(self._step_tuple)
            else:
                done, truncated = False, False
                obs = None
            accumulated_reward = 0.0
            while not (done or truncated):
                if obs is None:
                    action = random.choice(self.get_valid_actions())
                    obs, reward, done, truncated, info = self.env.step(action)
                    continue

                action, _ = model.predict(obs, deterministic=True, action_masks=self._action_mask_fn(self.env))
                if isinstance(action, np.ndarray):
                    action = action.item()
                obs, reward, done, truncated, info = self.env.step(action)
                accumulated_reward += reward

            ExperimentDeepCopyMCTSGymEnvWrapper.return_buffer.append(accumulated_reward)
            res = accumulated_reward


        env = self.env.unwrapped
        # append to buffers
        ExperimentDeepCopyMCTSGymEnvWrapper.final_balance_buffer.append(env.balance)
        ExperimentDeepCopyMCTSGymEnvWrapper.balance_always_buffer.append(env.balance_always)
        ExperimentDeepCopyMCTSGymEnvWrapper.balance_numerator_buffer.append(env.balance_numerator)

        ExperimentDeepCopyMCTSGymEnvWrapper.round_buffer.append(env.unwrapped.V[env.unwrapped.ROUND])

        ExperimentDeepCopyMCTSGymEnvWrapper.sanitation_buffer.append(env.V[env.SANITATION])
        ExperimentDeepCopyMCTSGymEnvWrapper.production_buffer.append(env.V[env.PRODUCTION])
        ExperimentDeepCopyMCTSGymEnvWrapper.education_buffer.append(env.V[env.EDUCATION])
        ExperimentDeepCopyMCTSGymEnvWrapper.quality_of_life_buffer.append(env.V[env.QUALITY_OF_LIFE])
        ExperimentDeepCopyMCTSGymEnvWrapper.population_growth_buffer.append(env.V[env.POPULATION_GROWTH])
        ExperimentDeepCopyMCTSGymEnvWrapper.environment_buffer.append(env.V[env.ENVIRONMENT])
        ExperimentDeepCopyMCTSGymEnvWrapper.population_buffer.append(env.V[env.POPULATION])
        ExperimentDeepCopyMCTSGymEnvWrapper.politics_buffer.append(env.V[env.POLITICS])

        return res

class ExperimentMctsAgent(GymctsNeuralAgent):

    _mcts_step = 0
    _experiment_prefix = ""

    def reset(self) -> None:
        super().reset()
        ExperimentMctsAgent._mcts_step = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def perform_mcts_step(self, *args, **kwargs):
        res = super().perform_mcts_step(*args, **kwargs)
        ExperimentMctsAgent._mcts_step += 1
        # log mean values of the buffers if they are not empty
        buffers_kv = [
            (f"{ExperimentMctsAgent._experiment_prefix}final_balance", ExperimentDeepCopyMCTSGymEnvWrapper.final_balance_buffer),
            (f"{ExperimentMctsAgent._experiment_prefix}balance_always", ExperimentDeepCopyMCTSGymEnvWrapper.balance_always_buffer),
            (f"{ExperimentMctsAgent._experiment_prefix}balance_numerator", ExperimentDeepCopyMCTSGymEnvWrapper.balance_numerator_buffer),
            (f"{ExperimentMctsAgent._experiment_prefix}round", ExperimentDeepCopyMCTSGymEnvWrapper.round_buffer),
            (f"{ExperimentMctsAgent._experiment_prefix}sanitation", ExperimentDeepCopyMCTSGymEnvWrapper.sanitation_buffer),
            (f"{ExperimentMctsAgent._experiment_prefix}production", ExperimentDeepCopyMCTSGymEnvWrapper.production_buffer),
            (f"{ExperimentMctsAgent._experiment_prefix}education", ExperimentDeepCopyMCTSGymEnvWrapper.education_buffer),
            (f"{ExperimentMctsAgent._experiment_prefix}quality_of_life", ExperimentDeepCopyMCTSGymEnvWrapper.quality_of_life_buffer),
            (f"{ExperimentMctsAgent._experiment_prefix}population_growth", ExperimentDeepCopyMCTSGymEnvWrapper.population_growth_buffer),
            (f"{ExperimentMctsAgent._experiment_prefix}environment", ExperimentDeepCopyMCTSGymEnvWrapper.environment_buffer),
            (f"{ExperimentMctsAgent._experiment_prefix}population", ExperimentDeepCopyMCTSGymEnvWrapper.population_buffer),
            (f"{ExperimentMctsAgent._experiment_prefix}politics", ExperimentDeepCopyMCTSGymEnvWrapper.politics_buffer),
        ]
        log_dict = {
            "mcts_step": ExperimentMctsAgent._mcts_step
        }
        for (key, buf) in buffers_kv:
            if len(buf) > 0:
                # example ExperimentDeepCopyMCTSGymEnvWrapper.politics_buffer -> buf_name = politics
                log_dict[key] = sum(buf) / len(buf) if len(buf) > 0 else 0
        log.info(f"MCTS step {ExperimentMctsAgent._mcts_step}: logging mean values of the buffers: {pprint.pformat(log_dict)}")
        wb.log(log_dict)
        return res

class LoggingWrapper(gym.Wrapper):

    logging_enabled = True
    num_timestep = 0

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # log.info(f"Step info: \n{pprint.pformat(log_dict)}")
        if terminated and LoggingWrapper.logging_enabled:
            LoggingWrapper.num_timestep += 1  # increment by number of environments if vectorized
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

        env = OekoEnv(render_mode="ansi")
        env = LoggingWrapper(env)
        env = OekoAuxRewardWrapper(env, scaling=0.1)
        env = OekoActionBuilderWrapper(env, auxilary_reward=False)
        obs, _ = env.reset()

        def action_mask_fn(env: OekoActionBuilderWrapper):
            return env.env.valid_action_mask()

        def action_mask_fn2(env: OekoActionBuilderWrapper):
            return env.env.env.valid_action_mask()

        env = ExperimentDeepCopyMCTSGymEnvWrapper(env, action_mask_fn=action_mask_fn)

        env.reset()

        nmcts_agent_kwargs = {
            "clear_mcts_tree_after_step": experiment_params["clear_mcts_tree_after_step"],
            "render_tree_after_step": experiment_params["render_tree_after_step"],
            "render_tree_max_depth": experiment_params["render_tree_max_depth"],
            "exclude_unvisited_nodes_from_render": experiment_params["exclude_unvisited_nodes_from_render"],
            "keep_whole_tree_till_initial_root": experiment_params["keep_whole_tree_till_initial_root"],
            "calc_number_of_simulations_per_step": experiment_params["calc_number_of_simulations_per_step"],
            "score_variate": experiment_params["score_variate"],
            "best_action_weight": experiment_params["best_action_weight"],
        }

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
            "seed": experiment_params["run_no"]  # using run_no as seed for reproducibility
        }


        agent = ExperimentMctsAgent(
            env=env,
            model_kwargs=masked_ppo_model_kwargs,
            **nmcts_agent_kwargs
        )
        ExperimentDeepCopyMCTSGymEnvWrapper.model = agent._model

        num_timesteps = experiment_params["num_timesteps"]

        LoggingWrapper.logging_enabled = True
        agent.learn(total_timesteps=num_timesteps,callback=WandbCallback(verbose=2))
        LoggingWrapper.logging_enabled = False

        # evaluate model performance without neural guidance
        log.info("##### Evaluating the learned policy without tree search #####")
        obs, info = env.reset()
        model = agent._model
        done, truncated = False, False
        action_history = []
        accumulated_reward = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True, action_masks=action_mask_fn2(env))
            if isinstance(action, np.ndarray):
                action = action.item()
            obs, reward, done, truncated, info = env.step(action)
            action_history.append(action)
            accumulated_reward += reward
        env.render()
        log.info(f"[no search] accumulated_reward: {accumulated_reward:.2f}, final_balance: {env.unwrapped.balance:.2f}, done_reason: {info['done_reason']}")

        eval_logs = {
            "eval_0_final_balance": env.unwrapped.balance,
            "eval_0_balance_always": env.unwrapped.balance_always,
            "eval_0_balance_numerator": env.unwrapped.balance_numerator,

            "eval_0_round": env.unwrapped.V[env.unwrapped.ROUND],

            "eval_0_sanitation": env.unwrapped.V[env.unwrapped.SANITATION],
            "eval_0_production": env.unwrapped.V[env.unwrapped.PRODUCTION],
            "eval_0_education": env.unwrapped.V[env.unwrapped.EDUCATION],
            "eval_0_quality_of_life": env.unwrapped.V[env.unwrapped.QUALITY_OF_LIFE],
            "eval_0_population_growth": env.unwrapped.V[env.unwrapped.POPULATION_GROWTH],
            "eval_0_environment": env.unwrapped.V[env.unwrapped.ENVIRONMENT],
            "eval_0_population": env.unwrapped.V[env.unwrapped.POPULATION],
            "eval_0_politics": env.unwrapped.V[env.unwrapped.POLITICS],

            "eval_0_done_reason": info['done_reason'],

            "eval_0_accumulated_reward": accumulated_reward,
        }
        wb.log(eval_logs)
        wb.log({"eval_0_action_history": action_history})
        log.info(f"[no search] Eval results: \n{pprint.pformat(eval_logs)}")
        log.info(f"[no search] Final balance: {env.unwrapped.balance}, Reason: {info['done_reason']}. ")



        # evaluate the performance of the learned policy with different number of simulations per step
        log.info("##### Evaluating the learned policy without tree search with 30 simulations per step #####")
        ExperimentMctsAgent._experiment_prefix = "exp_50_"
        actions_30 = agent.solve(num_simulations_per_step=50)
        env.reset()
        accumulated_reward = 0.0
        action_history = []
        for act in actions_30:
            obs, reward, done, truncated, info = env.step(act)
            accumulated_reward += reward
            action_history.append(act)
            if truncated:
                log.warning(f"Episode ended ended unexpectedly when performing the actions: {actions_30}")
                break
        env.render()
        log.info(f"[search 30] accumulated_reward: {accumulated_reward:.2f}, final_balance: {env.unwrapped.balance:.2f}, done_reason: {info['done_reason']}")

        eval_logs = {
            "eval_30_final_balance": env.unwrapped.balance,
            "eval_30_balance_always": env.unwrapped.balance_always,
            "eval_30_balance_numerator": env.unwrapped.balance_numerator,

            "eval_30_round": env.unwrapped.V[env.unwrapped.ROUND],

            "eval_30_sanitation": env.unwrapped.V[env.unwrapped.SANITATION],
            "eval_30_production": env.unwrapped.V[env.unwrapped.PRODUCTION],
            "eval_30_education": env.unwrapped.V[env.unwrapped.EDUCATION],
            "eval_30_quality_of_life": env.unwrapped.V[env.unwrapped.QUALITY_OF_LIFE],
            "eval_30_population_growth": env.unwrapped.V[env.unwrapped.POPULATION_GROWTH],
            "eval_30_environment": env.unwrapped.V[env.unwrapped.ENVIRONMENT],
            "eval_30_population": env.unwrapped.V[env.unwrapped.POPULATION],
            "eval_30_politics": env.unwrapped.V[env.unwrapped.POLITICS],

            "eval_30_done_reason": info['done_reason'],

            "eval_30_accumulated_reward": accumulated_reward,
        }
        wb.log(eval_logs)
        wb.log({"eval_30_action_history": actions_30})
        log.info(f"[search 30] Eval results: \n{pprint.pformat(eval_logs)}")
        log.info(f"[search 30] Final balance: {env.unwrapped.balance}, Reason: {info['done_reason']}. ")

        env.reset()
        # perform search with 100 simulations per step
        log.info("##### Evaluating the learned policy without tree search with 100 simulations per step #####")
        agent.reset()
        ExperimentMctsAgent._experiment_prefix = "exp_100_"
        actions_100 = agent.solve(num_simulations_per_step=100)

        env.reset()
        accumulated_reward = 0.0
        for act in actions_100:
            obs, reward, done, truncated, info = env.step(act)
            accumulated_reward += reward
            if truncated:
                log.warning(f"Episode ended ended unexpectedly when performing the actions: {actions_100}")
                break
        env.render()
        log.info(f"[search 100] accumulated_reward: {accumulated_reward:.2f}, final_balance: {env.unwrapped.balance:.2f}, done_reason: {info['done_reason']}")
        log.debug(f"Final info: \n{pprint.pformat(info)}")

        eval_logs = {
            "eval_100_final_balance": env.unwrapped.balance,
            "eval_100_balance_always": env.unwrapped.balance_always,
            "eval_100_balance_numerator": env.unwrapped.balance_numerator,

            "eval_100_round": env.unwrapped.V[env.unwrapped.ROUND],

            "eval_100_sanitation": env.unwrapped.V[env.unwrapped.SANITATION],
            "eval_100_production": env.unwrapped.V[env.unwrapped.PRODUCTION],
            "eval_100_education": env.unwrapped.V[env.unwrapped.EDUCATION],
            "eval_100_quality_of_life": env.unwrapped.V[env.unwrapped.QUALITY_OF_LIFE],
            "eval_100_population_growth": env.unwrapped.V[env.unwrapped.POPULATION_GROWTH],
            "eval_100_environment": env.unwrapped.V[env.unwrapped.ENVIRONMENT],
            "eval_100_population": env.unwrapped.V[env.unwrapped.POPULATION],
            "eval_100_politics": env.unwrapped.V[env.unwrapped.POLITICS],

            "eval_100_done_reason": info['done_reason'],

            "eval_100_accumulated_reward": accumulated_reward,
        }

        wb.log(eval_logs)
        wb.log({"eval_100_action_history": actions_100})
        log.info(f"[search 100] Eval results: \n{pprint.pformat(eval_logs)}")
        log.info(f"[search 100] Final balance: {env.unwrapped.balance}, Reason: {info['done_reason']}. ")


        # perform search with 150 simulations per step
        log.info("##### Evaluating the learned policy without tree search with 150 simulations per step #####")
        agent.reset()
        ExperimentMctsAgent._experiment_prefix = "exp_150_"
        actions_150 = agent.solve(num_simulations_per_step=150)

        env.reset()
        accumulated_reward = 0.0
        for act in actions_150:
            obs, reward, done, truncated, info = env.step(act)
            accumulated_reward += reward
            if truncated:
                log.warning(f"Episode ended ended unexpectedly when performing the actions: {actions_150}")
                break

        env.render()
        log.info(f"[search 150] accumulated_reward: {accumulated_reward:.2f}, final_balance: {env.unwrapped.balance:.2f}, done_reason: {info['done_reason']}")
        log.debug(f"Final info: \n{pprint.pformat(info)}")

        eval_logs = {
            "eval_150_final_balance": env.unwrapped.balance,
            "eval_150_balance_always": env.unwrapped.balance_always,
            "eval_150_balance_numerator": env.unwrapped.balance_numerator,

            "eval_150_round": env.unwrapped.V[env.unwrapped.ROUND],

            "eval_150_sanitation": env.unwrapped.V[env.unwrapped.SANITATION],
            "eval_150_production": env.unwrapped.V[env.unwrapped.PRODUCTION],
            "eval_150_education": env.unwrapped.V[env.unwrapped.EDUCATION],
            "eval_150_quality_of_life": env.unwrapped.V[env.unwrapped.QUALITY_OF_LIFE],
            "eval_150_population_growth": env.unwrapped.V[env.unwrapped.POPULATION_GROWTH],
            "eval_150_environment": env.unwrapped.V[env.unwrapped.ENVIRONMENT],
            "eval_150_population": env.unwrapped.V[env.unwrapped.POPULATION],
            "eval_150_politics": env.unwrapped.V[env.unwrapped.POLITICS],

            "eval_150_done_reason": info['done_reason'],

            "eval_150_accumulated_reward": accumulated_reward,
        }
        wb.log(eval_logs)
        wb.log({"eval_150_action_history": actions_150})
        log.info(f"[search 150] Eval results: \n{pprint.pformat(eval_logs)}")
        log.info(f"[search 150] Final balance: {env.unwrapped.balance}, Reason: {info['done_reason']}. ")

        log.info(f"eval 0 actions (len: {len(action_history)}): {action_history}")
        log.info(f"eval 50 actions (len: {len(actions_30)}): {actions_30}")
        log.info(f"eval 100 actions (len: {len(actions_100)}): {actions_100}")
        log.info(f"eval 150 actions (len: {len(actions_150)}): {actions_150}")
        
        


if __name__ == '__main__':
    #sweep_id = wb.sweep(experiment_sweep_config, project="gymcts-oekolopoly", entity="querry")
    #print(sweep_id)
    sweep_id = "0xub6wkd"
    wb.agent(sweep_id, function=perform_run, count=1, project="gymcts-oekolopoly", entity="querry")
