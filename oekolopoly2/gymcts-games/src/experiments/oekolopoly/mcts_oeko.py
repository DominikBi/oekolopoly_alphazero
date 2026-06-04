import pprint
from collections import deque
from typing import SupportsFloat, Any

import wandb as wb
from gymcts.logger import log

from gymcts.gymcts_agent import GymctsAgent
from gymcts.gymcts_deepcopy_wrapper import DeepCopyMCTSGymEnvWrapper
from gymnasium.core import WrapperActType, WrapperObsType

from oekolopoly.env.oeko_env import OekoEnv, OekoActionBuilderWrapper
from oekolopoly.env.oeko_wrappers import OekoAuxRewardWrapper
experiment_sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'eval_accumulated_reward',
        'goal': 'maximize'
    },
    'parameters': {
        "approach": {
            'values': ["MCTS"]
        },
        "mcts_simulation_budget_per_step": {
            'values': [50, 100, 150, 1000]
        },

        "score_variate": {
            'values': ["UCT_v0"]
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

        "run_no": {
            'values': [1, 2, 3, 4, 5]
        }
    }
}

class ExperimentDeepCopyMCTSGymEnvWrapper(DeepCopyMCTSGymEnvWrapper):
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

    def step(
            self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if action == 0:
            log_dict ={
                "round": self.env.unwrapped.V[self.env.unwrapped.ROUND],
                "unwrapped_sanitation": self.env.unwrapped.V[self.env.unwrapped.SANITATION],
                "unwrapped_production": self.env.unwrapped.V[self.env.unwrapped.PRODUCTION],
                "unwrapped_education": self.env.unwrapped.V[self.env.unwrapped.EDUCATION],
                "unwrapped_quality_of_life": self.env.unwrapped.V[self.env.unwrapped.QUALITY_OF_LIFE],
                "unwrapped_population_growth": self.env.unwrapped.V[self.env.unwrapped.POPULATION_GROWTH],
                "unwrapped_environment": self.env.unwrapped.V[self.env.unwrapped.ENVIRONMENT],
                "unwrapped_population": self.env.unwrapped.V[self.env.unwrapped.POPULATION],
                "unwrapped_politics": self.env.unwrapped.V[self.env.unwrapped.POLITICS],
            }
            wb.log(log_dict)
        return super().step(action)


    def __init__(self, env, action_mask_fn):
        super().__init__(env)
        self._action_mask_fn = action_mask_fn

    def rollout(self) -> float:
        res = super().rollout()
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

class ExperimentMctsAgent(GymctsAgent):

    _mcts_step = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def perform_mcts_step(self, *args, **kwargs):
        res = super().perform_mcts_step(*args, **kwargs)
        ExperimentMctsAgent._mcts_step += 1
        # log mean values of the buffers if they are not empty
        buffers_kv = [
            ("final_balance", ExperimentDeepCopyMCTSGymEnvWrapper.final_balance_buffer),
            ("balance_always", ExperimentDeepCopyMCTSGymEnvWrapper.balance_always_buffer),
            ("balance_numerator", ExperimentDeepCopyMCTSGymEnvWrapper.balance_numerator_buffer),
            ("round", ExperimentDeepCopyMCTSGymEnvWrapper.round_buffer),
            ("sanitation", ExperimentDeepCopyMCTSGymEnvWrapper.sanitation_buffer),
            ("production", ExperimentDeepCopyMCTSGymEnvWrapper.production_buffer),
            ("education", ExperimentDeepCopyMCTSGymEnvWrapper.education_buffer),
            ("quality_of_life", ExperimentDeepCopyMCTSGymEnvWrapper.quality_of_life_buffer),
            ("population_growth", ExperimentDeepCopyMCTSGymEnvWrapper.population_growth_buffer),
            ("environment", ExperimentDeepCopyMCTSGymEnvWrapper.environment_buffer),
            ("population", ExperimentDeepCopyMCTSGymEnvWrapper.population_buffer),
            ("politics", ExperimentDeepCopyMCTSGymEnvWrapper.politics_buffer),
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
        env = OekoAuxRewardWrapper(env, scaling=0.1)
        env = OekoActionBuilderWrapper(env, auxilary_reward=False)
        obs, _ = env.reset()

        def action_mask_fn(env: OekoActionBuilderWrapper):
            return env.env.valid_action_mask()

        env = ExperimentDeepCopyMCTSGymEnvWrapper(env, action_mask_fn=action_mask_fn)

        env.reset()

        mcts_agent_kwargs = {
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

        # 2. create the agent
        agent = ExperimentMctsAgent(
            env=env,
            **mcts_agent_kwargs
        )

        # 3. solve the environment
        actions = agent.solve()

        obs, info = env.reset()
        done, truncated = False, False
        accumulated_reward = 0
        _idx = 0
        for action in actions:
            obs, reward, done, truncated, info = env.step(action)
            accumulated_reward += reward

        env.render()

        # log final evaluation metrics
        eval_logs = {
            "eval_final_balance": env.unwrapped.balance,
            "eval_balance_always": env.unwrapped.balance_always,
            "eval_balance_numerator": env.unwrapped.balance_numerator,

            "eval_round": env.unwrapped.ROUND,
            "eval_sanitation": env.unwrapped.V[env.unwrapped.SANITATION],
            "eval_production": env.unwrapped.V[env.unwrapped.PRODUCTION],
            "eval_education": env.unwrapped.V[env.unwrapped.EDUCATION],
            "eval_quality_of_life": env.unwrapped.V[env.unwrapped.QUALITY_OF_LIFE],
            "eval_population_growth": env.unwrapped.V[env.unwrapped.POPULATION_GROWTH],
            "eval_environment": env.unwrapped.V[env.unwrapped.ENVIRONMENT],
            "eval_population": env.unwrapped.V[env.unwrapped.POPULATION],
            "eval_politics": env.unwrapped.V[env.unwrapped.POLITICS],

            "eval_accumulated_reward": accumulated_reward,

            "eval_done_reason": info['done_reason'],
        }
        wb.log(eval_logs)
        wb.log({"eval_action_history": actions})
        log.info(f"Eval results: \n{pprint.pformat(eval_logs)}")
        log.info(f"Final balance: {env.unwrapped.balance}, Reason: {info['done_reason']}. ")


if __name__ == '__main__':
    # sweep_id = wb.sweep(experiment_sweep_config, project="gymcts-oekolopoly", entity="querry")
    # print(sweep_id)
    sweep_id = "fey6b24u"
    wb.agent(sweep_id, function=perform_run, count=1, project="gymcts-oekolopoly", entity="querry")
