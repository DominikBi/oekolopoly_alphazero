import pprint
import wandb as wb
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit, TransformReward

from gymcts.gymcts_agent import GymctsAgent
from gymcts.gymcts_deepcopy_wrapper import DeepCopyMCTSGymEnvWrapper
from gymcts.logger import log
from sb3_contrib.common.wrappers import ActionMasker

log.setLevel(20)

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
            'values': [0.0]
        },

        "time_limit_max_episode_steps": {
            'values': [50]
        },

        "run_no": {
            'values': [1, 2, 3]
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
        # small negative reward for each step to encourage shorter solutions
        # this is not that useful in FrozenLake, since the agent will aim to fall into a hole in case no solution is found.
        # env = TransformReward(env, lambda r: -0.01 if r == 0 else r)
        env = TimeLimit(env, max_episode_steps=50) # limit the episode length to prevent infinite loops and too long MCTS simulations
        env = ActionMasker(env=env, action_mask_fn=action_mask_fn)
        env.reset()

        env = DeepCopyMCTSGymEnvWrapper(env, action_mask_fn=action_mask_fn)

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

        agent = GymctsAgent(
            env=env,
            **mcts_agent_kwargs,
        )

        env.reset()

        actions = agent.solve()

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

        if episode_return == 1.0:
            log.info(f"Environment solved in {episode_length} steps.")
        else:
            log.warning(f"Environment not solved in {episode_length} steps.")


if __name__ == '__main__':
    sweep_id = wb.sweep(experiment_sweep_config, project="test")
    wb.agent(sweep_id, function=perform_run, count=1, project="test")
