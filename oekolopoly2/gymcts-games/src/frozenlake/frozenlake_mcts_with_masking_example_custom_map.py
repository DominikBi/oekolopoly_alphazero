import gymnasium as gym
import numpy as np

from gymcts.gymcts_agent import GymctsAgent
from gymcts.gymcts_deepcopy_wrapper import DeepCopyMCTSGymEnvWrapper

from gymcts.logger import log

# set log level to 20 (INFO)
# set log level to 10 (DEBUG) to see more detailed information
log.setLevel(20)

if __name__ == '__main__':
    # 0. create the environment
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

    env = gym.make('FrozenLake-v1', desc=custom_map, is_slippery=True, render_mode="ansi")
    env.reset()

    # 1. wrap the environment with the deep copy wrapper or a custom gymcts wrapper
    env = DeepCopyMCTSGymEnvWrapper(env, action_mask_fn=action_mask_fn)

    # 2. create the agent
    agent = GymctsAgent(
        env=env,
        clear_mcts_tree_after_step=True,
        render_tree_after_step=True,
        number_of_simulations_per_step=150,
        exclude_unvisited_nodes_from_render=True
    )

    # 3. solve the environment
    actions = agent.solve()

    # 4. render the environment solution in the terminal
    print(env.render())
    for a in actions:
        obs, rew, term, trun, info = env.step(a)
        print(env.render())

    # 5. print the solution
    # read the solution from the info provided by the RecordEpisodeStatistics wrapper
    # (that DeepCopyMCTSGymEnvWrapper uses internally)
    episode_length = info["episode"]["l"]
    episode_return = info["episode"]["r"]

    if episode_return == 1.0:
        print(f"Environment solved in {episode_length} steps.")
    else:
        print(f"Environment not solved in {episode_length} steps.")