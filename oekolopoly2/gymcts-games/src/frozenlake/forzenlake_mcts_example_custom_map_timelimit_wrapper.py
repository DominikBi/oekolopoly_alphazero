import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from gymcts.gymcts_agent import GymctsAgent
from gymcts.gymcts_deepcopy_wrapper import DeepCopyMCTSGymEnvWrapper
from gymcts.logger import log

log.setLevel(20)

if __name__ == '__main__':
    custom_map = [
        "SFFFFF",
        "FHFFHF",
        "FFFHFF",
        "HFHFHF",
        "FFFHFF",
        "FHFHFG"
    ]

    env = gym.make('FrozenLake-v1', desc=custom_map, is_slippery=False, render_mode="ansi")
    env = TimeLimit(env, max_episode_steps=50)
    env.reset()

    env = DeepCopyMCTSGymEnvWrapper(env)

    agent = GymctsAgent(
        env=env,
        clear_mcts_tree_after_step=False,
        render_tree_after_step=True,
        number_of_simulations_per_step=50,
        exclude_unvisited_nodes_from_render=True
    )

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
