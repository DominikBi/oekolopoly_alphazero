from gymcts.gymcts_agent import GymctsAgent
from gymcts.gymcts_action_history_wrapper import ActionHistoryMCTSGymEnvWrapper

from oekolopoly.env.oeko_env import OekoEnv, OekoActionBuilderWrapper

if __name__ == '__main__':
    env = OekoEnv(render_mode="ansi")
    # env = OekoPerRoundRewardWrapper(env)
    # env = OekoAuxRewardWrapper(env)
    # env = NormalizeReward(env, gamma=0.99, epsilon=1e-8)
    env = OekoActionBuilderWrapper(env, auxilary_reward=True)
    obs, _ = env.reset()

    def action_mask_fn(env: OekoActionBuilderWrapper):
        return env.env.valid_action_mask()

    #env = DeepCopyMCTSGymEnvWrapper(env, action_mask_fn=action_mask_fn)
    env = ActionHistoryMCTSGymEnvWrapper(env, action_mask_fn=action_mask_fn)

    env.reset()

    # 2. create the agent
    agent = GymctsAgent(
        env=env,
        clear_mcts_tree_after_step=True,
        render_tree_after_step=True,
        number_of_simulations_per_step=3_000,
        exclude_unvisited_nodes_from_render=True
    )

    # 3. solve the environment
    actions = agent.solve()

