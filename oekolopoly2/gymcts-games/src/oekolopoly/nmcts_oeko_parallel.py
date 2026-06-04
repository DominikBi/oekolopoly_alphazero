import argparse
import copy
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from gymcts.gymcts_neural_agent import GymctsNeuralAgent
from sb3_contrib import MaskablePPO

from oekolopoly.nmcts_oeko import (
    NeuralRolloutOekoWrapper,
    evaluate_policy_only,
    load_policy,
    make_nmcts_env,
    print_result,
    summarize_episode,
    train_agent,
)

DEFAULT_MODEL_PATH = Path("models_nmcts_parallel") / "neural_mcts_oekolopoly_policy"
_PROCESS_MODEL = None


def init_process_worker(model_path):
    global _PROCESS_MODEL
    _PROCESS_MODEL = MaskablePPO.load(model_path)
    NeuralRolloutOekoWrapper.model = _PROCESS_MODEL


def rollout_from_env(env):
    return env.rollout()


class ParallelRolloutNeuralMctsAgent(GymctsNeuralAgent):
    def __init__(
        self,
        *args,
        parallel_workers=12,
        parallel_batch_size=12,
        parallel_backend="process",
        parallel_model_path=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.parallel_workers = max(1, parallel_workers)
        self.parallel_batch_size = parallel_batch_size or self.parallel_workers
        self.parallel_backend = parallel_backend
        self.parallel_model_path = parallel_model_path
        self._parallel_executor = None

    def vanilla_mcts_search(self, search_start_node=None, num_simulations=10) -> int:
        if search_start_node is None:
            search_start_node = self.search_root_node

        if self.parallel_workers <= 1 or num_simulations <= 1:
            return super().vanilla_mcts_search(
                search_start_node=search_start_node,
                num_simulations=num_simulations,
            )

        completed_simulations = 0
        executor = self._get_parallel_executor()
        while completed_simulations < num_simulations:
            batch_size = min(
                self.parallel_batch_size,
                num_simulations - completed_simulations,
            )
            leaf_nodes = []
            futures = []

            for _ in range(batch_size):
                leaf_node = self.navigate_to_leaf(from_node=search_start_node)

                if leaf_node.visit_count > 0 and not leaf_node.terminal:
                    self.expand_node(leaf_node)
                    leaf_node = leaf_node.get_random_child()

                self._load_state(leaf_node)
                rollout_env = copy.deepcopy(self.env)
                leaf_nodes.append(leaf_node)
                futures.append(executor.submit(rollout_from_env, rollout_env))

            for leaf_node, future in zip(leaf_nodes, futures):
                episode_return = future.result()
                self.backpropagation(
                    node=leaf_node,
                    episode_return=episode_return,
                )

            completed_simulations += batch_size

        if self.render_tree_after_step:
            self.show_mcts_tree()

        return search_start_node.get_best_action()

    def _get_parallel_executor(self):
        if self._parallel_executor is not None:
            return self._parallel_executor

        if self.parallel_backend == "thread":
            self._parallel_executor = ThreadPoolExecutor(
                max_workers=self.parallel_workers
            )
            return self._parallel_executor

        if self.parallel_model_path is None:
            raise ValueError("parallel_model_path must be set for process rollouts.")
        self._parallel_executor = ProcessPoolExecutor(
            max_workers=self.parallel_workers,
            initializer=init_process_worker,
            initargs=(self.parallel_model_path,),
        )
        return self._parallel_executor

    def shutdown_parallel_executor(self):
        if self._parallel_executor is not None:
            self._parallel_executor.shutdown(wait=True)
            self._parallel_executor = None


def build_parallel_agent(args, env):
    agent = ParallelRolloutNeuralMctsAgent(
        env=env,
        model_kwargs={
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "ent_coef": args.ent_coef,
            "vf_coef": args.vf_coef,
            "policy_kwargs": dict(net_arch=[args.net_arch, args.net_arch]),
            "normalize_advantage": True,
            "verbose": 1,
            "seed": args.seed,
        },
        clear_mcts_tree_after_step=True,
        render_tree_after_step=args.render_tree,
        render_tree_max_depth=args.render_tree_max_depth,
        number_of_simulations_per_step=args.simulations,
        exclude_unvisited_nodes_from_render=True,
        keep_whole_tree_till_initial_root=False,
        score_variate=args.score_variate,
        best_action_weight=args.best_action_weight,
        parallel_workers=args.parallel_workers,
        parallel_batch_size=args.parallel_batch_size,
        parallel_backend=args.parallel_backend,
        parallel_model_path=args.model_path,
    )
    NeuralRolloutOekoWrapper.model = agent._model
    return agent


def evaluate_with_search_fresh(agent, args):
    agent.reset()
    actions = agent.solve(num_simulations_per_step=args.simulations)

    replay_env = make_nmcts_env(
        render_mode="ansi" if args.render else None,
        reward_start_round=args.reward_start_round,
        reward_weight_power=args.reward_weight_power,
        survival_reward=args.survival_reward,
    )
    replay_env.reset()

    accumulated_reward = 0.0
    info = {}
    for action in actions:
        _obs, reward, terminated, truncated, info = replay_env.step(action)
        accumulated_reward += float(reward)
        if terminated or truncated:
            break

    return summarize_episode("nmcts", replay_env, info, accumulated_reward, actions)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Neural MCTS agent with parallel rollouts for Oekolopoly."
    )
    parser.add_argument("--timesteps", type=int, default=800_000)
    parser.add_argument("--simulations", type=int, default=500)
    parser.add_argument("--parallel-workers", type=int, default=12)
    parser.add_argument("--parallel-batch-size", type=int, default=12)
    parser.add_argument(
        "--parallel-backend", choices=["process", "thread"], default="process"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-tree", action="store_true")
    parser.add_argument("--render-tree-max-depth", type=int, default=2)

    parser.add_argument("--reward-start-round", type=int, default=10)
    parser.add_argument("--reward-weight-power", type=float, default=2.0)
    parser.add_argument("--survival-reward", type=float, default=1)

    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--net-arch", type=int, default=64)
    parser.add_argument("--score-variate", type=str, default="MuZero_v1")
    parser.add_argument("--best-action-weight", type=float, default=0.95)
    return parser.parse_args()


def main():
    start_time = time.perf_counter()
    args = parse_args()
    args.model_path = args.model_path.resolve()
    env = make_nmcts_env(
        render_mode="ansi" if args.render else None,
        reward_start_round=args.reward_start_round,
        reward_weight_power=args.reward_weight_power,
        survival_reward=args.survival_reward,
    )
    agent = build_parallel_agent(args, env)

    if args.eval_only:
        load_policy(args, agent)
    else:
        train_agent(args, agent)

    print_result(evaluate_policy_only(agent, env))
    try:
        print_result(evaluate_with_search_fresh(agent, args))
    finally:
        agent.shutdown_parallel_executor()

    if args.render:
        env.render()

    elapsed_seconds = time.perf_counter() - start_time
    print(f"Total runtime: {elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    main()
