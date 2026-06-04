import argparse
import copy
import random
import time
from pathlib import Path
from typing import Any, SupportsFloat

import numpy as np
from gymnasium.core import WrapperActType, WrapperObsType
from gymcts.gymcts_deepcopy_wrapper import DeepCopyMCTSGymEnvWrapper
from gymcts.gymcts_neural_agent import GymctsNeuralAgent

from oekolopoly.env.oeko_env import OekoActionBuilderWrapper, OekoEnv
from oekolopoly.env.oeko_wrappers import OekoRoundWeightedBalanceRewardWrapper

DEFAULT_MODEL_PATH = Path("models_nmcts") / "neural_mcts_oekolopoly_policy"


class NeuralRolloutOekoWrapper(DeepCopyMCTSGymEnvWrapper):
    model = None

    def __init__(self, env, action_mask_fn, model=None):
        super().__init__(env, action_mask_fn=action_mask_fn)
        NeuralRolloutOekoWrapper.model = model
        self._action_mask_fn = action_mask_fn

    def rollout(self) -> float:
        if NeuralRolloutOekoWrapper.model is None:
            return self._random_rollout()

        model = NeuralRolloutOekoWrapper.model
        if self._step_tuple:
            obs, _reward, terminated, truncated, _info = copy.deepcopy(self._step_tuple)
        else:
            obs = None
            terminated = False
            truncated = False

        accumulated_reward = 0.0
        while not (terminated or truncated):
            if obs is None:
                action = random.choice(self.get_valid_actions())
            else:
                action, _ = model.predict(
                    obs,
                    deterministic=False,
                    action_masks=self._action_mask_fn(self.env),
                )
                if isinstance(action, np.ndarray):
                    action = int(action.item())

            obs, reward, terminated, truncated, _info = self.env.step(action)
            accumulated_reward += float(reward)

        return accumulated_reward

    def _random_rollout(self) -> float:
        accumulated_reward = 0.0
        terminated = self.is_terminal()
        truncated = False

        while not (terminated or truncated):
            action = random.choice(self.get_valid_actions())
            _obs, reward, terminated, truncated, _info = self.step(action)
            accumulated_reward += float(reward)

        return accumulated_reward

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        return super().step(action)


def make_oeko_env(
    render_mode: str | None = None,
    reward_start_round: int = 10,
    reward_weight_power: float = 2.0,
    survival_reward: float = 0.25,
):
    env = OekoEnv(render_mode=render_mode)
    env = OekoRoundWeightedBalanceRewardWrapper(
        env,
        start_round=reward_start_round,
        weight_power=reward_weight_power,
        survival_reward=survival_reward,
    )
    return OekoActionBuilderWrapper(env, auxilary_reward=False)


def action_mask_fn(env: OekoActionBuilderWrapper):
    while not isinstance(env, OekoActionBuilderWrapper):
        env = env.env
    return env.valid_action_mask()


def make_nmcts_env(
    render_mode: str | None = None,
    reward_start_round: int = 10,
    reward_weight_power: float = 2.0,
    survival_reward: float = 0.25,
):
    env = make_oeko_env(
        render_mode=render_mode,
        reward_start_round=reward_start_round,
        reward_weight_power=reward_weight_power,
        survival_reward=survival_reward,
    )
    env = NeuralRolloutOekoWrapper(env, action_mask_fn=action_mask_fn)
    env.reset()
    return env


def build_agent(args, env):
    model_kwargs = {
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
    }

    agent = GymctsNeuralAgent(
        env=env,
        model_kwargs=model_kwargs,
        clear_mcts_tree_after_step=True,
        render_tree_after_step=args.render_tree,
        render_tree_max_depth=args.render_tree_max_depth,
        number_of_simulations_per_step=args.simulations,
        exclude_unvisited_nodes_from_render=True,
        keep_whole_tree_till_initial_root=False,
        score_variate=args.score_variate,
        best_action_weight=args.best_action_weight,
    )
    NeuralRolloutOekoWrapper.model = agent._model
    return agent


def train_agent(args, agent):
    agent.learn(total_timesteps=args.timesteps)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    agent._model.save(args.model_path)


def load_policy(args, agent):
    agent._model = agent._model.load(args.model_path, env=agent._model.env)
    NeuralRolloutOekoWrapper.model = agent._model


def evaluate_policy_only(agent, env):
    obs, _info = env.reset()
    terminated = False
    truncated = False
    actions = []
    accumulated_reward = 0.0

    while not (terminated or truncated):
        action, _ = agent._model.predict(
            obs,
            deterministic=True,
            action_masks=env.action_masks(),
        )
        if isinstance(action, np.ndarray):
            action = int(action.item())

        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        accumulated_reward += float(reward)

    return summarize_episode("policy", env, info, accumulated_reward, actions)


def evaluate_with_search(agent, env, simulations: int):
    env.reset()
    agent.reset()
    actions = agent.solve(num_simulations_per_step=simulations)

    env.reset()
    accumulated_reward = 0.0
    info = {}
    for action in actions:
        _obs, reward, terminated, truncated, info = env.step(action)
        accumulated_reward += float(reward)
        if terminated or truncated:
            break

    return summarize_episode("nmcts", env, info, accumulated_reward, actions)


def summarize_episode(label, env, info, accumulated_reward, actions):
    unwrapped = env.unwrapped
    return {
        "label": label,
        "round": int(unwrapped.V[unwrapped.ROUND]),
        "balance": float(unwrapped.balance),
        "balance_always": float(unwrapped.balance_always),
        "accumulated_reward": accumulated_reward,
        "done_reason": info.get("done_reason", ""),
        "actions": actions,
    }


def print_result(result):
    print(
        f"{result['label']}: "
        f"round={result['round']}, "
        f"balance={result['balance']:.3f}, "
        f"balance_always={result['balance_always']:.3f}, "
        f"reward={result['accumulated_reward']:.3f}, "
        f"done_reason={result['done_reason']}"
    )
    print(f"Actions: {result['actions']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Neural MCTS agent for Oekolopoly."
    )
    parser.add_argument("--timesteps", type=int, default=800_000)
    parser.add_argument("--simulations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-tree", action="store_true")
    parser.add_argument("--render-tree-max-depth", type=int, default=2)

    parser.add_argument("--reward-start-round", type=int, default=10)
    parser.add_argument("--reward-weight-power", type=float, default=2.0)
    parser.add_argument("--survival-reward", type=float, default=0.25)

    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=1.0)
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
    env = make_nmcts_env(
        render_mode="ansi" if args.render else None,
        reward_start_round=args.reward_start_round,
        reward_weight_power=args.reward_weight_power,
        survival_reward=args.survival_reward,
    )
    agent = build_agent(args, env)

    if args.eval_only:
        load_policy(args, agent)
    else:
        train_agent(args, agent)

    print_result(evaluate_policy_only(agent, env))
    print_result(evaluate_with_search(agent, env, simulations=args.simulations))

    if args.render:
        env.render()

    elapsed_seconds = time.perf_counter() - start_time
    print(f"Total runtime: {elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    main()
