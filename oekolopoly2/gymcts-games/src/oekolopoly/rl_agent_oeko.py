import argparse
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env

from oekolopoly.env.oeko_env import OekoActionBuilderWrapper, OekoEnv
from oekolopoly.env.oeko_wrappers import OekoRoundWeightedBalanceRewardWrapper

DEFAULT_MODEL_PATH = Path("models") / "maskable_ppo_oekolopoly_weighted"


def make_oeko_env(
    render_mode: str | None = None,
    reward_start_round: int = 10,
    reward_weight_power: float = 2.0,
    survival_reward: float = 1,
):
    env = OekoEnv(render_mode=render_mode)
    env = OekoRoundWeightedBalanceRewardWrapper(
        env,
        start_round=reward_start_round,
        weight_power=reward_weight_power,
        survival_reward=survival_reward,
    )
    env = OekoActionBuilderWrapper(env, auxilary_reward=False)

    def action_mask_fn(masked_env: OekoActionBuilderWrapper):
        return masked_env.valid_action_mask()

    return ActionMasker(env, action_mask_fn)


def train_agent(
    total_timesteps: int,
    model_path: Path,
    n_envs: int,
    seed: int,
    reward_start_round: int,
    reward_weight_power: float,
    survival_reward: float,
):
    env = make_vec_env(
        lambda: make_oeko_env(
            reward_start_round=reward_start_round,
            reward_weight_power=reward_weight_power,
            survival_reward=survival_reward,
        ),
        n_envs=n_envs,
        seed=seed,
    )

    model = MaskablePPO(
        "MlpPolicy",
        env,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        vf_coef=0.5,
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=1,
        seed=seed,
    )

    model.learn(total_timesteps=total_timesteps)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    env.close()
    return model


def evaluate_agent(
    model: MaskablePPO,
    episodes: int,
    render: bool,
    reward_start_round: int,
    reward_weight_power: float,
    survival_reward: float,
):
    episode_results = []

    for episode in range(episodes):
        env = make_oeko_env(
            render_mode="ansi" if render else None,
            reward_start_round=reward_start_round,
            reward_weight_power=reward_weight_power,
            survival_reward=survival_reward,
        )
        obs, _ = env.reset()
        terminated = False
        truncated = False
        accumulated_reward = 0.0
        actions = []

        while not (terminated or truncated):
            action_masks = env.action_masks()
            action, _ = model.predict(
                obs,
                deterministic=True,
                action_masks=action_masks,
            )
            if isinstance(action, np.ndarray):
                action = int(action.item())

            obs, reward, terminated, truncated, info = env.step(action)
            actions.append(action)
            accumulated_reward += float(reward)

        if render:
            env.render()

        unwrapped = env.unwrapped
        result = {
            "episode": episode + 1,
            "round": int(unwrapped.V[unwrapped.ROUND]),
            "balance": float(unwrapped.balance),
            "accumulated_reward": accumulated_reward,
            "done_reason": info.get("done_reason", ""),
            "actions": actions,
        }
        episode_results.append(result)
        env.close()

    return episode_results


def print_results(results):
    for result in results:
        print(
            f"Episode {result['episode']}: "
            f"round={result['round']}, "
            f"balance={result['balance']:.3f}, "
            f"reward={result['accumulated_reward']:.3f}, "
            f"done_reason={result['done_reason']}"
        )
        print(f"Actions: {result['actions']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a MaskablePPO RL agent for Oekolopoly."
    )
    parser.add_argument("--timesteps", type=int, default=800_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--reward-start-round", type=int, default=10)
    parser.add_argument("--reward-weight-power", type=float, default=2.0)
    parser.add_argument("--survival-reward", type=float, default=0.25)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.eval_only:
        model = MaskablePPO.load(args.model_path)
    else:
        model = train_agent(
            total_timesteps=args.timesteps,
            model_path=args.model_path,
            n_envs=args.n_envs,
            seed=args.seed,
            reward_start_round=args.reward_start_round,
            reward_weight_power=args.reward_weight_power,
            survival_reward=args.survival_reward,
        )

    results = evaluate_agent(
        model,
        episodes=args.episodes,
        render=args.render,
        reward_start_round=args.reward_start_round,
        reward_weight_power=args.reward_weight_power,
        survival_reward=args.survival_reward,
    )
    print_results(results)


if __name__ == "__main__":
    main()
