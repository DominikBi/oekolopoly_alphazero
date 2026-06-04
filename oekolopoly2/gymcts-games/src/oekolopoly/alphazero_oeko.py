import argparse
import copy
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from oekolopoly.env.oeko_env import OekoActionBuilderWrapper, OekoEnv
from oekolopoly.env.oeko_wrappers import OekoRoundWeightedBalanceRewardWrapper

DEFAULT_MODEL_PATH = Path("models_alphazero") / "alphazero_oekolopoly.pt"


def make_alphazero_env(
    render_mode: str | None = None,
    reward_start_round: int = 10,
    reward_weight_power: float = 2.0,
    survival_reward: float = 0.5,
) -> OekoActionBuilderWrapper:
    """Create the Oekolopoly env with the reward requested for optimization.

    OekoActionBuilderWrapper turns one round-level budget allocation into a
    sequence of small discrete actions. Its mask encodes the non-box budget
    constraint: the sum of spent points must not exceed the available points.
    """
    env = OekoEnv(render_mode=render_mode)
    env = OekoRoundWeightedBalanceRewardWrapper(
        env,
        start_round=reward_start_round,
        weight_power=reward_weight_power,
        survival_reward=survival_reward,
    )
    return OekoActionBuilderWrapper(env, auxilary_reward=False)


def find_action_builder(env) -> OekoActionBuilderWrapper:
    while not isinstance(env, OekoActionBuilderWrapper):
        env = env.env
    return env


def legal_action_mask(env) -> np.ndarray:
    return np.asarray(find_action_builder(env).valid_action_mask(), dtype=bool)


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.masked_fill(~mask, -1.0e9)
    return F.softmax(masked_logits, dim=-1)


class OekoPolicyValueNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.SiLU())
            last_dim = hidden_size
        self.body = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last_dim, action_dim)
        self.value_head = nn.Sequential(
            nn.Linear(last_dim, max(32, last_dim // 2)),
            nn.SiLU(),
            nn.Linear(max(32, last_dim // 2), 1),
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.body(obs)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value


class AlphaZeroModel:
    def __init__(
        self,
        obs_space,
        action_dim: int,
        hidden_sizes: list[int],
        learning_rate: float,
        value_scale: float,
        device: str,
    ):
        self.obs_dim = int(np.prod(obs_space.shape))
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes
        self.value_scale = float(value_scale)
        self.device = torch.device(device)
        self.net = OekoPolicyValueNet(self.obs_dim, action_dim, hidden_sizes).to(
            self.device
        )
        self.optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=learning_rate,
            weight_decay=1.0e-4,
        )

        nvec = np.asarray(obs_space.nvec, dtype=np.float32)
        nvec = np.maximum(nvec - 1.0, 1.0)
        self.obs_scale = torch.tensor(nvec, dtype=torch.float32, device=self.device)

    def normalize_obs_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        return obs / self.obs_scale

    def predict(self, obs: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
        self.net.eval()
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).view(
                1, -1
            )
            mask_t = torch.tensor(mask, dtype=torch.bool, device=self.device).view(
                1, -1
            )
            logits, value_norm = self.net(self.normalize_obs_tensor(obs_t))
            probs = masked_softmax(logits, mask_t)[0].cpu().numpy()
            value = float(value_norm[0].cpu().item() * self.value_scale)
        return probs, value

    def train_on_examples(
        self,
        examples: list[tuple[np.ndarray, np.ndarray, float]],
        epochs: int,
        batch_size: int,
    ) -> dict[str, float]:
        if not examples:
            return {
                "loss": float("nan"),
                "policy_loss": float("nan"),
                "value_loss": float("nan"),
            }

        obs_np = np.asarray([example[0] for example in examples], dtype=np.float32)
        policy_np = np.asarray([example[1] for example in examples], dtype=np.float32)
        value_np = np.asarray(
            [example[2] / self.value_scale for example in examples],
            dtype=np.float32,
        )

        dataset = TensorDataset(
            torch.tensor(obs_np, dtype=torch.float32),
            torch.tensor(policy_np, dtype=torch.float32),
            torch.tensor(value_np, dtype=torch.float32),
        )
        loader = DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=True,
        )

        self.net.train()
        metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
        count = 0
        for _ in range(epochs):
            for obs_batch, policy_batch, value_batch in loader:
                obs_batch = obs_batch.to(self.device)
                policy_batch = policy_batch.to(self.device)
                value_batch = value_batch.to(self.device)

                logits, value_pred = self.net(self.normalize_obs_tensor(obs_batch))
                log_probs = F.log_softmax(logits, dim=-1)
                policy_loss = -(policy_batch * log_probs).sum(dim=-1).mean()
                value_loss = F.smooth_l1_loss(value_pred, value_batch)
                loss = policy_loss + value_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
                self.optimizer.step()

                batch_count = int(obs_batch.shape[0])
                metrics["loss"] += float(loss.detach().cpu().item()) * batch_count
                metrics["policy_loss"] += (
                    float(policy_loss.detach().cpu().item()) * batch_count
                )
                metrics["value_loss"] += (
                    float(value_loss.detach().cpu().item()) * batch_count
                )
                count += batch_count

        if count == 0:
            return metrics
        return {key: value / count for key, value in metrics.items()}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "hidden_sizes": self.hidden_sizes,
                "value_scale": self.value_scale,
                "model_state": self.net.state_dict(),
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: Path,
        obs_space,
        learning_rate: float,
        device: str,
    ) -> "AlphaZeroModel":
        payload = torch.load(path, map_location=device)
        model = cls(
            obs_space=obs_space,
            action_dim=int(payload["action_dim"]),
            hidden_sizes=list(payload["hidden_sizes"]),
            learning_rate=learning_rate,
            value_scale=float(payload["value_scale"]),
            device=device,
        )
        if model.obs_dim != int(payload["obs_dim"]):
            raise ValueError(
                f"Checkpoint obs_dim={payload['obs_dim']} does not match env obs_dim={model.obs_dim}."
            )
        model.net.load_state_dict(payload["model_state"])
        return model


@dataclass
class SearchNode:
    env: Any
    obs: np.ndarray
    reward: float = 0.0
    terminal: bool = False
    prior: float = 0.0
    parent: "SearchNode | None" = None
    action: int | None = None

    def __post_init__(self) -> None:
        self.children: dict[int, SearchNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def expanded(self) -> bool:
        return bool(self.children)

    @property
    def mean_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class AlphaZeroMCTS:
    def __init__(
        self,
        model: AlphaZeroModel,
        simulations: int,
        c_puct: float,
        gamma: float,
        dirichlet_alpha: float,
        dirichlet_fraction: float,
    ):
        self.model = model
        self.simulations = simulations
        self.c_puct = c_puct
        self.gamma = gamma
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_fraction = dirichlet_fraction
        self.action_dim = model.action_dim

    def run(
        self,
        env,
        obs: np.ndarray,
        add_exploration_noise: bool,
    ) -> tuple[np.ndarray, SearchNode]:
        root = SearchNode(env=copy.deepcopy(env), obs=np.asarray(obs).copy())
        self._expand(root)
        if add_exploration_noise:
            self._add_root_noise(root)

        for _ in range(self.simulations):
            self._simulate(root)

        visit_counts = np.zeros(self.action_dim, dtype=np.float32)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count
        return visit_counts, root

    def _simulate(self, node: SearchNode) -> float:
        if node.terminal:
            node.visit_count += 1
            return 0.0

        if not node.expanded:
            value = self._expand(node)
            node.visit_count += 1
            node.value_sum += value
            return value

        child = self._select_child(node)
        future_value = self._simulate(child)
        value = child.reward + self.gamma * future_value
        node.visit_count += 1
        node.value_sum += value
        return value

    def _expand(self, node: SearchNode) -> float:
        if node.terminal:
            return 0.0

        mask = legal_action_mask(node.env)
        priors, value = self.model.predict(node.obs, mask)

        legal_actions = np.flatnonzero(mask)
        if len(legal_actions) == 0:
            node.terminal = True
            return 0.0

        for action in legal_actions:
            child_env = copy.deepcopy(node.env)
            obs, reward, terminated, truncated, _info = child_env.step(int(action))
            node.children[int(action)] = SearchNode(
                env=child_env,
                obs=np.asarray(obs).copy(),
                reward=float(reward),
                terminal=bool(terminated or truncated),
                prior=float(priors[int(action)]),
                parent=node,
                action=int(action),
            )
        return value

    def _select_child(self, node: SearchNode) -> SearchNode:
        parent_visits = max(1, node.visit_count)
        best_score = -float("inf")
        best_child = None
        for child in node.children.values():
            q_value = child.reward + self.gamma * child.mean_value
            exploration = (
                self.c_puct
                * child.prior
                * np.sqrt(parent_visits)
                / (1 + child.visit_count)
            )
            score = q_value + exploration
            if score > best_score:
                best_score = score
                best_child = child

        if best_child is None:
            raise RuntimeError("MCTS selection failed: expanded node has no children.")
        return best_child

    def _add_root_noise(self, root: SearchNode) -> None:
        actions = list(root.children.keys())
        if not actions:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        for action, noise_value in zip(actions, noise):
            child = root.children[action]
            child.prior = (
                1.0 - self.dirichlet_fraction
            ) * child.prior + self.dirichlet_fraction * float(noise_value)


def policy_from_visit_counts(
    visit_counts: np.ndarray,
    legal_mask: np.ndarray,
    temperature: float,
) -> np.ndarray:
    policy = np.zeros_like(visit_counts, dtype=np.float32)
    legal_actions = np.flatnonzero(legal_mask)
    if len(legal_actions) == 0:
        return policy

    legal_counts = visit_counts[legal_actions].astype(np.float64)
    if legal_counts.sum() <= 0:
        policy[legal_actions] = 1.0 / len(legal_actions)
        return policy

    if temperature <= 1.0e-8:
        best_action = int(legal_actions[int(np.argmax(legal_counts))])
        policy[best_action] = 1.0
        return policy

    adjusted = np.power(legal_counts, 1.0 / temperature)
    adjusted_sum = adjusted.sum()
    if adjusted_sum <= 0:
        policy[legal_actions] = 1.0 / len(legal_actions)
    else:
        policy[legal_actions] = adjusted / adjusted_sum
    return policy


def select_action_from_policy(policy: np.ndarray, deterministic: bool) -> int:
    if deterministic:
        return int(np.argmax(policy))
    return int(np.random.choice(len(policy), p=policy))


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    returns = []
    running_return = 0.0
    for reward in reversed(rewards):
        running_return = float(reward) + gamma * running_return
        returns.append(running_return)
    returns.reverse()
    return returns


def run_self_play_episode(
    model: AlphaZeroModel,
    args,
    training: bool,
) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], dict[str, Any]]:
    env = make_alphazero_env(
        render_mode=None,
        reward_start_round=args.reward_start_round,
        reward_weight_power=args.reward_weight_power,
        survival_reward=args.survival_reward,
    )
    obs, _info = env.reset()
    mcts = AlphaZeroMCTS(
        model=model,
        simulations=args.simulations,
        c_puct=args.c_puct,
        gamma=args.gamma,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_fraction=args.dirichlet_fraction,
    )

    observations: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    rewards: list[float] = []
    actions: list[int] = []
    info: dict[str, Any] = {}
    terminated = False
    truncated = False
    step_idx = 0

    while not (terminated or truncated) and step_idx < args.max_episode_steps:
        mask = legal_action_mask(env)
        visit_counts, _root = mcts.run(
            env,
            obs,
            add_exploration_noise=training,
        )
        temperature = args.temperature if step_idx < args.temperature_steps else 0.0
        policy = policy_from_visit_counts(visit_counts, mask, temperature)
        action = select_action_from_policy(policy, deterministic=not training)

        observations.append(np.asarray(obs).copy())
        policies.append(policy)

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        actions.append(action)
        step_idx += 1

    returns = discounted_returns(rewards, gamma=args.gamma)
    examples = [
        (observation, policy, episode_return)
        for observation, policy, episode_return in zip(observations, policies, returns)
    ]
    unwrapped = env.unwrapped
    summary = {
        "steps": step_idx,
        "round": int(unwrapped.V[unwrapped.ROUND]),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "balance": float(unwrapped.balance),
        "balance_always": float(unwrapped.balance_always),
        "return": float(sum(rewards)),
        "done_reason": info.get("done_reason", ""),
        "valid_move": info.get("valid_move", True),
        "actions": actions,
    }
    return examples, summary


def evaluate(model: AlphaZeroModel, args) -> dict[str, Any]:
    _examples, summary = run_self_play_episode(model, args, training=False)
    return summary


def parse_hidden_sizes(value: str) -> list[int]:
    hidden_sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not hidden_sizes:
        raise argparse.ArgumentTypeError("hidden sizes must not be empty")
    return hidden_sizes


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def build_model(args) -> AlphaZeroModel:
    env = make_alphazero_env(
        reward_start_round=args.reward_start_round,
        reward_weight_power=args.reward_weight_power,
        survival_reward=args.survival_reward,
    )
    return AlphaZeroModel(
        obs_space=env.observation_space,
        action_dim=env.action_space.n,
        hidden_sizes=args.hidden_sizes,
        learning_rate=args.learning_rate,
        value_scale=args.value_scale,
        device=args.device,
    )


def print_summary(prefix: str, summary: dict[str, Any]) -> None:
    print(
        f"{prefix}: steps={summary['steps']}, "
        f"round={summary['round']}, "
        f"return={summary['return']:.3f}, "
        f"balance={summary['balance']:.3f}, "
        f"balance_always={summary['balance_always']:.3f}, "
        f"terminated={summary['terminated']}, "
        f"done_reason={summary['done_reason']}"
    )


def train(args) -> AlphaZeroModel:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = (
        AlphaZeroModel.load(
            args.model_path,
            obs_space=make_alphazero_env().observation_space,
            learning_rate=args.learning_rate,
            device=args.device,
        )
        if args.load_model and args.model_path.exists()
        else build_model(args)
    )
    replay_buffer: deque[tuple[np.ndarray, np.ndarray, float]] = deque(
        maxlen=args.replay_buffer_size
    )

    for iteration in range(args.iterations):
        started = time.perf_counter()
        iteration_examples = 0
        last_summary: dict[str, Any] | None = None

        for _episode in range(args.episodes_per_iteration):
            examples, summary = run_self_play_episode(model, args, training=True)
            replay_buffer.extend(examples)
            iteration_examples += len(examples)
            last_summary = summary

        metrics = model.train_on_examples(
            list(replay_buffer),
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        model.save(args.model_path)

        eval_summary = evaluate(model, args)
        elapsed = time.perf_counter() - started
        print(
            f"iteration={iteration + 1}/{args.iterations}, "
            f"new_examples={iteration_examples}, "
            f"buffer={len(replay_buffer)}, "
            f"loss={metrics['loss']:.4f}, "
            f"policy_loss={metrics['policy_loss']:.4f}, "
            f"value_loss={metrics['value_loss']:.4f}, "
            f"elapsed={elapsed:.2f}s, "
            f"model={args.model_path}"
        )
        if last_summary is not None:
            print_summary("self-play", last_summary)
        print_summary("eval", eval_summary)

    return model


def run_smoke_test(args) -> None:
    args.iterations = 1
    args.episodes_per_iteration = 1
    args.simulations = 2
    args.epochs = 1
    args.batch_size = 16
    args.hidden_sizes = [32, 32]
    args.replay_buffer_size = 128
    args.max_episode_steps = 32
    args.temperature_steps = 32
    args.model_path = Path("models_alphazero") / "smoke_alphazero_oekolopoly.pt"

    model = train(args)
    summary = evaluate(model, args)

    assert summary["steps"] > 0, "Smoke test did not execute any environment step."
    assert summary["valid_move"], "Smoke test produced an invalid move."
    print_summary("smoke", summary)
    print("Smoke test passed.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "AlphaZero-style self-play for Oekolopoly using masked PUCT MCTS "
            "and the OekoRoundWeightedBalanceRewardWrapper reward."
        )
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--episodes-per-iteration", type=int, default=16)
    parser.add_argument("--simulations", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-buffer-size", type=int, default=50_000)
    parser.add_argument("--max-episode-steps", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")

    parser.add_argument("--reward-start-round", type=int, default=10)
    parser.add_argument("--reward-weight-power", type=float, default=2.0)
    parser.add_argument("--survival-reward", type=float, default=0.5)

    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-fraction", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-steps", type=int, default=80)

    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--value-scale", type=float, default=200.0)
    parser.add_argument(
        "--hidden-sizes",
        type=parse_hidden_sizes,
        default=parse_hidden_sizes("128,128"),
    )
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)
    args.model_path = args.model_path.resolve()

    if args.smoke_test:
        run_smoke_test(args)
        return

    if args.eval_only:
        model = AlphaZeroModel.load(
            args.model_path,
            obs_space=make_alphazero_env(
                reward_start_round=args.reward_start_round,
                reward_weight_power=args.reward_weight_power,
                survival_reward=args.survival_reward,
            ).observation_space,
            learning_rate=args.learning_rate,
            device=args.device,
        )
        print_summary("eval", evaluate(model, args))
        return

    train(args)


if __name__ == "__main__":
    main()
