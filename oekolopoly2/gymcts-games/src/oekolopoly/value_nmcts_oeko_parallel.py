import argparse
import copy
import json
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, SupportsFloat

import numpy as np
import torch
import torch.nn.functional as F
from botorch.acquisition import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gymnasium.core import WrapperActType, WrapperObsType
from gymcts.gymcts_agent import GymctsAgent, GymctsNode, log
from gymcts.gymcts_deepcopy_wrapper import DeepCopyMCTSGymEnvWrapper
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from oekolopoly.env.oeko_env import OekoActionBuilderWrapper, OekoEnv
from oekolopoly.env.oeko_wrappers import OekoRoundWeightedBalanceRewardWrapper
from oekolopoly.nmcts_oeko import action_mask_fn, summarize_episode

DEFAULT_MODEL_PATH = Path("models_value_nmcts_parallel") / "oeko_policy_value_model.pt"
VALUE_OBS_SCALE = [29, 29, 29, 29, 29, 29, 48, 37, 30, 36, 5]
ACTION_DIM = 9
MAX_TRAIN_SAMPLES = 200_0000

# Bayesian optimization search space. Bounds are normalized internally by BO,
# so changing these values is enough to adjust the tuning range.
BO_SEARCH_SPACE = [
    {"name": "exploration_weight", "type": "float", "low": 0.5, "high": 4.0},
    {"name": "puct_prior_temperature", "type": "float", "low": 0.5, "high": 2.0},
    {"name": "best_action_weight", "type": "float", "low": 0.80, "high": 0.999},
]


class RandomRolloutOekoWrapper(DeepCopyMCTSGymEnvWrapper):
    def __init__(self, env, action_mask_fn):
        super().__init__(env, action_mask_fn=action_mask_fn)

    def _auto_advance_if_needed(self) -> None:
        if self.is_terminal():
            return

        action_builder = find_action_builder(self.env)
        if action_builder.expose_next_round_action:
            return
        if not action_builder.auto_next_round_when_no_points:
            return
        if np.any(action_builder.valid_action_mask()[1:]):
            return

        step_tuple = action_builder._step_next_round()
        _obs, _reward, terminated, truncated, _info = step_tuple
        self._terminal_flag = terminated or truncated
        self._step_tuple = step_tuple

    def action_masks(self) -> np.ndarray | None:
        self._auto_advance_if_needed()
        return super().action_masks()

    def get_valid_actions(self) -> list[int]:
        self._auto_advance_if_needed()
        return super().get_valid_actions()

    def rollout(self, value_model=None) -> float:
        accumulated_reward = 0.0
        terminated = self.is_terminal()
        truncated = False

        while not (terminated or truncated):
            self._auto_advance_if_needed()
            terminated = self.is_terminal()
            if terminated:
                break
            action = self._select_rollout_action(value_model)
            if action is None:
                break
            _obs, reward, terminated, truncated, _info = self.step(action)
            accumulated_reward += float(reward)

        return accumulated_reward

    def _select_rollout_action(self, value_model) -> int | None:
        valid_actions = self.get_valid_actions()
        if not valid_actions:
            self._auto_advance_if_needed()
            if self.is_terminal():
                return None
            valid_actions = self.get_valid_actions()
            if not valid_actions:
                return None

        if value_model is None or len(valid_actions) <= 1:
            return random.choice(valid_actions)

        policy = value_model.predict_policy(obs_from_mcts_env(self), valid_actions)
        if policy.sum() > 0.0:
            return int(np.random.choice(len(policy), p=policy))

        child_observations = []
        for action in valid_actions:
            child_env = copy.deepcopy(self)
            child_env.step(action)
            child_observations.append(obs_from_mcts_env(child_env))

        scores = value_model.predict_batch(child_observations)
        max_score = max(scores)
        best_actions = [
            action for action, score in zip(valid_actions, scores) if score == max_score
        ]
        return best_actions[0]

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        return super().step(action)


def make_value_nmcts_env(
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
    env = OekoActionBuilderWrapper(
        env,
        auxilary_reward=False,
        expose_next_round_action=False,
        auto_next_round_when_no_points=True,
    )
    env = RandomRolloutOekoWrapper(env, action_mask_fn=action_mask_fn)
    env.reset()
    return env


def find_action_builder(env) -> OekoActionBuilderWrapper:
    action_builder = env
    while not isinstance(action_builder, OekoActionBuilderWrapper):
        action_builder = action_builder.env
    return action_builder


def obs_from_mcts_env(env) -> np.ndarray:
    action_builder = find_action_builder(env)
    oeko_env = action_builder.env.unwrapped
    values = oeko_env.V
    current_action = action_builder._current_action_dict
    # Pure state encoding: removed transition_reward to make observations generalizable
    return np.array(
        [
            values[oeko_env.SANITATION] + current_action["Sanitation"],
            values[oeko_env.PRODUCTION] + current_action["Production"],
            values[oeko_env.EDUCATION] + current_action["Education"],
            values[oeko_env.QUALITY_OF_LIFE] + current_action["Quality of Life"],
            values[oeko_env.POPULATION_GROWTH]
            + current_action["Population Growth"]
            + current_action["Population Growth extra"],
            values[oeko_env.ENVIRONMENT],
            values[oeko_env.POPULATION],
            values[oeko_env.POLITICS],
            values[oeko_env.ROUND],
            action_builder._available_action_points,
            current_action["Population Growth extra"],
        ],
        dtype=np.float32,
    )


def obs_from_node(node: GymctsNode) -> np.ndarray:
    return obs_from_mcts_env(node.state)


def rollout_from_env(env, value_model=None):
    return env.rollout(value_model)


_ROLLOUT_WORKER_VALUE_MODEL = None


def rollout_from_env_with_worker_model(env):
    return env.rollout(_ROLLOUT_WORKER_VALUE_MODEL)


class OekoPolicyValueNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_sizes: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim

        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(last_dim, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.SiLU())
            if i == 0:
                layers.append(nn.Dropout(0.1))
            last_dim = hidden_size

        self.body = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last_dim, action_dim)
        self.value_head = nn.Sequential(
            nn.Linear(last_dim, max(32, last_dim // 2)),
            nn.SiLU(),
            nn.Linear(max(32, last_dim // 2), 1),
        )

    def forward(self, x):
        features = self.body(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return policy_logits, value


class ValueModel:
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_sizes: list[int],
        learning_rate: float,
        device: str = "cpu",
    ):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes
        self.device = torch.device(device)
        self.model = OekoPolicyValueNetwork(input_dim, action_dim, hidden_sizes).to(
            self.device
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )
        self.obs_scale = torch.tensor(
            VALUE_OBS_SCALE,
            dtype=torch.float32,
            device=self.device,
        )
        self.target_mean = 0.0
        self.target_std = 1.0

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return obs / self.obs_scale

    def predict_batch(self, observations: list[np.ndarray]) -> list[float]:
        if not observations:
            return []
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(
                np.asarray(observations), dtype=torch.float32, device=self.device
            )
            _policy_logits, value_pred = self.model(self._normalize_obs(x))
            value_pred = value_pred * self.target_std + self.target_mean
        return value_pred.cpu().numpy().astype(float).tolist()

    def predict_policy(
        self,
        observation: np.ndarray,
        valid_actions: list[int],
        temperature: float = 1.0,
    ) -> np.ndarray:
        if not valid_actions:
            return np.zeros(self.action_dim, dtype=np.float32)

        mask = np.zeros(self.action_dim, dtype=bool)
        mask[np.asarray(valid_actions, dtype=np.int64)] = True

        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(
                np.asarray([observation]), dtype=torch.float32, device=self.device
            )
            mask_t = torch.tensor(mask[None, :], dtype=torch.bool, device=self.device)
            policy_logits, _value_pred = self.model(self._normalize_obs(x))
            policy_logits = policy_logits.masked_fill(~mask_t, -1.0e9)
            policy = F.softmax(policy_logits, dim=-1)[0].cpu().numpy()

        temperature = max(temperature, 1e-6)
        if temperature != 1.0:
            legal_policy = np.power(policy[mask], 1.0 / temperature)
            denominator = float(legal_policy.sum())
            if denominator > 0.0 and np.isfinite(denominator):
                policy = np.zeros_like(policy)
                policy[mask] = legal_policy / denominator

        return policy.astype(np.float32)

    def train_on_dataset(
        self,
        observations: list[np.ndarray],
        policy_targets: list[np.ndarray],
        value_targets: list[float],
        epochs: int,
        batch_size: int,
        validation_fraction: float,
    ) -> dict[str, float]:
        if not observations:
            return {
                "train_loss": float("nan"),
                "validation_loss": float("nan"),
                "train_mae": float("nan"),
                "validation_mae": float("nan"),
                "train_rmse": float("nan"),
                "validation_rmse": float("nan"),
                "train_corr": float("nan"),
                "validation_corr": float("nan"),
                "train_policy_loss": float("nan"),
                "validation_policy_loss": float("nan"),
            }

        y_np = np.asarray(value_targets, dtype=np.float32)
        pi_np = np.asarray(policy_targets, dtype=np.float32)
        self.target_mean = float(y_np.mean())
        self.target_std = float(max(y_np.std(), 1e-6))

        x_all = torch.tensor(np.asarray(observations), dtype=torch.float32)
        pi_all = torch.tensor(pi_np, dtype=torch.float32)
        y_all = torch.tensor(
            (y_np - self.target_mean) / self.target_std, dtype=torch.float32
        )

        val_size = 0
        if len(y_np) >= 10 and validation_fraction > 0.0:
            val_size = max(1, int(len(y_np) * validation_fraction))
            val_size = min(val_size, len(y_np) - 1)

        if val_size:
            indices = np.random.permutation(len(y_np))
            val_indices = torch.tensor(indices[:val_size], dtype=torch.long)
            train_indices = torch.tensor(indices[val_size:], dtype=torch.long)
            x_train = x_all.index_select(0, train_indices)
            pi_train = pi_all.index_select(0, train_indices)
            y_train = y_all.index_select(0, train_indices)
            x_val = x_all.index_select(0, val_indices)
            pi_val = pi_all.index_select(0, val_indices)
            y_val = y_all.index_select(0, val_indices)
        else:
            x_train = x_all
            pi_train = pi_all
            y_train = y_all
            x_val = None
            pi_val = None
            y_val = None

        loader = DataLoader(
            TensorDataset(x_train, pi_train, y_train),
            batch_size=min(batch_size, len(y_train)),
            shuffle=True,
        )

        self.model.train()
        loss_fn = nn.SmoothL1Loss()
        for _ in range(epochs):
            for batch_x, batch_pi, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_pi = batch_pi.to(self.device)
                batch_y = batch_y.to(self.device)
                policy_logits, value_pred = self.model(self._normalize_obs(batch_x))
                policy_loss = (
                    -(batch_pi * F.log_softmax(policy_logits, dim=-1))
                    .sum(dim=-1)
                    .mean()
                )
                value_loss = loss_fn(value_pred, batch_y)
                loss = policy_loss + value_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

        train_metrics = self._evaluate_value_dataset(
            x_train,
            pi_train,
            y_train,
            batch_size=batch_size,
            prefix="train",
        )
        val_metrics = {
            "validation_loss": float("nan"),
            "validation_mae": float("nan"),
            "validation_rmse": float("nan"),
            "validation_corr": float("nan"),
            "validation_policy_loss": float("nan"),
        }
        if x_val is not None and pi_val is not None and y_val is not None:
            val_metrics = self._evaluate_value_dataset(
                x_val,
                pi_val,
                y_val,
                batch_size=batch_size,
                prefix="validation",
            )
        return {**train_metrics, **val_metrics}

    def _evaluate_value_dataset(
        self,
        x: torch.Tensor,
        pi: torch.Tensor,
        y: torch.Tensor,
        batch_size: int,
        prefix: str,
    ) -> dict[str, float]:
        loader = DataLoader(
            TensorDataset(x, pi, y),
            batch_size=min(batch_size, len(y)),
            shuffle=False,
        )
        loss_fn = nn.SmoothL1Loss(reduction="sum")
        loss_sum = 0.0
        policy_loss_sum = 0.0
        abs_error_sum = 0.0
        squared_error_sum = 0.0
        pred_sum = 0.0
        target_sum = 0.0
        pred_squared_sum = 0.0
        target_squared_sum = 0.0
        pred_target_sum = 0.0
        count = 0

        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_pi, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_pi = batch_pi.to(self.device)
                batch_y = batch_y.to(self.device)
                policy_logits, pred_norm = self.model(self._normalize_obs(batch_x))
                value_loss_sum = float(loss_fn(pred_norm, batch_y).cpu().item())
                policy_loss = -(batch_pi * F.log_softmax(policy_logits, dim=-1)).sum(
                    dim=-1
                )
                policy_loss_sum += float(policy_loss.sum().cpu().item())
                loss_sum += value_loss_sum + float(policy_loss.sum().cpu().item())

                pred = pred_norm * self.target_std + self.target_mean
                target = batch_y * self.target_std + self.target_mean
                error = pred - target

                abs_error_sum += float(error.abs().sum().cpu().item())
                squared_error_sum += float((error * error).sum().cpu().item())
                pred_sum += float(pred.sum().cpu().item())
                target_sum += float(target.sum().cpu().item())
                pred_squared_sum += float((pred * pred).sum().cpu().item())
                target_squared_sum += float((target * target).sum().cpu().item())
                pred_target_sum += float((pred * target).sum().cpu().item())
                count += int(batch_y.numel())

        if count == 0:
            return {
                f"{prefix}_loss": float("nan"),
                f"{prefix}_mae": float("nan"),
                f"{prefix}_rmse": float("nan"),
                f"{prefix}_corr": float("nan"),
                f"{prefix}_policy_loss": float("nan"),
            }

        pred_mean = pred_sum / count
        target_mean = target_sum / count
        covariance = pred_target_sum / count - pred_mean * target_mean
        pred_variance = pred_squared_sum / count - pred_mean * pred_mean
        target_variance = target_squared_sum / count - target_mean * target_mean
        corr_denominator = math.sqrt(
            max(pred_variance, 0.0) * max(target_variance, 0.0)
        )
        corr = covariance / corr_denominator if corr_denominator > 0.0 else float("nan")

        return {
            f"{prefix}_loss": loss_sum / count,
            f"{prefix}_mae": abs_error_sum / count,
            f"{prefix}_rmse": math.sqrt(squared_error_sum / count),
            f"{prefix}_corr": corr,
            f"{prefix}_policy_loss": policy_loss_sum / count,
        }

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_type": "policy_value",
                "input_dim": self.input_dim,
                "action_dim": self.action_dim,
                "hidden_sizes": self.hidden_sizes,
                "model_state": self.model.state_dict(),
                "target_mean": self.target_mean,
                "target_std": self.target_std,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, learning_rate: float, device: str = "cpu"):
        payload = torch.load(path, map_location=device)
        if payload.get("model_type") != "policy_value":
            raise ValueError(
                "This checkpoint was saved by the old value-only model. "
                "Train a fresh policy/value checkpoint or choose a new --model-path."
            )
        if payload["input_dim"] != len(VALUE_OBS_SCALE):
            raise ValueError(
                f"Saved value model expects input_dim={payload['input_dim']}, "
                f"but the current feature vector has input_dim={len(VALUE_OBS_SCALE)}. "
                "Train a fresh value model or migrate the checkpoint."
            )
        value_model = cls(
            input_dim=payload["input_dim"],
            action_dim=int(payload["action_dim"]),
            hidden_sizes=list(payload["hidden_sizes"]),
            learning_rate=learning_rate,
            device=device,
        )
        value_model.model.load_state_dict(payload["model_state"])
        value_model.target_mean = float(payload["target_mean"])
        value_model.target_std = float(payload["target_std"])
        return value_model


def rollout_worker_payload(value_model: ValueModel) -> dict[str, Any]:
    return {
        "input_dim": value_model.input_dim,
        "action_dim": value_model.action_dim,
        "hidden_sizes": value_model.hidden_sizes,
        "model_state": {
            name: tensor.detach().cpu()
            for name, tensor in value_model.model.state_dict().items()
        },
        "target_mean": value_model.target_mean,
        "target_std": value_model.target_std,
    }


def initialize_rollout_worker(payload: dict[str, Any]) -> None:
    global _ROLLOUT_WORKER_VALUE_MODEL
    value_model = ValueModel(
        input_dim=payload["input_dim"],
        action_dim=int(payload["action_dim"]),
        hidden_sizes=list(payload["hidden_sizes"]),
        learning_rate=1e-3,
        device="cpu",
    )
    value_model.model.load_state_dict(payload["model_state"])
    value_model.target_mean = float(payload["target_mean"])
    value_model.target_std = float(payload["target_std"])
    value_model.model.eval()
    _ROLLOUT_WORKER_VALUE_MODEL = value_model


class ValueGuidedMCTSAgent(GymctsAgent):
    def __init__(
        self,
        *args,
        value_model: ValueModel | None = None,
        nn_weight: float = 0.0,
        exploration_weight: float = 1.0,
        tree_selection_policy: str = "nn",
        rollout_policy: str = "nn",
        puct_prior_temperature: float = 1.0,
        parallel_workers: int = 1,
        parallel_batch_size: int | None = None,
        parallel_backend: str = "process",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.value_model = value_model
        self.nn_weight = nn_weight
        self.exploration_weight = max(0.0, exploration_weight)
        self.tree_selection_policy = tree_selection_policy
        self.rollout_policy = rollout_policy
        self.puct_prior_temperature = max(1e-6, puct_prior_temperature)
        self.parallel_workers = max(1, parallel_workers)
        self.parallel_batch_size = parallel_batch_size or self.parallel_workers
        self.parallel_backend = parallel_backend
        self._parallel_executor = None
        self._reset_search_root_node = copy.deepcopy(self.search_root_node)
        self.training_observations: list[np.ndarray] = []
        self.training_policies: list[np.ndarray] = []
        self.training_targets: list[float] = []
        self.last_actions: list[int] = []
        self.last_accumulated_reward = 0.0
        self.last_info: dict[str, Any] = {}
        # Cache scale factors for efficient access in tree policy
        self.target_mean = value_model.target_mean if value_model else 0.0
        self.target_std = value_model.target_std if value_model else 1.0

    def reset(self) -> None:
        self.search_root_node = copy.deepcopy(self._reset_search_root_node)
        self.last_actions = []
        self.last_accumulated_reward = 0.0
        self.last_info = {}

    def _sync_node_with_env(self, node: GymctsNode) -> GymctsNode:
        self._load_state(node)
        node.valid_actions = self.env.get_valid_actions()
        node.terminal = self.env.is_terminal()
        node.state = self.env.get_state()
        return node

    def solve(
        self,
        num_simulations_per_step: int | None = None,
        render_tree_after_step: bool | None = None,
    ) -> list[int]:
        if num_simulations_per_step is None:
            num_simulations_per_step = self.number_of_simulations_per_step
        if render_tree_after_step is None:
            render_tree_after_step = self.render_tree_after_step

        log.debug(f"Solving from root node: {self.search_root_node}")
        current_node = self.search_root_node
        action_list = []
        accumulated_reward = 0.0
        info = {}

        idx = 0
        while True:
            current_node = self._sync_node_with_env(current_node)
            self.search_root_node = current_node
            if current_node.terminal:
                step_tuple = current_node.state._step_tuple
                if step_tuple is not None:
                    accumulated_reward += float(step_tuple[1])
                    info = step_tuple[4]
                break

            num_sims = self.calc_number_of_simulations_per_step(
                num_simulations_per_step,
                idx,
            )
            log.info(f"Performing MCTS step {idx} with {num_sims} simulations.")
            next_action, current_node = self.perform_mcts_step(
                num_simulations=num_sims,
                render_tree_after_step=render_tree_after_step,
            )
            if next_action is None:
                break

            log.info(f"selected action {next_action} after {num_sims} simulations.")
            action_list.append(next_action)
            step_tuple = current_node.state._step_tuple
            if step_tuple is not None:
                accumulated_reward += float(step_tuple[1])
                info = step_tuple[4]
            if current_node.terminal:
                break
            idx += 1

        self.last_actions = action_list
        self.last_accumulated_reward = accumulated_reward
        self.last_info = info
        return action_list

    def perform_mcts_step(
        self,
        search_start_node: GymctsNode | None = None,
        num_simulations: int | None = None,
        render_tree_after_step: bool | None = None,
    ) -> tuple[int | None, GymctsNode]:
        if render_tree_after_step is None:
            render_tree_after_step = self.render_tree_after_step
        if num_simulations is None:
            num_simulations = self.number_of_simulations_per_step
        if search_start_node is None:
            search_start_node = self.search_root_node

        search_start_node = self._sync_node_with_env(search_start_node)
        if search_start_node.terminal:
            self.search_root_node = search_start_node
            return None, search_start_node

        action = self.vanilla_mcts_search(
            search_start_node=search_start_node,
            num_simulations=num_simulations,
        )
        if action is None:
            search_start_node = self._sync_node_with_env(search_start_node)
            self.search_root_node = search_start_node
            return None, search_start_node

        next_node = search_start_node.children[action]
        if self.clear_mcts_tree_after_step:
            next_node.reset()
        elif not self.keep_whole_tree_till_initial_root:
            next_node.remove_parent()

        self.search_root_node = next_node
        return action, next_node

    def expand_node(self, node: GymctsNode) -> None:
        self._sync_node_with_env(node)
        if node.terminal or not node.valid_actions:
            node.children = {}
            return

        child_dict = {}
        uniform_prior = 1.0 / max(1, len(node.valid_actions))
        priors = self._policy_priors_for_node(node)
        for action in node.valid_actions:
            self._load_state(node)
            _obs, _reward, _terminal, _truncated, _info = self.env.step(action)
            child = GymctsNode(action=action, parent=node, env_reference=self.env)
            prior = float(priors.get(action, uniform_prior))
            child._selection_score_prior = prior
            child._puct_prior = prior
            child.mean_value = 0.0  # Neutral initialization: avoid optimistic bias
            child_dict[action] = child
        node.children = child_dict
        self._cache_child_nn_values(list(child_dict.values()))

    def _policy_priors_for_node(self, node: GymctsNode) -> dict[int, float]:
        valid_actions = list(node.valid_actions)
        if not valid_actions:
            return {}

        uniform_prior = 1.0 / len(valid_actions)
        if self.value_model is None or self.nn_weight <= 0.0:
            return {action: uniform_prior for action in valid_actions}

        policy = self.value_model.predict_policy(
            obs_from_node(node),
            valid_actions,
            temperature=self.puct_prior_temperature,
        )
        nn_blend = min(1.0, max(0.0, self.nn_weight))
        return {
            action: (1.0 - nn_blend) * uniform_prior + nn_blend * float(policy[action])
            for action in valid_actions
        }

    def navigate_to_leaf(self, from_node: GymctsNode) -> GymctsNode:
        if from_node.terminal or from_node.is_leaf():
            return from_node

        temp_node = from_node
        while temp_node.children:
            temp_node = self._select_best_tree_policy_child(temp_node)
        return temp_node

    def _select_best_tree_policy_child(self, node: GymctsNode) -> GymctsNode:
        """Select child with UCB or AlphaZero-style PUCT."""
        if not node.children:
            return node

        children = list(node.children.values())
        scores = [(child, self._tree_policy_score(child)) for child in children]
        max_score = max(score for _child, score in scores)
        best_children = [child for child, score in scores if score == max_score]
        return best_children[0]

    def _tree_policy_score(self, node: GymctsNode) -> float:
        if self.tree_selection_policy in ("puct", "nn"):
            return self._puct_score(node)
        return self._ucb_score(node)

    def _ucb_score(self, node: GymctsNode) -> float:
        exploitation = 0.0 if node.visit_count == 0 else node.get_score()
        exploration = (
            self.exploration_weight
            * GymctsNode.ubc_c
            * math.sqrt(
                2
                * math.log(max(2, node.parent.visit_count + 1))
                / (node.visit_count + 1)
            )
        )
        return exploitation + exploration

    def _puct_score(self, node: GymctsNode) -> float:
        """AlphaZero-style tree policy: Q(s,a) + c_puct * P(s,a) * sqrt(N)/(1+n)."""
        q_value = 0.0 if node.visit_count == 0 else node.get_score()
        prior = float(getattr(node, "_puct_prior", 0.0))
        parent_visits = max(1, node.parent.visit_count if node.parent else 1)
        exploration = (
            self.exploration_weight
            * GymctsNode.ubc_c
            * prior
            * math.sqrt(parent_visits)
            / (1 + node.visit_count)
        )
        return q_value + exploration

    def _record_exploitation_training_data(self, search_start_node: GymctsNode) -> None:
        """Record policy/value targets from searched nodes.

        CRITICAL: Training targets should ideally be actual episode returns from the
        root, not node.mean_value. node.mean_value has exploration bias:
        - Early tree expansion visits nodes with few rollouts
        - NN predictions can bias which nodes get visited
        - This creates circular dependency: NN → biased visit counts → biased targets

        WORKAROUND: Use node.mean_value but prefer node.max_value for exploration bonus.
        Ideally, separate NN training from search: train on collected full-episode
        trajectories with true discounted returns.
        """
        stack = [search_start_node]
        while stack:
            node = stack.pop()
            if node.children:
                total_child_visits = sum(
                    child.visit_count for child in node.children.values()
                )
                if total_child_visits > 0:
                    policy_target = np.zeros(ACTION_DIM, dtype=np.float32)
                    for action, child in node.children.items():
                        policy_target[action] = child.visit_count / total_child_visits

                    self.training_observations.append(obs_from_node(node))
                    self.training_policies.append(policy_target)
                    self.training_targets.append(node.mean_value)

                stack.extend(node.children.values())
            elif node.visit_count > 0:
                policy_target = np.zeros(ACTION_DIM, dtype=np.float32)
                valid_actions = list(node.valid_actions)
                if valid_actions:
                    policy_target[valid_actions] = 1.0 / len(valid_actions)
                self.training_observations.append(obs_from_node(node))
                self.training_policies.append(policy_target)
                self.training_targets.append(node.mean_value)

    @staticmethod
    def _node_transition_reward(node: GymctsNode) -> float:
        step_tuple = node.state._step_tuple
        return 0.0 if step_tuple is None else float(step_tuple[1])

    def _update_node_statistics(self, node: GymctsNode, episode_return: float) -> None:
        node.mean_value = node.mean_value + (episode_return - node.mean_value) / (
            node.visit_count + 1
        )
        node.visit_count += 1
        node.max_value = max(node.max_value, episode_return)
        node.min_value = min(node.min_value, episode_return)

    def _backpropagate_suffix_returns(
        self, leaf_node: GymctsNode, rollout_return_after_leaf: float
    ) -> None:
        node = leaf_node
        suffix_return = rollout_return_after_leaf
        while node is not None:
            if not node.is_root():
                suffix_return += self._node_transition_reward(node)
            self._update_node_statistics(node, suffix_return)
            node = node.parent

    def _cache_child_nn_values(self, children: list[GymctsNode]) -> None:
        if not children:
            return
        if self.value_model is None:
            return

        missing_children = [
            child for child in children if not hasattr(child, "_selection_nn_value")
        ]
        if not missing_children:
            return
        observations = [obs_from_node(child) for child in missing_children]
        nn_predictions = self.value_model.predict_batch(observations)
        for child, nn_value in zip(missing_children, nn_predictions):
            child._selection_nn_value = float(nn_value)

    def _node_nn_value(self, node: GymctsNode) -> float:
        """Return NN prediction normalized (as output from model)."""
        if self.value_model is None or self.nn_weight <= 0.0:
            return 0.0
        return self._node_nn_prediction(node)  # Already normalized

    def _node_nn_prediction(self, node: GymctsNode) -> float:
        if self.value_model is None:
            return 0.0
        if not hasattr(node, "_selection_nn_value"):
            node._selection_nn_value = float(
                self.value_model.predict_batch([obs_from_node(node)])[0]
            )
        return float(node._selection_nn_value)

    def _rollout_value_model(self) -> ValueModel | None:
        if self.rollout_policy != "nn":
            return None
        return self.value_model

    def vanilla_mcts_search(
        self, search_start_node=None, num_simulations=10
    ) -> int | None:
        if search_start_node is None:
            search_start_node = self.search_root_node

        completed_simulations = 0
        while completed_simulations < num_simulations:
            batch_size = min(
                self.parallel_batch_size if self.parallel_workers > 1 else 1,
                num_simulations - completed_simulations,
            )
            leaf_nodes = []
            rollout_envs = []
            futures = []

            for _ in range(batch_size):
                leaf_node = self.navigate_to_leaf(from_node=search_start_node)
                if leaf_node.visit_count > 0 and not leaf_node.terminal:
                    self.expand_node(leaf_node)
                    if leaf_node.children:
                        leaf_node = self._select_best_tree_policy_child(leaf_node)

                self._load_state(leaf_node)
                leaf_nodes.append(leaf_node)

                rollout_env = copy.deepcopy(self.env)
                rollout_envs.append(rollout_env)
                if self.parallel_workers > 1:
                    if (
                        self.parallel_backend == "process"
                        and self._rollout_value_model() is not None
                    ):
                        futures.append(
                            self._get_parallel_executor().submit(
                                rollout_from_env_with_worker_model, rollout_env
                            )
                        )
                    else:
                        futures.append(
                            self._get_parallel_executor().submit(
                                rollout_from_env,
                                rollout_env,
                                self._rollout_value_model(),
                            )
                        )
                else:
                    futures.append(None)

            for leaf_node, rollout_env, future in zip(
                leaf_nodes,
                rollout_envs,
                futures,
            ):
                rollout_return_after_leaf = (
                    rollout_from_env(rollout_env, self._rollout_value_model())
                    if future is None
                    else float(future.result())
                )
                self._backpropagate_suffix_returns(
                    leaf_node,
                    rollout_return_after_leaf,
                )

            completed_simulations += batch_size

        if not search_start_node.children and not search_start_node.terminal:
            self.expand_node(search_start_node)
        if search_start_node.terminal or not search_start_node.children:
            return None

        self._record_exploitation_training_data(search_start_node)

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
        elif self._rollout_value_model() is not None:
            self._parallel_executor = ProcessPoolExecutor(
                max_workers=self.parallel_workers,
                initializer=initialize_rollout_worker,
                initargs=(rollout_worker_payload(self._rollout_value_model()),),
            )
        else:
            self._parallel_executor = ProcessPoolExecutor(
                max_workers=self.parallel_workers
            )
        return self._parallel_executor

    def shutdown_parallel_executor(self):
        if self._parallel_executor is not None:
            self._parallel_executor.shutdown(wait=True)
            self._parallel_executor = None

    def pop_training_data(self):
        observations = self.training_observations
        policies = self.training_policies
        targets = self.training_targets
        self.training_observations = []
        self.training_policies = []
        self.training_targets = []
        return observations, policies, targets


def evaluate_value_search(agent: ValueGuidedMCTSAgent, args):
    agent.reset()
    actions = agent.solve(num_simulations_per_step=args.simulations)
    agent._load_state(agent.search_root_node)

    return summarize_episode(
        "value-nmcts",
        agent.env,
        agent.last_info,
        agent.last_accumulated_reward,
        actions,
    )


def print_result_without_actions(result):
    print(
        f"{result['label']}: "
        f"round={result['round']}, "
        f"balance={result['balance']:.3f}, "
        f"balance_always={result['balance_always']:.3f}, "
        f"reward={result['accumulated_reward']:.3f}, "
        f"done_reason={result['done_reason']}"
    )


def parse_hidden_sizes(value: str) -> list[int]:
    hidden_sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not hidden_sizes:
        raise argparse.ArgumentTypeError("hidden sizes must not be empty")
    return hidden_sizes


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Iterative value-network guided MCTS for Oekolopoly."
    )
    parser.add_argument("--iterations", type=positive_int, default=5)
    parser.add_argument("--simulations", type=positive_int, default=5000)
    parser.add_argument("--parallel-workers", type=positive_int, default=12)
    parser.add_argument("--parallel-batch-size", type=positive_int, default=12)
    parser.add_argument(
        "--parallel-backend", choices=["process", "thread"], default="process"
    )
    parser.add_argument(
        "--tree-selection-policy", choices=["puct", "nn", "mcts"], default="puct"
    )
    parser.add_argument("--rollout-policy", choices=["nn", "random"], default="nn")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-tree", action="store_true")
    parser.add_argument("--render-tree-max-depth", type=int, default=2)

    parser.add_argument("--reward-start-round", type=int, default=10)
    parser.add_argument("--reward-weight-power", type=float, default=2.0)
    parser.add_argument("--survival-reward", type=float, default=0.25)

    parser.add_argument("--nn-start-weight", type=float, default=0.0)
    parser.add_argument("--nn-final-weight", type=float, default=0.4)
    parser.add_argument(
        "--exploration-weight",
        "--exploration-start-weight",
        "--exploration-final-weight",
        dest="exploration_weight",
        type=float,
        default=2.5,
    )
    parser.add_argument("--puct-prior-temperature", type=float, default=1.1)
    parser.add_argument("--training-window-iterations", type=int, default=5)
    parser.add_argument("--train-epochs", type=positive_int, default=8)
    parser.add_argument("--train-batch-size", type=positive_int, default=4096)
    parser.add_argument("--validation-fraction", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument(
        "--hidden-sizes",
        type=parse_hidden_sizes,
        default=parse_hidden_sizes("64,64,32,32"),
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--best-action-weight", type=float, default=0.99)
    parser.add_argument("--bo", action="store_true")
    parser.add_argument("--bo-trials", type=positive_int, default=25)
    parser.add_argument("--bo-initial-random", type=positive_int, default=6)
    parser.add_argument("--bo-candidates", type=positive_int, default=1024)
    parser.add_argument(
        "--bo-objective-metric",
        choices=["balance_always", "balance", "accumulated_reward", "round"],
        default="balance_always",
    )
    parser.add_argument(
        "--bo-results-path",
        type=Path,
        default=Path("models_value_nmcts_parallel") / "bo_results.jsonl",
    )
    return parser.parse_args()


def nn_weight_for_iteration(args, iteration_idx: int) -> float:
    if args.iterations <= 1:
        return args.nn_final_weight
    fraction = iteration_idx / (args.iterations - 1)
    return args.nn_start_weight + fraction * (
        args.nn_final_weight - args.nn_start_weight
    )


def exploration_weight_for_iteration(args, iteration_idx: int) -> float:
    return max(0.0, args.exploration_weight)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def sample_training_data(
    observations: list[np.ndarray],
    policies: list[np.ndarray],
    targets: list[float],
    max_samples: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    if len(targets) <= max_samples:
        return observations, policies, targets

    indices = np.random.choice(len(targets), size=max_samples, replace=False)
    return (
        [observations[i] for i in indices],
        [policies[i] for i in indices],
        [targets[i] for i in indices],
    )


def flatten_training_window(
    training_window: list[tuple[list[np.ndarray], list[np.ndarray], list[float]]],
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    observations: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    targets: list[float] = []
    for window_observations, window_policies, window_targets in training_window:
        observations.extend(window_observations)
        policies.extend(window_policies)
        targets.extend(window_targets)
    return observations, policies, targets


@dataclass(frozen=True)
class BOParam:
    name: str
    kind: str
    low: float
    high: float


def load_bo_search_space() -> list[BOParam]:
    return [
        BOParam(
            name=str(spec["name"]),
            kind=str(spec["type"]),
            low=float(spec["low"]),
            high=float(spec["high"]),
        )
        for spec in BO_SEARCH_SPACE
    ]


def decode_bo_point(x: np.ndarray, search_space: list[BOParam]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for value, param in zip(x, search_space):
        clipped = float(np.clip(value, 0.0, 1.0))
        if param.kind == "log_float":
            low = math.log(param.low)
            high = math.log(param.high)
            decoded = math.exp(low + clipped * (high - low))
        else:
            decoded = param.low + clipped * (param.high - param.low)

        if param.kind == "int":
            decoded = int(round(decoded))
            decoded = max(int(param.low), min(int(param.high), decoded))
        params[param.name] = decoded
    return params


def encode_bo_params(
    args: argparse.Namespace, search_space: list[BOParam]
) -> np.ndarray:
    values = []
    for param in search_space:
        raw_value = float(getattr(args, param.name))
        if param.kind == "log_float":
            low = math.log(param.low)
            high = math.log(param.high)
            encoded = (math.log(raw_value) - low) / (high - low)
        else:
            encoded = (raw_value - param.low) / (param.high - param.low)
        values.append(float(np.clip(encoded, 0.0, 1.0)))
    return np.asarray(values, dtype=np.float64)


def propose_bo_point(
    observed_x: list[np.ndarray],
    observed_y: list[float],
    search_space: list[BOParam],
    rng: np.random.Generator,
    candidate_count: int,
) -> np.ndarray:
    if len(observed_y) < 2:
        return rng.random(len(search_space))
    if np.std(observed_y) < 1.0e-8:
        return rng.random(len(search_space))

    x_train = torch.tensor(
        np.asarray(observed_x),
        dtype=torch.double,
    )
    y_train = torch.tensor(
        np.asarray(observed_y)[:, None],
        dtype=torch.double,
    )

    gp = SingleTaskGP(
        x_train,
        y_train,
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    candidates = torch.tensor(
        rng.random((candidate_count, len(search_space))),
        dtype=torch.double,
    )

    acquisition_fn = qLogExpectedImprovement(
        model=gp,
        best_f=float(max(observed_y)),
    )
    with torch.no_grad():
        acquisition = acquisition_fn(candidates.unsqueeze(-2))
    return candidates[int(torch.argmax(acquisition).item())].cpu().numpy()


def apply_bo_params(args: argparse.Namespace, params: dict[str, Any]) -> None:
    for name, value in params.items():
        setattr(args, name, value)


def metric_from_result(result: dict[str, Any], metric_name: str) -> float:
    if metric_name not in result:
        raise ValueError(
            f"Unknown BO objective metric '{metric_name}'. "
            f"Available metrics: {sorted(result.keys())}"
        )
    return float(result[metric_name])


def write_bo_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True) + "\n")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    start_time = time.perf_counter()
    args.device = resolve_device(args.device)
    print(args.seed)
    print(f"device={args.device}")
    set_global_seed(args.seed)
    args.model_path = args.model_path.resolve()

    env = make_value_nmcts_env(
        render_mode="ansi" if args.render else None,
        reward_start_round=args.reward_start_round,
        reward_weight_power=args.reward_weight_power,
        survival_reward=args.survival_reward,
    )
    input_dim = len(VALUE_OBS_SCALE)
    value_model = (
        ValueModel.load(
            args.model_path, learning_rate=args.learning_rate, device=args.device
        )
        if args.load_model
        else ValueModel(
            input_dim=input_dim,
            action_dim=ACTION_DIM,
            hidden_sizes=args.hidden_sizes,
            learning_rate=args.learning_rate,
            device=args.device,
        )
    )

    training_window: list[tuple[list[np.ndarray], list[np.ndarray], list[float]]] = []
    final_result: dict[str, Any] | None = None
    try:
        for iteration_idx in range(args.iterations):
            nn_weight = nn_weight_for_iteration(args, iteration_idx)
            exploration_weight = exploration_weight_for_iteration(args, iteration_idx)
            # Iteration 0: Use random rollout (NN not trained yet)
            # Iteration 1+: Use NN-guided rollout (NN has training data now)
            rollout_policy = "random" if iteration_idx == 0 else args.rollout_policy
            print(
                f"\nIteration {iteration_idx + 1}/{args.iterations}: "
                f"mcts_simulations={args.simulations}, "
                f"nn_weight={nn_weight:.3f}, "
                f"exploration_weight={exploration_weight:.3f}, "
                f"tree_selection_policy={args.tree_selection_policy}, "
                f"puct_prior_temperature={args.puct_prior_temperature:.3f}, "
                f"rollout_policy={rollout_policy}"
            )

            env = make_value_nmcts_env(
                render_mode="ansi" if args.render else None,
                reward_start_round=args.reward_start_round,
                reward_weight_power=args.reward_weight_power,
                survival_reward=args.survival_reward,
            )
            agent = ValueGuidedMCTSAgent(
                env=env,
                value_model=value_model,
                nn_weight=nn_weight,
                exploration_weight=exploration_weight,
                tree_selection_policy=args.tree_selection_policy,
                rollout_policy=rollout_policy,
                puct_prior_temperature=args.puct_prior_temperature,
                clear_mcts_tree_after_step=True,
                render_tree_after_step=args.render_tree,
                render_tree_max_depth=args.render_tree_max_depth,
                number_of_simulations_per_step=args.simulations,
                exclude_unvisited_nodes_from_render=True,
                score_variate="UCT_v0",
                best_action_weight=args.best_action_weight,
                parallel_workers=args.parallel_workers,
                parallel_batch_size=args.parallel_batch_size,
                parallel_backend=args.parallel_backend,
            )

            try:
                result = evaluate_value_search(agent, args)
                new_observations, new_policies, new_targets = agent.pop_training_data()
            finally:
                agent.shutdown_parallel_executor()

            training_window.append((new_observations, new_policies, new_targets))
            if args.training_window_iterations > 0:
                training_window = training_window[-args.training_window_iterations :]

            (
                dataset_observations,
                dataset_policies,
                dataset_targets,
            ) = flatten_training_window(training_window)
            train_observations, train_policies, train_targets = sample_training_data(
                dataset_observations,
                dataset_policies,
                dataset_targets,
                max_samples=MAX_TRAIN_SAMPLES,
            )
            metrics = value_model.train_on_dataset(
                train_observations,
                train_policies,
                train_targets,
                epochs=args.train_epochs,
                batch_size=args.train_batch_size,
                validation_fraction=args.validation_fraction,
            )
            value_model.save(args.model_path)

            print(
                f"New samples: {len(new_targets)}, "
                f"window_iterations={len(training_window)}, "
                f"window_samples={len(dataset_targets)}, "
                f"used_for_training={len(train_targets)}, "
                f"train_loss={metrics['train_loss']:.5f}, "
                f"train_policy_loss={metrics['train_policy_loss']:.5f}, "
                f"validation_loss={metrics['validation_loss']:.5f}, "
                f"validation_policy_loss={metrics['validation_policy_loss']:.5f}, "
                f"train_mae={metrics['train_mae']:.3f}, "
                f"validation_mae={metrics['validation_mae']:.3f}, "
                f"train_rmse={metrics['train_rmse']:.3f}, "
                f"validation_rmse={metrics['validation_rmse']:.3f}, "
                f"train_corr={metrics['train_corr']:.3f}, "
                f"validation_corr={metrics['validation_corr']:.3f}, "
                f"model={args.model_path}"
            )
            print_result_without_actions(result)
            final_result = result

            if args.render:
                agent.env.render()
    finally:
        elapsed_seconds = time.perf_counter() - start_time
        print(f"\nTotal runtime: {elapsed_seconds:.2f} seconds")

    if final_result is None:
        raise RuntimeError("Training did not produce an evaluation result.")
    final_result["runtime_seconds"] = elapsed_seconds
    return final_result


def run_bayesian_optimization(args: argparse.Namespace) -> dict[str, Any]:
    start_time = time.perf_counter()
    search_space = load_bo_search_space()
    rng = np.random.default_rng(args.seed)
    observed_x: list[np.ndarray] = []
    observed_y: list[float] = []
    best_payload: dict[str, Any] | None = None

    for trial_idx in range(args.bo_trials):
        if trial_idx == 0:
            x = encode_bo_params(args, search_space)
        elif trial_idx < args.bo_initial_random:
            x = rng.random(len(search_space))
        else:
            x = propose_bo_point(
                observed_x,
                observed_y,
                search_space,
                rng,
                candidate_count=args.bo_candidates,
            )

        params = decode_bo_point(x, search_space)
        trial_args = copy.deepcopy(args)
        apply_bo_params(trial_args, params)
        trial_args.load_model = False
        trial_args.render = False
        trial_args.render_tree = False
        trial_args.seed = args.seed
        trial_args.model_path = (
            args.model_path.parent
            / "bo_trials"
            / f"{args.model_path.stem}_trial_{trial_idx + 1:02d}.pt"
        )

        print(f"\nBO trial {trial_idx + 1}/{args.bo_trials}: {params}")
        result = run_training(trial_args)
        score = metric_from_result(result, args.bo_objective_metric)

        observed_x.append(x)
        observed_y.append(score)
        payload = {
            "trial": trial_idx + 1,
            "score": score,
            "objective_metric": args.bo_objective_metric,
            "params": params,
            "result": {key: value for key, value in result.items() if key != "actions"},
            "model_path": str(trial_args.model_path),
        }
        write_bo_result(args.bo_results_path, payload)

        if best_payload is None or score > float(best_payload["score"]):
            best_payload = payload
            print(f"New BO best: score={score:.3f}, " f"model={trial_args.model_path}")
        else:
            print(f"BO score={score:.3f}, best={best_payload['score']:.3f}")

    elapsed_seconds = time.perf_counter() - start_time
    if best_payload is None:
        raise RuntimeError("Bayesian optimization did not run any trials.")

    print(
        f"\nBO finished in {elapsed_seconds:.2f} seconds. "
        f"Best score={best_payload['score']:.3f}, "
        f"params={best_payload['params']}, "
        f"model={best_payload['model_path']}"
    )
    return best_payload


def main():
    args = parse_args()
    if args.bo:
        run_bayesian_optimization(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
