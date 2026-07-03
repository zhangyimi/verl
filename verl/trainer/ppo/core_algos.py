# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = ["register_adv_est", "get_adv_estimator_fn", "AdvantageEstimator"]

from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig
from verl.utils import as_torch_index, group_mean_std
from verl.utils.import_utils import deprecated
from verl.workers.config import ActorConfig

PolicyLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        torch.Tensor,  # response_mask
        str,  # loss_agg_mode
        Optional[DictConfig | ActorConfig],  # config
        torch.Tensor | None,  # rollout_log_probs
    ],
    tuple[torch.Tensor, dict[str, Any]],
]

POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}


def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
    """Register a policy loss function with the given name.

    Args:
        name (str): The name to register the policy loss function under.

    Returns:
        function: Decorator function that registers the policy loss function.
    """

    def decorator(func: PolicyLossFn) -> PolicyLossFn:
        POLICY_LOSS_REGISTRY[name] = func
        return func

    return decorator


def get_policy_loss_fn(name):
    """Get the policy loss with a given name.

    Args:
        name: `(str)`
            The name of the policy loss.

    Returns:
        `(callable)`: The policy loss function.
    """
    loss_name = name
    if loss_name not in POLICY_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(POLICY_LOSS_REGISTRY.keys())}"
        )
    return POLICY_LOSS_REGISTRY[loss_name]


class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GPG = "gpg"
    RLOO_VECTORIZED = "rloo_vectorized"
    GRPO_VECTORIZED = "grpo_vectorized"
    OPTIMAL_TOKEN_BASELINE = "optimal_token_baseline"
    TIR_OPTIMAL_TOKEN_BASELINE = "tir_optimal_token_baseline"


ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}


def register_adv_est(name_or_enum: str | AdvantageEstimator) -> Any:
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(
                f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}"
            )
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        """Update the KL coefficient based on current KL divergence.

        Args:
            current_kl (float): Current KL divergence value.
            n_steps (int): Number of steps taken.
        """
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        """Update method for fixed KL controller (no-op).

        Args:
            current_kl (float): Current KL divergence value (unused).
            n_steps (int): Number of steps taken (unused).
        """
        pass


def get_kl_controller(kl_ctrl):
    """Factory function to create appropriate KL controller based on configuration.

    Args:
        kl_ctrl: Configuration object containing KL controller settings.

    Returns:
        KL controller instance (FixedKLController or AdaptiveKLController).

    Raises:
        NotImplementedError: If controller type is not supported.
        AssertionError: If adaptive controller horizon is not positive.
    """
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


@register_adv_est(AdvantageEstimator.GAE)  # or simply: @register_adv_est("gae")
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        nextvalues = 0
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam_ = delta + gamma * lam * lastgaelam

            # skip values and TD-error on observation tokens
            nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
            lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam

            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(AdvantageEstimator.GRPO)  # or simply: @register_adv_est("grpo")
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.GRPO_VECTORIZED)
def compute_grpo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized GRPO（outcome-only）:
      For each group g:
      a_i = \\frac{r_i - \\mu_g}{\\sigma_g} (or without dividing by \\sigma_g),
      then broadcast the scalar across the token dimension (multiplied by response_mask).。
    """
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        g = as_torch_index(index, device=scores.device)
        mean_g, std_g, _ = group_mean_std(scores, g, eps=epsilon, device=scores.device)
        if norm_adv_by_std_in_grpo:
            scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
        else:
            scalars = scores - mean_g[g]
        advantages = scalars.unsqueeze(-1) * response_mask
        return advantages, advantages


@register_adv_est(AdvantageEstimator.GRPO_PASSK)  # or simply: @register_adv_est("grpo_passk")
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        config: (AlgoConfig) algorithm settings, which contains "norm_adv_by_std_in_grpo"

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    assert config is not None
    # if True, normalize advantage by std within group
    norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(
                    f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}."
                )
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


@register_adv_est(
    AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
)  # or simply: @register_adv_est("reinforce_plus_plus_baseline")
def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO)  # or simply: @register_adv_est("rloo")
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (
                    response_num - 1
                )
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.OPO)  # or simply: @register_adv_est("opo")
def compute_opo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.stack(id2score[idx])
                len_tensor = torch.stack(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS)  # or simply: @register_adv_est("reinforce_plus_plus")
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, config: Optional[AlgoConfig] = None, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    assert config is not None
    gamma = config.gamma
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.REMAX)  # or simply: @register_adv_est("remax")
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,
    reward_baselines: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.GPG)  # or simply: @register_adv_est("gpg")
def compute_gpg_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    f_norm: float = 1.0,
    alpha: float = 1.0,
    config=None,
    **kwargs,
):
    """
    Compute advantage for GPG, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.ndarray)`
            shape: (bs,)
        epsilon: (float)
        f_norm: (float)
        alpha: (float)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        m = torch.count_nonzero(scores)
        alpha = bsz / m.clamp(min=1)

        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = alpha * (scores[i] - id2mean[index[i]]) / (f_norm)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO_VECTORIZED)  # or simply: @register_adv_est("rloo_vectorized")
def compute_rloo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        inv = torch.from_numpy(np.unique(index, return_inverse=True)[1]).to(scores.device)

        c = torch.bincount(inv)[inv].to(scores.dtype)
        adv = ((c * scores - torch.bincount(inv, weights=scores)[inv]) / (c - 1).clamp_min(1)) * (c > 1)

        adv = adv.unsqueeze(-1) * response_mask

    return adv, adv


@register_adv_est(AdvantageEstimator.OPTIMAL_TOKEN_BASELINE)
def compute_optimal_token_baseline_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    old_log_probs: torch.Tensor,
    sum_pi_squared: torch.Tensor,
    rollout_is_weights: torch.Tensor = None,
    handle_zero_tail: bool = True,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages using Optimal Token Baseline (OTB).

    Unlike the group mean based baseline which uses a single baseline per trajectory,
    this computes a unique baseline for each timestep using cumulative path variance.

    Theory:
        For each timestep t in each prompt group:
            B_t* = E[G_t × W_t] / E[W_t]
        where W_t = Σ_{j=1}^t ||s_j||² (cumulative path-variance proxy)
        and ||s_j||² = 1 - 2π_j + Σπ²

    The cumulative sum W_t captures the "realized energy" of trajectory has been up to timestep t,
    giving higher weight to predicting rewards on high-variance paths.

    Args:
        token_level_rewards: Rewards at each token position [shape: (bs, response_length)]
        response_mask: Binary mask for valid tokens (1) vs padding (0) [shape: (bs, response_length)]
        index: Prompt indices for grouping trajectories from same prompt [shape: (bs,)]
        old_log_probs: Log probabilities from training policy during generation [shape: (bs, response_length)]
        sum_pi_squared: Sum of squared probabilities over vocabulary Σπ² [shape: (bs, response_length)]
        rollout_is_weights: Pre-computed IS weights for W correction [shape: (bs, response_length)],
            None if not using IS
        handle_zero_tail: If True, zero baselines will be set in the portion of the longest trajectory
            that extends beyond the second-longest trajectory in the prompt group.
            Default: True
        epsilon: Small constant for numerical stability (default: 1e-8)

    Returns:
        advantages: OTB advantage estimates [shape: (bs, response_length)]
        returns: Cumulative rewards (returns) from each position [shape: (bs, response_length)]

    Note on Rollout Importance Sampling:
        When rollout_is_weights is provided, W_t is scaled by ρ̄²(t) to minimize MSE under truncated IS:
            B_t* = Σ[G_t × ρ̄²(t) × W_t] / Σ[ρ̄²(t) × W_t]
    """
    with torch.no_grad():
        batch_size, seq_len = token_level_rewards.shape
        device = token_level_rewards.device

        # Compute returns (reward-to-go) for each timestep
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        # Step 1: Compute w_per_timestep = 1 - 2π_t + Σπ²)
        pi_t = torch.exp(old_log_probs)
        w_per_timestep = 1 - 2 * pi_t + sum_pi_squared

        # Step 2: Apply rollout importance sampling correction (if enabled)
        if rollout_is_weights is not None:
            # Scale W by ρ̄² to minimize MSE under truncated IS
            w_per_timestep = w_per_timestep * (rollout_is_weights**2)

        # Step 3: Compute cumulative path-variance proxy: W_t = Σ_{j=1}^t w_j
        # This measures accumulated variance from the start of the trajectory up to timestep t
        w_cumulative = (w_per_timestep * response_mask).cumsum(dim=-1)

        # Group trajectories by prompt
        prompt_groups = defaultdict(list)
        for i in range(batch_size):
            prompt_groups[index[i]].append(i)

        # Initialize baselines tensor [batch_size, seq_len]
        baselines = torch.zeros_like(returns)

        # Compute per-step baseline for each prompt group
        for _, trajectory_indices in prompt_groups.items():
            N = len(trajectory_indices)
            if N == 1:
                # Single trajectory - no baseline (advantage = return)
                continue

            traj_idx = torch.tensor(trajectory_indices, device=device)

            # Extract group data [N, seq_len]
            returns_group = returns[traj_idx]
            w_cumulative_group = w_cumulative[traj_idx]
            mask_group = response_mask[traj_idx]

            # Compute per-timestep baseline: B_t = Σ[G_t × W_t] / Σ[W_t]
            # where W_t = Σ_{j=1}^t ||s_j||² (cumulative path variance)
            # Shape: [seq_len]
            numerator = (returns_group * w_cumulative_group * mask_group).sum(dim=0)  # Sum over trajectories
            denominator = (w_cumulative_group * mask_group).sum(dim=0) + epsilon

            baseline_per_step = numerator / denominator  # [seq_len]

            # Assign to all trajectories in this group
            baselines[traj_idx] = baseline_per_step.unsqueeze(0).expand(N, -1)

            if handle_zero_tail:
                # Optionally zero out the portion of the longest trajectory that extends
                # beyond the second-longest trajectory in the prompt group.
                response_lengths = mask_group.sum(dim=-1)
                sorted_lengths, _ = torch.sort(response_lengths)
                max_length = int(sorted_lengths[-1].item())
                second_max_length = int(sorted_lengths[-2].item())
                max_length_idx = (response_lengths == max_length).nonzero(as_tuple=True)[0]
                if max_length_idx.numel() == 1 and max_length > second_max_length:
                    max_length_traj_idx = trajectory_indices[int(max_length_idx[0])]
                    baselines[max_length_traj_idx, second_max_length:] = 0.0

        # Compute advantages: A_t = G_t - B_t
        advantages = (returns - baselines) * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE)
def compute_multi_turn_optimal_token_baseline_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    old_log_probs: torch.Tensor,
    sum_pi_squared: torch.Tensor,
    rollout_is_weights: torch.Tensor = None,
    handle_zero_tail: bool = True,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages using Optimal Token Baseline (OTB).

    Unlike the group mean based baseline which uses a single baseline per trajectory,
    this computes a unique baseline for each timestep using cumulative path variance.

    Theory:
        For each timestep t in each prompt group:
            B_t* = E[G_t × W_t] / E[W_t]
        where W_t = Σ_{j=1}^t ||s_j||² (cumulative path-variance proxy)
        and ||s_j||² = 1 - 2π_j + Σπ²

    The cumulative sum W_t captures the "realized energy" of trajectory has been up to timestep t,
    giving higher weight to predicting rewards on high-variance paths.

    Args:
        token_level_rewards: Rewards at each token position [shape: (bs, response_length)]
        response_mask: Binary mask for valid tokens (1) vs padding (0) [shape: (bs, response_length)]
        index: Prompt indices for grouping trajectories from same prompt [shape: (bs,)]
        old_log_probs: Log probabilities from training policy during generation [shape: (bs, response_length)]
        sum_pi_squared: Sum of squared probabilities over vocabulary Σπ² [shape: (bs, response_length)]
        rollout_is_weights: Pre-computed IS weights for W correction [shape: (bs, response_length)],
            None if not using IS
        handle_zero_tail: If True, zero baselines will be set in the portion of the longest trajectory
            that extends beyond the second-longest trajectory in the prompt group.
            Default: False
        epsilon: Small constant for numerical stability (default: 1e-8)

    Returns:
        advantages: OTB advantage estimates [shape: (bs, response_length)]
        returns: Cumulative rewards (returns) from each position [shape: (bs, response_length)]

    Note on Rollout Importance Sampling:
        When rollout_is_weights is provided, W_t is scaled by ρ̄²(t) to minimize MSE under truncated IS:
            B_t* = Σ[G_t × ρ̄²(t) × W_t] / Σ[ρ̄²(t) × W_t]
    """
    with torch.no_grad():
        # Compute returns (reward-to-go) for each timestep
        token_returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        # Step 1: Compute w_per_timestep = 1 - 2π_t + Σπ²)
        pi_t = torch.exp(old_log_probs)
        w_per_timestep = 1 - 2 * pi_t + sum_pi_squared

        # Step 2: Apply rollout importance sampling correction (if enabled)
        if rollout_is_weights is not None:
            # Scale W by ρ̄² to minimize MSE under truncated IS
            w_per_timestep = w_per_timestep * (rollout_is_weights**2)

        # Step 3: Compute cumulative path-variance proxy: W_t = Σ_{j=1}^t w_j
        # This measures accumulated variance from the start of the trajectory up to timestep t
        w_cumulative = (w_per_timestep * response_mask).cumsum(dim=-1)

        # Step 4: Concatenate returns and w_cumulative for each trajectory
        # This allows us to compute baseline per timestep for each trajectory
        response_lengths = response_mask.sum(dim=-1).to(dtype=torch.long)  # [shape: (bs * n, )]
        max_response_length = int(response_lengths.max().item()) if response_lengths.numel() > 0 else 0
        all_w_values = w_cumulative.new_zeros(
            (len(response_lengths), max_response_length)
        )  # [shape: (bs * n, max_response_length)]
        all_returns = torch.zeros_like(all_w_values)
        for i in range(len(response_lengths)):
            length = int(response_lengths[i].item())
            if length == 0:
                continue
            mask = response_mask[i].bool()
            all_w_values[i, :length] = w_cumulative[i, mask]
            all_returns[i, :length] = token_returns[i, mask]

        # Group trajectories by prompt
        prompt_groups = defaultdict(list)
        for i in range(len(response_lengths)):
            if response_lengths[i] == 0:
                continue
            prompt_groups[index[i]].append(i)

        # Compute optimal baseline for each prompt group
        baselines = torch.zeros_like(all_returns)

        for _, trajectory_indices in prompt_groups.items():
            N = len(trajectory_indices)
            traj_idx = torch.tensor(trajectory_indices, device=all_returns.device)

            if N == 1:
                # Single trajectory - no baseline (keep original reward as advantage)
                baselines[traj_idx[0]] = 0.0
                continue

            # Extract group data
            w_group = all_w_values[traj_idx]  # [shape: (N, max_response_length)]
            R_group = all_returns[traj_idx]  # [shape: (N, max_response_length)]
            # Direct optimal baseline - single value for all in group
            b_star = (R_group * w_group).sum(dim=0) / (w_group.sum(dim=0) + epsilon)
            # Convert to match baselines dtype (epsilon can cause float64 promotion)
            baselines[traj_idx] = b_star.to(baselines.dtype)

            if handle_zero_tail:
                # Optionally zero out the portion of the longest trajectory that extends
                # beyond the second-longest trajectory in the prompt group.
                response_lengths_group = response_lengths[traj_idx]
                sorted_lengths, _ = torch.sort(response_lengths_group)
                max_length = int(sorted_lengths[-1].item())
                second_max_length = int(sorted_lengths[-2].item())
                max_length_idx = (response_lengths_group == max_length).nonzero(as_tuple=True)[0]
                if max_length_idx.numel() == 1 and max_length > second_max_length:
                    max_length_traj_idx = trajectory_indices[int(max_length_idx[0])]
                    baselines[max_length_traj_idx, second_max_length:] = 0.0

        # Compute advantages
        all_advantages = all_returns - baselines  # [shape: (bs * n, max_response_length)]

        advantages = torch.zeros_like(token_returns)  # [shape: (bs * n, turn * response_length)]
        for i in range(len(response_lengths)):
            if response_lengths[i] == 0:
                continue
            advantages[i, response_mask[i].bool()] = all_advantages[i, : response_lengths[i]]

        advantages = advantages * response_mask  # [shape: (bs * n * turn, response_length)]

    return advantages, token_returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    """Compute token-level rewards with KL penalty.

    Args:
        token_level_scores (torch.Tensor): Token-level reward scores.
        old_log_prob (torch.Tensor): Log probabilities from current policy.
        ref_log_prob (torch.Tensor): Log probabilities from reference policy.
        kl_ratio (float): KL penalty coefficient.

    Returns:
        torch.Tensor: Token-level rewards with KL penalty applied.
    """
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None,
):
    """
    Aggregate the loss across global batch to ensure the loss is invariant to fsdp/megatron parallelism.

    NOTE: The returned loss has different behaviors for different backend:
    - FSDP: the loss is directly used for backward.
    - Megatron: the loss should be scaled by `num_microbatches` and `cp_size` for pp schedule.

    Args:
        loss_mat: micro batch loss matrix, (bs, response_length)
        loss_mask: micro batch loss mask, (bs, response_length)
        loss_agg_mode: method to aggregate the loss matrix into a scalar
        dp_size: data parallel size
        batch_num_tokens: number of valid tokens in global batch
        global_batch_size: global batch size
        loss_scale_factor: scale factor for "seq-mean-token-sum-norm" mode. If None, uses loss_mask.shape[-1].
            Set this to a constant value to ensure consistent normalization throughout training.

    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        if batch_num_tokens is None:
            if dp_size > 1:
                raise ValueError("(global) batch_num_tokens is required when dp_size > 1")
            batch_num_tokens = loss_mask.sum()
        loss = verl_F.masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size
    elif loss_agg_mode in ["seq-mean-token-sum", "seq-mean-token-sum-norm"]:
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()  # exclude fully masked sequences
        if global_batch_size is None:
            if dp_size > 1:
                raise ValueError("global_batch_size is required when dp_size > 1")
            global_batch_size = seq_mask.sum()
        if loss_agg_mode == "seq-mean-token-sum-norm":
            if loss_scale_factor is None:
                horizon = loss_mask.shape[-1]
                loss_scale_factor = horizon
            if torch.is_tensor(loss_scale_factor):
                seq_scale = loss_scale_factor.to(device=seq_losses.device, dtype=seq_losses.dtype)
                # Guard against the adaptive tail-denom inverting into a length AMPLIFIER:
                # a sequence longer than its denom (e.g. tail_denom < actual response length)
                # gets seq_losses/denom weighted >1 "unit" -> positive-feedback length runaway,
                # the exact failure this length-norm is meant to prevent. Floor each per-seq
                # denom at its own token count so every sequence contributes <=1 unit (a
                # too-long sequence is divided by its own length -> neutral, never amplified).
                seq_token_count = loss_mask.sum(dim=-1).to(dtype=seq_scale.dtype)
                seq_scale = torch.maximum(seq_scale, seq_token_count)
                seq_losses = seq_losses / seq_scale.clamp_min(1e-8)
                loss_scale_factor = None
        loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size  # seq-mean
        if loss_agg_mode == "seq-mean-token-sum-norm":
            if loss_scale_factor is not None:
                loss /= loss_scale_factor
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_mask = torch.sum(loss_mask, dim=-1)  # per-sequence token count
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_mask + 1e-8)  # token-mean
        seq_mask = (seq_mask > 0).float()  # exclude fully masked sequences
        if global_batch_size is None:
            if dp_size > 1:
                raise ValueError("global_batch_size is required when dp_size > 1")
            global_batch_size = seq_mask.sum()
        loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size  # seq-mean
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def _masked_mean_or_zero(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=values.dtype)
    denom = mask_f.sum()
    if denom.detach().item() <= 0:
        return values.new_zeros(())
    return (values * mask_f).sum() / denom.clamp_min(1.0)


def _masked_sum_or_zero(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=values.dtype)
    if mask_f.sum().detach().item() <= 0:
        return values.new_zeros(())
    return (values * mask_f).sum()


def _global_sum_scalar(value: torch.Tensor, dp_group=None) -> torch.Tensor:
    value = value.clone()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM, group=dp_group)
    return value


def _align_token_objective(raw_delta: torch.Tensor, loss_type: str, beta: float) -> torch.Tensor:
    delta = torch.clamp(raw_delta, min=-20.0, max=20.0)
    if loss_type == "huber":
        return F.smooth_l1_loss(delta, torch.zeros_like(delta), reduction="none", beta=beta)
    if loss_type == "mse":
        return torch.square(delta)
    if loss_type == "abs":
        return torch.abs(delta)
    token_obj = torch.exp(delta) - delta - 1.0
    return torch.clamp(token_obj, min=-10.0, max=10.0)


def _delta_window(
    response_mask: torch.Tensor,
    raw_delta: torch.Tensor,
    delta_min: float | None,
    delta_max: float | None,
    abs_delta_min: float | None,
    abs_delta_max: float | None,
) -> torch.Tensor:
    window = response_mask.clone()
    if delta_min is not None:
        window = window & (raw_delta >= float(delta_min))
    if delta_max is not None:
        window = window & (raw_delta <= float(delta_max))
    abs_delta = torch.abs(raw_delta)
    if abs_delta_min is not None:
        window = window & (abs_delta >= float(abs_delta_min))
    if abs_delta_max is not None:
        window = window & (abs_delta <= float(abs_delta_max))
    return window


def _segment_mismatch_gate(
    response_mask: torch.Tensor,
    raw_delta: torch.Tensor,
    advantages: torch.Tensor,
    *,
    segment_size: int,
    neg_delta_threshold: float,
    neg_adv_max: float,
    neg_weight: float,
    severe_delta_threshold: float | None,
    bad_delta_threshold: float,
    bad_fraction_threshold: float | None,
    severe_weight: float,
    pos_delta_threshold: float,
    pos_adv_min: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    segment_size = max(int(segment_size), 1)
    neg_weight = min(max(float(neg_weight), 0.0), 1.0)
    severe_weight = min(max(float(severe_weight), 0.0), 1.0)

    response_mask_f = response_mask.to(dtype=raw_delta.dtype)
    pg_weight = torch.ones_like(raw_delta)
    neg_segment_mask = torch.zeros_like(response_mask, dtype=torch.bool)
    severe_segment_mask = torch.zeros_like(response_mask, dtype=torch.bool)
    pos_segment_mask = torch.zeros_like(response_mask, dtype=torch.bool)
    neg_segment_count = raw_delta.new_zeros(())
    severe_segment_count = raw_delta.new_zeros(())
    pos_segment_count = raw_delta.new_zeros(())

    seq_len = raw_delta.shape[-1]
    detached_delta = raw_delta.detach()
    detached_adv = advantages.detach()

    for start in range(0, seq_len, segment_size):
        end = min(start + segment_size, seq_len)
        seg_mask = response_mask[:, start:end]
        seg_mask_f = response_mask_f[:, start:end]
        token_count = seg_mask_f.sum(dim=-1)
        valid_seg = token_count > 0
        denom = token_count.clamp_min(1.0)

        seg_delta = (detached_delta[:, start:end] * seg_mask_f).sum(dim=-1) / denom
        seg_adv = (detached_adv[:, start:end] * seg_mask_f).sum(dim=-1) / denom
        seg_bad_fraction = (((detached_delta[:, start:end] < bad_delta_threshold) & seg_mask).float()).sum(
            dim=-1
        ) / denom

        neg_segment = valid_seg & (seg_delta < neg_delta_threshold) & (seg_adv < neg_adv_max)
        severe_segment = torch.zeros_like(neg_segment)
        if severe_delta_threshold is not None:
            severe_segment = severe_segment | (seg_delta < float(severe_delta_threshold))
        if bad_fraction_threshold is not None:
            severe_segment = severe_segment | (seg_bad_fraction > float(bad_fraction_threshold))
        severe_segment = valid_seg & severe_segment & (seg_adv < neg_adv_max)
        pos_segment = valid_seg & (seg_delta < pos_delta_threshold) & (seg_adv > pos_adv_min)

        neg_segment_count = neg_segment_count + neg_segment.to(dtype=raw_delta.dtype).sum()
        severe_segment_count = severe_segment_count + severe_segment.to(dtype=raw_delta.dtype).sum()
        pos_segment_count = pos_segment_count + pos_segment.to(dtype=raw_delta.dtype).sum()

        seg_weight = torch.ones_like(seg_delta)
        seg_weight = torch.where(neg_segment, torch.full_like(seg_weight, neg_weight), seg_weight)
        seg_weight = torch.where(severe_segment, torch.full_like(seg_weight, severe_weight), seg_weight)
        pg_weight[:, start:end] = pg_weight[:, start:end] * torch.where(
            seg_mask,
            seg_weight.unsqueeze(-1),
            torch.ones_like(pg_weight[:, start:end]),
        )
        neg_segment_mask[:, start:end] = neg_segment.unsqueeze(-1) & seg_mask
        severe_segment_mask[:, start:end] = severe_segment.unsqueeze(-1) & seg_mask
        pos_segment_mask[:, start:end] = pos_segment.unsqueeze(-1) & seg_mask

    return (
        pg_weight,
        neg_segment_mask,
        severe_segment_mask,
        pos_segment_mask,
        neg_segment_count,
        severe_segment_count,
        pos_segment_count,
    )


# Per-step GLOBAL response-length references, set by the actor worker once
# before the micro-batch loop.  Advantage P70 and sequence-normalization P95
# are separate controls; sharing one global P95 was the original bug.
_SEQ_NORM_P95_GLOBAL = None
_ADV_LENGTH_REF_GLOBAL = None


def set_seq_norm_p95_global(value):
    global _SEQ_NORM_P95_GLOBAL
    _SEQ_NORM_P95_GLOBAL = value


def set_adv_length_ref_global(value):
    global _ADV_LENGTH_REF_GLOBAL
    _ADV_LENGTH_REF_GLOBAL = value


def compute_length_quantile_refs(
    response_lengths: torch.Tensor,
    *,
    adv_enabled: bool,
    adv_quantile: float,
    seq_enabled: bool,
    seq_quantile: float,
) -> tuple[float | None, float | None]:
    """Compute independent advantage/seq-norm references from one global set.

    ``response_lengths`` must already contain all active sequences in the DP
    update.  Keeping this helper pure makes global-partition invariance easy to
    test without initializing torch.distributed.
    """

    lengths = response_lengths.detach().float().reshape(-1)
    lengths = lengths[torch.isfinite(lengths) & lengths.gt(0)]
    if lengths.numel() == 0:
        return None, None

    def quantile(q: float) -> float:
        q = min(max(float(q), 1e-6), 1.0)
        return float(torch.quantile(lengths, q).item())

    adv_ref = quantile(adv_quantile) if adv_enabled else None
    seq_ref = quantile(seq_quantile) if seq_enabled else None
    return adv_ref, seq_ref


def validate_global_length_population(
    rank_payloads: list[dict], *, batch_size_per_dp: int, dp_size: int
) -> list[float]:
    """Validate and flatten one whole actor-update response population.

    R3/ADV-P70 expects every dispatched sample to have a non-empty response.
    Failing here prevents a silently biased percentile when a rank drops or
    omits samples before PPO mini/micro-batch splitting.
    """

    rank_counts = [
        {
            "rank": rank,
            "total": int(payload.get("total", -1)),
            "active": int(payload.get("active", -1)),
            "length_count": len(payload.get("lengths", [])),
        }
        for rank, payload in enumerate(rank_payloads)
    ]
    expected_total = int(batch_size_per_dp) * int(dp_size)
    actual_total = sum(item["total"] for item in rank_counts)
    actual_active = sum(item["active"] for item in rank_counts)
    valid = (
        len(rank_payloads) == int(dp_size)
        and actual_total == expected_total
        and actual_active == actual_total
        and all(
            item["total"] == int(batch_size_per_dp)
            and item["active"] == item["total"]
            and item["length_count"] == item["active"]
            for item in rank_counts
        )
    )
    if not valid:
        raise RuntimeError(
            "adaptive length population invariant failed: "
            f"expected_total={expected_total}, actual_total={actual_total}, "
            f"actual_active={actual_active}, dp_size={dp_size}, rank_counts={rank_counts}"
        )
    return [float(value) for payload in rank_payloads for value in payload["lengths"]]


@torch.no_grad()
def apply_group_recentered_adv_length_norm(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    uids: Any,
    *,
    quantile: float,
    alpha: float,
    min_scale: float,
    max_scale: float,
    expected_group_size: int,
    mode: str = "neg_only",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply global length damping while preserving GRPO's per-group zero mean.

    GRPO produces one scalar advantage per response and centers those scalars
    within each prompt/uid group.  Scaling only negative responses inside the
    policy loss breaks that invariant and creates a positive group bias.  This
    function runs once on the global driver batch: compute P(q), damp long
    negative sequence advantages, then re-center every uid group before actor
    dispatch.  Returns are intentionally outside this function and unchanged.
    """

    if advantages.ndim != 2 or response_mask.shape != advantages.shape:
        raise ValueError(
            "driver_group_recenter requires advantages/response_mask with identical [batch,response] shape; "
            f"got advantages={tuple(advantages.shape)}, mask={tuple(response_mask.shape)}"
        )
    if mode != "neg_only":
        raise ValueError(f"driver_group_recenter supports only neg_only mode, got {mode!r}")
    if not (0.0 < float(quantile) <= 1.0):
        raise ValueError(f"advantage length quantile must be in (0,1], got {quantile}")
    if not np.isfinite(alpha) or float(alpha) < 0.0:
        raise ValueError(f"advantage length alpha must be finite and non-negative, got {alpha}")
    if not (0.0 <= float(min_scale) <= float(max_scale) <= 1.0):
        raise ValueError(f"advantage scales must satisfy 0 <= min <= max <= 1, got {min_scale}, {max_scale}")
    if int(expected_group_size) < 2:
        raise ValueError(f"GRPO driver_group_recenter requires group size >=2, got {expected_group_size}")

    mask = response_mask.bool()
    batch_size = advantages.shape[0]
    uid_values = np.asarray(uids, dtype=object).reshape(-1).tolist()
    if len(uid_values) != batch_size:
        raise ValueError(f"uid count {len(uid_values)} does not match advantage batch {batch_size}")
    invalid_uid_indices = []
    for index, uid in enumerate(uid_values):
        invalid = uid is None or (isinstance(uid, str) and not uid.strip())
        if isinstance(uid, (float, np.floating)) and not np.isfinite(uid):
            invalid = True
        try:
            hash(uid)
        except TypeError:
            invalid = True
        if invalid:
            invalid_uid_indices.append(index)
    if invalid_uid_indices:
        raise ValueError(f"missing/invalid uid values at batch indices {invalid_uid_indices}")

    response_lengths = mask.sum(dim=-1)
    empty_indices = torch.nonzero(response_lengths.eq(0), as_tuple=False).reshape(-1).cpu().tolist()
    if empty_indices:
        raise ValueError(f"driver_group_recenter found empty responses at batch indices {empty_indices}")
    if not torch.isfinite(advantages[mask]).all():
        raise ValueError("driver_group_recenter received non-finite active advantages")

    lengths_float = response_lengths.float()
    seq_advantages = (advantages.float() * mask).sum(dim=-1) / lengths_float
    deviations = (advantages.float() - seq_advantages.unsqueeze(-1)).abs().masked_fill(~mask, 0.0)
    tolerance = 1e-5 + 1e-4 * seq_advantages.abs()
    nonconstant = torch.nonzero(deviations.amax(dim=-1) > tolerance, as_tuple=False).reshape(-1)
    if nonconstant.numel() > 0:
        raise ValueError(
            "driver_group_recenter requires one constant GRPO advantage per response; "
            f"nonconstant batch indices={nonconstant.cpu().tolist()}"
        )

    groups: dict[Any, list[int]] = defaultdict(list)
    for index, uid in enumerate(uid_values):
        groups[uid].append(index)
    group_sizes = [len(indices) for indices in groups.values()]
    if not group_sizes or len(set(group_sizes)) != 1 or group_sizes[0] != int(expected_group_size):
        raise ValueError(
            "driver_group_recenter requires complete, equally-sized uid groups; "
            f"expected_group_size={expected_group_size}, observed_group_sizes={group_sizes}"
        )

    ref_len = float(torch.quantile(lengths_float, float(quantile)).item())
    if not np.isfinite(ref_len) or ref_len <= 0.0:
        raise ValueError(f"invalid global response-length reference: {ref_len}")
    length_scale = torch.pow(torch.clamp(ref_len / lengths_float, max=1.0), float(alpha))
    length_scale = torch.clamp(length_scale, min=float(min_scale), max=float(max_scale))
    negative = seq_advantages < 0
    effective_scale = torch.where(negative, length_scale, torch.ones_like(length_scale))
    scaled_seq_advantages = seq_advantages * effective_scale

    recentered = scaled_seq_advantages.clone()
    pre_means = []
    post_means = []
    for indices in groups.values():
        index = torch.tensor(indices, device=advantages.device, dtype=torch.long)
        pre_mean = scaled_seq_advantages[index].mean()
        recentered[index] = scaled_seq_advantages[index] - pre_mean
        post_mean = recentered[index].mean()
        pre_means.append(pre_mean)
        post_means.append(post_mean)

    pre_means_tensor = torch.stack(pre_means)
    post_means_tensor = torch.stack(post_means)
    if not torch.isfinite(recentered).all() or float(post_means_tensor.abs().max().item()) > 1e-5:
        raise RuntimeError(
            "driver_group_recenter failed to produce finite zero-mean groups; "
            f"max_abs_group_mean={float(post_means_tensor.abs().max().item())}"
        )

    output = torch.where(mask, recentered.to(advantages.dtype).unsqueeze(-1), advantages)
    actually_scaled = negative & effective_scale.lt(0.999)
    metrics = {
        "actor/adv_length_norm_ref_len": ref_len,
        "actor/adv_length_norm_quantile": float(quantile),
        "actor/adv_length_norm_scale_mean": float(effective_scale.mean().item()),
        "actor/adv_length_norm_scaled_seq_fraction": float(actually_scaled.float().mean().item()),
        "actor/adv_length_norm_group_mean_abs_pre": float(pre_means_tensor.abs().mean().item()),
        "actor/adv_length_norm_group_mean_abs_post": float(post_means_tensor.abs().mean().item()),
        "actor/adv_length_norm_group_count": float(len(groups)),
        "actor/adv_length_norm_group_size": float(group_sizes[0]),
    }
    return output, metrics


def _apply_gap_guard_policy_loss(
    *,
    config: ActorConfig,
    policy_loss_fn,
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str,
    rollout_is_weights: torch.Tensor | None,
    rollout_log_prob: torch.Tensor | None,
    dp_group=None,
) -> tuple[torch.Tensor, dict, torch.Tensor | None]:
    pg_advantages = advantages
    pg_response_mask = response_mask
    raw_delta = None
    seg_gate_pos = None
    stats = {}

    if rollout_log_prob is not None:
        rollout_log_prob = rollout_log_prob.detach()
        raw_delta = log_prob - rollout_log_prob

    # Independent global references.  P70 for advantage scaling must not
    # silently change seq-norm P95 (or vice versa).
    seq_norm_p95 = None
    adv_length_ref = None
    adv_length_stage = str(getattr(config, "adv_length_norm_stage", "loss"))
    if adv_length_stage not in {"disabled", "loss", "driver_group_recenter"}:
        raise ValueError(f"invalid adv_length_norm_stage={adv_length_stage!r}")
    _need_adv_ref = bool(getattr(config, "adv_length_norm_enable", False)) and adv_length_stage == "loss"
    _need_seq_ref = loss_agg_mode == "seq-mean-token-sum-norm" and bool(
        getattr(config, "seq_norm_adaptive_enable", False)
    )
    if _need_seq_ref:
        if _SEQ_NORM_P95_GLOBAL is not None:
            seq_norm_p95 = float(_SEQ_NORM_P95_GLOBAL)
        else:
            _p95_lengths = response_mask.to(dtype=log_prob.dtype).sum(dim=-1).clamp_min(1.0)
            _p95_active = _p95_lengths[response_mask.any(dim=-1)]
            if _p95_active.numel() > 0:
                _p95_q = min(max(float(getattr(config, "seq_norm_adaptive_quantile", 0.95)), 1e-6), 1.0)
                seq_norm_p95 = float(torch.quantile(_p95_active.detach().float(), _p95_q).item())

    if _need_adv_ref:
        if _ADV_LENGTH_REF_GLOBAL is not None:
            adv_length_ref = float(_ADV_LENGTH_REF_GLOBAL)
        else:
            _adv_lengths = response_mask.to(dtype=log_prob.dtype).sum(dim=-1).clamp_min(1.0)
            _adv_active = _adv_lengths[response_mask.any(dim=-1)]
            if _adv_active.numel() > 0:
                _adv_q = min(max(float(getattr(config, "adv_length_norm_quantile", 0.95)), 1e-6), 1.0)
                adv_length_ref = float(torch.quantile(_adv_active.detach().float(), _adv_q).item())

    if (
        raw_delta is not None
        and bool(getattr(config, "adv_length_norm_enable", False))
        and adv_length_stage == "loss"
    ):
        response_lengths = response_mask.to(dtype=log_prob.dtype).sum(dim=-1).clamp_min(1.0)
        # Follow the configured global/local batch quantile so the scale measures "how much
        # longer than the CURRENT typical length", instead of a fixed 4096 that everyone overshoots
        # late in training (which pins the neg-advantage scale at min_scale for all sequences).
        if adv_length_ref is not None:
            ref_len = max(float(adv_length_ref), 1.0)
        else:
            ref_len = max(float(getattr(config, "adv_length_norm_ref_len", 4096)), 1.0)
        alpha = max(float(getattr(config, "adv_length_norm_alpha", 0.5)), 0.0)
        min_scale = min(max(float(getattr(config, "adv_length_norm_min_scale", 0.5)), 0.0), 1.0)
        max_scale = max(float(getattr(config, "adv_length_norm_max_scale", 1.0)), min_scale)
        adv_scale = torch.pow(torch.clamp(ref_len / response_lengths, max=1.0), alpha)
        adv_scale = torch.clamp(adv_scale, min=min_scale, max=max_scale)
        mode = str(getattr(config, "adv_length_norm_mode", "neg_only")).strip().lower()
        scale_mat = adv_scale.unsqueeze(-1)
        if mode == "all":
            pg_advantages = advantages * scale_mat
        elif mode == "pos_only":
            pg_advantages = torch.where(advantages > 0, advantages * scale_mat, advantages)
        else:
            pg_advantages = torch.where(advantages < 0, advantages * scale_mat, advantages)
        seq_has_tokens = response_mask.any(dim=-1)
        stats["actor/adv_length_norm_scale_mean"] = _masked_mean_or_zero(adv_scale.float(), seq_has_tokens).item()
        stats["actor/adv_length_norm_scaled_seq_fraction"] = _masked_mean_or_zero(
            (adv_scale < 0.999).float(), seq_has_tokens
        ).item()
        stats["actor/adv_length_norm_ref_len"] = float(ref_len)
        stats["actor/adv_length_norm_quantile"] = float(
            getattr(config, "adv_length_norm_quantile", 0.95)
        )

    if raw_delta is not None and bool(getattr(config, "pg_mask_enable", False)):
        pg_mask_bad = torch.zeros_like(response_mask, dtype=torch.bool)
        pg_delta_min = getattr(config, "pg_mask_delta_min", None)
        pg_delta_max = getattr(config, "pg_mask_delta_max", None)
        if pg_delta_min is not None:
            pg_mask_bad = pg_mask_bad | (raw_delta.detach() < float(pg_delta_min))
        if pg_delta_max is not None:
            pg_mask_bad = pg_mask_bad | (raw_delta.detach() > float(pg_delta_max))
        pg_mask_bad = pg_mask_bad & response_mask
        if pg_delta_min is not None or pg_delta_max is not None:
            bad_weight = min(max(float(getattr(config, "pg_mask_bad_weight", 0.0)), 0.0), 1.0)
            pg_response_mask = response_mask.to(dtype=log_prob.dtype) * torch.where(
                pg_mask_bad,
                torch.full_like(log_prob, bad_weight),
                torch.ones_like(log_prob),
            )
            stats["actor/pg_mask_bad_fraction"] = _masked_mean_or_zero(pg_mask_bad.float(), response_mask).item()
            stats["actor/pg_mask_weight_mean"] = _masked_mean_or_zero(pg_response_mask.float(), response_mask).item()
            stats["actor/pg_mask_bad_weight"] = bad_weight

    seg_gate_enable = bool(getattr(config, "seg_gate_enable", False))
    seg_gate_pos_enable = bool(getattr(config, "seg_gate_pos_enable", False)) or bool(
        getattr(config, "vllm_align_segment_pos_enable", False)
    )
    if raw_delta is not None and (seg_gate_enable or seg_gate_pos_enable):
        (
            seg_weight,
            seg_gate_neg,
            seg_gate_severe,
            seg_gate_pos,
            seg_gate_neg_segment_count,
            seg_gate_severe_segment_count,
            seg_gate_pos_segment_count,
        ) = _segment_mismatch_gate(
            response_mask=response_mask,
            raw_delta=raw_delta.detach(),
            advantages=advantages,
            segment_size=int(getattr(config, "seg_gate_size", 128)),
            neg_delta_threshold=float(getattr(config, "seg_gate_neg_delta_threshold", -0.5)),
            neg_adv_max=float(getattr(config, "seg_gate_neg_adv_max", 0.0)),
            neg_weight=float(getattr(config, "seg_gate_neg_weight", 0.3)),
            severe_delta_threshold=getattr(config, "seg_gate_severe_delta_threshold", -1.5),
            bad_delta_threshold=float(getattr(config, "seg_gate_bad_delta_threshold", -6.0)),
            bad_fraction_threshold=getattr(config, "seg_gate_bad_fraction_threshold", 0.02),
            severe_weight=float(getattr(config, "seg_gate_severe_weight", 0.1)),
            pos_delta_threshold=float(getattr(config, "seg_gate_pos_delta_threshold", -0.5)),
            pos_adv_min=float(getattr(config, "seg_gate_pos_adv_min", 0.0)),
        )
        if seg_gate_enable:
            pg_response_mask = pg_response_mask.to(dtype=log_prob.dtype) * seg_weight
            stats["actor/seg_gate_weight_mean"] = _masked_mean_or_zero(seg_weight.float(), response_mask).item()
            stats["actor/seg_gate_neg_fraction"] = _masked_mean_or_zero(seg_gate_neg.float(), response_mask).item()
            stats["actor/seg_gate_severe_fraction"] = _masked_mean_or_zero(
                seg_gate_severe.float(), response_mask
            ).item()
            stats["actor/seg_gate_neg_weight"] = float(getattr(config, "seg_gate_neg_weight", 0.3))
            stats["actor/seg_gate_severe_weight"] = float(getattr(config, "seg_gate_severe_weight", 0.1))
            stats["actor/seg_gate_size"] = float(getattr(config, "seg_gate_size", 128))
            stats["actor/seg_gate_neg_segment_count"] = seg_gate_neg_segment_count.item()
            stats["actor/seg_gate_severe_segment_count"] = seg_gate_severe_segment_count.item()
        if seg_gate_pos is not None:
            stats["actor/seg_gate_pos_fraction"] = _masked_mean_or_zero(seg_gate_pos.float(), response_mask).item()
            stats["actor/seg_gate_pos_segment_count"] = seg_gate_pos_segment_count.item()

    if raw_delta is not None and bool(getattr(config, "seq_mismatch_gate_enable", False)):
        response_mask_f = response_mask.to(dtype=log_prob.dtype)
        seq_active = response_mask.any(dim=-1)
        seq_count = response_mask_f.sum(dim=-1).clamp_min(1.0)
        seq_mismatch_delta = ((old_log_prob.detach() - rollout_log_prob) * response_mask_f).sum(dim=-1) / seq_count
        seq_prox_delta = ((log_prob.detach() - old_log_prob.detach()) * response_mask_f).sum(dim=-1) / seq_count
        seq_adv = (advantages.detach() * response_mask_f).sum(dim=-1) / seq_count

        mismatch_bad = torch.zeros_like(seq_active, dtype=torch.bool)
        gate_delta_min = getattr(config, "seq_mismatch_gate_delta_min", None)
        gate_delta_max = getattr(config, "seq_mismatch_gate_delta_max", None)
        if gate_delta_min is not None:
            mismatch_bad = mismatch_bad | (seq_mismatch_delta < float(gate_delta_min))
        if gate_delta_max is not None:
            mismatch_bad = mismatch_bad | (seq_mismatch_delta > float(gate_delta_max))

        neg_prox_bad = torch.zeros_like(seq_active, dtype=torch.bool)
        neg_prox_delta_max = getattr(config, "seq_mismatch_gate_neg_prox_delta_max", None)
        if neg_prox_delta_max is not None:
            neg_adv_max = float(getattr(config, "seq_mismatch_gate_neg_adv_max", 0.0))
            neg_prox_bad = (seq_adv < neg_adv_max) & (seq_prox_delta > float(neg_prox_delta_max))

        bad_seq = seq_active & (mismatch_bad | neg_prox_bad)
        bad_weight = min(max(float(getattr(config, "seq_mismatch_gate_bad_weight", 0.3)), 0.0), 1.0)
        seq_weight = torch.where(
            bad_seq,
            torch.full_like(seq_mismatch_delta, bad_weight),
            torch.ones_like(seq_mismatch_delta),
        )
        pg_response_mask = pg_response_mask.to(dtype=log_prob.dtype) * torch.where(
            response_mask,
            seq_weight.unsqueeze(-1),
            torch.ones_like(pg_response_mask, dtype=log_prob.dtype),
        )
        stats["actor/seq_mismatch_gate_weight_mean"] = _masked_mean_or_zero(seq_weight.float(), seq_active).item()
        stats["actor/seq_mismatch_gate_bad_seq_fraction"] = _masked_mean_or_zero(bad_seq.float(), seq_active).item()
        stats["actor/seq_mismatch_gate_mismatch_bad_seq_fraction"] = _masked_mean_or_zero(
            (mismatch_bad & seq_active).float(), seq_active
        ).item()
        stats["actor/seq_mismatch_gate_neg_prox_bad_seq_fraction"] = _masked_mean_or_zero(
            (neg_prox_bad & seq_active).float(), seq_active
        ).item()
        stats["actor/seq_mismatch_gate_seq_mismatch_delta_mean"] = _masked_mean_or_zero(
            seq_mismatch_delta.detach(), seq_active
        ).item()
        stats["actor/seq_mismatch_gate_seq_prox_delta_mean"] = _masked_mean_or_zero(
            seq_prox_delta.detach(), seq_active
        ).item()
        stats["actor/seq_mismatch_gate_bad_weight"] = bad_weight
        if gate_delta_min is not None:
            stats["actor/seq_mismatch_gate_delta_min"] = float(gate_delta_min)
        if gate_delta_max is not None:
            stats["actor/seq_mismatch_gate_delta_max"] = float(gate_delta_max)
        if neg_prox_delta_max is not None:
            stats["actor/seq_mismatch_gate_neg_prox_delta_max"] = float(neg_prox_delta_max)

    saved_global_batch_info = dict(getattr(config, "global_batch_info", {}) or {})
    if hasattr(config, "global_batch_info"):
        if (
            loss_agg_mode == "seq-mean-token-sum-norm"
            and bool(getattr(config, "seq_norm_adaptive_enable", False))
        ):
            response_lengths = response_mask.to(dtype=log_prob.dtype).sum(dim=-1).clamp_min(1.0)
            seq_active = response_mask.any(dim=-1)
            active_lengths = response_lengths[seq_active]
            if active_lengths.numel() > 0:
                q = min(max(float(getattr(config, "seq_norm_adaptive_quantile", 0.95)), 1e-6), 1.0)
                p_batch = torch.quantile(active_lengths.detach().float(), q)
                # p_used = GLOBAL whole-batch P95 (computed once per update across the DP group);
                # falls back to this micro-batch's local P95 only if unavailable. Per-micro-batch
                # local P95 is unreliable late in training: token-budgeted micro-batches hold only
                # ~2-3 long sequences, so the local "P95" ~= the longest of 3. p_batch is kept as
                # the local-P95 stat for comparison.
                p_used = float(seq_norm_p95) if seq_norm_p95 is not None else float(p_batch.detach().item())

                # Bands: 2048, 4096, then ceil(p95 / 4096) * 4096, capped at the padded response
                # width (= max_response_length). The denom tracks length across the full range
                # instead of saturating at a fixed top band.
                max_resp = float(response_mask.shape[-1])
                if p_used < 2048.0:
                    short_denom = 2048.0
                elif p_used < 4096.0:
                    short_denom = 4096.0
                else:
                    _blocks = int(p_used // 4096.0)
                    if p_used > _blocks * 4096.0:
                        _blocks += 1
                    short_denom = min(float(_blocks) * 4096.0, max_resp)
                tail_denom = float(getattr(config, "seq_norm_adaptive_tail_denom", 20480.0))
                # Tail = sequences above the batch p95 (relative outliers).
                #   tail_power == 0 (default): legacy constant tail_denom (back-compat / "just cap").
                #   tail_power  > 0: outlier-damping. The denom grows super-linearly with how far L
                #     overshoots p95, so contribution ~ (p95/L)^(tail_power-1): a relative outlier
                #     (others 2K, you 20K) is progressively DOWN-weighted. Adaptive -- keyed on
                #     L/p95, so it only bites the rare long-vs-its-own-batch sequence, not a batch
                #     in which everything is long.
                tail_power = float(getattr(config, "seq_norm_adaptive_tail_power", 0.0))
                if tail_power > 0.0:
                    overshoot = (response_lengths / max(p_used, 1.0)).clamp_min(1.0)
                    tail_scale = short_denom * torch.pow(overshoot, tail_power)
                else:
                    tail_scale = torch.full_like(response_lengths, tail_denom)
                loss_scale_factor = torch.where(
                    response_lengths <= p_used,
                    torch.full_like(response_lengths, short_denom),
                    tail_scale,
                )
                config.global_batch_info["loss_scale_factor"] = loss_scale_factor
                stats["actor/seq_norm_adaptive_p_quantile"] = q
                stats["actor/seq_norm_adaptive_p_batch"] = float(p_batch.detach().item())
                stats["actor/seq_norm_adaptive_p_used"] = p_used
                stats["actor/seq_norm_adaptive_short_denom"] = short_denom
                stats["actor/seq_norm_adaptive_tail_denom"] = tail_denom
                stats["actor/seq_norm_adaptive_tail_fraction"] = _masked_mean_or_zero(
                    (response_lengths > p_used).float(), seq_active
                ).item()
                stats["actor/seq_norm_adaptive_denom_mean"] = _masked_mean_or_zero(
                    loss_scale_factor.float(), seq_active
                ).item()
        if loss_agg_mode == "token-mean":
            weighted_count = pg_response_mask.to(dtype=log_prob.dtype).sum().clamp_min(1.0)
            config.global_batch_info["batch_num_tokens"] = _global_sum_scalar(weighted_count, dp_group).clamp_min(1.0)
        elif loss_agg_mode == "seq-mean-token-mean":
            active_seq = (pg_response_mask.to(dtype=log_prob.dtype).sum(dim=-1) > 0).to(dtype=log_prob.dtype).sum()
            config.global_batch_info["global_batch_size"] = _global_sum_scalar(active_seq, dp_group).clamp_min(1.0)
    try:
        if raw_delta is not None and bool(getattr(config, "seq_tbpo_enable", False)):
            response_mask_f = response_mask.to(dtype=log_prob.dtype)
            seq_active = response_mask.any(dim=-1)
            seq_count = response_mask_f.sum(dim=-1).clamp_min(1.0)
            seq_adv = (pg_advantages * response_mask_f).sum(dim=-1) / seq_count
            seq_mask_weight = (pg_response_mask.to(dtype=log_prob.dtype) * response_mask_f).sum(dim=-1) / seq_count

            seq_prox_logratio = ((log_prob - old_log_prob.detach()) * response_mask_f).sum(dim=-1) / seq_count
            seq_prox_logratio = torch.clamp(seq_prox_logratio, min=-20.0, max=20.0)
            seq_ratio = torch.exp(seq_prox_logratio)

            seq_mismatch_logratio = ((old_log_prob.detach() - rollout_log_prob.detach()) * response_mask_f).sum(
                dim=-1
            ) / seq_count
            tis_cap = max(float(getattr(config, "seq_tbpo_tis_imp_ratio_cap", 2.0)), 1.0)
            log_tis_cap = torch.log(log_prob.new_tensor(tis_cap))
            seq_mismatch_weight = torch.exp(torch.clamp(seq_mismatch_logratio, min=-log_tis_cap, max=log_tis_cap))

            pos_high = max(float(getattr(config, "seq_tbpo_clip_ratio_high", 0.001)), 0.0)
            neg_low = max(float(getattr(config, "seq_tbpo_neg_clip_ratio_low", 0.001)), 0.0)
            neg_high = max(float(getattr(config, "seq_tbpo_neg_clip_ratio_high", 0.001)), 0.0)
            pos_clipped_ratio = torch.clamp(seq_ratio, min=0.0, max=1.0 + pos_high)
            neg_clipped_ratio = torch.clamp(seq_ratio, min=max(0.0, 1.0 - neg_low), max=1.0 + neg_high)
            seq_clipped_ratio = torch.where(seq_adv >= 0, pos_clipped_ratio, neg_clipped_ratio)

            seq_loss_vec = -seq_mismatch_weight * seq_clipped_ratio * seq_adv * seq_mask_weight
            pg_loss = _masked_mean_or_zero(seq_loss_vec, seq_active)
            pg_metrics = {
                "actor/pg_clipfrac": _masked_mean_or_zero(
                    ((seq_adv >= 0) & (seq_ratio > 1.0 + pos_high)).float(), seq_active
                ).item(),
                "actor/ppo_kl": _masked_mean_or_zero((-seq_prox_logratio).detach(), seq_active).item(),
                "actor/pg_clipfrac_lower": _masked_mean_or_zero(
                    ((seq_adv < 0) & ((seq_ratio < max(0.0, 1.0 - neg_low)) | (seq_ratio > 1.0 + neg_high))).float(),
                    seq_active,
                ).item(),
            }
            stats["actor/seq_tbpo_enable"] = 1.0
            stats["actor/seq_tbpo_seq_ratio_mean"] = _masked_mean_or_zero(seq_ratio.detach(), seq_active).item()
            stats["actor/seq_tbpo_seq_ratio_clipped_mean"] = _masked_mean_or_zero(
                seq_clipped_ratio.detach(), seq_active
            ).item()
            stats["actor/seq_tbpo_mismatch_weight_mean"] = _masked_mean_or_zero(
                seq_mismatch_weight.detach(), seq_active
            ).item()
            stats["actor/seq_tbpo_seq_prox_logratio_mean"] = _masked_mean_or_zero(
                seq_prox_logratio.detach(), seq_active
            ).item()
            stats["actor/seq_tbpo_seq_mismatch_logratio_mean"] = _masked_mean_or_zero(
                seq_mismatch_logratio.detach(), seq_active
            ).item()
            stats["actor/seq_tbpo_seq_mask_weight_mean"] = _masked_mean_or_zero(
                seq_mask_weight.detach(), seq_active
            ).item()
            stats["actor/seq_tbpo_clip_ratio_high"] = pos_high
            stats["actor/seq_tbpo_neg_clip_ratio_low"] = neg_low
            stats["actor/seq_tbpo_neg_clip_ratio_high"] = neg_high
            stats["actor/seq_tbpo_tis_imp_ratio_cap"] = tis_cap
        else:
            pg_loss, pg_metrics = policy_loss_fn(
                old_log_prob=old_log_prob,
                log_prob=log_prob,
                advantages=pg_advantages,
                response_mask=pg_response_mask,
                loss_agg_mode=loss_agg_mode,
                config=config,
                rollout_is_weights=rollout_is_weights,
            )
    finally:
        if hasattr(config, "global_batch_info"):
            config.global_batch_info.clear()
            config.global_batch_info.update(saved_global_batch_info)

    stats.update(pg_metrics)
    response_mask_f = response_mask.to(dtype=log_prob.dtype)
    token_weight = pg_response_mask.to(dtype=log_prob.dtype) * response_mask_f
    pressure = torch.abs(pg_advantages.detach()) * token_weight
    total_pressure = pressure.sum().clamp_min(1e-12)
    stats["actor_diag/pg_effective_token_fraction"] = _masked_mean_or_zero(token_weight, response_mask).item()
    stats["actor_diag/pg_pressure_share_neg_adv"] = (
        _masked_sum_or_zero(pressure, (advantages.detach() < 0) & response_mask) / total_pressure
    ).item()
    stats["actor_diag/pg_pressure_share_pos_adv"] = (
        _masked_sum_or_zero(pressure, (advantages.detach() > 0) & response_mask) / total_pressure
    ).item()

    return pg_loss, stats, seg_gate_pos


def _add_gap_guard_aux_losses(
    *,
    config: ActorConfig,
    policy_loss: torch.Tensor,
    log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor | None,
    response_mask: torch.Tensor,
    advantages: torch.Tensor,
    loss_agg_mode: str,
    seg_gate_pos: torch.Tensor | None,
    dp_group=None,
    local_segment_align_step_mean: torch.Tensor | None = None,
    local_segment_align_step_clip_ratio: torch.Tensor | None = None,
    local_segment_align_tail_mass: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    if rollout_log_prob is None:
        return policy_loss, {}

    rollout_log_prob = rollout_log_prob.detach()
    raw_delta = log_prob - rollout_log_prob
    stats = {}

    def add_align_loss(prefix: str, coef: float, *, hard: bool = False) -> None:
        nonlocal policy_loss
        if coef <= 0.0:
            return
        key = "vllm_align_hard" if hard else "vllm_align"
        loss_type = str(getattr(config, f"{key}_loss_type", "huber" if hard else "k3")).strip().lower()
        token_obj = _align_token_objective(raw_delta, loss_type, float(getattr(config, f"{key}_huber_beta", 1.0)))
        loss_cap = getattr(config, f"{key}_loss_cap", None)
        if loss_cap is not None:
            token_obj = torch.clamp(token_obj, max=float(loss_cap))
        align_window = _delta_window(
            response_mask=response_mask,
            raw_delta=raw_delta,
            delta_min=getattr(config, f"{key}_delta_min", None),
            delta_max=getattr(config, f"{key}_delta_max", None),
            abs_delta_min=getattr(config, f"{key}_abs_delta_min", None),
            abs_delta_max=getattr(config, f"{key}_abs_delta_max", None),
        )
        adv_min = getattr(config, f"{key}_adv_min", None)
        adv_max = getattr(config, f"{key}_adv_max", None)
        if adv_min is not None:
            align_window = align_window & (advantages.detach() >= float(adv_min))
        if adv_max is not None:
            align_window = align_window & (advantages.detach() <= float(adv_max))
        if (not hard) and bool(getattr(config, "vllm_align_segment_pos_enable", False)):
            if seg_gate_pos is None:
                align_window = align_window & torch.zeros_like(response_mask, dtype=torch.bool)
            else:
                align_window = align_window & seg_gate_pos

        token_weight = align_window.to(dtype=token_obj.dtype)
        align_count = token_weight.sum()
        response_count = response_mask.to(dtype=token_obj.dtype).sum().clamp_min(1.0)
        if bool(getattr(config, f"{key}_normalize_by_window", False)):
            min_fraction = float(getattr(config, f"{key}_min_window_fraction", 0.0)) if not hard else 0.0
            denom = torch.maximum(align_count, response_count * max(min_fraction, 0.0)).clamp_min(1.0)
            align_loss = (token_obj * token_weight).sum() / denom
            if align_count.detach().item() <= 0:
                align_loss = token_obj.sum() * 0.0
        else:
            align_loss = agg_loss(
                loss_mat=token_obj * token_weight,
                loss_mask=response_mask,
                loss_agg_mode=loss_agg_mode,
                **config.global_batch_info,
            )
        policy_loss = policy_loss + coef * align_loss

        window_fraction = _masked_mean_or_zero(align_window.float(), response_mask)
        gap_mean = _masked_mean_or_zero(raw_delta.detach(), align_window)
        stats[f"{prefix}_loss"] = align_loss.detach().item()
        stats[f"{prefix}_coef"] = coef
        stats[f"{prefix}_window_fraction"] = window_fraction.detach().item()
        stats[f"{prefix}_logprob_gap_mean"] = gap_mean.detach().item()
        stats[f"{prefix}_active_weight_sum"] = align_count.detach().item()
        stats[f"{prefix}_denom_token_count"] = (
            torch.maximum(align_count, response_count * float(getattr(config, f"{key}_min_window_fraction", 0.0)))
            .clamp_min(1.0)
            .detach()
            .item()
        )
        if (not hard) and bool(getattr(config, "vllm_align_segment_pos_enable", False)):
            stats[f"{prefix}_segment_pos_gated"] = 1.0

    if bool(getattr(config, "vllm_align_enable", False)):
        add_align_loss("actor/vllm_align", float(getattr(config, "vllm_align_coef", 0.0)), hard=False)
    if bool(getattr(config, "vllm_align_hard_enable", False)):
        add_align_loss("actor/vllm_align_hard", float(getattr(config, "vllm_align_hard_coef", 0.0)), hard=True)

    if bool(getattr(config, "delta_mean_guard_enable", False)):
        coef = float(getattr(config, "delta_mean_guard_coef", 0.0))
        if coef > 0.0:
            target = float(getattr(config, "delta_mean_guard_target", -0.01))
            mode = str(getattr(config, "delta_mean_guard_mode", "token")).strip().lower()
            guard_mask = _delta_window(
                response_mask=response_mask,
                raw_delta=raw_delta,
                delta_min=getattr(config, "delta_mean_guard_delta_min", None),
                delta_max=getattr(config, "delta_mean_guard_delta_max", None),
                abs_delta_min=None,
                abs_delta_max=None,
            )
            adv_min = getattr(config, "delta_mean_guard_adv_min", None)
            adv_max = getattr(config, "delta_mean_guard_adv_max", None)
            if adv_min is not None:
                guard_mask = guard_mask & (advantages.detach() >= float(adv_min))
            if adv_max is not None:
                guard_mask = guard_mask & (advantages.detach() <= float(adv_max))

            guard_weight = guard_mask.to(dtype=raw_delta.dtype)
            guard_count = guard_weight.sum()
            if guard_count.detach().item() <= 0:
                guard_loss = raw_delta.sum() * 0.0
                guard_mean = raw_delta.sum().detach() * 0.0
                active_seq_fraction = raw_delta.sum().detach() * 0.0
            elif mode == "seq":
                seq_count = guard_weight.sum(dim=-1)
                seq_active = seq_count > 0
                seq_delta = (raw_delta * guard_weight).sum(dim=-1) / seq_count.clamp_min(1.0)
                seq_penalty = F.relu(target - seq_delta).pow(2)
                guard_loss = _masked_mean_or_zero(seq_penalty, seq_active)
                guard_mean = _masked_mean_or_zero(seq_delta.detach(), seq_active)
                active_seq_fraction = _masked_mean_or_zero(seq_active.float(), response_mask.any(dim=-1))
            else:
                guard_mean = (raw_delta * guard_weight).sum() / guard_count.clamp_min(1.0)
                guard_loss = F.relu(target - guard_mean).pow(2)
                active_seq_fraction = _masked_mean_or_zero(
                    (guard_weight.sum(dim=-1) > 0).float(), response_mask.any(dim=-1)
                )

            loss_cap = getattr(config, "delta_mean_guard_loss_cap", None)
            if loss_cap is not None:
                guard_loss = torch.clamp(guard_loss, max=float(loss_cap))
            policy_loss = policy_loss + coef * guard_loss
            stats["actor/delta_mean_guard_loss"] = guard_loss.detach().item()
            stats["actor/delta_mean_guard_coef"] = coef
            stats["actor/delta_mean_guard_target"] = target
            stats["actor/delta_mean_guard_mean_delta"] = guard_mean.detach().item()
            stats["actor/delta_mean_guard_active_fraction"] = (
                _masked_mean_or_zero(guard_mask.float(), response_mask).detach().item()
            )
            stats["actor/delta_mean_guard_active_seq_fraction"] = active_seq_fraction.detach().item()
            stats["actor/delta_mean_guard_contribution"] = (coef * guard_loss.detach()).item()

    if bool(getattr(config, "local_segment_align_enable", False)):
        coef = float(getattr(config, "local_segment_align_coef", 0.0))
        if coef > 0.0:
            target = float(getattr(config, "local_segment_align_target", -0.03))
            segment_size = max(int(getattr(config, "local_segment_align_size", 128)), 1)
            response_mask_f = response_mask.to(dtype=raw_delta.dtype)
            response_lengths = response_mask_f.sum(dim=-1).clamp_min(1.0)
            seq_active = response_mask.any(dim=-1)
            active_lengths = response_lengths[seq_active]

            gate_enabled = bool(getattr(config, "local_segment_align_length_gate_enable", False))
            gate_quantile = min(
                max(float(getattr(config, "local_segment_align_length_gate_quantile", 0.95)), 1e-6), 1.0
            )
            gate_min = float(getattr(config, "local_segment_align_length_gate_min", 8192.0))
            gate_mean_min = float(getattr(config, "local_segment_align_length_gate_mean_min", 7000.0))
            gate_clip_ratio_min = float(getattr(config, "local_segment_align_length_gate_clip_ratio_min", 0.02))
            if active_lengths.numel() > 0:
                gate_p = torch.quantile(active_lengths.detach().float(), gate_quantile)
                gate_p_value = float(gate_p.detach().item())
            else:
                gate_p_value = float("nan")

            local_seq_count = seq_active.to(dtype=raw_delta.dtype).sum()
            local_token_sum = (response_lengths * seq_active.to(dtype=raw_delta.dtype)).sum()
            max_response_len = float(response_mask.shape[-1])
            local_clip_count = ((response_lengths >= max_response_len) & seq_active).to(dtype=raw_delta.dtype).sum()
            global_seq_count = _global_sum_scalar(local_seq_count, dp_group).clamp_min(1.0)
            global_token_sum = _global_sum_scalar(local_token_sum, dp_group)
            global_clip_count = _global_sum_scalar(local_clip_count, dp_group)
            gate_mean = global_token_sum / global_seq_count
            gate_clip_ratio = global_clip_count / global_seq_count
            if gate_enabled:
                gate_active_tensor = (gate_mean >= gate_mean_min) & (gate_clip_ratio >= gate_clip_ratio_min)
                gate_factor = gate_active_tensor.to(dtype=raw_delta.dtype)
                gate_active = bool(gate_active_tensor.detach().item())
            else:
                gate_factor = raw_delta.new_tensor(1.0)
                gate_active = True

            adaptive_tail_enabled = bool(getattr(config, "local_segment_align_adaptive_tail_enable", False))
            adaptive_tail_clip_gate_enabled = bool(
                getattr(config, "local_segment_align_adaptive_tail_clip_gate_enable", False)
            )
            adaptive_tail_uniform_weight = bool(
                getattr(config, "local_segment_align_adaptive_tail_uniform_weight", False)
            )
            adaptive_offset = float(getattr(config, "local_segment_align_adaptive_tail_offset", 4096.0))
            adaptive_min_start = float(getattr(config, "local_segment_align_adaptive_tail_min_start", 8192.0))
            adaptive_max_start = float(getattr(config, "local_segment_align_adaptive_tail_max_start", 14336.0))
            adaptive_width = max(float(getattr(config, "local_segment_align_adaptive_tail_width", 4096.0)), 1e-6)
            adaptive_tail_mass_min = float(getattr(config, "local_segment_align_adaptive_tail_mass_min", 0.02))
            adaptive_clip_ratio_min = float(
                getattr(config, "local_segment_align_adaptive_tail_clip_ratio_min", gate_clip_ratio_min)
            )
            if local_segment_align_step_mean is not None and local_segment_align_step_mean.numel() > 0:
                step_mean = local_segment_align_step_mean.to(dtype=raw_delta.dtype).flatten()[0]
            else:
                step_mean = gate_mean.detach().to(dtype=raw_delta.dtype)
            if local_segment_align_step_clip_ratio is not None and local_segment_align_step_clip_ratio.numel() > 0:
                step_clip_ratio = local_segment_align_step_clip_ratio.to(dtype=raw_delta.dtype).flatten()[0]
            else:
                step_clip_ratio = gate_clip_ratio.detach().to(dtype=raw_delta.dtype)
            adaptive_start = torch.clamp(
                step_mean + raw_delta.new_tensor(adaptive_offset), min=adaptive_min_start, max=adaptive_max_start
            )
            adaptive_full = adaptive_start + raw_delta.new_tensor(adaptive_width)
            length_weight = torch.clamp((response_lengths - adaptive_start) / adaptive_width, min=0.0, max=1.0)
            length_weight = torch.where(seq_active, length_weight, torch.zeros_like(length_weight))
            if local_segment_align_tail_mass is not None and local_segment_align_tail_mass.numel() > 0:
                tail_mass = local_segment_align_tail_mass.to(dtype=raw_delta.dtype).flatten()[0]
            else:
                tail_mass = _masked_mean_or_zero(length_weight, seq_active)
            if adaptive_tail_enabled:
                adaptive_gate_active_tensor = tail_mass >= adaptive_tail_mass_min
                if adaptive_tail_clip_gate_enabled:
                    adaptive_gate_active_tensor = adaptive_gate_active_tensor & (step_clip_ratio >= adaptive_clip_ratio_min)
                adaptive_gate_factor = adaptive_gate_active_tensor.to(dtype=raw_delta.dtype)
                gate_factor = gate_factor * adaptive_gate_factor
                adaptive_gate_active = bool(adaptive_gate_active_tensor.detach().item())
                if adaptive_tail_uniform_weight:
                    length_weight = torch.where(
                        seq_active, torch.ones_like(length_weight), torch.zeros_like(length_weight)
                    )
            else:
                adaptive_gate_factor = raw_delta.new_tensor(1.0)
                adaptive_gate_active = False
                length_weight = torch.ones_like(response_lengths, dtype=raw_delta.dtype)
                tail_mass = _masked_mean_or_zero(length_weight, seq_active)

            kl_gate_enabled = bool(getattr(config, "local_segment_align_kl_gate_enable", False))
            kl_gate_start = float(getattr(config, "local_segment_align_kl_gate_start", 0.01))
            kl_gate_full = float(getattr(config, "local_segment_align_kl_gate_full", 0.02))
            kl_gate_min_factor = float(getattr(config, "local_segment_align_kl_gate_min_factor", 0.0))
            kl_gate_max_factor = float(getattr(config, "local_segment_align_kl_gate_max_factor", 1.0))
            kl_gate_min_factor = min(max(kl_gate_min_factor, 0.0), 1.0)
            kl_gate_max_factor = min(max(kl_gate_max_factor, 0.0), 1.0)
            kl_gate_max_factor = max(kl_gate_max_factor, kl_gate_min_factor)
            kl_gate_value = -_masked_mean_or_zero(raw_delta.detach(), response_mask)
            if kl_gate_enabled:
                denom = max(kl_gate_full - kl_gate_start, 1e-6)
                kl_gate_progress = torch.clamp((kl_gate_value - kl_gate_start) / denom, min=0.0, max=1.0)
                kl_gate_factor = raw_delta.new_tensor(kl_gate_min_factor) + kl_gate_progress.to(
                    dtype=raw_delta.dtype
                ) * (kl_gate_max_factor - kl_gate_min_factor)
                gate_factor = gate_factor * kl_gate_factor
            else:
                kl_gate_factor = raw_delta.new_tensor(1.0)
                kl_gate_progress = raw_delta.new_tensor(1.0)

            seq_len = raw_delta.shape[-1]
            segment_losses = []
            segment_deltas = []
            segment_active = []
            segment_unweighted_active = []
            for start in range(0, seq_len, segment_size):
                end = min(start + segment_size, seq_len)
                seg_mask_f = response_mask_f[:, start:end]
                token_count = seg_mask_f.sum(dim=-1)
                valid_seg = token_count > 0
                seg_delta = (raw_delta[:, start:end] * seg_mask_f).sum(dim=-1) / token_count.clamp_min(1.0)
                seg_penalty = F.relu(target - seg_delta).pow(2)
                loss_cap = getattr(config, "local_segment_align_loss_cap", None)
                if loss_cap is not None:
                    seg_penalty = torch.clamp(seg_penalty, max=float(loss_cap))
                weighted_penalty = seg_penalty * length_weight
                segment_losses.append(torch.where(valid_seg, weighted_penalty, torch.zeros_like(weighted_penalty)))
                segment_deltas.append(seg_delta.detach())
                segment_active.append(valid_seg)
                segment_unweighted_active.append(valid_seg & (seg_penalty.detach() > 0))

            all_losses = torch.stack(segment_losses, dim=-1)
            all_deltas = torch.stack(segment_deltas, dim=-1)
            all_valid = torch.stack(segment_active, dim=-1)
            all_unweighted_active = torch.stack(segment_unweighted_active, dim=-1)
            valid_count = all_valid.to(dtype=raw_delta.dtype).sum()
            if valid_count.detach().item() <= 0:
                align_loss = raw_delta.sum() * 0.0
                align_loss_raw = raw_delta.sum() * 0.0
                mean_delta = raw_delta.sum().detach() * 0.0
                active_fraction = raw_delta.sum().detach() * 0.0
                weighted_active_fraction = raw_delta.sum().detach() * 0.0
            else:
                align_loss_raw = all_losses.sum() / valid_count.clamp_min(1.0)
                align_loss = align_loss_raw * gate_factor
                mean_delta = _masked_mean_or_zero(all_deltas, all_valid)
                active_fraction = _masked_mean_or_zero(all_unweighted_active.float(), all_valid)
                weighted_active_fraction = _masked_mean_or_zero((all_losses.detach() > 0).float(), all_valid)

            policy_loss = policy_loss + coef * align_loss
            stats["actor/local_segment_align_loss"] = align_loss.detach().item()
            stats["actor/local_segment_align_loss_raw"] = align_loss_raw.detach().item()
            stats["actor/local_segment_align_coef"] = coef
            stats["actor/local_segment_align_contribution"] = (coef * align_loss.detach()).item()
            stats["actor/local_segment_align_target"] = target
            stats["actor/local_segment_align_size"] = float(segment_size)
            stats["actor/local_segment_align_mean_delta"] = mean_delta.detach().item()
            stats["actor/local_segment_align_active_segment_fraction"] = active_fraction.detach().item()
            stats["actor/local_segment_align_weighted_segment_fraction"] = weighted_active_fraction.detach().item()
            stats["actor/local_segment_align_valid_segment_count"] = valid_count.detach().item()
            stats["actor/local_segment_align_length_gate_enable"] = float(gate_enabled)
            stats["actor/local_segment_align_length_gate_active"] = float(gate_active)
            stats["actor/local_segment_align_length_gate_quantile"] = gate_quantile
            stats["actor/local_segment_align_length_gate_min"] = gate_min
            stats["actor/local_segment_align_length_gate_p"] = gate_p_value
            stats["actor/local_segment_align_length_gate_mean"] = gate_mean.detach().item()
            stats["actor/local_segment_align_length_gate_mean_min"] = gate_mean_min
            stats["actor/local_segment_align_length_gate_clip_ratio"] = gate_clip_ratio.detach().item()
            stats["actor/local_segment_align_length_gate_clip_ratio_min"] = gate_clip_ratio_min
            stats["actor/local_segment_align_adaptive_tail_enable"] = float(adaptive_tail_enabled)
            stats["actor/local_segment_align_adaptive_gate_active"] = float(adaptive_gate_active)
            stats["actor/local_segment_align_adaptive_gate_factor"] = adaptive_gate_factor.detach().item()
            stats["actor/local_segment_align_adaptive_start"] = adaptive_start.detach().item()
            stats["actor/local_segment_align_adaptive_full"] = adaptive_full.detach().item()
            stats["actor/local_segment_align_adaptive_tail_mass"] = tail_mass.detach().item()
            stats["actor/local_segment_align_adaptive_tail_mass_min"] = adaptive_tail_mass_min
            stats["actor/local_segment_align_adaptive_step_mean"] = step_mean.detach().item()
            stats["actor/local_segment_align_adaptive_step_clip_ratio"] = step_clip_ratio.detach().item()
            stats["actor/local_segment_align_adaptive_clip_gate_enable"] = float(adaptive_tail_clip_gate_enabled)
            stats["actor/local_segment_align_adaptive_clip_ratio_min"] = adaptive_clip_ratio_min
            stats["actor/local_segment_align_adaptive_tail_uniform_weight"] = float(adaptive_tail_uniform_weight)
            stats["actor/local_segment_align_adaptive_length_weight_mean"] = _masked_mean_or_zero(
                length_weight.detach(), seq_active
            ).item()
            stats["actor/local_segment_align_kl_gate_enable"] = float(kl_gate_enabled)
            stats["actor/local_segment_align_kl_gate_value"] = kl_gate_value.detach().item()
            stats["actor/local_segment_align_kl_gate_start"] = kl_gate_start
            stats["actor/local_segment_align_kl_gate_full"] = kl_gate_full
            stats["actor/local_segment_align_kl_gate_min_factor"] = kl_gate_min_factor
            stats["actor/local_segment_align_kl_gate_max_factor"] = kl_gate_max_factor
            stats["actor/local_segment_align_kl_gate_factor"] = kl_gate_factor.detach().item()
            stats["actor/local_segment_align_kl_gate_progress"] = kl_gate_progress.detach().item()

    return policy_loss, stats



@deprecated("verl.trainer.ppo.core_algos.compute_policy_loss_vanilla")
def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def _compute_policy_loss_vanilla_base(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get(  # Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
        "clip_ratio_c", 3.0
    )

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("vanilla")  # type: ignore[arg-type]
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    rollout_log_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    assert config is not None
    assert not isinstance(config, AlgoConfig)
    response_mask = response_mask.to(torch.bool)
    pg_loss, pg_metrics, seg_gate_pos = _apply_gap_guard_policy_loss(
        config=config,
        policy_loss_fn=_compute_policy_loss_vanilla_base,
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        rollout_is_weights=rollout_is_weights,
        rollout_log_prob=rollout_log_probs,
        dp_group=None,
    )
    pg_loss, aux_metrics = _add_gap_guard_aux_losses(
        config=config,
        policy_loss=pg_loss,
        log_prob=log_prob,
        rollout_log_prob=rollout_log_probs,
        response_mask=response_mask,
        advantages=advantages,
        loss_agg_mode=loss_agg_mode,
        seg_gate_pos=seg_gate_pos,
        dp_group=None,
    )
    pg_metrics.update(aux_metrics)
    return pg_loss, pg_metrics


@register_policy_loss("dppo_tv")
def compute_policy_loss_dppo_tv(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for DPPO-Binary-TV.

    See https://arxiv.org/pdf/2602.04879 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    # Note: the clip_ratio is different from the standard PPO, it is the TV divergence threshold for DPPO.
    clip_divergence = config.clip_ratio
    clip_divergence_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_divergence
    clip_divergence_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_divergence

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Instead of dual-clip PPO, we use truncated importance sampling (TIS) to clip the policy loss.
    # However, a large threshold is recommended to avoid performance degradation due to the truncation bias.
    # See Section 5.4 in https://arxiv.org/pdf/2602.04879 for more details.
    clip_ratio_c = config.get("clip_ratio_c", 20.0)
    truncated_ratio = torch.clamp(ratio, max=clip_ratio_c)
    truncated_ratio = truncated_ratio.detach()

    # Compute valid mask for DPPO-Binary-TV
    prob = torch.exp(log_prob)
    old_prob = torch.exp(old_log_prob)
    valid_positive_mask = (prob - old_prob) <= clip_divergence_high
    valid_negative_mask = (prob - old_prob) >= -clip_divergence_low
    valid_mask = torch.where(advantages > 0, valid_positive_mask, valid_negative_mask)
    valid_mask = valid_mask.detach().float()

    pg_losses = -advantages * truncated_ratio * log_prob * valid_mask

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    pg_clipfrac = verl_F.masked_mean((1.0 - valid_mask).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((ratio > clip_ratio_c).float() * valid_mask, response_mask)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("dppo_kl")
def compute_policy_loss_dppo_kl(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for DPPO-Binary-KL.

    See https://arxiv.org/pdf/2602.04879 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    # Note: the clip_ratio is different from the standard PPO, it is the KL divergence threshold for DPPO.
    clip_divergence = config.clip_ratio
    clip_divergence_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_divergence
    clip_divergence_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_divergence

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Instead of dual-clip PPO, we use truncated importance sampling (TIS) to clip the policy loss.
    # However, a large threshold is recommended to avoid performance degradation due to the truncation bias.
    # See Section 5.4 in https://arxiv.org/pdf/2602.04879 for more details.
    clip_ratio_c = config.get("clip_ratio_c", 20.0)
    truncated_ratio = torch.clamp(ratio, max=clip_ratio_c)
    truncated_ratio = truncated_ratio.detach()

    # Compute valid mask for DPPO-Binary-KL
    prob = torch.exp(log_prob)
    old_prob = torch.exp(old_log_prob)
    binary_kl = old_prob * (old_log_prob - log_prob) + (1 - old_prob) * torch.log(
        (1.0 - old_prob + 1e-8) / (1.0 - prob + 1e-8)
    )
    valid_positive_mask = (binary_kl <= clip_divergence_high) | (prob <= old_prob)
    valid_negative_mask = (binary_kl <= clip_divergence_low) | (prob >= old_prob)
    valid_mask = torch.where(advantages > 0, valid_positive_mask, valid_negative_mask)
    valid_mask = valid_mask.detach().float()

    pg_losses = -advantages * truncated_ratio * log_prob * valid_mask

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard DPPO)
    pg_clipfrac = verl_F.masked_mean((1.0 - valid_mask).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((ratio > clip_ratio_c).float() * valid_mask, response_mask)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("gspo")
def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GSPO.

    See https://arxiv.org/pdf/2507.18071 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # clamp for numerical stability

    # finaly exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean", **config.global_batch_info
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("sapo")
def compute_policy_loss_sapo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the smoothed policy objective and related metrics for SAPO.

    See https://arxiv.org/pdf/2511.20347 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For SAPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)

    # temperature for positive and negative token updates
    tau_pos = torch.as_tensor(config.tau_pos, dtype=advantages.dtype, device=advantages.device)
    tau_neg = torch.as_tensor(config.tau_neg, dtype=advantages.dtype, device=advantages.device)

    def gate_function(x, tau):
        """The gating function used in SAPO"""
        return torch.sigmoid(tau * (x - 1.0)) * (4.0 / tau)

    # compute IS at token level:
    # r_{i,t}(θ) = π_θ(y_{i,t}|x, y_{i,<t}) / π_θold(y_{i,t}|x, y_{i,<t})]
    # In log space: log(r_{i,t}(θ)) = log_prob - ol_log_prob
    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    # finally exp() to remove log and get r_{i,t}(θ)
    ratio = torch.exp(negative_approx_kl)

    # tau_{i,t} is tau_pos if adv > 0 else tau_neg
    taus = torch.where(
        condition=advantages > 0,
        input=tau_pos,  # if A_{i,t} > 0 we set to tau_pos
        other=tau_neg,  # if A_{i,t} <= 0 we set to tau_neg
    )

    # compute the gates f_{i,t}(r_{i,t}(θ)) at token level
    gates = gate_function(ratio, taus)

    # compute policy gradient loss
    pg_losses = -gates * advantages

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # for SAPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean", **config.global_batch_info
    )

    # For compatibility, return zero for both pg_clipfrac and pg_clipfrac_lower (not used in SAPO)
    pg_clipfrac = torch.tensor(0.0, device=pg_loss.device)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)
    # compute KL for metrics tracking
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    # return metrics dict
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }

    return pg_loss, pg_metrics


@register_policy_loss("gpg")
def compute_policy_loss_gpg(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Adapted from
    https://github.com/AMAP-ML/GPG/blob/main/VisualThinker-R1-Zero/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py#L495
    Args:
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    return:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via GPG
    """
    assert config is not None
    pg_losses = -log_prob * advantages

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    return pg_loss, {}


@register_policy_loss("clip_cov")
def compute_policy_loss_clip_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        clip_cvo_ratio (float, optional):
            Ratio for clipping the covariance. Defaults to 0.0002.
        clip_cov_lb (float, optional):
            Lower bound for clipping covariance. Defaults to 1.0.
        clip_cov_ub (float, optional):
            Upper bound for clipping covariance. Defaults to 5.0.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    clip_cov_ratio = config.policy_loss.clip_cov_ratio if config.policy_loss.clip_cov_ratio is not None else 0.0002
    cliprange = config.clip_ratio
    cliprange_low = config.clip_ratio_low if config.clip_ratio_low is not None else cliprange
    cliprange_high = config.clip_ratio_high if config.clip_ratio_high is not None else cliprange
    clip_cov_ub = config.policy_loss.clip_cov_ub if config.policy_loss.clip_cov_ub is not None else 5.0
    clip_cov_lb = config.policy_loss.clip_cov_lb if config.policy_loss.clip_cov_lb is not None else 1.0

    assert clip_cov_ratio > 0, "clip_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    corr = torch.ones_like(advantages)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)

    cov_all = (advantages - verl_F.masked_mean(advantages, response_mask)) * (
        log_prob - verl_F.masked_mean(log_prob.detach(), response_mask)
    )
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    clip_num = max(int(clip_cov_ratio * response_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)

    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[: min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0

    pg_clipfrac = verl_F.masked_mean((corr == 0).float(), response_mask)

    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("kl_cov")
def compute_policy_loss_kl_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        kl_cov_ratio (float, optional):
            Ratio for selecting the top-k covariance values. Defaults to 0.0002.
        ppo_kl_coef (float, optional):
            Coefficient for the KL penalty term in the loss. Defaults to 1.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    kl_cov_ratio = config.policy_loss.kl_cov_ratio if config.policy_loss.kl_cov_ratio is not None else 0.0002
    ppo_kl_coef = config.policy_loss.ppo_kl_coef if config.policy_loss.ppo_kl_coef is not None else 1.0

    assert kl_cov_ratio > 0, "kl_cov_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    abs_kl = negative_approx_kl.abs()
    ratio = torch.exp(negative_approx_kl)
    ppo_kl_abs = verl_F.masked_mean(negative_approx_kl.abs(), response_mask)
    pg_losses1 = -advantages * ratio
    pg_losses_kl = -advantages * ratio + ppo_kl_coef * abs_kl
    pg_losses = pg_losses1

    all_valid = response_mask > 0
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_prob[all_valid].detach().reshape(-1).cpu()

    k = min(kl_cov_ratio, len(all_valid_adv))

    if k != 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        k_percent_nums = max(1, int(len(cov_lst_all) * kl_cov_ratio))
        large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices

        if len(large_cov_idxs) != 0:
            large_cov_idxs = all_valid_idx[large_cov_idxs]
            pg_losses[large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]] = pg_losses_kl[
                large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]
            ]

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    pg_metrics = {
        "actor/ppo_kl": ppo_kl_abs.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("geo_mean")
def compute_policy_loss_geo_mean(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GMPO.

    Adapted from paper https://arxiv.org/abs/2507.20673
    https://github.com/callsys/GMPO/blob/main/train_zero_math_gmpo.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            not used
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability (uncomment it if you like)
    # negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Clipping at token-level & Clipping wider
    sgn_advantage = torch.sign(advantages)
    negative_approx_kl_clamp = torch.clamp(negative_approx_kl, -cliprange_low, cliprange_high)
    negative_approx_kl_min = torch.min(sgn_advantage * negative_approx_kl, sgn_advantage * negative_approx_kl_clamp)
    negative_approx_kl_min = sgn_advantage * negative_approx_kl_min

    # Geometric-Mean Policy Optimization
    response_mask_sum = response_mask.sum(dim=-1)
    ratio = torch.exp((negative_approx_kl_min * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8))
    # we only support sequence level advantage for now,
    # otherwise, below would be not consistent with the paper
    advantage = (advantages * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
    pg_losses = -advantage * ratio

    # Apply rollout correction weights if provided
    # For geo_mean, IS weights are 2D (batch_size, seq_length) and need to be aggregated to sequence level
    if rollout_is_weights is not None:
        # Aggregate token-level weights to sequence level using geometric mean for consistency
        # Note: rollout_is_weights is always 2D regardless of aggregation mode
        seq_is_weights = torch.exp(
            (torch.log(rollout_is_weights + 1e-10) * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
        )
        pg_losses = pg_losses * seq_is_weights

    pg_loss = torch.mean(pg_losses)

    # higher: ratio is too large that need clamp to clip_high (when adv > 0)
    clipped = torch.ne(negative_approx_kl, negative_approx_kl_clamp)
    pg_clipfrac = verl_F.masked_mean((clipped * (advantages > 0)).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((clipped * (advantages < 0)).float(), response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("cispo")
def compute_policy_loss_cispo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for CISPO.

    See https://arxiv.org/pdf/2506.13585 for more details.
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    # Compute importance sampling ratio: π_θ / π_θ_old
    negative_approx_kl = log_prob - old_log_prob
    # Clamp for numerical stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # CISPO: Clip the importance sampling weights
    # KEY: Apply stop gradient to the clipped ratio
    # This prevents gradients from flowing through the ratio computation and clipping
    # Gradients only flow through log_prob in the final loss term
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clipped_ratio_sg = clipped_ratio.detach()

    # CISPO objective function (to maximize): J = sg(clip(ratio)) * A * log π_θ
    # Loss function (to minimize): L = -J = -sg(clip(ratio)) * A * log_prob
    pg_losses = -clipped_ratio_sg * advantages * log_prob

    # Track clipping statistics
    pg_clipfrac = verl_F.masked_mean((ratio != clipped_ratio).float(), response_mask)

    # Apply rollout importance sampling weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in CISPO)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob. Optionally using straight through to bind k2 on other
    kl penalty compute method for unbiased KL gradient estimation.
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    forward_score = kl_penalty_forward(logprob, ref_logprob, kl_penalty)
    if not kl_penalty.endswith("+") or kl_penalty in ("mse", "k2"):
        return forward_score

    """
    The expectation of k1 and k3 estimator is the expectaed value of KL, but the expected gradient of k1 and k3
    estimator is not the expectaed gradient of KL. On the other hand k2 estimator gives right gradient estimator, 
    so we use a straight through trick here if the kl_penalty method ends with '+', .e.g., k3+. 
    """
    backward_score = 0.5 * (logprob - ref_logprob).square()

    return backward_score - backward_score.detach() + forward_score.detach()


def kl_penalty_forward(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        """Compute importance weights for resampling based on scores.

        Args:
            scores (torch.Tensor): Tensor of scores to compute weights from.
            reweight_method (str): Method for computing weights ('pow', 'max_min', 'max_random').
            weight_pow (float): Power exponent for 'pow' method.

        Returns:
            torch.Tensor: Computed importance weights.

        Raises:
            ValueError: If reweight_method is not supported.
        """
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data


def compute_policy_loss_reinforce(
    rollout_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-sum",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute REINFORCE-style policy gradient loss with optional IS correction.

    This function implements policy gradient (REINFORCE) with optional importance
    sampling correction for rollout-training policy mismatch.

    Mathematical formulation:
        Without IS (rollout_is_weights=None):
            L = -E[log π(a|s) * A(s,a)]
            Gradient: ∇_θ L = -E[∇log π(a|s) * A] (standard REINFORCE)

        With IS (rollout_is_weights provided):
            L = -E_π_rollout[w * log π(a|s) * A(s,a)]
            where w = π_current / π_rollout (truncated IS weight)
            Gradient: ∇_θ L = -E[w * ∇log π(a|s) * A] (IS-corrected policy gradient)

    Args:
        rollout_log_prob: Log probabilities from rollout policy (e.g., vLLM BF16).
            Shape: (batch_size, seq_length). Used for KL computation.
        log_prob: Log probabilities from current training policy.
            Shape: (batch_size, seq_length)
        advantages: Advantage estimates for each token.
            Shape: (batch_size, seq_length)
        response_mask: Mask indicating valid tokens (1 for valid, 0 for padding).
            Shape: (batch_size, seq_length). Should already include rejection sampling.
        loss_agg_mode: Loss aggregation strategy (see agg_loss for details).
        config: Actor config (required for global_batch_info).
        rollout_is_weights: Pre-computed IS weights (π_current / π_rollout).
            Shape: (batch_size, seq_length). None to disable IS correction.

    Returns:
        Tuple of (loss, metrics):
            loss: Scalar policy gradient loss
            metrics: Dictionary with "actor/ppo_kl"

    Note:
        Unlike PPO (compute_policy_loss_vanilla), this function:
        - Does NOT use PPO clipping
        - Uses log π(a|s) directly (not ratio)
        - IS weights are applied as multiplicative factor
    """
    assert config is not None, "ActorConfig must be provided for REINFORCE loss"

    # Compute pure policy gradient loss with optional IS correction
    # Standard REINFORCE: L = -E[log π(a|s) * A]
    # With IS: L = -E[w * log π(a|s) * A] where w = π_current / π_rollout
    if rollout_is_weights is not None:
        # IS-corrected policy gradient: L = -E[stopgrad(w) · log π · A]
        pg_losses = -advantages * log_prob * rollout_is_weights
    else:
        # Standard REINFORCE: L = -E[log π · A]
        pg_losses = -advantages * log_prob

    # Aggregate loss
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    # Compute KL divergence between current and rollout policy
    negative_approx_kl = log_prob - rollout_log_prob
    kl_divergence = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_metrics = {
        "actor/ppo_kl": kl_divergence.detach().item(),
    }

    return pg_loss, pg_metrics


@register_policy_loss("bypass_mode")
def compute_policy_loss_bypass_mode(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Bypass mode policy loss supporting both REINFORCE and PPO-clip.

    This function is the entry point for bypass mode, where old_log_prob = rollout_log_prob.
    It computes IS weights and rejection masks, then dispatches to either REINFORCE or
    PPO-clip loss based on the loss_type configuration.

    IMPORTANT - Bypass mode semantics:
        In bypass mode, the trainer sets old_log_prob = rollout_log_prob.
        This means:
        - For REINFORCE: We use IS weights w = π_current / π_rollout explicitly
        - For PPO-clip: The PPO ratio π_current / π_old = π_current / π_rollout
          already incorporates the IS correction through clipping, so we do NOT
          apply additional IS weights (would be double-counting)

    Loss types:
        - "ppo_clip" (default): PPO clipped objective (compute_policy_loss_vanilla)
            L = -E[min(r*A, clip(r)*A)] where r = π_current / π_rollout
            Note: IS weights are NOT applied (clipping handles the ratio)
        - "reinforce": REINFORCE-style policy gradient with IS correction
            L = -E[w * log π(a|s) * A] where w = π_current / π_rollout

    Args:
        old_log_prob: In bypass mode, this is actually rollout_log_prob.
            Shape: (batch_size, seq_length)
        log_prob: Current policy log probabilities.
            Shape: (batch_size, seq_length)
        advantages: Advantage estimates.
            Shape: (batch_size, seq_length)
        response_mask: Valid token mask (1=valid, 0=padding).
            Shape: (batch_size, seq_length)
        loss_agg_mode: Loss aggregation mode (passed to underlying loss function).
        config: Actor config containing rollout_correction settings in policy_loss.
        rollout_is_weights: Pre-computed IS weights (ignored, computed internally).

    Config options (in config.policy_loss.rollout_correction):
        loss_type: "ppo_clip" (default) or "reinforce"
        rollout_is: IS aggregation level ("token", "sequence", or None)
        rollout_is_threshold: Upper threshold for truncating IS weights (default: 2.0)
        rollout_rs: Rejection sampling level (see rollout_corr_helper for supported modes)
        rollout_rs_threshold: Threshold specification for rejection sampling
        rollout_is_batch_normalize: Whether to normalize IS weights to mean=1.0

    Returns:
        Tuple of (loss, metrics):
            loss: Scalar policy loss
            metrics: Dictionary with rollout correction metrics and actor/ppo_kl
    """
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_rejection_mask

    assert config is not None, "config is required for bypass_mode loss"

    # Extract rollout_correction config from policy_loss
    rollout_corr_config = config.policy_loss.get("rollout_correction", None) if hasattr(config, "policy_loss") else None

    if rollout_corr_config is None:
        raise ValueError(
            "rollout_correction config not found in policy_loss. "
            "When using loss_mode='bypass_mode', ensure rollout_correction config is passed."
        )

    # Extract parameters
    loss_type = rollout_corr_config.get("loss_type", "ppo_clip")
    rollout_is = rollout_corr_config.get("rollout_is", None)
    rollout_is_threshold = rollout_corr_config.get("rollout_is_threshold", 2.0)
    rollout_is_batch_normalize = rollout_corr_config.get("rollout_is_batch_normalize", False)
    rollout_rs = rollout_corr_config.get("rollout_rs", None)
    rollout_rs_threshold = rollout_corr_config.get("rollout_rs_threshold", None)

    # In bypass mode: old_log_prob IS rollout_log_prob
    rollout_log_prob = old_log_prob

    # Compute IS weights and rejection mask
    # Note: For PPO-clip, we still compute IS weights for metrics, but don't apply them
    with torch.no_grad():
        rollout_is_weights_proto, modified_response_mask, rollout_metrics = (
            compute_rollout_correction_and_rejection_mask(
                old_log_prob=log_prob,  # Current policy (for IS ratio: π_current / π_rollout)
                rollout_log_prob=rollout_log_prob,  # Rollout policy
                response_mask=response_mask,
                rollout_is=rollout_is,
                rollout_is_threshold=rollout_is_threshold,
                rollout_is_batch_normalize=rollout_is_batch_normalize,
                rollout_rs=rollout_rs,
                rollout_rs_threshold=rollout_rs_threshold,
            )
        )

    # Extract IS weights tensor (or None if disabled)
    computed_is_weights = rollout_is_weights_proto.batch["rollout_is_weights"] if rollout_is_weights_proto else None

    # Apply rejection mask (RS + veto)
    effective_mask = modified_response_mask

    # Dispatch to appropriate loss function based on loss_type
    if loss_type == "reinforce":
        # REINFORCE: Apply IS weights explicitly
        pg_loss, pg_metrics = compute_policy_loss_reinforce(
            rollout_log_prob=rollout_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=computed_is_weights,
        )

    elif loss_type == "ppo_clip":
        # PPO-clip: The ratio π_current/π_old = π_current/π_rollout already handles IS
        # DO NOT apply IS weights - would be double-counting!
        # The clipping mechanism constrains the effective IS ratio
        pg_loss, pg_metrics = compute_policy_loss_vanilla(  # type: ignore[call-arg]
            old_log_prob=rollout_log_prob,  # = old_log_prob in bypass mode
            log_prob=log_prob,
            advantages=advantages,
            response_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=None,  # Explicitly None - no IS weights for PPO-clip
        )

    else:
        raise ValueError(f"Invalid loss_type: {loss_type}. Must be 'reinforce' or 'ppo_clip'.")

    # Merge rollout correction metrics
    pg_metrics.update(rollout_metrics)

    return pg_loss, pg_metrics
