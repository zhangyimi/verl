# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Megatron Actor.
In megatron actor, the differences are:
1. We only make minibatch

Note that our model doesn't have to be `MegatronModule` because we don't share embedding in the last layer
"""

import itertools
import logging
import os
from functools import partial
from typing import Iterable

import torch
import torch.distributed
import torch.nn.functional as F
from megatron.core import parallel_state as mpu
from megatron.core.distributed import finalize_model_grads

# from megatron.core.optimizer import DistributedOptimizer
from megatron.core.optimizer import DistributedOptimizer
from megatron.core.pipeline_parallel import get_forward_backward_func
from omegaconf import OmegaConf
from torch import nn

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id, get_torch_device
from verl.utils.import_utils import deprecated
from verl.utils.megatron.pipeline_parallel import make_batch_generator
from verl.utils.megatron.router_replay_patch import RouterReplay, RouterReplayAction
from verl.utils.megatron.router_replay_utils import (
    RouterReplayHelper,
    merge_router_topk_indices,
    pp_gather,
    reorder_and_merge_vpp_layers,
    set_router_replay_data,
)
from verl.utils.megatron.tensor_parallel import vocab_parallel_entropy, vocab_parallel_log_probs_from_logits
from verl.utils.megatron_utils import get_megatron_mtp_loss, get_model_config, unwrap_model
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils import tensordict_utils as tu
from verl.utils.torch_functional import broadcast_dict_tensor
from verl.workers.actor import BasePPOActor
from verl.workers.config import MtpConfig

__all__ = ["MegatronPPOActor"]


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


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
    value = value.detach().clone()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM, group=dp_group)
    return value


_LOCAL_SEGMENT_ALIGN_GATE_STATE: dict[int, dict[str, float]] = {}


def _ema_beta_from_span(span: float) -> float:
    span = max(float(span), 1.0)
    return (span - 1.0) / (span + 1.0)


def _scalar_int_or_none(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() <= 0:
            return None
        return int(value.detach().flatten()[0].item())
    if hasattr(value, "data"):
        value = value.data
    if hasattr(value, "flat"):
        try:
            value = value.flat[0]
        except Exception:
            return None
    elif isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    try:
        return int(value)
    except Exception:
        return None


def _period_allows(step: int | None, period: int, fallback_counter: int) -> bool:
    period = int(period)
    if period <= 1:
        return True
    value = fallback_counter if step is None else step
    return (int(value) % period) == 0


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


def _apply_gap_guard_policy_loss(
    *,
    config,
    policy_loss_fn,
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str,
    rollout_is_weights: torch.Tensor | None,
    rollout_log_prob: torch.Tensor | None,
) -> tuple[torch.Tensor, dict, torch.Tensor | None]:
    pg_advantages = advantages
    pg_response_mask = response_mask
    raw_delta = None
    seg_gate_pos = None
    stats = {}

    if rollout_log_prob is not None:
        rollout_log_prob = rollout_log_prob.detach()
        raw_delta = log_prob - rollout_log_prob

    if raw_delta is not None and bool(getattr(config, "adv_length_norm_enable", False)):
        response_lengths = response_mask.to(dtype=log_prob.dtype).sum(dim=-1).clamp_min(1.0)
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
        stats["actor/adv_length_norm_ref_len"] = float(getattr(config, "adv_length_norm_ref_len", 4096))

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

    # Keep local normalization consistent with the actual weighted mask used by PG.
    saved_global_batch_info = dict(getattr(config, "global_batch_info", {}) or {})
    if hasattr(config, "global_batch_info"):
        if loss_agg_mode == "token-mean":
            config.global_batch_info["batch_num_tokens"] = (
                pg_response_mask.to(dtype=log_prob.dtype).sum().clamp_min(1.0)
            )
        elif loss_agg_mode == "seq-mean-token-mean":
            active_seq = (pg_response_mask.to(dtype=log_prob.dtype).sum(dim=-1) > 0).to(dtype=log_prob.dtype)
            config.global_batch_info["global_batch_size"] = active_seq.sum().clamp_min(1.0)
    try:
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
    config,
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
    local_segment_align_global_step=None,
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
                loss_mat=token_obj * token_weight, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
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

            activation_gate_enabled = bool(getattr(config, "local_segment_align_activation_gate_enable", False))
            activation_clip_min = float(getattr(config, "local_segment_align_activation_clip_min", 0.005))
            activation_clip_period = max(int(getattr(config, "local_segment_align_activation_clip_period", 4)), 1)
            activation_raw_ratio = float(getattr(config, "local_segment_align_activation_raw_ratio", 1.3))
            activation_raw_min = float(getattr(config, "local_segment_align_activation_raw_min", 5e-5))
            activation_raw_period = max(int(getattr(config, "local_segment_align_activation_raw_period", 3)), 1)
            activation_ema_span = float(getattr(config, "local_segment_align_activation_ema_span", 20.0))
            activation_baseline_span = float(getattr(config, "local_segment_align_activation_baseline_span", 100.0))
            activation_factor = raw_delta.new_tensor(1.0)
            activation_clip_active = False
            activation_raw_active = False
            activation_clip_condition = False
            activation_raw_condition = False
            activation_raw_ema = 0.0
            activation_raw_baseline = 0.0
            activation_raw_ratio_value = 0.0
            activation_counter = 0
            activation_step = _scalar_int_or_none(local_segment_align_global_step)

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
                if activation_gate_enabled:
                    state = _LOCAL_SEGMENT_ALIGN_GATE_STATE.setdefault(id(config), {})
                    activation_counter = int(state.get("counter", 0))
                    state["counter"] = activation_counter + 1
                    raw_value = float(align_loss_raw.detach().item())
                    if "raw_ema" not in state:
                        activation_raw_ema = raw_value
                        activation_raw_baseline = raw_value
                    else:
                        beta = _ema_beta_from_span(activation_ema_span)
                        base_beta = _ema_beta_from_span(activation_baseline_span)
                        activation_raw_ema = beta * float(state["raw_ema"]) + (1.0 - beta) * raw_value
                        activation_raw_baseline = base_beta * float(state["raw_baseline"]) + (1.0 - base_beta) * raw_value
                    state["raw_ema"] = activation_raw_ema
                    state["raw_baseline"] = activation_raw_baseline
                    activation_raw_ratio_value = activation_raw_ema / max(activation_raw_baseline, 1e-12)
                    activation_clip_condition = float(step_clip_ratio.detach().item()) >= activation_clip_min
                    activation_raw_condition = (
                        activation_raw_ratio_value >= activation_raw_ratio and activation_raw_ema >= activation_raw_min
                    )
                    activation_clip_active = activation_clip_condition and _period_allows(
                        activation_step, activation_clip_period, activation_counter
                    )
                    activation_raw_active = activation_raw_condition and _period_allows(
                        activation_step, activation_raw_period, activation_counter
                    )
                    activation_factor = raw_delta.new_tensor(1.0 if (activation_clip_active or activation_raw_active) else 0.0)
                align_loss = align_loss_raw * gate_factor * activation_factor
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
            stats["actor/local_segment_align_activation_gate_enable"] = float(activation_gate_enabled)
            stats["actor/local_segment_align_activation_factor"] = activation_factor.detach().item()
            stats["actor/local_segment_align_activation_clip_active"] = float(activation_clip_active)
            stats["actor/local_segment_align_activation_raw_active"] = float(activation_raw_active)
            stats["actor/local_segment_align_activation_clip_condition"] = float(activation_clip_condition)
            stats["actor/local_segment_align_activation_raw_condition"] = float(activation_raw_condition)
            stats["actor/local_segment_align_activation_clip_min"] = activation_clip_min
            stats["actor/local_segment_align_activation_clip_period"] = float(activation_clip_period)
            stats["actor/local_segment_align_activation_raw_ratio"] = activation_raw_ratio
            stats["actor/local_segment_align_activation_raw_min"] = activation_raw_min
            stats["actor/local_segment_align_activation_raw_period"] = float(activation_raw_period)
            stats["actor/local_segment_align_activation_raw_ema"] = float(activation_raw_ema)
            stats["actor/local_segment_align_activation_raw_baseline"] = float(activation_raw_baseline)
            stats["actor/local_segment_align_activation_raw_ratio_value"] = float(activation_raw_ratio_value)
            stats["actor/local_segment_align_activation_step"] = float(-1 if activation_step is None else activation_step)

    return policy_loss, stats


@deprecated("legacy worker implementation is deprecated and will be removed in v0.8.0")
class MegatronPPOActor(BasePPOActor):
    def __init__(
        self,
        config,
        model_config,
        hf_config,
        tf_config,
        actor_module: nn.ModuleList,
        actor_optimizer: DistributedOptimizer,
        mtp_config: MtpConfig = None,
    ):
        """MeagtronPPOActor class. This class implements the simple PPO logics when the model is built with Megatron.

        Args:
            config (OmegaConf): the basic config that contains the hyper-parameters of PPO Actor. It must contain

                ``ppo_micro_batch_size_per_gpu``: micro batch size when updating ppo.

                ``ppo_mini_batch_size``: minibatch size when updating ppo using the batch data.

                ``ppo_epochs``: number of epochs to update the actor using the batch data.

                ``shuffle``: whether to shuffle the data after each ppo epoch.

                ``clip_ratio``: clip ratio of the ppo algorithm. See https://arxiv.org/abs/1707.06347.

                ``entropy_coeff``: entropy coefficient of the PPO loss. See https://arxiv.org/abs/1707.06347.
            model_config (OmegaConf): model configuration. It must contains ``model_config.vocab_size`` and
                ``model_config.hidden_size``
            hf_config (PretrainedConfig): huggingface config
            tf_config (TransformerConfig): mcore transformer config
            mtp_config (MtpConfig): mtp config, default None
            actor_module (nn.ModuleList): actor module is a ModuleList that contains a list of nn.Module in this
                pp stage.
                each nn.Module in this rank holds a vpp module chunk. See https://arxiv.org/pdf/2104.04473.pdf for
                more details.
                The actor module has some constraints to follow in order to use the updating logics implemented here

                1. It must implement unpad_input before any computation and pad_input after all the computation.
                Remove padding is an
                optimization that removes the padding tokens. See unpad_input and pad_input function in flash-attn
                (https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py).

                2. Each pp stage must return the hidden state with the same shape [total_nnz, 1, hidden_size],
                where total_nnz is the number of valid tokens in this batch. If sequence parallel is enabled, the size
                of the hidden state is [total_nnz // tp, 1, hidden_size].
            actor_optimizer (DistributedOptimizer): currently, we only support DistributedOptimizer in Megatron.
                It implements
                zero1 optimizer that shards the optimizer state across dp ranks.

        >>> from megatron.training import get_model
        >>> from megatron.optimizer import get_megatron_optimizer
        >>> actor_module = get_model(megatron_actor_model_provider, wrap_with_ddp=True)
        >>> actor_module = nn.ModuleList(actor_module)
        >>> actor_optimizer = get_megatron_optimizer(actor_module)
        >>> actor = MegatronPPOActor(config=config,
        >>>                          model_config=actor_model_config,
        >>>                          hf_config=hf_config,
        >>>                          tf_config=tf_config,
        >>>                          actor_module=actor_module,
        >>>                          actor_optimizer=actor_optimizer)
        """
        super().__init__(config)
        self._validate_config(config)
        self.model_config = model_config
        self.hf_config = hf_config
        self.tf_config = tf_config
        self.mtp_config = mtp_config
        self.actor_module = actor_module
        self.actor_optimizer: DistributedOptimizer = actor_optimizer

        if self.mtp_config:
            assert self.mtp_config.enable, "MTP requires mtp_config.enable to be True"

        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if getattr(self.mtp_config, "enable", False) and self.use_fused_kernels:
            self.use_fused_kernels = False
            logger.warning_once(
                "MTP is not compatible with fused kernels for now. Automatically disable use_fused_kernels."
            )
        if self.use_fused_kernels and not getattr(self.config, "overlap_moe_expert_parallel_comm", False):
            # do not patch if overlap_moe_expert_parallel_comm is enabled
            logger.warning_once(
                "Recommend to disable use_fused_kernels since the fused kernel's performance is broken for triton>=3.3"
                "Unless you are using a very old version of triton < 3.3"
            )
            from verl.models.mcore.model_forward_fused import patch_fused_forward

            for model in self.actor_module:
                patch_fused_forward(model)
        else:
            from verl.models.mcore.mtp_patch import patch_postprocess

            for model in self.actor_module:
                if self.mtp_config:
                    from verl.models.mcore.mtp_patch import patch_mtp_layer_get_embeddings

                    patch_postprocess(model)

                    if self.mtp_config.detach_encoder:
                        patch_mtp_layer_get_embeddings(model)

        self.optimizer_step_args = OmegaConf.create(
            {
                "skip_grad": None,
                "overlap_dp_param_comm": False,
                "overlap_dp_grad_comm": False,
                "gradient_accumulation_steps": 1,
                "sequence_parallel": self.tf_config.sequence_parallel,
                "DDP_impl": "local",
                "layernorm_allreduce_bucket_threshold": 0,
                "reduce_grads_use_alltoall": False,
            }
        )

        self.router_replay = self.config.router_replay
        self.enable_routing_replay = self.router_replay.mode != "disabled"
        if self.enable_routing_replay:
            self.mini_layer_topk_idx_list = []

        config = get_model_config(self.actor_module[0])
        print(config)
        config.finalize_model_grads_func = finalize_model_grads

    def _validate_config(self, config) -> None:
        """Validate config options not implemented for Megatron backend"""
        assert config.get("ulysses_sequence_parallel_size", 1) == 1
        if config.get("shuffle", False):
            assert config.data_loader_seed is not None, "If shuffle dataloader, seed must be manually set"
        if config.megatron.tensor_model_parallel_size == 1:
            print("[Warining] Because actor tp size == 1, set sp to False")
            config.megatron.sequence_parallel = False
        self.config = config

    @GPUMemoryLogger(role="megatron actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            DataProto: torch.Tensor: the log_prob tensor
        """
        prev_modes = [m.training for m in self.actor_module]
        for module in self.actor_module:
            module.eval()
        use_dynamic_bsz = data.meta_info.get("use_dynamic_bsz", False)
        micro_batch_size = data.meta_info.get("micro_batch_size", None)
        max_token_len = data.meta_info.get("max_token_len", None)
        if use_dynamic_bsz:
            assert max_token_len is not None, "max_token_len must be set when use_dynamic_bsz is True"
            max_token_len = max_token_len * self.config.megatron.context_parallel_size
        else:
            assert micro_batch_size is not None, (
                "micro batch size is needed for forward compute when use_dynamic_bsz is False"
            )

        def compute_logprobs_fn(output, data, use_dynamic_bsz=False, indices=None):
            response = data["responses"]
            response_length = response.size(1)
            log_probs = output["log_probs"][:, -response_length - 1 : -1].contiguous()
            return {"log_probs": log_probs}

        # We make recompute_old_log_prob by default here.
        # TODO (zhangchi.usc1992): actually, this function should only return log_prob and this logic should be
        # handled by user outside
        recompute_old_log_prob = self.config.get("recompute_old_log_prob", True)

        entropys = torch.Tensor()
        if recompute_old_log_prob:
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]

            if self.enable_routing_replay and self.config.router_replay.mode == "R3":
                assert "routed_experts" in data.batch.keys(), "routed_experts must be in data.batch.keys()"
                select_keys.append("routed_experts")

            batch = data.select(batch_keys=select_keys).batch
            input_ids = batch["input_ids"]
            batch_size = input_ids.size(0)
            response = batch["responses"]
            response_length = response.size(1)
            with torch.no_grad():
                output = self.forward_backward_batch(
                    data,
                    forward_only=True,
                    post_process_fn=compute_logprobs_fn,
                    calculate_entropy=calculate_entropy,
                    use_dynamic_bsz=use_dynamic_bsz,
                    micro_batch_size=micro_batch_size,
                    max_token_len=max_token_len,
                )
                if mpu.is_pipeline_last_stage(ignore_virtual=True):
                    # only on last rank. It should be on every tp rank
                    if calculate_entropy:
                        log_probs = [o[0]["log_probs"] for o in output["output"]]  # (bs, seq_size)
                    else:
                        log_probs = [o["log_probs"] for o in output["output"]]  # (bs, seq_size)
                    log_probs = torch.cat(log_probs, dim=0).to(torch.float32)
                    if use_dynamic_bsz:
                        indices = output["indices"]
                        indices = list(itertools.chain.from_iterable(indices))
                        assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
                        revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                        log_probs = log_probs[revert_indices]
                else:
                    log_probs = torch.empty(
                        size=(batch_size, response_length), dtype=torch.float32, device=input_ids.device
                    )
                log_probs = log_probs.to(get_device_id())
                # broadcast across pp ranks
                torch.distributed.broadcast(
                    tensor=log_probs,
                    src=mpu.get_pipeline_model_parallel_last_rank(),
                    group=mpu.get_pipeline_model_parallel_group(),
                    async_op=False,
                )
                log_probs = log_probs.to("cpu")
                if calculate_entropy:
                    # Note that o[0] is metrics, o[1] is entropy
                    if mpu.is_pipeline_last_stage(ignore_virtual=True):
                        entropys = torch.cat([o[1] for o in output["output"]], dim=0)
                        entropys = entropys.to(torch.float32)
                        if use_dynamic_bsz:
                            indices = output["indices"]
                            indices = list(itertools.chain.from_iterable(indices))
                            assert len(indices) == entropys.size(0), f"{len(indices)} vs. {entropys.size()}"
                            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                            entropys = entropys[revert_indices]
                    else:
                        entropys = torch.empty(
                            size=(batch_size, response_length), dtype=torch.float32, device=input_ids.device
                        )
                    # broadcast across pp ranks
                    entropys = entropys.to(get_device_id())
                    torch.distributed.broadcast(
                        tensor=entropys,
                        src=mpu.get_pipeline_model_parallel_last_rank(),
                        group=mpu.get_pipeline_model_parallel_group(),
                        async_op=False,
                    )
                    entropys = entropys.to("cpu")
                layers_topk_idx = None

                if RouterReplayHelper.is_r2_record_action(self.tf_config):
                    # (bs, max_seq_len/response_len,local_layer_num,topk)
                    layers_topk_idx = output["mini_layer_topk_idx_tensor"].to(torch.uint8)
                    if use_dynamic_bsz:
                        indices = output["indices"]
                        indices = list(itertools.chain.from_iterable(indices))
                        assert len(indices) == layers_topk_idx.size(0), f"{len(indices)} vs. {layers_topk_idx.size()}"
                        revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                        layers_topk_idx = layers_topk_idx[revert_indices]
                    layers_topk_idx = pp_gather(layers_topk_idx, self.tf_config)
        # add empty cache after each compute
        get_torch_device().empty_cache()

        for module, mode in zip(self.actor_module, prev_modes, strict=False):
            module.train(mode)
        return log_probs, entropys, layers_topk_idx

    def make_minibatch_iterator(self, data: DataProto) -> Iterable[DataProto]:
        """Make minibatch iterator for updating the actor

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64, where
                ``sequence_length = prompt_length + response_length``

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64

                ``responses``: tensor of shape [batch_size, response_length]. torch.int64. Note that
                responses = input_ids[:, -response_length:]

                ``old_log_probs``: tensor of shape [batch_size, response_length]. torch.float32. The log probability
                of responses.

                ``advantages``: tensor of shape [batch_size, response_length]. torch.float32. The advantages of
                responses.
                See PPO paper for details. https://arxiv.org/abs/1707.06347

        Returns:

        """
        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "response_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        for optional_key in (
            "local_segment_align_step_mean",
            "local_segment_align_step_clip_ratio",
            "local_segment_align_tail_mass",
        ):
            if optional_key in data.batch.keys():
                select_keys.append(optional_key)
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")
        self.has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        # router replay
        if self.enable_routing_replay:
            select_keys.append("routed_experts")
        if self.has_multi_modal_inputs:
            data = data.select(select_keys, ["multi_modal_inputs"])
        else:
            data = data.select(batch_keys=select_keys)

        return data.make_iterator(
            mini_batch_size=self.config.ppo_mini_batch_size,
            epochs=self.config.ppo_epochs,
            seed=self.config.data_loader_seed,
            dataloader_kwargs={"shuffle": self.config.shuffle},
        )

    def forward_backward_batch(
        self,
        data: DataProto,
        forward_only=False,
        post_process_fn=None,
        calculate_entropy=False,
        use_dynamic_bsz=False,
        micro_batch_size=None,
        max_token_len=None,
        mini_batch_size=None,
    ):
        """
        We assume:
        - The model takes input: (input_ids, attention_mask, position_ids). No rmpad for the input
        - The communication shape is (total_nnz_pad_to_sp // tp_size, 1, hidden_size) if sequence parallel is enabled
        """
        # broadcast from last pp rank to all other pp ranks
        # TODO: actually, we just need to control the sampling order.
        data.to(get_device_id())
        data.batch = data.batch.contiguous()
        mini_batch = data
        broadcast_dict_tensor(
            mini_batch.batch,
            src=mpu.get_pipeline_model_parallel_last_rank(),
            group=mpu.get_pipeline_model_parallel_group(),
        )
        mini_batch.to("cpu")
        # split into micro-batches
        mini_batch.batch["attention_mask"] = mini_batch.batch["attention_mask"].to(bool)
        self.has_multi_modal_inputs = "multi_modal_inputs" in mini_batch.non_tensor_batch.keys()
        if self.has_multi_modal_inputs:
            mini_batch.batch["multi_modal_inputs"] = mini_batch.non_tensor_batch["multi_modal_inputs"]
            mini_batch.batch["multi_modal_inputs_idx"] = torch.Tensor(
                list(range(len(mini_batch.non_tensor_batch["multi_modal_inputs"])))
            ).to(torch.int64)

        if mini_batch.batch["position_ids"].dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
            mini_batch.batch["position_ids"] = mini_batch.batch["position_ids"][
                :, 0
            ]  # mcore patch recompute qwen2vl's pos ids during forward

        indices = None
        temperature = data.meta_info["temperature"]
        if use_dynamic_bsz:
            assert max_token_len is not None, "max_token_len must be set when use_dynamic_bsz is True"
            dp_group = mpu.get_data_parallel_group()
            vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
            if vpp_size is not None and vpp_size > 1:
                microbatch_group_size_per_vp_stage = self.tf_config.microbatch_group_size_per_vp_stage
                micro_batches, indices = rearrange_micro_batches(
                    batch=mini_batch.batch,
                    num_batches_divided_by=microbatch_group_size_per_vp_stage,
                    max_token_len=max_token_len,
                    dp_group=dp_group,
                )
                assert len(micro_batches) % self.tf_config.microbatch_group_size_per_vp_stage == 0, (
                    f"micro_batches {micro_batches} must be divisible by microbatch_group_size_per_vp_stage "
                    f"{microbatch_group_size_per_vp_stage} for megatron backend"
                )
            else:
                micro_batches, indices = rearrange_micro_batches(
                    batch=mini_batch.batch, max_token_len=max_token_len, dp_group=dp_group
                )
            total_seqlen = max_token_len
        else:
            assert micro_batch_size is not None, (
                "micro_batch_size is needed to be passed in when not using dynamic batch size"
            )
            micro_batches = mini_batch.batch.split(micro_batch_size)
            seq_len = micro_batches[0]["input_ids"].shape[1]
            total_seqlen = micro_batch_size * seq_len
        # compute input shapes for pp stages
        n_micro_batch = len(micro_batches)

        forward_backward_func = get_forward_backward_func()

        def loss_func(output, data, meta_info):
            # For memory efficiency
            # We move calculation of entropy to compute_log_probs, forward_only == True
            log_probs = None
            entropy = None
            if isinstance(output, dict):
                log_probs = output["log_probs"]
                if "entropy" in output:
                    entropy = output["entropy"]
            else:
                assert isinstance(output, torch.Tensor)
                log_probs = output

            device = log_probs.device
            metrics = {}
            if forward_only:
                if post_process_fn is None:
                    pass
                    # metrics["logits"] = output
                else:
                    stats = post_process_fn(output, data)
                    metrics.update(stats)
                if not calculate_entropy:
                    return torch.tensor(1.0, device=device), metrics

            responses = data["responses"]
            response_length = responses.size(1)
            response_mask = data["response_mask"].to(bool)
            loss_agg_mode = self.config.loss_agg_mode
            # compute policy loss
            log_prob = log_probs[:, -response_length - 1 : -1].contiguous()
            ret_entropy = None
            stats = {}
            if not forward_only:
                old_log_prob = data["old_log_probs"]
                advantages = data["advantages"]

                entropy_coeff = self.config.entropy_coeff
                loss_agg_mode = self.config.loss_agg_mode

                loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                policy_loss_fn = get_policy_loss_fn(loss_mode)

                # Extract pre-computed rollout correction weights if present
                # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                rollout_is_weights = data.get("rollout_is_weights", None)
                rollout_log_prob = data.get("rollout_log_probs", None)
                pg_loss, pg_metrics, seg_gate_pos = _apply_gap_guard_policy_loss(
                    config=self.config,
                    policy_loss_fn=policy_loss_fn,
                    old_log_prob=old_log_prob,
                    log_prob=log_prob,
                    advantages=advantages,
                    response_mask=response_mask,
                    loss_agg_mode=loss_agg_mode,
                    rollout_is_weights=rollout_is_weights,
                    rollout_log_prob=rollout_log_prob,
                )
                stats.update(pg_metrics)

                # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                    # Compute metrics using CURRENT policy π_θ vs π_rollout
                    # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                    rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                        log_prob=log_prob,
                        rollout_log_prob=rollout_log_prob,
                        response_mask=response_mask,
                    )
                    stats.update(rollout_corr_metrics)

                stats["actor/pg_loss"] = pg_loss.detach().item()
                policy_loss = pg_loss

            if calculate_entropy:
                entropy = output["entropy"][:, -response_length - 1 : -1].contiguous()
                if not forward_only:
                    entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                    stats["actor/entropy"] = entropy_loss.detach().item()
                    entropy_coeff = meta_info["entropy_coeff"]
                    if entropy_coeff != 0:
                        policy_loss = pg_loss - entropy_coeff * entropy_loss
                else:
                    ret_entropy = entropy

            if forward_only:
                policy_loss = torch.tensor(1.0, device=device)
            else:
                if self.config.use_kl_loss:
                    ref_log_prob = data["ref_log_prob"]
                    # compute kl loss
                    kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                    kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)

                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    metrics["actor/kl_loss"] = kl_loss.detach().item()
                    metrics["actor/kl_coef"] = self.config.kl_loss_coef

                policy_loss, aux_metrics = _add_gap_guard_aux_losses(
                    config=self.config,
                    policy_loss=policy_loss,
                    log_prob=log_prob,
                    rollout_log_prob=data.get("rollout_log_probs", None),
                    response_mask=response_mask,
                    advantages=advantages,
                    loss_agg_mode=loss_agg_mode,
                    seg_gate_pos=seg_gate_pos,
                    dp_group=mpu.get_data_parallel_group(),
                    local_segment_align_step_mean=data.get("local_segment_align_step_mean", None),
                    local_segment_align_step_clip_ratio=data.get("local_segment_align_step_clip_ratio", None),
                    local_segment_align_tail_mass=data.get("local_segment_align_tail_mass", None),
                    local_segment_align_global_step=tu.get_non_tensor_data(data, "global_steps", None),
                )
                stats.update(aux_metrics)

                # return loss and stats

            append_to_dict(metrics, stats)
            return policy_loss, [metrics, ret_entropy]

        def forward_step(batch_iter, model, return_schedule_plan: bool = False):
            """
            Args:
                batch_iter: the batch iterator
                model: the model
                return_schedule_plan: whether to return the schedule plan, for 1f1b overlap
            """
            if return_schedule_plan:
                assert self.tf_config.overlap_moe_expert_parallel_comm, (
                    "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
                )
                # TODO: Fix this
                assert not calculate_entropy, "calculate_entropy must be disabled to return the schedule plan"
                from megatron.core.models.gpt.gpt_model import GPTModel

                assert isinstance(model, GPTModel), "model must be a GPTModel"
                assert self.use_fused_kernels, "use_fused_kernels must be enabled to return the schedule plan"
                # TODO: support VLM with MoE
                from verl.models.mcore.model_forward_1f1b_overlap import gptmodel_forward_1f1b_overlap

            batch = next(batch_iter)
            batch = batch.to(get_device_id())
            batch = batch.contiguous()

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"].to(bool)
            position_ids = batch["position_ids"]

            unwrapped_model = unwrap_model(model)
            if hasattr(unwrapped_model, "vp_stage"):
                vp_rank = unwrapped_model.vp_stage
            else:
                vp_rank = 0

            multi_modal_inputs = {}
            if "multi_modal_inputs" in batch:
                from verl.utils.model import extract_multi_modal_inputs

                indices = batch.get("multi_modal_inputs_idx", None)
                multi_modal_inputs = extract_multi_modal_inputs(batch["multi_modal_inputs"], indices)
            responses = batch["responses"]
            response_length = responses.size(1)
            label = position_ids.clone()
            label[:, -response_length - 1 : -1] = responses
            label_mask = attention_mask.clone()
            label_mask[:, : -response_length - 1] = False
            label_mask[:, -1] = False

            if RouterReplayHelper.is_replay_backward_action(self.tf_config, vp_rank):
                router_instance_list = RouterReplayHelper.get_micro_batch_router_list(self.tf_config, vp_rank)
                for router in router_instance_list:
                    router.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

            if RouterReplayHelper.is_replay_forward_action(self.tf_config, vp_rank):
                layers_topk_idx = batch["routed_experts"]
                set_router_replay_data(layers_topk_idx, attention_mask, self.tf_config, vp_rank)

            from verl.models.mcore import get_mcore_forward_fn, get_mcore_forward_fused_fn

            if self.use_fused_kernels:
                forward_fn = get_mcore_forward_fused_fn(self.hf_config)
                if return_schedule_plan:
                    forward_fn = gptmodel_forward_1f1b_overlap
                # return dict of [logits, entropy]
                output = forward_fn(
                    model=model,
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    labels=label,
                    labels_mask=label_mask,
                    temperature=temperature,
                    multi_modal_inputs=multi_modal_inputs,
                )
            else:
                forward_fn = get_mcore_forward_fn(self.hf_config)

                def logits_processor(logits, label, label_mask):
                    assert logits.shape[:2] == label.shape[:2]
                    assert label.shape == label_mask.shape
                    logits.div_(temperature)
                    ret = {}
                    if calculate_entropy:
                        logits_bak = logits.clone()
                        # # disable the hint until the fused_kernel is optimized for triton>=3.3
                        # logger.warning_once(
                        #     "For memory-efficient computation, enable fused kernels via "
                        #     "`actor_rollout_ref.model.use_fused_kernels=True`. "
                        #     "The current `clone()` operation ensures correctness but increases memory usage."
                        # )
                        entropy = vocab_parallel_entropy(logits)
                        ret["entropy"] = entropy
                    else:
                        logits_bak = logits
                    log_probs = vocab_parallel_log_probs_from_logits(logits_bak, label)
                    log_probs = log_probs.masked_fill(~label_mask, 0.0)
                    ret["log_probs"] = log_probs
                    return ret

                logits_processor_args = {"label": label, "label_mask": label_mask}
                output = forward_fn(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    multi_modal_inputs=multi_modal_inputs,
                    logits_processor=logits_processor,
                    logits_processor_args=logits_processor_args,
                    data_format="thd" if self.config.megatron.use_remove_padding else "bshd",
                    mtp_config=None if forward_only else self.mtp_config,
                )

            if forward_only:
                meta_info = None
            else:
                clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                meta_info = {
                    "clip_ratio": self.config.clip_ratio,
                    "entropy_coeff": self.config.entropy_coeff,
                    "clip_ratio_c": clip_ratio_c,
                }

            if RouterReplayHelper.is_r2_record_action(self.tf_config, vp_rank):
                merge_router_topk_indices(
                    attention_mask, input_ids, self.mini_layer_topk_idx_list, self.tf_config, vp_rank
                )

            if RouterReplayHelper.is_replay_forward_action(self.tf_config, vp_rank):
                router_instance_list = RouterReplayHelper.get_micro_batch_router_list(self.tf_config, vp_rank)
                for router in router_instance_list:
                    router.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)

            return output, partial(loss_func, data=batch, meta_info=meta_info)

        # batch should be a list of batches inside micro-batches
        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.actor_module))

        # TODO: we may use the new schedule instead
        # for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.actor_module,
                num_microbatches=n_micro_batch,
                seq_length=total_seqlen,  # no use when input_shapes was set
                micro_batch_size=1,  # no use when input_shapes was set
                forward_only=forward_only,
            )
        else:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.actor_module,
                num_microbatches=n_micro_batch,
                seq_length=total_seqlen,  # in use for pp = 1
                micro_batch_size=1,  # in use for pp = 1
                forward_only=forward_only,
            )
        # loss_reduces contains the stats returned from loss_func

        if self.has_multi_modal_inputs:
            data.batch.pop("multi_modal_inputs")
            data.batch.pop("multi_modal_inputs_idx")
            data.non_tensor_batch.pop("multi_modal_inputs")

        losses_reduced = {"output": losses_reduced}
        if use_dynamic_bsz:
            losses_reduced["indices"] = indices
        if RouterReplayHelper.is_r2_record_action(self.tf_config):
            if self.tf_config.virtual_pipeline_model_parallel_size is not None:
                # config = self.actor_module[0].module.module.config
                vp_size = len(self.actor_module)
                microbatch_group_size_per_vp_stage = self.tf_config.microbatch_group_size_per_vp_stage
                bs = n_micro_batch
                losses_reduced["mini_layer_topk_idx_tensor"] = reorder_and_merge_vpp_layers(
                    self.mini_layer_topk_idx_list, bs, vp_size, microbatch_group_size_per_vp_stage
                )
            else:
                losses_reduced["mini_layer_topk_idx_tensor"] = torch.cat(self.mini_layer_topk_idx_list, dim=0)
            self.mini_layer_topk_idx_list = []

        # Collect and pass MTP metrics to losses_reduced
        if not forward_only and self.mtp_config and self.mtp_config.enable_train:
            metrics = get_megatron_mtp_loss(n_micro_batch)
            losses_reduced["mtp_losses"] = [metrics]

        return losses_reduced

    @GPUMemoryLogger(role="megatron actor", logger=logger)
    def update_policy(self, dataloader: Iterable[DataProto], enable_mtp: bool = False) -> dict:
        """Update the policy with an iterator of DataProto

        Args:
            dataloader (Iterable[DataProto]): an iterator over the DataProto that returns by ``make_minibatch_iterator``
                The keys of each data batch is described in the make_minibatch_iterator.

            enable_mtp (bool, optional): whether to enable MTP communication

        Returns:
            Dict: a dictionary containing the statistics. Note that the statistics are only valid in the last pp stage
            and users have to combine the output in each dp rank manually.

        """
        metrics = {}
        for data in dataloader:
            if self.config.router_replay.mode in ["R2", "R3"]:
                RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
            self.actor_optimizer.zero_grad()
            # use use_contiguous_buffers_in_local_ddp and no overlap_dp_param_comm
            for chunk in self.actor_module:
                # if use distributed optimizer, zero grad buffer will be handled by optimizer
                chunk.zero_grad_buffer()

            calculate_entropy = self.config.get("calculate_entropy", False) or (self.config.entropy_coeff != 0)
            if data.meta_info.get("micro_batch_size", None) is not None:
                micro_batch_size = data.meta_info["micro_batch_size"]
            else:
                micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
            max_token_len = None
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.config.megatron.context_parallel_size
            metric_micro_batch = self.forward_backward_batch(
                data,
                calculate_entropy=calculate_entropy,
                use_dynamic_bsz=self.config.use_dynamic_bsz,
                micro_batch_size=micro_batch_size,
                max_token_len=max_token_len,
                mini_batch_size=self.config.ppo_mini_batch_size,
            )

            mtp_losses = metric_micro_batch.get("mtp_losses", None)
            if mtp_losses is not None:
                # mtp_losses is now in format: [{"mtp_losses/mtp_1_loss": [value1], "mtp_losses/mtp_2_loss": [value2]}]
                for mtp_metrics_dict in mtp_losses:
                    append_to_dict(metrics, mtp_metrics_dict)

            metric_micro_batch = metric_micro_batch["output"]
            for metric in metric_micro_batch:
                # Note that o[0] is metrics, o[1] is entropy, o[2] is response_mask
                append_to_dict(metrics, metric[0])  # append the metric from this micro-batch to global metrics.

            update_successful, grad_norm, num_zeros_in_grad = self.actor_optimizer.step()
            data = {"actor/grad_norm": grad_norm}
            append_to_dict(metrics, data)

            if update_successful:
                # allgather already execute in optimizer.step in new megatron
                pass
            else:
                raise NotImplementedError

            if self.config.router_replay.mode in ["R2", "R3"]:
                RouterReplay.clear_global_router_replay_action()
                RouterReplay.clear_global_indices()

        self.actor_optimizer.zero_grad()
        self._maybe_recalibrate_input_amax()
        get_torch_device().empty_cache()
        return metrics

    def _maybe_recalibrate_input_amax(self):
        """Periodically refresh the W4A4/W4A8 activation amax (gated by VERL_W4A4_RECALIB_EVERY).

        The Megatron actor freezes ``input_quantizer._amax`` after one-shot init calibration. For
        W4A4 the activation range drifts as responses grow, so a stale amax mis-scales FP4
        activations and the (vLLM-static) input_scale exported at each weight sync is stale too.
        Every N policy updates we re-run modelopt's max-calibration on the current model to refresh
        the activation amax (FSDP-equivalent adaptive amax). No-op unless the env flag is set and
        the module was calibrated (W4A4/W4A8). Collective across DP/EP/TP — all ranks call it in
        lockstep (one update_policy per global step).
        """
        every = int(getattr(self.config, "recalib_every", 0) or 0)
        if every <= 0:  # env fallback (config is the reliable path; env covers ad-hoc runs)
            every = int(os.environ.get("VERL_W4A4_RECALIB_EVERY", "0") or "0")
        if every <= 0:
            return
        self._recalib_call_count = getattr(self, "_recalib_call_count", 0) + 1
        if self._recalib_call_count % every != 0:
            return
        ctx = None
        for m in self.actor_module:
            ctx = getattr(m, "_qat_recalib_ctx", None)
            if ctx is not None:
                break
        if ctx is None:
            return
        from verl.utils.modelopt.qat_utils import recalibrate_input_amax

        n = recalibrate_input_amax(self.actor_module, ctx["model_path"], ctx["calib_prompts"])
        if torch.distributed.get_rank() == 0:
            print(
                f"[QAT-RECALIB] refreshed activation amax on {n} chunk(s) at "
                f"update #{self._recalib_call_count} (every={every})",
                flush=True,
            )
