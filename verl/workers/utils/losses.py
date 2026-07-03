# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl.trainer.diffusion.diffusion_algos import kl_penalty_image
from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.utils.padding import no_padding_2_padding


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        # log_prob and loss mask are nested tensors of shape [bsz, j1]
        # for each sample, loss mask shape is [1, prompt_length + response_length]
        loss_mask = data["loss_mask"]

        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()

        # left-shift the loss mask by one token to align with log_prob
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        # NOTE: loss is averaged over all tokens in the batch across all data parallel groups,
        # For FSDP backend, the loss is directly used for backward; while for Megatron backend,
        # the loss should be scaled by `num_microbatches` for pp schedule.
        loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
    else:
        response_mask = data["response_mask"].to(bool)
        loss = -masked_sum(log_prob, response_mask) / batch_num_tokens * dp_size

    return loss, {}


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
    if dp_group is not None and torch.distributed.is_available() and torch.distributed.is_initialized():
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
                # ActorConfig is frozen in workers, so keep this stateless inside the loss.
                # The batch P95 is already stable enough for the coarse two-bucket denominator.
                p_used = float(p_batch.detach().item())

                t1 = float(getattr(config, "seq_norm_adaptive_threshold_1", 4096.0))
                t2 = float(getattr(config, "seq_norm_adaptive_threshold_2", 8192.0))
                t3 = float(getattr(config, "seq_norm_adaptive_threshold_3", 12288.0))
                if p_used < t1:
                    short_denom = float(getattr(config, "seq_norm_adaptive_denom_1", 8192.0))
                elif p_used < t2:
                    short_denom = float(getattr(config, "seq_norm_adaptive_denom_2", 12288.0))
                elif p_used < t3:
                    short_denom = float(getattr(config, "seq_norm_adaptive_denom_3", 16384.0))
                else:
                    short_denom = float(getattr(config, "seq_norm_adaptive_denom_4", 20480.0))
                tail_denom = float(getattr(config, "seq_norm_adaptive_tail_denom", 20480.0))
                loss_scale_factor = torch.where(
                    response_lengths <= p_used,
                    torch.full_like(response_lengths, short_denom),
                    torch.full_like(response_lengths, tail_denom),
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


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Computes ppo loss from model output (log_prob, entropy, values, etc. ) and old_log_probs from data."""
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    # assumes that if any of the global batch info is set, the policy_loss_fn will
    # normalize using dp_size/global_bsz/global_token; in this case, metric aggregation should be SUM
    # to reflect the mean loss over the global batch
    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    metrics = {}

    # select fields and convert to padded tensor
    fields = ["response_mask", "old_log_probs", "advantages"]
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "rollout_log_probs" in data:
        fields.append("rollout_log_probs")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    data = data.select(*fields).to_padded_tensor()

    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)
    rollout_log_prob = data.get("rollout_log_probs", None)

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics, seg_gate_pos = _apply_gap_guard_policy_loss(
        config=config,
        policy_loss_fn=policy_loss_fn,
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        rollout_is_weights=rollout_is_weights,
        rollout_log_prob=rollout_log_prob,
        dp_group=dp_group,
    )

    # AggregationType.MEAN for pg metrics: assumes policy_loss_fn normalizes by local_bsz/local_tokens
    # Ex: in compute_policy_loss_vanilla, pg_metrics are pg_clipfrac, ppo_kl, pg_clipfrac_lower
    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss

    # add entropy loss
    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
        )
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    policy_loss, aux_metrics = _add_gap_guard_aux_losses(
        config=config,
        policy_loss=policy_loss,
        log_prob=log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
        advantages=advantages,
        loss_agg_mode=loss_agg_mode,
        seg_gate_pos=seg_gate_pos,
        dp_group=dp_group,
        local_segment_align_step_mean=data.get("local_segment_align_step_mean", None),
        local_segment_align_step_clip_ratio=data.get("local_segment_align_step_clip_ratio", None),
        local_segment_align_tail_mass=data.get("local_segment_align_tail_mass", None),
        local_segment_align_global_step=tu.get_non_tensor_data(data, "global_steps", None),
    )
    metrics.update(Metric.from_dict(aux_metrics, aggregation=AggregationType.MEAN))

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    """value loss

    Args:
        config: CriticConfig
        model_output: model output from the model
        data: the input to the model
        dp_group: data paralle group

    Returns:
        value loss
    """
    vpreds = no_padding_2_padding(model_output["values"], data)  # (bsz, response_length)

    # select fields and convert to padded tensor
    data = data.select("values", "returns", "response_mask").to_padded_tensor()
    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
    )

    metrics = {}

    metrics.update(
        {
            "critic/vf_loss": vf_loss.detach().item(),
            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
        }
    )

    return vf_loss, metrics


def diffusion_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Compute loss for diffusion model"""
    log_prob = model_output["log_probs"]

    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    metrics = {}

    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "flow_grpo")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=None,
    )

    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=AggregationType.MEAN)
    policy_loss = pg_loss

    if config.use_kl_loss:
        ref_prev_sample_mean = data["ref_prev_sample_mean"]
        prev_sample_mean = model_output["prev_sample_mean"]
        std_dev_t = model_output["std_dev_t"]
        kl_loss = kl_penalty_image(
            prev_sample_mean=prev_sample_mean, ref_prev_sample_mean=ref_prev_sample_mean, std_dev_t=std_dev_t
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=AggregationType.MEAN)
        metrics["kl_coef"] = config.kl_loss_coef

    gradient_accumulation_steps = tu.get_non_tensor_data(data, "gradient_accumulation_steps", default=None)
    policy_loss = policy_loss / gradient_accumulation_steps

    return policy_loss, metrics
