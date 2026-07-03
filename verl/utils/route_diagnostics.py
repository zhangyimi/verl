"""Rollout-vs-FSDP MoE routing diagnostics and FSDP R3 replay.

The diagnostic is deliberately opt-in via ``VERL_ROUTE_DIAG=1``.  It records
only top-k expert ids and aggregate counts; full router logits never leave the
layer hook.  The same hook also implements production R3 replay when the FSDP
engine requests it.  Production replay deliberately keeps the hook active
until backward has completed so activation-checkpoint recomputation observes
the same rollout routes as the original forward.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch


_GATE_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.gate$")


def route_diag_enabled() -> bool:
    return os.environ.get("VERL_ROUTE_DIAG", "0") == "1"


def _nested_values(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.values() if getattr(tensor, "is_nested", False) else tensor


def compact_routed_experts(value: Any) -> torch.Tensor:
    """Validate one rollout route buffer and store expert ids as ``uint8``.

    Qwen3-30B has 128 experts, so an expert id is losslessly represented by one
    byte.  Casting before allocating/padding the per-sample route tensor avoids
    an 8x host-memory and Ray-transfer expansion when a backend returns int64.
    An all-zero row is a supported padding sentinel, but an entirely all-zero
    buffer cannot be a valid top-k capture (a real top-k set has distinct ids).
    """

    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"unsupported routed_experts type: {type(value)}")
    if value.ndim != 3:
        raise ValueError(f"expected routed_experts [tokens,layers,topk], got {tuple(value.shape)}")
    if value.shape[-1] <= 1:
        raise ValueError(f"invalid routed_experts top-k dimension: {tuple(value.shape)}")
    if value.numel() == 0 or int(torch.count_nonzero(value)) == 0:
        raise RuntimeError("rollout returned an empty/all-zero routed_experts buffer")
    if value.dtype.is_floating_point or value.dtype == torch.bool:
        raise TypeError(f"routed_experts must contain integer ids, got {value.dtype}")
    route_min = int(value.min())
    route_max = int(value.max())
    if route_min < 0 or route_max > 255:
        raise ValueError(f"routed_experts ids must fit uint8, got range [{route_min},{route_max}]")
    return value.to(torch.uint8)


def _route_scope_masks(micro_batch, rollout: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build fixed-prompt and response-predictor masks in unpadded token order.

    veRL currently keeps ``loss_mask`` as a dense ``[batch, max_response]``
    tensor after converting the rest of the batch to no-padding jagged tensors.
    Older paths may instead provide a full-sequence jagged loss mask.  Support
    both representations explicitly.
    """

    loss_mask = micro_batch.get("loss_mask", None)
    if loss_mask is None:
        raise KeyError("route diagnostics require loss_mask in the FSDP micro-batch")

    rollout_values = _nested_values(rollout)
    token_count = rollout_values.shape[0]
    loss_values = _nested_values(loss_mask).bool().reshape(-1)

    # Full-sequence jagged representation: loss_mask aligns one-to-one with
    # routed tokens, so a within-sequence left shift gives predictor positions.
    if getattr(loss_mask, "is_nested", False) and loss_values.numel() == token_count:
        offsets = loss_mask.offsets().detach().cpu().tolist()
        response = torch.zeros_like(loss_values, dtype=torch.bool)
        for start, end in zip(offsets[:-1], offsets[1:], strict=True):
            if end - start > 1:
                response[start : end - 1] = loss_values[start + 1 : end]
        return {"prompt": ~loss_values, "response": response}

    # Current no-padding representation: dense loss_mask contains response
    # positions only.  Recover prompt/response boundaries from each unpadded
    # sequence's offsets and response-token count.  This exactly mirrors
    # no_padding_2_padding(): [seq_end-resp_len-1 : seq_end-1].
    offsets_tensor = None
    if getattr(rollout, "is_nested", False):
        offsets_tensor = rollout.offsets()
    else:
        input_ids = micro_batch.get("input_ids", None)
        if input_ids is not None and getattr(input_ids, "is_nested", False):
            offsets_tensor = input_ids.offsets()
    if offsets_tensor is None:
        raise ValueError("dense response loss_mask requires jagged routed_experts or input_ids offsets")

    offsets = offsets_tensor.detach().cpu().tolist()
    batch_size = len(offsets) - 1
    if getattr(loss_mask, "is_nested", False):
        loss_offsets = loss_mask.offsets().detach().cpu().tolist()
        response_lengths = [
            int(loss_values[start:end].sum().item())
            for start, end in zip(loss_offsets[:-1], loss_offsets[1:], strict=True)
        ]
    else:
        if loss_mask.ndim == 1:
            loss_mask = loss_mask.unsqueeze(0)
        if loss_mask.shape[0] != batch_size:
            raise ValueError(
                f"loss_mask batch={loss_mask.shape[0]} does not match routed batch={batch_size}"
            )
        response_lengths = loss_mask.bool().reshape(batch_size, -1).sum(dim=-1).detach().cpu().tolist()

    prompt = torch.zeros(token_count, dtype=torch.bool, device=rollout_values.device)
    response = torch.zeros_like(prompt)
    for start, end, response_length in zip(offsets[:-1], offsets[1:], response_lengths, strict=True):
        response_length = int(response_length)
        sequence_length = end - start
        if response_length < 0 or response_length >= sequence_length:
            raise ValueError(
                f"invalid response length {response_length} for unpadded sequence length {sequence_length}"
            )
        prompt_end = end - response_length
        prompt[start:prompt_end] = True
        if response_length:
            response[prompt_end - 1 : end - 1] = True
    return {"prompt": prompt, "response": response}


class FSDPRouteCapture:
    """Capture or replay FSDP top-k routes from Qwen MoE gate outputs.

    Replay is intentionally implemented at the gate output instead of inside a
    particular transformers MoE block.  Non-selected logits are masked to
    ``-inf`` while the selected logits are left untouched.  Consequently the
    downstream top-k uses the rollout expert ids and the usual normalization
    computes weights from the current FSDP gate scores for those experts.  This
    matches R3 semantics and keeps gradients to the selected gate logits.
    """

    def __init__(self, module: torch.nn.Module, topk: int = 8):
        self.topk = int(topk)
        self.active = False
        self.collect_metrics = True
        self._routes: dict[int, torch.Tensor] = {}
        self._margins: dict[int, torch.Tensor] = {}
        self._seen_layers: set[int] = set()
        self._execution_counts: dict[int, int] = {}
        self._forced_routes: torch.Tensor | None = None
        self._allow_reentry = False
        self._require_complete_routes = False
        self._require_checkpoint_reentry = False
        self._handles = []
        self.layer_ids: list[int] = []

        matches: list[tuple[int, str, torch.nn.Module]] = []
        for name, submodule in module.named_modules():
            match = _GATE_RE.search(name)
            if match is not None:
                matches.append((int(match.group(1)), name, submodule))

        matches.sort(key=lambda item: item[0])
        if not matches:
            raise RuntimeError("VERL_ROUTE_DIAG=1 but no '*.layers.<id>.mlp.gate' modules were found")
        if len({layer_id for layer_id, _, _ in matches}) != len(matches):
            names = [name for _, name, _ in matches]
            raise RuntimeError(f"duplicate MoE gate layer ids in route diagnostics: {names}")

        self.layer_ids = [layer_id for layer_id, _, _ in matches]
        self._layer_to_route_column = {layer_id: idx for idx, layer_id in enumerate(self.layer_ids)}
        for layer_id, _, submodule in matches:
            self._handles.append(submodule.register_forward_hook(self._make_hook(layer_id)))

        print(
            f"[route-replay] registered {len(self.layer_ids)} FSDP MoE gate hooks; "
            f"layers={self.layer_ids[0]}..{self.layer_ids[-1]} topk={self.topk}",
            flush=True,
        )

    def _make_hook(self, layer_id: int):
        def hook(_module, _inputs, output):
            if not self.active:
                return
            already_seen = layer_id in self._seen_layers
            if already_seen and not self._allow_reentry:
                raise RuntimeError(f"route-diag gate layer {layer_id} executed more than once in one forward")
            self._seen_layers.add(layer_id)
            self._execution_counts[layer_id] = self._execution_counts.get(layer_id, 0) + 1

            raw_scores = output[0] if isinstance(output, tuple) else output
            if not isinstance(raw_scores, torch.Tensor) or raw_scores.ndim < 2:
                raise TypeError(f"unexpected gate output for layer {layer_id}: {type(raw_scores)}")
            scores = raw_scores.reshape(-1, raw_scores.shape[-1])
            if scores.shape[-1] <= self.topk:
                raise ValueError(
                    f"gate layer {layer_id} has {scores.shape[-1]} experts, need at least topk+1={self.topk + 1}"
                )

            # Checkpoint recomputation executes the gate again during backward.
            # Capture diagnostic counts only on the original forward, while
            # applying the forced mask on every execution.
            if self.collect_metrics and not already_seen:
                values, indices = torch.topk(scores.float(), k=self.topk + 1, dim=-1, largest=True, sorted=True)
                if scores.shape[-1] > 256:
                    raise ValueError(
                        "route diagnostics currently store expert ids as uint8, but num_experts exceeds 256"
                    )
                self._routes[layer_id] = indices[:, : self.topk].to(torch.uint8)
                self._margins[layer_id] = (values[:, self.topk - 1] - values[:, self.topk]).detach()

            if self._forced_routes is None:
                return

            route_column = self._layer_to_route_column[layer_id]
            forced_ids = self._forced_routes[:, route_column, :].to(device=scores.device, dtype=torch.long)
            if forced_ids.shape != (scores.shape[0], self.topk):
                raise ValueError(
                    f"forced route shape {tuple(forced_ids.shape)} at layer {layer_id}, "
                    f"expected {(scores.shape[0], self.topk)}"
                )
            valid = forced_ids.ne(0).any(dim=-1)
            if valid.any() and forced_ids[valid].max() >= scores.shape[-1]:
                raise ValueError(f"forced expert id outside [0,{scores.shape[-1]}) at layer {layer_id}")
            selected = torch.zeros_like(scores, dtype=torch.bool)
            selected.scatter_(1, forced_ids.clamp(min=0, max=scores.shape[-1] - 1), True)
            forced_scores = scores.masked_fill(valid.unsqueeze(-1) & ~selected, -torch.inf)
            forced_scores = forced_scores.reshape_as(raw_scores)
            if isinstance(output, tuple):
                return (forced_scores, *output[1:])
            return forced_scores

        return hook

    def begin(
        self,
        micro_batch=None,
        *,
        force_rollout_routes: bool = False,
        collect_metrics: bool = True,
        allow_reentry: bool = False,
        require_complete_routes: bool = False,
        require_checkpoint_reentry: bool = False,
    ) -> None:
        if self.active:
            raise RuntimeError("route diagnostic capture is already active")
        self._routes.clear()
        self._margins.clear()
        self._seen_layers.clear()
        self._execution_counts.clear()
        self._forced_routes = None
        self.collect_metrics = bool(collect_metrics)
        self._allow_reentry = bool(allow_reentry)
        self._require_complete_routes = bool(require_complete_routes)
        self._require_checkpoint_reentry = bool(require_checkpoint_reentry)
        if self._require_checkpoint_reentry and not self._allow_reentry:
            raise ValueError("checkpoint route replay requires allow_reentry=True")
        if force_rollout_routes:
            if micro_batch is None or micro_batch.get("routed_experts", None) is None:
                raise KeyError("forced route replay requires routed_experts in the FSDP micro-batch")
            forced_routes = _nested_values(micro_batch["routed_experts"])
            if forced_routes.ndim != 3:
                raise ValueError(
                    f"expected forced routes [tokens,layers,topk], got {tuple(forced_routes.shape)}"
                )
            if forced_routes.shape[1:] != (len(self.layer_ids), self.topk):
                raise ValueError(
                    "forced route layer/topk mismatch: "
                    f"routes={tuple(forced_routes.shape)}, layers={len(self.layer_ids)}, topk={self.topk}"
                )
            if forced_routes.dtype.is_floating_point or forced_routes.dtype == torch.bool:
                raise TypeError(f"forced routes must contain integer ids, got {forced_routes.dtype}")
            valid_rows = forced_routes.ne(0).any(dim=-1)
            if valid_rows.any():
                valid_ids = forced_routes[valid_rows].to(torch.long)
                if valid_ids.min() < 0 or valid_ids.max() >= 256:
                    raise ValueError("forced expert id is outside the uint8 range [0,256)")
                duplicate = valid_ids.sort(dim=-1).values.diff(dim=-1).eq(0).any(dim=-1)
                if duplicate.any():
                    raise ValueError("forced routes contain duplicate expert ids in a valid top-k row")

            if self._require_complete_routes:
                valid_all_layers = valid_rows.all(dim=-1)
                scope_masks = _route_scope_masks(micro_batch, micro_batch["routed_experts"])
                missing = {
                    scope: int((scope_mask.to(valid_all_layers.device) & ~valid_all_layers).sum().item())
                    for scope, scope_mask in scope_masks.items()
                }
                empty = {
                    scope: int(scope_mask.sum().item())
                    for scope, scope_mask in scope_masks.items()
                    if int(scope_mask.sum().item()) == 0
                }
                if empty or any(count > 0 for count in missing.values()):
                    raise RuntimeError(
                        "R3 requires complete prompt/response-predictor routes; "
                        f"empty_scopes={empty}, missing_rows={missing}. "
                        "Disable rollout prefix caching and verify route capture."
                    )

            self._forced_routes = forced_routes
        self.active = True

    def abort(self) -> None:
        self.active = False
        self._routes.clear()
        self._margins.clear()
        self._seen_layers.clear()
        self._execution_counts.clear()
        self._forced_routes = None
        self._allow_reentry = False
        self._require_complete_routes = False
        self._require_checkpoint_reentry = False

    def end_replay(self) -> None:
        """End a replay spanning forward and backward/checkpoint recomputation."""

        if not self.active:
            raise RuntimeError("route replay is not active")
        missing = [layer_id for layer_id in self.layer_ids if layer_id not in self._seen_layers]
        if missing:
            self.abort()
            raise RuntimeError(f"route replay hooks did not fire for layers: {missing}")
        if self._require_checkpoint_reentry:
            not_recomputed = [
                layer_id for layer_id in self.layer_ids if self._execution_counts.get(layer_id, 0) < 2
            ]
            if not_recomputed:
                counts = {layer_id: self._execution_counts.get(layer_id, 0) for layer_id in not_recomputed}
                self.abort()
                raise RuntimeError(
                    "R3 checkpoint replay expected every MoE gate to execute in forward and backward "
                    f"recomputation; layers with fewer than two executions: {counts}"
                )
        self.abort()

    @torch.no_grad()
    def finish(self, micro_batch) -> dict[str, float]:
        """Compare captured FSDP routes with rollout routes in one micro-batch."""

        self.active = False
        try:
            missing = [layer_id for layer_id in self.layer_ids if layer_id not in self._seen_layers]
            if missing:
                raise RuntimeError(f"route diagnostic hooks did not fire for layers: {missing}")
            if not self.collect_metrics:
                return {}

            rollout = micro_batch.get("routed_experts", None)
            if rollout is None or micro_batch.get("loss_mask", None) is None:
                raise KeyError("route diagnostics require routed_experts and loss_mask in the FSDP micro-batch")

            rollout_values = _nested_values(rollout)
            if rollout_values.ndim != 3:
                raise ValueError(f"expected rollout routes [tokens,layers,topk], got {tuple(rollout_values.shape)}")
            if rollout_values.shape[1] != len(self.layer_ids) or rollout_values.shape[2] != self.topk:
                raise ValueError(
                    "rollout/FSDP route shape mismatch: "
                    f"rollout={tuple(rollout_values.shape)}, fsdp_layers={len(self.layer_ids)}, topk={self.topk}"
                )

            token_count = rollout_values.shape[0]
            for layer_id in self.layer_ids:
                if self._routes[layer_id].shape != (token_count, self.topk):
                    raise ValueError(
                        f"layer {layer_id} captured {tuple(self._routes[layer_id].shape)}, "
                        f"expected {(token_count, self.topk)}"
                    )

            metrics: dict[str, float] = {}
            scope_masks = _route_scope_masks(micro_batch, rollout)

            for scope, scope_mask in scope_masks.items():
                exact_columns = []
                valid_columns = []
                rows = []
                for route_layer, layer_id in enumerate(self.layer_ids):
                    train_ids = self._routes[layer_id]
                    rollout_ids = rollout_values[:, route_layer, :].to(device=train_ids.device, dtype=torch.uint8)
                    # Padded/missing route rows are all-zero; a real top-k set cannot contain duplicate zeros.
                    valid = scope_mask.to(train_ids.device) & rollout_ids.ne(0).any(dim=-1)

                    train_sorted = train_ids.sort(dim=-1).values
                    rollout_sorted = rollout_ids.sort(dim=-1).values
                    exact = train_sorted.eq(rollout_sorted).all(dim=-1) & valid
                    intersection = (
                        train_ids.unsqueeze(-1).eq(rollout_ids.unsqueeze(-2)).any(dim=-1).sum(dim=-1)
                    )
                    flip = valid & ~exact
                    match = exact
                    margin = self._margins[layer_id]
                    rows.append(
                        torch.stack(
                            [
                                (intersection * valid).sum().float(),
                                exact.sum().float(),
                                valid.sum().float(),
                                (margin * flip).sum().float(),
                                (margin * match).sum().float(),
                                flip.sum().float(),
                                match.sum().float(),
                            ]
                        )
                    )
                    exact_columns.append(exact)
                    valid_columns.append(valid)

                counts = torch.stack(rows, dim=0)
                exact_matrix = torch.stack(exact_columns, dim=-1)
                valid_matrix = torch.stack(valid_columns, dim=-1)
                path_exact = torch.cumprod(exact_matrix.to(torch.int32), dim=-1).bool() & valid_matrix
                path_valid = torch.cumprod(valid_matrix.to(torch.int32), dim=-1).bool()
                all_valid = valid_matrix.all(dim=-1)
                any_flip = all_valid & ~exact_matrix.all(dim=-1)

                counts_cpu = counts.detach().cpu()
                path_exact_cpu = path_exact.sum(dim=0).detach().cpu()
                path_valid_cpu = path_valid.sum(dim=0).detach().cpu()
                for row_idx, layer_id in enumerate(self.layer_ids):
                    prefix = f"route_diag_raw/{scope}/layer_{layer_id:02d}"
                    metrics[f"{prefix}/intersection"] = float(counts_cpu[row_idx, 0])
                    metrics[f"{prefix}/exact"] = float(counts_cpu[row_idx, 1])
                    metrics[f"{prefix}/count"] = float(counts_cpu[row_idx, 2])
                    metrics[f"{prefix}/margin_flip_sum"] = float(counts_cpu[row_idx, 3])
                    metrics[f"{prefix}/margin_match_sum"] = float(counts_cpu[row_idx, 4])
                    metrics[f"{prefix}/flip_count"] = float(counts_cpu[row_idx, 5])
                    metrics[f"{prefix}/match_count"] = float(counts_cpu[row_idx, 6])
                    metrics[f"{prefix}/path_exact"] = float(path_exact_cpu[row_idx])
                    metrics[f"{prefix}/path_count"] = float(path_valid_cpu[row_idx])

                metrics[f"route_diag_raw/{scope}/token_any_flip"] = float(any_flip.sum().detach().cpu())
                metrics[f"route_diag_raw/{scope}/token_all_layer_valid"] = float(all_valid.sum().detach().cpu())
            return metrics
        finally:
            self.abort()


def route_replay_counterfactual_metrics(
    rollout_log_probs: torch.Tensor,
    natural_log_probs: torch.Tensor,
    replay_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, float]:
    """Measure how much forced rollout routes close the actor/rollout gap.

    All statistics are response-token-only.  The primary decision metric is
    ``logprob_abs_mean_recovery``: the fraction of the baseline mean absolute
    log-probability gap removed by routing replay.  Positive values indicate
    recovery; values near zero mean route ids are not the main mismatch source.
    """

    if not (
        rollout_log_probs.shape
        == natural_log_probs.shape
        == replay_log_probs.shape
        == response_mask.shape
    ):
        raise ValueError(
            "route replay counterfactual shape mismatch: "
            f"rollout={tuple(rollout_log_probs.shape)}, natural={tuple(natural_log_probs.shape)}, "
            f"replay={tuple(replay_log_probs.shape)}, mask={tuple(response_mask.shape)}"
        )

    mask = response_mask.bool()
    rollout = rollout_log_probs.float()[mask]
    natural = natural_log_probs.float()[mask]
    replay = replay_log_probs.float()[mask]
    finite = torch.isfinite(rollout) & torch.isfinite(natural) & torch.isfinite(replay)
    rollout, natural, replay = rollout[finite], natural[finite], replay[finite]
    if rollout.numel() == 0:
        raise RuntimeError("route replay counterfactual has no finite response-token log probabilities")

    natural_abs = (natural - rollout).abs()
    replay_abs = (replay - rollout).abs()
    route_change = (replay - natural).abs()
    natural_ratio_abs = (natural - rollout).abs()
    replay_ratio_abs = (replay - rollout).abs()

    def stats(prefix: str, values: torch.Tensor) -> dict[str, float]:
        quantiles = torch.quantile(values, torch.tensor([0.5, 0.9, 0.95, 0.99], device=values.device))
        return {
            f"{prefix}/mean": float(values.mean()),
            f"{prefix}/p50": float(quantiles[0]),
            f"{prefix}/p90": float(quantiles[1]),
            f"{prefix}/p95": float(quantiles[2]),
            f"{prefix}/p99": float(quantiles[3]),
            f"{prefix}/max": float(values.max()),
        }

    prefix = "route_replay_cf"
    metrics = {
        f"{prefix}/token_count": float(rollout.numel()),
        f"{prefix}/finite_fraction": float(finite.float().mean()),
        f"{prefix}/improved_token_fraction": float((replay_abs < natural_abs).float().mean()),
        f"{prefix}/worsened_token_fraction": float((replay_abs > natural_abs).float().mean()),
        f"{prefix}/natural/ratio_outside_2x_fraction": float(
            (natural_ratio_abs > np.log(2.0)).float().mean()
        ),
        f"{prefix}/replay/ratio_outside_2x_fraction": float((replay_ratio_abs > np.log(2.0)).float().mean()),
    }
    metrics.update(stats(f"{prefix}/natural/logprob_abs", natural_abs))
    metrics.update(stats(f"{prefix}/replay/logprob_abs", replay_abs))
    metrics.update(stats(f"{prefix}/route_only/logprob_abs_change", route_change))

    natural_mean = float(natural_abs.mean())
    replay_mean = float(replay_abs.mean())
    natural_p95 = float(torch.quantile(natural_abs, 0.95))
    replay_p95 = float(torch.quantile(replay_abs, 0.95))
    metrics[f"{prefix}/logprob_abs_mean_recovery"] = (
        (natural_mean - replay_mean) / natural_mean if natural_mean > 0 else float("nan")
    )
    metrics[f"{prefix}/logprob_abs_p95_recovery"] = (
        (natural_p95 - replay_p95) / natural_p95 if natural_p95 > 0 else float("nan")
    )
    return metrics


def _flatten_numbers(value: Any) -> Iterable[float]:
    if value is None:
        return
    if isinstance(value, torch.Tensor):
        for item in value.detach().cpu().reshape(-1).tolist():
            yield float(item)
        return
    if isinstance(value, np.ndarray):
        for item in value.reshape(-1).tolist():
            yield float(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_numbers(item)
        return
    if isinstance(value, (int, float, np.number)):
        yield float(value)


def finalize_route_diag_metrics(worker_metrics: dict[str, Any]) -> dict[str, float]:
    """Reduce nested per-microbatch/per-rank raw counts into W&B ratios."""

    raw = {
        key: sum(_flatten_numbers(value))
        for key, value in worker_metrics.items()
        if key.startswith("route_diag_raw/")
    }
    if not raw:
        return {}

    metrics: dict[str, float] = {}
    all_layer_ids: set[int] = set()
    for scope in ("prompt", "response"):
        layer_ids = sorted(
            {
                int(match.group(1))
                for key in raw
                if (match := re.match(rf"route_diag_raw/{scope}/layer_(\d+)/", key)) is not None
            }
        )
        all_layer_ids.update(layer_ids)
        overlap_num = exact_num = layer_denom = 0.0

        for layer_id in layer_ids:
            raw_prefix = f"route_diag_raw/{scope}/layer_{layer_id:02d}"
            out_prefix = f"route_diag/{scope}/layer_{layer_id:02d}"
            count = raw.get(f"{raw_prefix}/count", 0.0)
            intersection = raw.get(f"{raw_prefix}/intersection", 0.0)
            exact = raw.get(f"{raw_prefix}/exact", 0.0)
            flip_count = raw.get(f"{raw_prefix}/flip_count", 0.0)
            match_count = raw.get(f"{raw_prefix}/match_count", 0.0)
            path_count = raw.get(f"{raw_prefix}/path_count", 0.0)

            metrics[f"{out_prefix}/set_overlap"] = intersection / (8.0 * count) if count else float("nan")
            metrics[f"{out_prefix}/exact_match"] = exact / count if count else float("nan")
            metrics[f"{out_prefix}/flip_fraction"] = 1.0 - exact / count if count else float("nan")
            metrics[f"{out_prefix}/path_exact_match"] = (
                raw.get(f"{raw_prefix}/path_exact", 0.0) / path_count if path_count else float("nan")
            )
            metrics[f"{out_prefix}/margin_flip_mean"] = (
                raw.get(f"{raw_prefix}/margin_flip_sum", 0.0) / flip_count if flip_count else float("nan")
            )
            metrics[f"{out_prefix}/margin_match_mean"] = (
                raw.get(f"{raw_prefix}/margin_match_sum", 0.0) / match_count if match_count else float("nan")
            )

            overlap_num += intersection
            exact_num += exact
            layer_denom += count

        metrics[f"route_diag/{scope}/set_overlap_mean"] = (
            overlap_num / (8.0 * layer_denom) if layer_denom else float("nan")
        )
        metrics[f"route_diag/{scope}/exact_match_mean"] = (
            exact_num / layer_denom if layer_denom else float("nan")
        )
        metrics[f"route_diag/{scope}/flip_fraction_mean"] = (
            1.0 - exact_num / layer_denom if layer_denom else float("nan")
        )
        all_valid = raw.get(f"route_diag_raw/{scope}/token_all_layer_valid", 0.0)
        metrics[f"route_diag/{scope}/token_any_flip_fraction"] = (
            raw.get(f"route_diag_raw/{scope}/token_any_flip", 0.0) / all_valid if all_valid else float("nan")
        )
        metrics[f"route_diag/{scope}/token_count"] = all_valid

    # Backward-compatible response aliases make the primary decision metrics easy to compare.
    for name in ("set_overlap_mean", "exact_match_mean", "flip_fraction_mean", "token_any_flip_fraction"):
        metrics[f"route_diag/{name}"] = metrics.get(f"route_diag/response/{name}", float("nan"))
    metrics["route_diag/response_token_count"] = metrics.get("route_diag/response/token_count", 0.0)
    metrics["route_diag/num_layers"] = float(len(all_layer_ids))
    return metrics
