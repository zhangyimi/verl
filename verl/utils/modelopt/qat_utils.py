# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""High-level QAT workflow helpers for Megatron backend."""

import logging
import os


logger = logging.getLogger(__name__)
_QAT_DEBUG_ENABLED = os.environ.get("VERL_QAT_DEBUG", "0") == "1"


def _qat_debug(message: str) -> None:
    if _QAT_DEBUG_ENABLED:
        print(f"[QAT-DEBUG] {message}", flush=True)


def patch_provider_for_qat(provider):
    """Patch the Megatron-Bridge provider to support QAT quantized layers."""
    from megatron.bridge.models.conversion.param_mapping import AutoMapping
    from megatron.bridge.models.gpt_provider import modelopt_transformer_layer_spec

    from verl.utils.modelopt.megatron_qat_patch import apply_qat_patch

    provider.transformer_layer_spec = modelopt_transformer_layer_spec
    apply_qat_patch()
    AutoMapping.register_module_type("QuantColumnParallelLinear", "column")
    AutoMapping.register_module_type("QuantRowParallelLinear", "row")


def _get_qat_field(qat_config, key, default=None):
    """Extract a field from qat_config, supporting both dict and object-style access."""
    if isinstance(qat_config, dict):
        return qat_config.get(key, default)
    return getattr(qat_config, key, default)


def apply_qat_to_modules(modules, qat_config, model_path=None):
    """Apply ModelOpt fake quantization to a list of Megatron module chunks.

    For W4A4/W4A8, the input_quantizer needs a calibrated per-layer ``_amax`` so the
    exported (static, global) ``input_scale`` is correct. We do this the modelopt-canonical
    way: pass a real-data ``forward_loop`` to ``mtq.quantize`` so each input_quantizer
    collects its true activation amax during calibration mode.

    NOTE: a plain forward in QUANT mode does NOT collect amax for a dynamic NVFP4 input
    quantizer — only modelopt's calibration mode (entered via ``forward_loop``) does. The
    previous random-token "warmup" left amax at a constant garbage value: all observed
    layers used amax=100, substantially larger than values obtained with real calibration.
    """
    from verl.utils.modelopt.quantize import apply_qat, build_quantize_config

    qat_mode = _get_qat_field(qat_config, "mode", "w4a16")
    ignore_patterns = _get_qat_field(qat_config, "ignore_patterns", None)
    if ignore_patterns is not None:
        ignore_patterns = list(ignore_patterns)

    needs_calib = qat_mode in ("w4a4", "w4a8")
    calib_data_path = _get_qat_field(qat_config, "calib_data_path", None)
    calib_size = _get_qat_field(qat_config, "calib_size", 32) or 32
    calib_prompts = _load_calib_prompts(calib_data_path, n=calib_size) if needs_calib else None

    if needs_calib and calib_prompts and model_path is not None:
        import modelopt.torch.quantization as mtq

        config = build_quantize_config(qat_mode, ignore_patterns)
        for i in range(len(modules)):
            forward_loop = _build_calib_forward_loop(model_path, calib_prompts)
            mtq.quantize(modules[i], config, forward_loop=forward_loop)
        _report_input_amax(modules)
        # Stash context so the actor can periodically REFRESH the (static) activation amax
        # during training (gated by VERL_W4A4_RECALIB_EVERY). The Megatron actor freezes the
        # input_quantizer._amax after this one-shot calibration; for W4A4 the activation range
        # drifts as responses grow, so a stale amax mis-scales FP4 activations. See
        # recalibrate_input_amax() for the FSDP-equivalent adaptive-amax mechanism.
        for m in modules:
            m._qat_recalib_ctx = {
                "model_path": model_path,
                "calib_prompts": calib_prompts,
                "qat_mode": qat_mode,
            }
    else:
        if needs_calib:
            logger.warning(
                "W4A4/W4A8 calibration skipped "
                f"(model_path={model_path}, n_prompts={len(calib_prompts) if calib_prompts else 0}). "
                "input_scale will be uncalibrated and accuracy may be degraded."
            )
        for i in range(len(modules)):
            modules[i] = apply_qat(modules[i], qat_mode, ignore_patterns=ignore_patterns)

    return modules


def _load_calib_prompts(calib_data_path: str = None, n: int = 32, max_chars: int = 4096) -> list[str]:
    """Load real calibration prompts from a parquet (qat.calib_data_path, fallback TRAIN_FILE env)."""
    train_file = calib_data_path or os.environ.get("TRAIN_FILE")
    if not train_file or not os.path.exists(train_file):
        _qat_debug(
            "_load_calib_prompts: calib data not found "
            f"(calib_data_path={calib_data_path}, TRAIN_FILE={os.environ.get('TRAIN_FILE')})"
        )
        return []
    try:
        import pandas as pd

        df = pd.read_parquet(train_file)
        col = "prompt" if "prompt" in df.columns else df.columns[0]
        prompts = []
        for v in df[col].tolist():
            if isinstance(v, (list, tuple)):  # chat format [{role, content}, ...]
                txt = " ".join(m.get("content", "") for m in v if isinstance(m, dict))
            else:
                txt = str(v)
            txt = txt.strip()
            if txt:
                prompts.append(txt[:max_chars])
            if len(prompts) >= n:
                break
        _qat_debug(f"_load_calib_prompts: loaded {len(prompts)} prompts from {train_file}")
        return prompts
    except Exception as e:
        logger.warning("Failed to load QAT calibration prompts (%s: %s)", type(e).__name__, e)
        return []


def _build_calib_forward_loop(model_path, calib_prompts, seq_len: int = 512):
    """Build a modelopt forward_loop closure that runs real prompts through the Megatron model.

    Matches the input format the Megatron GPTModel forward accepts
    (``input_ids``, ``position_ids``, causal ``attention_mask``).
    """
    import torch

    def _context_parallel_size():
        try:
            from megatron.core import parallel_state

            return max(1, int(parallel_state.get_context_parallel_world_size()))
        except Exception:
            return 1

    def forward_loop(model):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = "cuda"
        cp_size = _context_parallel_size()
        seq_multiple = max(1, 2 * cp_size)
        aligned_seq_len = ((seq_len + seq_multiple - 1) // seq_multiple) * seq_multiple
        pad_token_id = tok.pad_token_id
        if pad_token_id is None:
            pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0
            tok.pad_token_id = pad_token_id
        _qat_debug(
            f"calib forward_loop: seq_len={aligned_seq_len}, "
            f"cp_size={cp_size}, seq_multiple={seq_multiple}, pad_token_id={pad_token_id}"
        )
        model.eval()
        with torch.no_grad():
            for p in calib_prompts:
                enc = tok(
                    p,
                    return_tensors="pt",
                    truncation=True,
                    max_length=aligned_seq_len,
                    padding="max_length",
                )
                if int(enc.attention_mask.sum().item()) < 2:
                    continue
                ids = enc.input_ids.to(device)
                L = ids.shape[1]
                position_ids = torch.arange(L, device=device, dtype=torch.long).unsqueeze(0)
                attention_mask = torch.triu(torch.ones(1, 1, L, L, device=device, dtype=torch.bool), diagonal=1)
                model(input_ids=ids, position_ids=position_ids, attention_mask=attention_mask)
        model.train()

    return forward_loop


def _report_input_amax(modules) -> None:
    """Log a few calibrated input_quantizer._amax values to confirm calibration worked
    (should be per-layer varied, NOT a constant)."""
    vals = []
    for module in modules:
        for name, sub in module.named_modules():
            iq = getattr(sub, "input_quantizer", None)
            if iq is None or not getattr(iq, "is_enabled", True):
                continue
            amax = getattr(iq, "_amax", None)
            if amax is not None and amax.numel():
                vals.append((name, float(amax.float().reshape(-1)[0].item())))
    if not vals:
        _qat_debug("_report_input_amax: no calibrated input_quantizer found")
        return
    distinct = len(set(round(v, 4) for _, v in vals))
    sample = vals[:6]
    _qat_debug(
        f"_report_input_amax: {len(vals)} input_quantizers calibrated, "
        f"{distinct} distinct amax (>1 = good). sample={[(n.split('.')[-1], round(v, 3)) for n, v in sample]}"
    )


def recalibrate_input_amax(modules, model_path, calib_prompts) -> int:
    """Periodically refresh the (static) activation/input amax of a quantized Megatron model.

    The Megatron W4A4/W4A8 actor calibrates ``input_quantizer._amax`` ONCE at init and then
    freezes it. As RL responses grow, the activation range drifts and the frozen amax mis-scales
    the FP4 activations (vLLM reads this as a static ``input_scale`` at each weight sync, so the
    rollout inherits the stale scale). This re-runs modelopt's own ``max_calibrate`` on the
    *current* model over the fixed calibration prompts to recollect a fresh activation amax —
    the FSDP-equivalent of an actor-side running/adaptive amax + a vLLM-side static-refreshed
    scale. ``max_calibrate`` handles the DP/EP/TP amax all-reduce(MAX) sync via each module's
    ``parallel_state`` (exactly what FSDP did manually with ``sync_qat_input_amax``).

    Only the ACTIVATION amax is changed: weight amax is saved and restored so the weight
    fake-quant grid stays identical to the (working) frozen-weight w4a16 path — this isolates
    the experiment to the activation scale and avoids destabilizing QAT with a mid-training
    weight-grid shift.

    Must be called collectively by all ranks in the DP/EP/TP groups (it issues all_reduce).

    Returns the number of module chunks recalibrated (0 if skipped).
    """
    if not calib_prompts or model_path is None:
        return 0
    from modelopt.torch.quantization.model_calib import max_calibrate

    # 1) Save weight amax — refresh ONLY the activation amax.
    saved = []
    for module in modules:
        for _name, sub in module.named_modules():
            wq = getattr(sub, "weight_quantizer", None)
            if wq is not None and getattr(wq, "_amax", None) is not None:
                saved.append((wq, wq._amax.detach().clone()))

    # 2) Re-run modelopt max-calibration on the current model (enable stats -> forward prompts
    #    -> finish -> sync DP/EP/TP). Same forward_loop used at init, so behavior matches init
    #    calibration but reflects the CURRENT (trained) weights' activation distribution.
    n = 0
    for module in modules:
        forward_loop = _build_calib_forward_loop(model_path, calib_prompts)
        max_calibrate(module, forward_loop, distributed_sync=True)
        n += 1

    # 3) Restore weight amax (activation amax stays refreshed).
    for wq, amax in saved:
        cur = getattr(wq, "_amax", None)
        if cur is not None and cur.shape == amax.shape:
            cur.data.copy_(amax)

    _report_input_amax(modules)
    return n


def export_qat_weights(per_tensor_param, modules, qat_mode, bridge):
    """Process exported weights through QATWeightExporter for quantized weight sync."""
    from verl.utils.modelopt.qat_weight_exporter import QATWeightExporter

    qat_weight_exporter = QATWeightExporter(modules, bridge, qat_mode)
    return qat_weight_exporter.process_weights_iterator(per_tensor_param)
