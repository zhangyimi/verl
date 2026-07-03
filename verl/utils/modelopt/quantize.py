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

"""ModelOpt NVFP4 quantization config and application for Megatron QAT.

Adapted for modelopt >= 0.44 (list-based ``quant_cfg`` API). The base
configs ship with modelopt and are list-of-dict entries; ``ignore_patterns``
override them by appending ``{"quantizer_name": <pat>, "enable": False}``
to the tail (later entries win).
"""

import copy
import os

import modelopt.torch.quantization as mtq
import torch.nn as nn

_IGNORE_PATTERN_MAPPING = {
    "lm_head": "*output_layer*",
    "*mlp.gate": "*router*",
    "*self_attn*": "*self_attention*",
}


def _ignore_patterns_to_entries(ignore_patterns: list[str]) -> list[dict]:
    entries = []
    for pattern in ignore_patterns:
        key = _IGNORE_PATTERN_MAPPING.get(pattern, pattern)
        entries.append({"quantizer_name": key, "enable": False})
    return entries


def _disable_input_quantizer(quant_cfg_list: list[dict]) -> None:
    """Remove the input quantizer entry entirely.

    Important: just setting ``enable=False`` is NOT sufficient — modelopt's
    ``export_hf_checkpoint`` still emits ``input_activations`` in the resulting
    HF ``quantization_config`` when the entry is present, which makes vLLM
    treat the checkpoint as W4A4 and apply input quantization on top of
    weight-only-trained weights → garbage outputs. Remove the entry instead.
    """
    quant_cfg_list[:] = [
        e for e in quant_cfg_list if not (e.get("quantizer_name") == "*input_quantizer" and "parent_class" not in e)
    ]


def _make_input_quantizer_dynamic(quant_cfg) -> int:
    """Make the activation (input) quantizer PER-FORWARD DYNAMIC (recompute amax every forward,
    no frozen calibrated ``_amax``).

    Rationale: the W4A4 baseline freezes the input_quantizer amax at init calibration, whereas the
    FSDP W4A4 path (which grows response length to ~6.6k before crashing) uses an ADAPTIVE
    running-max activation amax, and the vLLM rollout uses runtime-dynamic activation quant. Making
    the actor's activation amax dynamic matches both. Gated by ``VERL_W4A4_DYNAMIC_ACT_AMAX=1`` so
    the default (frozen) behaviour is unchanged.

    Handles modelopt>=0.44 list-based ``quant_cfg`` (entries ``{"quantizer_name","cfg"}``) and the
    older dict form.

    WARNING: enabling this alone breaks a rollout configured for static activation scales. A
    dynamic actor input_quantizer exports no static ``input_scale``, but the static W4A4 rollout config
    (``nvfp4_w4a4_megatron.json`` input_activations ``dynamic:false``) expects one → vLLM uses
    garbage scales → garbage rollouts → DAPO filter_groups exhausts → crash. To use this you MUST
    also switch the rollout to dynamic activation (``input_activations.dynamic: local``), matching
    FSDP. Also note single-node forward tests showed frozen amax is benign and baseline IS≈0.998,
    so this switch alone is unlikely to resolve a length stall.
    """
    n = 0
    if isinstance(quant_cfg, list):
        for e in quant_cfg:
            if not (isinstance(e, dict) and "input" in str(e.get("quantizer_name", ""))):
                continue
            target = e["cfg"] if isinstance(e.get("cfg"), dict) else e
            if target.get("enable", True) is not False:
                target["type"] = "dynamic"
                n += 1
    elif isinstance(quant_cfg, dict):
        for k, v in quant_cfg.items():
            if "input_quantizer" in str(k) and isinstance(v, dict) and v.get("enable", True):
                v["type"] = "dynamic"
                n += 1
    return n


def build_quantize_config(
    qat_mode: str,
    ignore_patterns: list[str] | None = None,
) -> dict:
    """Build a complete ModelOpt quantization config for ``mtq.quantize``.

    Supported modes:
      - ``w4a16``: NVFP4 weight-only (FP4 E2M1 weights, no input quant).
      - ``w4a4`` : NVFP4 W4A4 (FP4 E2M1 weights + FP4 E2M1 input, block 16 dynamic).
      - ``w4a8`` : NVFP4 W4A8 (FP4 E2M1 weights + FP8 E4M3 input).
    """
    if qat_mode == "w4a4":
        # modelopt's NVFP4_DEFAULT_CFG is already W4A4 (block 16 dynamic on both weight + input).
        cfg = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
        # Accept any truthy string. Note: when injected via Ray runtime_env env_vars (the only
        # reliable way to reach Megatron workers — they do NOT inherit the launcher shell env),
        # the value MUST be a string, so prefer a non-numeric token like "enabled" (hydra coerces
        # bare 1/true/on to int/bool, which Ray rejects for env_vars).
        if os.environ.get("VERL_W4A4_DYNAMIC_ACT_AMAX", "0").strip().lower() in ("1", "true", "on", "yes", "y", "enable", "enabled"):
            n = _make_input_quantizer_dynamic(cfg["quant_cfg"])
            print(
                f"[QAT] VERL_W4A4_DYNAMIC_ACT_AMAX=1 -> {n} input_quantizer entries set to dynamic "
                f"(per-forward activation amax, no frozen calib); matches FSDP adaptive amax + vLLM dynamic act-quant",
                flush=True,
            )
    elif qat_mode == "w4a16":
        # Start from W4A4 default and disable the input_quantizer entry.
        cfg = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
        _disable_input_quantizer(cfg["quant_cfg"])
    elif qat_mode == "w4a8":
        cfg = copy.deepcopy(mtq.W4A8_NVFP4_FP8_CFG)
    else:
        raise ValueError(f"Unsupported qat_mode: {qat_mode!r}. Use 'w4a16', 'w4a4', or 'w4a8'.")

    if ignore_patterns:
        cfg["quant_cfg"].extend(_ignore_patterns_to_entries(ignore_patterns))

    return cfg


def apply_qat(
    model: nn.Module,
    qat_mode: str,
    ignore_patterns: list[str] | None = None,
) -> nn.Module:
    """Apply Quantization-Aware Training to a Megatron model."""
    config = build_quantize_config(qat_mode, ignore_patterns)
    mtq.quantize(model, config)
    return model
