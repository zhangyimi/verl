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
