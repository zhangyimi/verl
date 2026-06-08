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
    previous random-token "warmup" left amax at a constant garbage value (verified job
    2155941: 252 layers all amax=100, 6-200x too large, ppl +24% vs +13% with real calib).
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
    else:
        if needs_calib:
            print(
                f"[QAT-DEBUG] WARNING: W4A4/W4A8 calibration skipped "
                f"(model_path={model_path}, n_prompts={len(calib_prompts) if calib_prompts else 0}). "
                f"input_scale will be UNCALIBRATED — accuracy will be degraded.",
                flush=True,
            )
        for i in range(len(modules)):
            modules[i] = apply_qat(modules[i], qat_mode, ignore_patterns=ignore_patterns)

    return modules


def _load_calib_prompts(calib_data_path: str = None, n: int = 32, max_chars: int = 4096) -> list[str]:
    """Load real calibration prompts from a parquet (qat.calib_data_path, fallback TRAIN_FILE env)."""
    import os

    train_file = calib_data_path or os.environ.get("TRAIN_FILE")
    if not train_file or not os.path.exists(train_file):
        print(
            f"[QAT-DEBUG] _load_calib_prompts: calib data not found "
            f"(calib_data_path={calib_data_path}, TRAIN_FILE={os.environ.get('TRAIN_FILE')})",
            flush=True,
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
        print(f"[QAT-DEBUG] _load_calib_prompts: loaded {len(prompts)} prompts from {train_file}", flush=True)
        return prompts
    except Exception as e:
        print(f"[QAT-DEBUG] _load_calib_prompts: failed ({type(e).__name__}: {e})", flush=True)
        return []


def _build_calib_forward_loop(model_path, calib_prompts, seq_len: int = 512):
    """Build a modelopt forward_loop closure that runs real prompts through the Megatron model.

    Matches the input format the Megatron GPTModel forward accepts
    (``input_ids``, ``position_ids``, causal ``attention_mask``).
    """
    import torch

    def forward_loop(model):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = "cuda"
        model.eval()
        with torch.no_grad():
            for p in calib_prompts:
                ids = tok(p, return_tensors="pt", truncation=True, max_length=seq_len).input_ids.to(device)
                if ids.shape[1] < 2:
                    continue
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
        print("[QAT-DEBUG] _report_input_amax: no calibrated input_quantizer found", flush=True)
        return
    distinct = len(set(round(v, 4) for _, v in vals))
    sample = vals[:6]
    print(
        f"[QAT-DEBUG] _report_input_amax: {len(vals)} input_quantizers calibrated, "
        f"{distinct} distinct amax (>1 = good). sample={[(n.split('.')[-1], round(v, 3)) for n, v in sample]}",
        flush=True,
    )


def export_qat_weights(per_tensor_param, modules, qat_mode, bridge):
    """Process exported weights through QATWeightExporter for quantized weight sync."""
    from verl.utils.modelopt.qat_weight_exporter import QATWeightExporter

    qat_weight_exporter = QATWeightExporter(modules, bridge, qat_mode)
    return qat_weight_exporter.process_weights_iterator(per_tensor_param)
