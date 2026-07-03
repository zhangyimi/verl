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

"""vLLM ModelOpt NVFP4 patches for dynamic weight updates (Marlin backend)."""

import os
from typing import Optional

import inspect

import torch
from torch.nn import Parameter

from verl.utils.device import get_device_name


_QAT_DEBUG_ENABLED = os.environ.get("VERL_QAT_DEBUG", "0") == "1"


def _qat_debug(message: str) -> None:
    if _QAT_DEBUG_ENABLED:
        print(f"[QAT-DEBUG] {message}", flush=True)


def _save_param_meta(layer: torch.nn.Module, param_name: str):
    if not hasattr(layer, "_hf_param_meta"):
        layer._hf_param_meta = {}

    param = getattr(layer, param_name, None)
    if param is None:
        return

    meta = {
        "shape": tuple(param.shape),
        "dtype": param.dtype,
        "device": str(param.device),
        "param_class": type(param),
    }

    if hasattr(param, "_input_dim"):
        meta["input_dim"] = param._input_dim
    if hasattr(param, "_output_dim"):
        meta["output_dim"] = param._output_dim

    layer._hf_param_meta[param_name] = meta


def _create_param_from_meta(
    module: torch.nn.Module,
    param_name: str,
    meta: dict,
    device: Optional[torch.device] = None,
) -> Parameter:
    shape = meta["shape"]
    dtype = meta["dtype"]
    dev = device or meta.get("device", get_device_name())
    param_class = meta.get("param_class", Parameter)

    weight_loaders = getattr(module, "_weight_loaders", {})
    weight_loader = weight_loaders.get(param_name)

    data = torch.empty(shape, dtype=dtype, device=dev)

    if param_class is not Parameter and weight_loader is not None:
        kwargs = {"data": data, "weight_loader": weight_loader}
        if "input_dim" in meta:
            kwargs["input_dim"] = meta["input_dim"]
        if "output_dim" in meta:
            kwargs["output_dim"] = meta["output_dim"]
        new_param = param_class(**kwargs)
    else:
        new_param = Parameter(data, requires_grad=False)
        if weight_loader is not None:
            new_param.weight_loader = weight_loader

    return new_param


def _check_first_call(layer: torch.nn.Module) -> bool:
    count = getattr(layer, "_process_weights_call_count", 0)
    layer._process_weights_call_count = count + 1
    return count == 0


def _save_weight_loaders(layer: torch.nn.Module, param_names: list[str]):
    if not hasattr(layer, "_weight_loaders"):
        layer._weight_loaders = {}
    for pname in param_names:
        param = getattr(layer, pname, None)
        if param is not None and hasattr(param, "weight_loader"):
            layer._weight_loaders[pname] = param.weight_loader


def _update_ref_or_create(layer, ref_name, new_data):
    refs = getattr(layer, "_marlin_tensor_refs", {})
    ref = refs.get(ref_name)
    if ref is not None:
        ref.copy_(new_data)
        setattr(layer, ref_name, Parameter(ref, requires_grad=False))
    else:
        t = new_data.clone() if isinstance(new_data, torch.Tensor) else torch.tensor(new_data)
        setattr(layer, ref_name, Parameter(t, requires_grad=False))


class ModelOptParamMetaDict(dict):
    """Dict-like parameter store with metadata-based rebuild and tensor swap."""

    def __init__(self, model: torch.nn.Module, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device

        actual_model = model
        if hasattr(model, "model"):
            actual_model = model.model
        self._model = actual_model

        self._layer_meta_cache: dict[str, dict] = {}
        self._tensor_swap_layers: dict[str, dict] = {}

        self._build_mappings()

        for name, param in actual_model.named_parameters():
            self[name] = param

    def _build_mappings(self):
        for layer_name, module in self._model.named_modules():
            if not hasattr(module, "_hf_param_meta"):
                continue

            self._layer_meta_cache[layer_name] = {
                "module": module,
                "meta": module._hf_param_meta,
            }

            marlin_refs = getattr(module, "_marlin_tensor_refs", {})
            for param_name, meta in module._hf_param_meta.items():
                if param_name in marlin_refs:
                    key = f"{layer_name}.{param_name}" if layer_name else param_name
                    self._tensor_swap_layers[key] = {
                        "module": module,
                        "param_name": param_name,
                        "marlin_ref": marlin_refs[param_name],
                        "hf_meta": meta,
                    }

    def _try_rebuild(self, key: str) -> Optional[Parameter]:
        parts = key.rsplit(".", 1)
        if len(parts) != 2:
            return None
        layer_name, param_name = parts
        if layer_name not in self._layer_meta_cache:
            return None
        cache_entry = self._layer_meta_cache[layer_name]
        module = cache_entry["module"]
        meta = cache_entry["meta"]
        if param_name not in meta:
            return None
        if hasattr(module, param_name):
            param = getattr(module, param_name)
            if param is not None:
                return param
        new_param = _create_param_from_meta(module, param_name, meta[param_name], self.device)
        module.register_parameter(param_name, new_param)
        return new_param

    def prepare_for_reload(self) -> None:
        """Replace kernel-format tensors with HF-shape tensors for reload."""
        for _key, swap_info in self._tensor_swap_layers.items():
            module = swap_info["module"]
            param_name = swap_info["param_name"]
            hf_meta = swap_info["hf_meta"]
            if hasattr(module, param_name):
                new_param = _create_param_from_meta(module, param_name, hf_meta, self.device)
                setattr(module, param_name, new_param)

    def __getitem__(self, key: str) -> Parameter:
        if key in dict.keys(self):
            return super().__getitem__(key)
        param = self._try_rebuild(key)
        if param is not None:
            self[key] = param
            return param
        raise KeyError(f"Parameter not found: {key}")

    def __contains__(self, key: str) -> bool:
        if super().__contains__(key):
            return True
        parts = key.rsplit(".", 1)
        if len(parts) == 2:
            layer_name, param_name = parts
            if layer_name in self._layer_meta_cache:
                if param_name in self._layer_meta_cache[layer_name]["meta"]:
                    return True
        return False

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default


_DENSE_HF_PARAMS = ["weight", "weight_scale", "input_scale", "weight_scale_2"]


_KERNEL_FINAL_ATTRS = (
    "weight",
    "weight_scale",
    "weight_global_scale",
    "input_global_scale",
    "alpha",
    "input_global_scale_inv",
)


_DENSE_PWAL_CALL_COUNT = 0


def _modelopt_dense_process_weights(self, layer: torch.nn.Module) -> None:
    """Refit-friendly replacement for ModelOptNvFp4LinearMethod.process_weights_after_loading.

    Strategy: save HF-format metadata, replicate the stock rename/derivation, delegate the
    backend-specific transform to ``self.kernel.process_weights_after_loading`` (so vLLM picks
    Marlin / Cutlass / FlashInfer correctly per hardware), then save refs to the kernel-format
    tensors for CUDA Graph stability across refits.

    First call:  save HF metadata + weight_loaders; rename input_scale → input_global_scale,
                 weight_scale_2 → weight_global_scale; compute alpha + input_global_scale_inv;
                 invoke kernel pwal; save final refs.
    Subsequent:  same flow, but copy_() into saved refs where shapes match.
    """
    is_first_call = _check_first_call(layer)
    global _DENSE_PWAL_CALL_COUNT
    _DENSE_PWAL_CALL_COUNT += 1
    if _DENSE_PWAL_CALL_COUNT <= 3 or _DENSE_PWAL_CALL_COUNT % 100 == 0:
        _qat_debug(
            f"_modelopt_dense_process_weights call #{_DENSE_PWAL_CALL_COUNT} "
            f"is_first={is_first_call} layer={type(layer).__name__} "
            f"weight.shape={tuple(layer.weight.shape) if hasattr(layer, 'weight') else 'NO_WEIGHT'}"
        )

    if is_first_call:
        for pname in _DENSE_HF_PARAMS:
            _save_param_meta(layer, pname)
        _save_weight_loaders(layer, _DENSE_HF_PARAMS)

    # Mirror the stock method's rename + derive logic. (See vllm 0.20
    # ModelOptNvFp4LinearMethod.process_weights_after_loading.)
    input_global_scale = layer.input_scale.max().to(torch.float32)
    weight_global_scale = layer.weight_scale_2.max().to(torch.float32)
    alpha = input_global_scale * weight_global_scale
    input_global_scale_inv = (1.0 / input_global_scale).to(torch.float32)

    layer.input_global_scale = Parameter(input_global_scale, requires_grad=False)
    layer.weight_global_scale = Parameter(weight_global_scale, requires_grad=False)
    layer.alpha = Parameter(alpha, requires_grad=False)
    layer.input_global_scale_inv = Parameter(input_global_scale_inv, requires_grad=False)
    if hasattr(layer, "input_scale"):
        del layer.input_scale
    if hasattr(layer, "weight_scale_2"):
        del layer.weight_scale_2

    # Let the kernel (Marlin / Cutlass / FlashInfer / Emulation) do its backend-specific
    # transform on layer.weight / layer.weight_scale / etc.
    self.kernel.process_weights_after_loading(layer)

    # Save refs to whatever the kernel produced, so subsequent refits can copy_() instead
    # of recreating Parameter objects (preserves CUDA Graph addresses).
    if is_first_call:
        layer._marlin_tensor_refs = {}
        for attr in _KERNEL_FINAL_ATTRS:
            if hasattr(layer, attr):
                layer._marlin_tensor_refs[attr] = getattr(layer, attr).data
    else:
        for attr in _KERNEL_FINAL_ATTRS:
            if not hasattr(layer, attr):
                continue
            new_data = getattr(layer, attr).data
            ref = layer._marlin_tensor_refs.get(attr)
            if ref is not None and ref.shape == new_data.shape and ref.dtype == new_data.dtype:
                ref.copy_(new_data)
                setattr(layer, attr, Parameter(ref, requires_grad=False))
            else:
                # Shape/dtype changed (rare) — accept new tensor, update ref
                layer._marlin_tensor_refs[attr] = new_data


def _marlin_repack_experts(packed, perm, size_k, size_n, num_experts):
    import vllm._custom_ops as ops

    result = []
    for i in range(num_experts):
        qweight = packed[i].view(torch.int32).T.contiguous()
        result.append(
            ops.gptq_marlin_repack(
                b_q_weight=qweight,
                perm=perm,
                size_k=size_k,
                size_n=size_n,
                num_bits=4,
                is_a_8bit=False,
            )
        )
    return torch.stack(result)


def _marlin_process_scales_experts(scale_hf, param_dtype, size_k, size_n, group_size, num_experts):
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import nvfp4_marlin_process_scales

    result = []
    scale_factors = []
    scales = scale_hf.to(param_dtype)
    for i in range(num_experts):
        s = marlin_permute_scales(s=scales[i].T, size_k=size_k, size_n=size_n, group_size=group_size, is_a_8bit=False)
        # vLLM 0.20: returns (processed_scales, scale_factor)
        processed, sf = nvfp4_marlin_process_scales(s, a_dtype=param_dtype)
        result.append(processed)
        scale_factors.append(sf)
    # Return both stacked scales and the (constant) scale_factor; callers must divide global scales accordingly.
    # For MoE we assume scale_factor is constant across experts.
    return torch.stack(result), scale_factors[0] if scale_factors else 1.0


_MOE_HF_PARAMS = [
    "w13_weight",
    "w2_weight",
    "w13_weight_scale",
    "w2_weight_scale",
    "w13_weight_scale_2",
    "w2_weight_scale_2",
    "w13_input_scale",
    "w2_input_scale",
]

_MOE_KERNEL_INIT_COUNT = 0
_MOE_SCALE_DEBUG_COUNTS = {}


def _w13_scale2_mode() -> str:
    mode = os.environ.get("VERL_MOE_W13_SCALE2_MODE", "col0").strip().lower()
    aliases = {
        "0": "col0",
        "w1": "col0",
        "gate": "col0",
        "gate_proj": "col0",
        "1": "col1",
        "w3": "col1",
        "up": "col1",
        "up_proj": "col1",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"col0", "col1", "max", "mean"}:
        _qat_debug(f"unknown VERL_MOE_W13_SCALE2_MODE={mode!r}; falling back to col0")
        return "col0"
    return mode


def _get_marlin_moe_backend():
    try:
        from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
    except Exception as exc:  # pragma: no cover - depends on the runtime vLLM build.
        _qat_debug(f"unable to import NvFp4MoeBackend for MoE Marlin forcing: {exc}")
        return None
    return NvFp4MoeBackend.MARLIN


def _ensure_moe_activation_scale_attrs(layer: torch.nn.Module) -> None:
    # vLLM 0.20 ModelOpt MoE quant config reads these attributes unconditionally.
    # W4A16 is weight-only, so None is the expected activation-scale sentinel.
    if not hasattr(layer, "w13_input_scale"):
        layer.w13_input_scale = None
    if not hasattr(layer, "w2_input_scale"):
        layer.w2_input_scale = None


def _modelopt_moe_marlin_convert(self, layer: torch.nn.Module, is_first_call: bool) -> None:
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import nvfp4_marlin_process_global_scale

    group_size = 16
    e = layer.num_experts
    k = layer.hidden_size
    n = layer.intermediate_size_per_partition
    device = layer.w13_weight.device
    param_dtype = layer.params_dtype

    if is_first_call:
        layer.workspace = marlin_make_workspace_new(device, 4)

    perm = torch.empty(0, dtype=torch.int, device=device)
    size_n_w13, size_k_w13 = n * 2, k
    size_n_w2, size_k_w2 = k, n

    # Repack weights
    w13_weight_marlin = _marlin_repack_experts(layer.w13_weight.data, perm, size_k_w13, size_n_w13, e)
    w2_weight_marlin = _marlin_repack_experts(layer.w2_weight.data, perm, size_k_w2, size_n_w2, e)

    # Process scales (vLLM 0.20: helper returns (stacked_scales, scale_factor))
    w13_weight_scale_marlin, w13_scale_factor = _marlin_process_scales_experts(
        layer.w13_weight_scale.data,
        param_dtype,
        size_k_w13,
        size_n_w13,
        group_size,
        e,
    )
    w2_weight_scale_marlin, w2_scale_factor = _marlin_process_scales_experts(
        layer.w2_weight_scale.data,
        param_dtype,
        size_k_w2,
        size_n_w2,
        group_size,
        e,
    )

    # Process global scales  (w13_weight_scale_2 is already (E,) after common processing).
    # vLLM 0.20: caller must divide global scale by the scale_factor returned by nvfp4_marlin_process_scales.
    w13_scale_2_processed = nvfp4_marlin_process_global_scale(
        layer.w13_weight_scale_2.data.to(param_dtype), param_dtype
    )
    w13_scale_2_processed = (w13_scale_2_processed / w13_scale_factor).to(torch.float32)
    w2_scale_2_processed = nvfp4_marlin_process_global_scale(
        layer.w2_weight_scale_2.data.to(param_dtype), param_dtype
    )
    w2_scale_2_processed = (w2_scale_2_processed / w2_scale_factor).to(torch.float32)

    if is_first_call:
        layer.w13_weight = Parameter(w13_weight_marlin, requires_grad=False)
        layer.w2_weight = Parameter(w2_weight_marlin, requires_grad=False)
        layer.w13_weight_scale = Parameter(w13_weight_scale_marlin, requires_grad=False)
        layer.w2_weight_scale = Parameter(w2_weight_scale_marlin, requires_grad=False)
        layer.w13_weight_scale_2 = Parameter(w13_scale_2_processed, requires_grad=False)
        layer.w2_weight_scale_2 = Parameter(w2_scale_2_processed, requires_grad=False)
        if not hasattr(layer, "_marlin_tensor_refs"):
            layer._marlin_tensor_refs = {}
        for rn in [
            "w13_weight",
            "w2_weight",
            "w13_weight_scale",
            "w2_weight_scale",
            "w13_weight_scale_2",
            "w2_weight_scale_2",
        ]:
            layer._marlin_tensor_refs[rn] = getattr(layer, rn).data
    else:
        for rn, nd in [
            ("w13_weight", w13_weight_marlin),
            ("w2_weight", w2_weight_marlin),
            ("w13_weight_scale", w13_weight_scale_marlin),
            ("w2_weight_scale", w2_weight_scale_marlin),
            ("w13_weight_scale_2", w13_scale_2_processed),
            ("w2_weight_scale_2", w2_scale_2_processed),
        ]:
            _update_ref_or_create(layer, rn, nd)

    # vLLM 0.20's ModelOpt FusedMoE quant config builder unconditionally
    # reads these attributes.  Weight-only W4A16 has no activation scales,
    # and the initial dummy-load path for W4A4 may not have them yet either.
    # Keep the attributes present and use None as vLLM's weight-only signal.
    if not hasattr(layer, "w13_input_scale"):
        layer.w13_input_scale = None
    if not hasattr(layer, "w2_input_scale"):
        layer.w2_input_scale = None


_MOE_KERNEL_FORMAT_ATTRS = (
    "w13_weight",
    "w13_weight_scale",
    "w13_weight_scale_2",
    "w13_input_scale",
    "w2_weight",
    "w2_weight_scale",
    "w2_weight_scale_2",
    "w2_input_scale",
)


def _contiguous_moe_tensor(data: torch.Tensor) -> torch.Tensor:
    data = data.detach()
    if not data.is_contiguous():
        data = data.clone().contiguous()
    return data


def _store_moe_ref(layer: torch.nn.Module, attr: str) -> None:
    value = getattr(layer, attr)
    layer._marlin_tensor_refs[attr] = value.data if isinstance(value, Parameter) else value


def _replace_or_ref_moe_attr(layer: torch.nn.Module, attr: str, new_data, is_first_call: bool, replace_parameter) -> None:
    if new_data is None:
        if not hasattr(layer, "_marlin_tensor_refs"):
            layer._marlin_tensor_refs = {}
        layer._marlin_tensor_refs.pop(attr, None)
        setattr(layer, attr, None)
        return

    data = new_data.data if isinstance(new_data, Parameter) else new_data
    if not isinstance(data, torch.Tensor):
        data = torch.as_tensor(data, device=getattr(layer, attr).device)

    if not hasattr(layer, "_marlin_tensor_refs"):
        layer._marlin_tensor_refs = {}

    ref = layer._marlin_tensor_refs.get(attr)
    if ref is not None and not is_first_call and ref.shape == data.shape and ref.dtype == data.dtype and ref.is_contiguous():
        try:
            ref.copy_(data)
            setattr(layer, attr, Parameter(ref, requires_grad=False))
            return
        except RuntimeError as exc:
            if "more than one element of the written-to tensor" not in str(exc):
                raise

    replace_parameter(layer, attr, _contiguous_moe_tensor(data))
    _store_moe_ref(layer, attr)


def _refresh_moe_attr_refs(layer: torch.nn.Module, is_first_call: bool) -> None:
    if not hasattr(layer, "_marlin_tensor_refs"):
        layer._marlin_tensor_refs = {}

    for attr in _MOE_KERNEL_FORMAT_ATTRS:
        if not hasattr(layer, attr):
            continue
        value = getattr(layer, attr)
        data = value.data if isinstance(value, Parameter) else value
        if not isinstance(data, torch.Tensor):
            continue
        if not data.is_contiguous():
            data = _contiguous_moe_tensor(data)
            setattr(layer, attr, Parameter(data, requires_grad=False))
        layer._marlin_tensor_refs[attr] = data


def _modelopt_moe_kernel_format_convert(
    self,
    layer: torch.nn.Module,
    is_first_call: bool,
    require_activation_scales: bool = True,
) -> None:
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import convert_to_nvfp4_moe_kernel_format
    from vllm.model_executor.utils import replace_parameter

    if require_activation_scales:
        for required in ("w13_input_scale", "w2_input_scale"):
            if not hasattr(layer, required) or getattr(layer, required) is None:
                raise RuntimeError(
                    f"ModelOpt NVFP4 MoE W4A4 requires {required}; check QAT config and weight reload."
                )

    raw_w13_weight_scale_2 = layer.w13_weight_scale_2.data
    w13_scale2_mode = _w13_scale2_mode()
    if raw_w13_weight_scale_2.dim() == 2:
        if self.moe.is_act_and_mul and raw_w13_weight_scale_2.shape[1] > 1:
            col0 = raw_w13_weight_scale_2[:, 0].detach().float()
            col1 = raw_w13_weight_scale_2[:, 1].detach().float()
            max_abs_diff = (col0 - col1).abs().max().item()
            max_rel_diff = ((col0 - col1).abs() / col0.abs().clamp_min(1e-12)).max().item()
            if not torch.allclose(raw_w13_weight_scale_2[:, 0], raw_w13_weight_scale_2[:, 1]):
                _qat_debug(
                    "w1_weight_scale_2 and w3_weight_scale_2 differ; "
                    f"using W13 scale2 mode={w13_scale2_mode}. max_abs_diff={max_abs_diff:.6g} "
                    f"max_rel_diff={max_rel_diff:.6g}"
                )
        if w13_scale2_mode == "col1" and raw_w13_weight_scale_2.shape[1] > 1:
            w13_weight_scale_2 = raw_w13_weight_scale_2[:, 1].contiguous()
        elif w13_scale2_mode == "max":
            w13_weight_scale_2 = raw_w13_weight_scale_2.max(dim=1).values.contiguous()
        elif w13_scale2_mode == "mean":
            w13_weight_scale_2 = raw_w13_weight_scale_2.float().mean(dim=1).to(raw_w13_weight_scale_2.dtype).contiguous()
        else:
            w13_weight_scale_2 = raw_w13_weight_scale_2[:, 0].contiguous()
    else:
        w13_weight_scale_2 = raw_w13_weight_scale_2.contiguous()

    global _MOE_SCALE_DEBUG_COUNTS
    debug_key = (bool(is_first_call), str(getattr(self, 'nvfp4_backend', None)))
    debug_count = _MOE_SCALE_DEBUG_COUNTS.get(debug_key, 0)
    if debug_count < 8:
        def _stat(name, value):
            data = value.data if isinstance(value, Parameter) else value
            if data is None:
                return f"{name}=None"
            if not isinstance(data, torch.Tensor):
                return f"{name}=non_tensor"
            d = data.detach().float()
            return f"{name}.shape={tuple(data.shape)} min={d.min().item():.6g} max={d.max().item():.6g} mean={d.mean().item():.6g}"

        _qat_debug(
            "MoE scale stats before convert: "
            f"is_first={is_first_call} backend={getattr(self, 'nvfp4_backend', None)} "
            + _stat("w13_scale_2", w13_weight_scale_2)
            + "; "
            + _stat("w2_scale_2", getattr(layer, "w2_weight_scale_2", None))
            + "; "
            + _stat("w13_input_scale", getattr(layer, "w13_input_scale", None))
            + "; "
            + _stat("w2_input_scale", getattr(layer, "w2_input_scale", None))
        )
        _MOE_SCALE_DEBUG_COUNTS[debug_key] = debug_count + 1


    (
        w13,
        w13_scale,
        w13_scale_2,
        a13_scale,
        w2,
        w2_scale,
        w2_scale_2,
        a2_scale,
    ) = convert_to_nvfp4_moe_kernel_format(
        nvfp4_backend=self.nvfp4_backend,
        layer=layer,
        w13=layer.w13_weight,
        w13_scale=layer.w13_weight_scale,
        w13_scale_2=w13_weight_scale_2,
        a13_scale=layer.w13_input_scale,
        w2=layer.w2_weight,
        w2_scale=layer.w2_weight_scale,
        w2_scale_2=layer.w2_weight_scale_2,
        a2_scale=layer.w2_input_scale,
        is_act_and_mul=self.moe.is_act_and_mul,
    )

    converted = (
        ("w13_weight", w13),
        ("w13_weight_scale", w13_scale),
        ("w13_weight_scale_2", w13_scale_2),
        ("w13_input_scale", a13_scale),
        ("w2_weight", w2),
        ("w2_weight_scale", w2_scale),
        ("w2_weight_scale_2", w2_scale_2),
        ("w2_input_scale", a2_scale),
    )
    for attr, tensor in converted:
        _replace_or_ref_moe_attr(layer, attr, tensor, is_first_call, replace_parameter)

    # CUDA-graph compatibility (W4A4 non-Marlin only, opt-in via env): on subsequent
    # weight syncs refresh quant_config scales in place + reuse the kernel instead of
    # rebuilding them (rebuild -> new tensor addresses -> stale reads under cuda graph).
    if (
        not is_first_call
        and _moe_cudagraph_inplace_enabled()
        and not _backend_is_marlin(self)
        and _update_moe_quant_config_inplace(self, layer)
    ):
        pass  # reused moe_quant_config + kernel in place (cuda-graph safe)
    else:
        _init_modelopt_moe_kernel(self, layer)

    fused_experts = getattr(getattr(self.moe_kernel, "fused_experts", None), "process_weights_after_loading", None)
    if fused_experts is not None:
        fused_experts(layer)

    _refresh_moe_attr_refs(layer, is_first_call)


# --- CUDA-graph-compatible quant_config refresh (port of FSDP commit f8c98f76) ---
# On weight sync, the W4A4 (non-Marlin) MoE path must refresh the EXISTING
# moe_quant_config + kernel rather than rebuilding them. CUDA graphs capture tensor
# addresses at capture time; replacing moe_quant_config (new tensors -> new addresses)
# makes the captured graph read STALE scales -> garbage, which is exactly why the
# Megatron W4A4 path previously required enforce_eager=True.
#   g1_alphas/g2_alphas/w1_scale/w2_scale reference the layer params (already kept
#   address-stable via _replace_or_ref_moe_attr) -> same-object, nothing to do.
#   a1_gscale/a2_gscale are freshly computed (1/input_scale) each call -> THESE are
#   the addresses that break cuda graph, so we copy_ them into the persistent object.
# CutlassExpertsFp4.process_weights_after_loading fuses input_scale into w_scale_2
# IN-PLACE (mul_), at the same stable address, so the fuse stays cuda-graph safe.
# Gated behind VERL_MOE_CUDAGRAPH_INPLACE_QC=1 + non-Marlin backend so the running
# w4a16 (Marlin) / bf16 baselines are byte-for-byte unchanged.
_NVFP4_QC_SCALE_ATTRS = ("g1_alphas", "g2_alphas", "a1_gscale", "a2_gscale", "w1_scale", "w2_scale")
_MOE_INPLACE_REFRESH_COUNT = 0


def _moe_cudagraph_inplace_enabled() -> bool:
    import os

    # Default ON: the in-place refresh is the desired behavior for W4A4 (it is what
    # lets cuda graph stay enabled). Gated to the non-Marlin path below, and the
    # helper safely falls back to a full rebuild if state is missing, so w4a16/bf16
    # are unaffected. Set VERL_MOE_CUDAGRAPH_INPLACE_QC=0 to force the old rebuild path.
    return os.environ.get("VERL_MOE_CUDAGRAPH_INPLACE_QC", "1") != "0"


def _backend_is_marlin(self) -> bool:
    be = getattr(self, "nvfp4_backend", None)
    return be is not None and getattr(be, "name", "") == "MARLIN"


def _update_moe_quant_config_inplace(self, layer: torch.nn.Module) -> bool:
    """Refresh moe_quant_config scale tensors in place (preserving addresses for
    CUDA-graph stability) and reuse the existing kernel. Returns True on success,
    False to fall back to a full kernel rebuild (first call / missing state /
    shape or dtype change that prevents in-place copy)."""
    global _MOE_INPLACE_REFRESH_COUNT

    def _diag(msg):
        if _MOE_INPLACE_REFRESH_COUNT < 6:
            _qat_debug(f"inplace-qc {msg}")

    old_qc = getattr(self, "moe_quant_config", None)
    kernel = getattr(self, "kernel", None)
    if old_qc is None or kernel is None:
        _diag(f"BAIL->rebuild: moe_quant_config={'None' if old_qc is None else 'set'} "
              f"kernel={'None' if kernel is None else 'set'} (self={type(self).__name__})")
        _MOE_INPLACE_REFRESH_COUNT += 1
        return False
    new_qc = self.get_fused_moe_quant_config(layer)
    if new_qc is None:
        _diag("BAIL->rebuild: get_fused_moe_quant_config returned None")
        _MOE_INPLACE_REFRESH_COUNT += 1
        return False
    copied = []
    for attr in _NVFP4_QC_SCALE_ATTRS:
        old_val = getattr(old_qc, attr, None)
        new_val = getattr(new_qc, attr, None)
        if old_val is None or new_val is None:
            continue
        if not (isinstance(old_val, torch.Tensor) and isinstance(new_val, torch.Tensor)):
            continue
        if old_val is new_val:
            # same object (references an already in-place-updated layer param)
            continue
        if old_val.shape != new_val.shape or old_val.dtype != new_val.dtype:
            _diag(f"BAIL->rebuild: {attr} shape/dtype changed "
                  f"({tuple(old_val.shape)}/{old_val.dtype} -> {tuple(new_val.shape)}/{new_val.dtype})")
            _MOE_INPLACE_REFRESH_COUNT += 1
            return False  # address cannot be preserved -> caller does full rebuild
        old_val.copy_(new_val)
        copied.append(attr)
    del new_qc
    # Keep the SAME moe_quant_config + kernel objects (stable addresses); make sure
    # the kernel's experts read this (unchanged) object.
    self.moe_kernel = kernel
    fe = getattr(kernel, "fused_experts", None)
    if fe is not None and hasattr(fe, "quant_config"):
        fe.quant_config = old_qc
    if _MOE_INPLACE_REFRESH_COUNT < 4:
        _qat_debug(
            "MoE quant_config refreshed IN-PLACE (cuda-graph safe); "
            f"copied={copied} reused_kernel={type(kernel).__name__}"
        )
        _MOE_INPLACE_REFRESH_COUNT += 1
    return True


def _init_modelopt_moe_kernel(self, layer: torch.nn.Module) -> None:
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import make_nvfp4_moe_kernel

    global _MOE_KERNEL_INIT_COUNT

    import os

    # _modelopt_moe_marlin_convert stores expert weights/scales in Marlin layout.
    # Only W4A16 can use MarlinExperts; W4A4 activation quantization requires a
    # non-Marlin kernel-format conversion path, which vLLM20 ModelOpt does not
    # expose through this patch.
    force_marlin = os.environ.get("VERL_QAT_MODE", "").lower() == "w4a16"
    marlin_backend = _get_marlin_moe_backend() if force_marlin else None
    if marlin_backend is not None:
        self.nvfp4_backend = marlin_backend

    # Debug/override hook: force a specific NVFP4 MoE backend (e.g. EMULATION,
    # VLLM_CUTLASS) to test W4A4 activation-quant engagement. Per-job via env.
    _backend_override = os.environ.get("VERL_NVFP4_MOE_BACKEND", "").strip()
    if _backend_override:
        from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend

        try:
            self.nvfp4_backend = NvFp4MoeBackend[_backend_override.upper()]
            marlin_backend = self.nvfp4_backend if self.nvfp4_backend == NvFp4MoeBackend.MARLIN else None
            if _MOE_KERNEL_INIT_COUNT < 4:
                _qat_debug(f"forced NVFP4 MoE backend = {self.nvfp4_backend}")
        except KeyError:
            _qat_debug(f"unknown VERL_NVFP4_MOE_BACKEND={_backend_override!r}")

    self.moe_quant_config = self.get_fused_moe_quant_config(layer)
    if self.moe_quant_config is None:
        return

    kwargs = {
        "moe_quant_config": self.moe_quant_config,
        "moe_config": self.moe,
        "experts_cls": self.experts_cls,
    }
    sig = inspect.signature(make_nvfp4_moe_kernel)
    if "nvfp4_backend" in sig.parameters:
        backend = marlin_backend if marlin_backend is not None else getattr(self, 'nvfp4_backend', None)
        if backend is not None:
            kwargs["nvfp4_backend"] = backend
    if "routing_tables" in sig.parameters and hasattr(layer, "_maybe_init_expert_routing_tables"):
        kwargs["routing_tables"] = layer._maybe_init_expert_routing_tables()
    if "shared_experts" in sig.parameters and hasattr(layer, "shared_experts"):
        kwargs["shared_experts"] = layer.shared_experts

    result = make_nvfp4_moe_kernel(**kwargs)
    if isinstance(result, tuple):
        kernel = result[0]
        self.kernel = kernel
        if len(result) > 1:
            self.use_inplace = result[1]
    else:
        kernel = result
        self.kernel = kernel

    # vLLM 0.20's ModelOpt MoE path reads moe_kernel in apply_monolithic.
    self.moe_kernel = kernel

    _MOE_KERNEL_INIT_COUNT += 1
    if _MOE_KERNEL_INIT_COUNT <= 4:
        _qat_debug(
            "ModelOpt MoE kernel init "
            f"#{_MOE_KERNEL_INIT_COUNT}: requested_backend={marlin_backend}, "
            f"kernel={type(kernel).__module__}.{type(kernel).__name__}"
        )


def _modelopt_moe_process_weights(self, layer: torch.nn.Module) -> None:
    is_first_call = _check_first_call(layer)

    if is_first_call:
        for pname in _MOE_HF_PARAMS:
            _save_param_meta(layer, pname)
        _save_weight_loaders(layer, _MOE_HF_PARAMS)

    import os

    mode = os.environ.get("VERL_QAT_MODE", "").lower()

    # Backend override MUST be applied here, before the convert — the convert
    # formats tensors for self.nvfp4_backend, so setting it later (in
    # _init_modelopt_moe_kernel) would mismatch tensor layout vs kernel.
    # Applies to the W4A4 path; w4a16 keeps marlin unless explicitly overridden.
    _backend_override = os.environ.get("VERL_NVFP4_MOE_BACKEND", "").strip()
    if _backend_override and mode != "w4a16":
        from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend

        try:
            self.nvfp4_backend = NvFp4MoeBackend[_backend_override.upper()]
            if _MOE_KERNEL_INIT_COUNT < 4:
                _qat_debug(f"(pre-convert) forced NVFP4 MoE backend = {self.nvfp4_backend}")
        except KeyError:
            _qat_debug(f"unknown VERL_NVFP4_MOE_BACKEND={_backend_override!r}")

    if mode == "w4a16":
        marlin_backend = _get_marlin_moe_backend()
        if marlin_backend is not None:
            self.nvfp4_backend = marlin_backend
        _ensure_moe_activation_scale_attrs(layer)
        _modelopt_moe_kernel_format_convert(self, layer, is_first_call, require_activation_scales=False)
        return

    _modelopt_moe_kernel_format_convert(self, layer, is_first_call)


def _modelopt_kv_process_weights(self, layer) -> None:
    """
    Replacement for BaseKVCacheMethod.process_weights_after_loading.
    Doesn't delete k_scale, v_scale, q_scale, prob_scale to allow
    for dynamic updates during refit.
    """
    from vllm.platforms import current_platform

    if layer.kv_cache_dtype != "auto" and not layer.calculate_kv_scales:
        if layer.k_scale > 0.0 and layer.v_scale > 0.0:
            k_scale = layer.k_scale.to("cpu").tolist()
            v_scale = layer.v_scale.to("cpu").tolist()
            if current_platform.is_fp8_fnuz():
                k_scale *= 2
                v_scale *= 2
        elif layer.k_scale < 0.0 and layer.v_scale < 0.0:
            k_scale = 1.0
            v_scale = 1.0
        else:
            assert layer.k_scale > 0.0
            scale_to_duplicate = max(layer.k_scale, layer.v_scale)
            k_scale = scale_to_duplicate.to("cpu").tolist()
            v_scale = scale_to_duplicate.to("cpu").tolist()
            if current_platform.is_fp8_fnuz():
                k_scale *= 2
                v_scale *= 2

        if not isinstance(k_scale, float) or not isinstance(v_scale, float):
            raise ValueError("Only support per-tensor scaling factor for fp8 KV cache")

        if layer.q_scale < 0.0:
            layer._q_scale.copy_(k_scale)
            layer._q_scale_float = k_scale

        layer._k_scale.copy_(k_scale)
        layer._v_scale.copy_(v_scale)
        layer._k_scale_float = k_scale
        layer._v_scale_float = v_scale

    if layer.q_scale > 0.0:
        q_scale = layer.q_scale
        if current_platform.is_fp8_fnuz():
            q_scale *= 2
        layer.calculate_kv_scales = False
    else:
        q_scale = 1.0
    if layer.prob_scale > 0.0:
        prob_scale = layer.prob_scale
        if current_platform.is_fp8_fnuz():
            prob_scale *= 2
    else:
        prob_scale = 1.0

    is_singleton_float = (
        lambda x: isinstance(x, float) or isinstance(x, torch.Tensor) and x.numel() == 1 and x.is_floating_point()
    )
    if not is_singleton_float(q_scale) or not is_singleton_float(prob_scale):
        raise ValueError("Only support per-tensor scaling factor for fp8-quantized Q/prob")

    layer._q_scale.copy_(q_scale)
    layer._q_scale_float = q_scale.item() if isinstance(q_scale, torch.Tensor) else q_scale
    layer._prob_scale.copy_(prob_scale)


_patched = False


def prepare_modelopt_for_weight_reload(model, device=None):
    """Prepare ModelOpt model for weight reloading. Call ONCE before each reload cycle."""
    _qat_debug(f"prepare_modelopt_for_weight_reload entered, model={type(model).__name__}")
    inner_model = model
    if hasattr(model, "model"):
        inner_model = model.model

    param_meta = ModelOptParamMetaDict(inner_model, device=device)

    param_meta.prepare_for_reload()
    cache_size = len(param_meta._layer_meta_cache) if hasattr(param_meta, "_layer_meta_cache") else -1
    _qat_debug(f"prepare_modelopt_for_weight_reload: cache size = {cache_size}")

    restored_count = 0
    for layer_name, cache_entry in param_meta._layer_meta_cache.items():
        module = cache_entry["module"]
        for param_name, pm in cache_entry["meta"].items():
            existing = getattr(module, param_name, None)
            if existing is not None:
                hf_shape = tuple(pm["shape"])
                hf_dtype = pm["dtype"]
                if (
                    tuple(existing.shape) == hf_shape
                    and existing.dtype == hf_dtype
                    and hasattr(existing, "weight_loader")
                ):
                    continue
            new_param = _create_param_from_meta(module, param_name, pm, device)
            module.register_parameter(param_name, new_param)
            restored_count += 1
    _qat_debug(
        f"prepare_modelopt_for_weight_reload: restored {restored_count} params "
        f"(cache had {cache_size} layers)"
    )

    inner_model._param_meta_for_restore = param_meta
    return param_meta


def modelopt_process_weights_after_loading(model):
    """Trigger weight post-processing for all quantized layers after load_weights."""
    dense_count = 0
    moe_count = 0
    wrapped_count = 0
    missing_process_count = 0

    actual_model = model
    if hasattr(model, "model"):
        actual_model = model.model

    for module in actual_model.modules():
        if hasattr(module, "scheme"):
            module.scheme.process_weights_after_loading(module)
            dense_count += 1

        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None and not hasattr(module, "scheme"):
            # After vLLM maybe_init_modular_kernel(), quant_method may be a
            # FusedMoEModularMethod wrapper. The ModelOpt reload hook lives on
            # the wrapped method, and the wrapper experts need the refreshed
            # quant_config for inference.
            actual_qm = quant_method
            is_wrapped = hasattr(quant_method, "old_quant_method")
            if is_wrapped:
                wrapped_count += 1
                actual_qm = quant_method.old_quant_method

            if hasattr(actual_qm, "process_weights_after_loading"):
                if "KVCache" in actual_qm.__class__.__name__:
                    continue
                actual_qm.process_weights_after_loading(module)
                if is_wrapped and hasattr(actual_qm, "moe_quant_config"):
                    quant_method.moe_quant_config = actual_qm.moe_quant_config
                    if hasattr(quant_method, "fused_experts"):
                        inner = quant_method.fused_experts
                        if hasattr(inner, "fused_experts"):
                            inner.fused_experts.quant_config = actual_qm.moe_quant_config
                moe_count += 1
            else:
                missing_process_count += 1

    if hasattr(actual_model, "_param_meta_for_restore"):
        del actual_model._param_meta_for_restore
        torch.cuda.empty_cache()

    _qat_debug(
        "modelopt_process_weights_after_loading: "
        f"dense_count={dense_count}, moe_count={moe_count}, "
        f"wrapped_count={wrapped_count}, missing_process_count={missing_process_count}"
    )
    return dense_count + moe_count

def apply_modelopt_nvfp4_patches(mode: str = "w4a4"):
    """Apply ModelOpt NVFP4 patches to support dynamic weight updates. Call before model loading.

    Args:
        mode: "w4a4" (default) keeps FlashInfer/CUTLASS NVFP4 kernels which quantize input to FP4.
              "w4a16" forces Marlin backend on Linear layers — ``apply_fp4_marlin_linear`` does
              NOT touch the input (BF16 in, dequant weights, do BF16 GEMM) so activations stay
              full precision.
    """
    global _patched

    if _patched:
        return

    from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4FusedMoE,
        ModelOptNvFp4LinearMethod,
    )

    ModelOptNvFp4LinearMethod.process_weights_after_loading = _modelopt_dense_process_weights
    ModelOptNvFp4FusedMoE.process_weights_after_loading = _modelopt_moe_process_weights
    BaseKVCacheMethod.process_weights_after_loading = _modelopt_kv_process_weights

    if mode == "w4a16":
        # vLLM 0.20 picks the kernel via init_nvfp4_linear_kernel() inside
        # ModelOptNvFp4LinearMethod.__init__. The default kernel on GB200 is
        # FlashInferCutlass which always FP4-quantizes the input → wrong for W4A16.
        # The Marlin kernel (apply_fp4_marlin_linear) takes BF16 input directly = true W4A16.
        # VLLM_NVFP4_GEMM_BACKEND is read inside init_nvfp4_linear_kernel().
        import os

        os.environ["VLLM_NVFP4_GEMM_BACKEND"] = "marlin"
        # vLLM20 selects the MoE experts implementation in ModelOptNvFp4FusedMoE.__init__.
        # This env is checked by select_nvfp4_moe_backend before the default SM100 TRTLLM path.
        os.environ["VLLM_TEST_FORCE_FP8_MARLIN"] = "1"
        os.environ["VLLM_USE_FLASHINFER_MOE_FP4"] = "0"
        _qat_debug("set VLLM_NVFP4_GEMM_BACKEND=marlin and force MoE Marlin (w4a16 mode)")

    _patched = True
