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

# To support different vLLM versions, we add the model into SUPPORTED_MOE_MODELS separately to avoid triggering
# unsupported issues.
import os

import torch

SUPPORTED_MOE_MODELS = []
_QAT_DEBUG_ENABLED = os.environ.get("VERL_QAT_DEBUG", "0") == "1"


def _qat_debug(message: str) -> None:
    if _QAT_DEBUG_ENABLED:
        print(f"[QAT-DEBUG] {message}", flush=True)

try:
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM, DeepseekV3ForCausalLM

    SUPPORTED_MOE_MODELS.append(DeepseekV2ForCausalLM)
    SUPPORTED_MOE_MODELS.append(DeepseekV3ForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.mixtral import MixtralForCausalLM

    SUPPORTED_MOE_MODELS.append(MixtralForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen2MoeForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3MoeForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_vl_moe import Qwen3MoeLLMForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3MoeLLMForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_next import Qwen3NextForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3NextForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.kimi_vl import KimiVLForConditionalGeneration

    SUPPORTED_MOE_MODELS.append(KimiVLForConditionalGeneration)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3_5MoeForCausalLM)
except ImportError:
    pass


def patch_vllm_moe_model_weight_loader(model):
    # Patch vLLM MoE expert loaders for online weight sync.  vLLM20 may pass a
    # CUDAGraphWrapper or the inner Qwen3MoeModel here, so use structural
    # traversal instead of relying only on outer ForCausalLM classes.
    if model is None:
        return

    def _iter_layer_containers(root):
        queue = [root]
        seen = set()
        while queue:
            obj = queue.pop(0)
            if obj is None:
                continue
            obj_id = id(obj)
            if obj_id in seen:
                continue
            seen.add(obj_id)
            layers = getattr(obj, "layers", None)
            if layers is not None:
                yield obj, layers
            for attr in ("model", "language_model", "runnable", "module"):
                child = getattr(obj, attr, None)
                if child is not None and id(child) not in seen:
                    queue.append(child)

    def _copy_same_or_transposed(target, loaded_weight):
        candidates = [loaded_weight]
        if loaded_weight.ndim >= 2:
            candidates.append(loaded_weight.transpose(-1, -2).contiguous())
        for candidate in candidates:
            if candidate.ndim != target.ndim:
                continue
            if any(candidate.shape[i] > target.shape[i] for i in range(candidate.ndim)):
                continue
            view = target
            for dim, size in enumerate(candidate.shape):
                if view.shape[dim] != size:
                    view = view.narrow(dim, 0, size)
            view.copy_(candidate.to(device=view.device, dtype=view.dtype, non_blocking=True))
            return True
        return False

    def _orient_2d_source(source, rows, cols):
        source = source.contiguous()
        if tuple(source.shape) == (rows, cols):
            return source
        if tuple(source.shape) == (cols, rows):
            return source.transpose(0, 1).contiguous()
        return None

    def _swap_w13_to_w31(source):
        return source.reshape(-1, 2, source.shape[-2] // 2, source.shape[-1]).flip(dims=[1]).reshape(source.shape)

    def _trtllm_bf16_cache(experts_module, name):
        attr = f"_verl_trtllm_bf16_{name}_cache"
        cache = getattr(experts_module, attr, None)
        if cache is None:
            cache = {}
            setattr(experts_module, attr, cache)
        return cache

    def _convert_trtllm_bf16_w13(experts_module, source):
        from flashinfer.fused_moe.core import (
            _maybe_get_cached_w3_w1_permute_indices,
            convert_to_block_layout,
        )

        source = _swap_w13_to_w31(source.contiguous())
        cache = _trtllm_bf16_cache(experts_module, "permute")
        permute_indices = _maybe_get_cached_w3_w1_permute_indices(
            cache,
            source.view(torch.uint8),
            128,
        )
        source = source.clone().view(torch.uint8)[permute_indices.to(source.device)].contiguous()
        return convert_to_block_layout(source.view(torch.uint8), 128).view(torch.bfloat16).contiguous()

    def _convert_trtllm_bf16_w2(experts_module, source):
        from flashinfer.fused_moe.core import (
            convert_to_block_layout,
            get_w2_permute_indices_with_cache,
        )

        source = source.contiguous()
        cache = _trtllm_bf16_cache(experts_module, "permute")
        permute_indices = get_w2_permute_indices_with_cache(
            cache,
            source.view(torch.uint8),
            128,
        )
        source = source.clone().view(torch.uint8)[permute_indices.to(source.device)].contiguous()
        return convert_to_block_layout(source.view(torch.uint8), 128).view(torch.bfloat16).contiguous()

    def _copy_trtllm_packed_moe_tensor(experts_module, target, loaded_weight, shard_id, local_expert_id):
        if target.ndim != 3 or loaded_weight.ndim != 2 or shard_id not in ("w1", "w2", "w3"):
            return False
        if target.dtype != torch.bfloat16:
            return False

        if shard_id in ("w1", "w3"):
            hidden_blocks, packed_intermediate, block = target.shape
            hidden_size = hidden_blocks * block
            if packed_intermediate % 2 != 0:
                return False
            intermediate_size = packed_intermediate // 2
            source = _orient_2d_source(loaded_weight, intermediate_size, hidden_size)
            if source is None or local_expert_id is None:
                return False
            source = source.to(device=target.device, dtype=target.dtype, non_blocking=True).contiguous().clone()
            staged_w13 = _trtllm_bf16_cache(experts_module, "w13").setdefault(int(local_expert_id), {})
            staged_w13[shard_id] = source
            if "w1" not in staged_w13 or "w3" not in staged_w13:
                return True

            # vLLM's TRTLLM BF16 backend converts the full [w1; w3] tensor by
            # first swapping to [w3; w1], then applying FlashInfer's epilogue
            # tile/block layout. Online reload receives the two shards
            # separately, so stage one half until both are available.
            source = torch.cat((staged_w13.pop("w1"), staged_w13.pop("w3")), dim=0)
            if not staged_w13:
                _trtllm_bf16_cache(experts_module, "w13").pop(int(local_expert_id), None)
            converted = _convert_trtllm_bf16_w13(experts_module, source)
            if converted.shape != target.shape:
                return False
            target.copy_(converted.to(device=target.device, dtype=target.dtype, non_blocking=True))
            return True

        intermediate_blocks, hidden_size, block = target.shape
        intermediate_size = intermediate_blocks * block
        source = _orient_2d_source(loaded_weight, hidden_size, intermediate_size)
        if source is None:
            return False
        source = source.to(device=target.device, dtype=target.dtype, non_blocking=True)
        converted = _convert_trtllm_bf16_w2(experts_module, source)
        if converted.shape != target.shape:
            return False
        target.copy_(converted.to(device=target.device, dtype=target.dtype, non_blocking=True))
        return True

    def _local_expert_id(experts_module, expert_id):
        if expert_id is None:
            return None
        try:
            local_id = experts_module._map_global_expert_id_to_local_expert_id(int(expert_id))
        except Exception:
            local_id = expert_id
        if isinstance(local_id, torch.Tensor):
            local_id = local_id.item()
        return int(local_id)

    def _direct_copy_moe_tensor(experts_module, param, loaded_weight, shard_id=None, expert_id=None):
        if not isinstance(loaded_weight, torch.Tensor) or not isinstance(param, torch.Tensor):
            return False
        param_data = param.data
        target_data = param_data

        if expert_id is not None and param_data.ndim >= loaded_weight.ndim + 1:
            local_id = _local_expert_id(experts_module, expert_id)
            if local_id is None or local_id < 0 or local_id >= param_data.shape[0]:
                return False
            target_data = param_data[local_id]
        elif param_data.ndim != loaded_weight.ndim:
            return False

        if _copy_trtllm_packed_moe_tensor(experts_module, target_data, loaded_weight, shard_id, local_id if expert_id is not None else None):
            return True

        if _copy_same_or_transposed(target_data, loaded_weight):
            return True

        if shard_id in ("w1", "w3"):
            shard_idx = 0 if shard_id == "w1" else 1
            candidates = [loaded_weight]
            if loaded_weight.ndim >= 2:
                candidates.append(loaded_weight.transpose(-1, -2).contiguous())
            for candidate in candidates:
                if candidate.ndim != target_data.ndim:
                    continue
                for shard_dim in range(candidate.ndim):
                    shard_size = candidate.shape[shard_dim]
                    start = shard_idx * shard_size
                    if start + shard_size > target_data.shape[shard_dim]:
                        continue
                    target = target_data.narrow(shard_dim, start, shard_size)
                    if _copy_same_or_transposed(target, candidate):
                        return True

        return False

    def _debug_direct_moe_loader(experts_module, kind, param, loaded_weight, shard_id, expert_id):
        if not isinstance(loaded_weight, torch.Tensor) or not isinstance(param, torch.Tensor):
            return
        counter_name = f"_verl_direct_moe_loader_{kind}_count"
        count = getattr(experts_module, counter_name, 0)
        if count >= 8:
            return
        setattr(experts_module, counter_name, count + 1)
        _qat_debug(
            f"direct MoE loader {kind}: layer={getattr(experts_module, 'layer_name', 'unknown')} "
            f"param_shape={tuple(param.shape)} loaded_shape={tuple(loaded_weight.shape)} "
            f"shard_id={shard_id} expert_id={expert_id}"
        )

    def _quant_config_name(experts_module):
        quant_config = getattr(experts_module, "quant_config", None)
        if quant_config is None:
            return None
        get_name = getattr(quant_config, "get_name", None)
        return get_name() if get_name is not None else str(quant_config)

    enable_direct_unquantized_loader = os.environ.get("VERL_ENABLE_DIRECT_UNQUANTIZED_MOE_LOADER", "1").lower() in (
        "1",
        "true",
        "yes",
    )

    def _make_direct_moe_weight_loader(fallback_loader, experts_module, force_direct: bool = False):
        def _direct_moe_weight_loader(param, loaded_weight, *args, **kwargs):
            shard_id = kwargs.get("shard_id")
            expert_id = kwargs.get("expert_id")
            for arg in args:
                if shard_id is None and arg in ("w1", "w2", "w3"):
                    shard_id = arg
            return_success = kwargs.get("return_success", False)
            direct_allowed = force_direct or (
                enable_direct_unquantized_loader and _quant_config_name(experts_module) is None
            )
            if direct_allowed and _direct_copy_moe_tensor(
                experts_module,
                param,
                loaded_weight,
                shard_id=shard_id,
                expert_id=expert_id,
            ):
                _debug_direct_moe_loader(
                    experts_module,
                    "forced_hit" if force_direct else "hit",
                    param,
                    loaded_weight,
                    shard_id,
                    expert_id,
                )
                return True if return_success else None
            _debug_direct_moe_loader(
                experts_module,
                "forced_fallback" if force_direct else "fallback",
                param,
                loaded_weight,
                shard_id,
                expert_id,
            )
            return fallback_loader(param, loaded_weight, *args, **kwargs)

        return _direct_moe_weight_loader

    patched_layers = 0
    patched_params = 0

    for _container, layers in _iter_layer_containers(model):
        for layer_idx, layer in enumerate(layers):
            mlp = None
            for mlp_attr in ("mlp", "block_sparse_moe"):
                mlp = getattr(layer, mlp_attr, None)
                if mlp is not None:
                    break
            if mlp is None:
                continue

            experts = getattr(mlp, "experts", None)
            if experts is None or not hasattr(experts, "weight_loader"):
                continue
            if not hasattr(experts, "layer_name"):
                experts.layer_name = f"layer_{layer_idx}.mlp.experts"

            original_experts_weight_loader = experts.weight_loader
            experts_cls = type(experts)
            if not getattr(experts_cls, "_verl_direct_moe_class_loader_patched", False):
                original_class_weight_loader = experts_cls.weight_loader

                def _direct_class_weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id, return_success=False):
                    force_direct = "input_scale" in str(weight_name)
                    direct_allowed = force_direct or (
                        enable_direct_unquantized_loader and _quant_config_name(self) is None
                    )
                    if direct_allowed and _direct_copy_moe_tensor(
                        self,
                        param,
                        loaded_weight,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    ):
                        _debug_direct_moe_loader(
                            self,
                            "class_forced_hit" if force_direct else "class_hit",
                            param,
                            loaded_weight,
                            shard_id,
                            expert_id,
                        )
                        return True if return_success else None
                    return original_class_weight_loader(
                        self,
                        param,
                        loaded_weight,
                        weight_name,
                        shard_id,
                        expert_id,
                        return_success=return_success,
                    )

                _direct_class_weight_loader.supports_moe_loading = True
                experts_cls.weight_loader = _direct_class_weight_loader
                experts_cls._verl_direct_moe_class_loader_patched = True

            if not getattr(experts, "_verl_direct_moe_weight_loader_patched", False):
                experts.weight_loader = _make_direct_moe_weight_loader(original_experts_weight_loader, experts)
                experts._verl_direct_moe_weight_loader_patched = True
                patched_layers += 1

            moe_param_tokens = ("w13_weight", "w2_weight", "w13_input_scale", "w2_input_scale")
            for name, param in mlp.named_parameters():
                if "experts" not in name or not any(token in name for token in moe_param_tokens):
                    continue
                if getattr(param, "_verl_direct_moe_loader_patched", False):
                    continue
                fallback_loader = getattr(param, "weight_loader", None) or original_experts_weight_loader
                param.weight_loader = _make_direct_moe_weight_loader(
                    fallback_loader,
                    experts,
                    force_direct="input_scale" in name,
                )
                param._verl_direct_moe_loader_patched = True
                patched_params += 1

    count = getattr(model, "_verl_direct_moe_patch_log_count", 0)
    if count < 4:
        _qat_debug(
            "patch_vllm_moe_model_weight_loader: "
            f"patched_layers={patched_layers} patched_params={patched_params} "
            f"model={type(model).__name__}"
        )
        setattr(model, "_verl_direct_moe_patch_log_count", count + 1)

def patch_vllm_unquantized_moe_process_weights_after_loading():
    try:
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
            UnquantizedFusedMoEMethod,
        )
    except Exception as exc:
        _qat_debug(f"skip unquantized MoE process patch unavailable: {exc}")
        return False

    if getattr(UnquantizedFusedMoEMethod, "_verl_skip_kernel_layout_process_patched", False):
        return True

    original_process = UnquantizedFusedMoEMethod.process_weights_after_loading

    def _param_data(param):
        if isinstance(param, torch.nn.Parameter):
            return param.data
        if isinstance(param, torch.Tensor):
            return param
        return None

    def _iter_moe_modules(layer):
        seen = set()
        queue = [layer]
        for attr in ("experts", "mlp", "block_sparse_moe"):
            child = getattr(layer, attr, None)
            if child is not None:
                queue.append(child)
        while queue:
            module = queue.pop(0)
            if module is None or id(module) in seen:
                continue
            seen.add(id(module))
            yield module
            for attr in ("experts", "mlp", "block_sparse_moe"):
                child = getattr(module, attr, None)
                if child is not None and id(child) not in seen:
                    queue.append(child)

    def _kernel_layout_shapes(layer):
        shapes = []
        for module in _iter_moe_modules(layer):
            for name in ("w13_weight", "w2_weight"):
                data = _param_data(getattr(module, name, None))
                if data is not None:
                    shapes.append(f"{name}={tuple(data.shape)}")
                    if data.ndim >= 4:
                        return shapes, True
        return shapes, False

    def _patched_process_weights_after_loading(self, layer):
        shapes, is_kernel_layout = _kernel_layout_shapes(layer)
        if is_kernel_layout:
            count = getattr(UnquantizedFusedMoEMethod, "_verl_skip_kernel_layout_process_count", 0)
            if count < 8:
                _qat_debug(
                    "skip unquantized MoE process_weights_after_loading for existing kernel layout: "
                    + ", ".join(shapes[:4])
                )
            setattr(UnquantizedFusedMoEMethod, "_verl_skip_kernel_layout_process_count", count + 1)
            return None
        return original_process(self, layer)

    UnquantizedFusedMoEMethod._verl_original_process_weights_after_loading = original_process
    UnquantizedFusedMoEMethod.process_weights_after_loading = _patched_process_weights_after_loading
    UnquantizedFusedMoEMethod._verl_skip_kernel_layout_process_patched = True
    _qat_debug("patched UnquantizedFusedMoEMethod.process_weights_after_loading")
    return True
