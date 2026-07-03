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
import ctypes
import json
import logging
import os
import platform
import re
import signal
import threading
from types import MethodType
from typing import Any, Literal, Optional, get_args

import torch
from vllm.outputs import RequestOutput

from verl.utils.device import is_npu_available
from verl.utils.vllm import TensorLoRARequest, VLLMHijack
from verl.utils.vllm.patch import (
    patch_vllm_moe_model_weight_loader,
    patch_vllm_unquantized_moe_process_weights_after_loading,
)
from verl.utils.vllm.vllm_fp8_utils import apply_vllm_fp8_patches, is_fp8_model, load_quanted_weights

try:
    from vllm_omni.diffusion.worker.diffusion_worker import CustomPipelineWorkerExtension

    from verl.utils.vllm_omni import OmniTensorLoRARequest, VLLMOmniHijack

    _VLLM_OMNI_AVAILABLE = True
except (ImportError, RuntimeError):  # optional stack; ImportError if missing, RuntimeError e.g. diffusers/transformers
    CustomPipelineWorkerExtension = None  # type: ignore[assignment]
    OmniTensorLoRARequest = None  # type: ignore[assignment]
    VLLMOmniHijack = None  # type: ignore[assignment]
    _VLLM_OMNI_AVAILABLE = False

# Use object as fallback base so the class definition is always valid even when
# vllm_omni is not installed (None is not a valid base class).
_OmniWorkerBase = CustomPipelineWorkerExtension if _VLLM_OMNI_AVAILABLE else object

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
_QAT_DEBUG_ENABLED = os.environ.get("VERL_QAT_DEBUG", "0") == "1"


def _qat_debug(message: str) -> None:
    if _QAT_DEBUG_ENABLED:
        print(f"[QAT-DEBUG] {message}", flush=True)

# magic numbers that ensure we are using the same LoRA adapter during the rollout and training process
VLLM_LORA_INT_ID = 123
VLLM_LORA_NAME = "123"
VLLM_LORA_PATH = "simon_lora_path"

VLLM_ASCEND_REQUIRED_ENV_VARS = {"VLLM_ALL2ALL_BACKEND": "flashinfer_all2allv", "VLLM_ASCEND_ENABLE_NZ": "0"}


def set_death_signal():
    """Kill the current process when the parent process exits."""
    if platform.system() != "Linux":
        return
    libc = ctypes.CDLL("libc.so.6")
    libc.prctl(1, signal.SIGKILL)
    if os.getppid() == 1:
        os.kill(os.getpid(), signal.SIGKILL)


def get_device_uuid(device_id: int) -> str:
    from vllm.platforms import current_platform

    # Convert torch.npu.current_device to its corresponding ASCEND_RT_VISIBLE_DEVICES.
    if is_npu_available:
        if os.getenv("ASCEND_RT_VISIBLE_DEVICES") is not None:
            npu_visible_devices = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
            assert device_id < len(npu_visible_devices), f"device_id {device_id} must less than {npu_visible_devices}"
            return "NPU-" + npu_visible_devices[device_id]
        else:
            return f"NPU-{device_id}"
    else:
        return current_platform.get_device_uuid(device_id)


def get_vllm_max_lora_rank(lora_rank: int):
    """
    For vLLM, automatically adjusts the `max_lora_rank` to the nearest allowed value.
    The allowed values are retrieved from vLLM's MaxLoRARanks type definition.
    """
    assert lora_rank > 0, f"lora_rank must be greater than 0, get {lora_rank}"

    try:
        from vllm.config.lora import MaxLoRARanks
    except Exception:
        # FIXME: migrate vllm version https://github.com/vllm-project/vllm/blob/main/vllm/config/lora.py#L25
        MaxLoRARanks = Literal[1, 8, 16, 32, 64, 128, 256, 320, 512]

    vllm_max_lora_ranks = sorted(get_args(MaxLoRARanks))
    if lora_rank > vllm_max_lora_ranks[-1]:
        raise ValueError(f"lora_rank must be less than or equal to {vllm_max_lora_ranks[-1]}, but got {lora_rank}")

    for rank in vllm_max_lora_ranks:
        if lora_rank <= rank:
            return rank


# https://github.com/vllm-project/vllm/issues/13175
def monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        logits = original_compute_logits(*args, **kwargs)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


_MOE_SCALE_NAME_RE = re.compile(
    r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(input_scale|weight_scale_2)$"
)
_MOE_LAYER_IDX_RE = re.compile(r"layers\.(\d+)\b")


def _write_fused_scale(param, idx, val) -> int:
    """Write a per-expert scalar scale into a fused MoE scale param. Returns 1 on success."""
    if param is None or val is None:
        return 0
    data = param.data if hasattr(param, "data") else param
    try:
        data[idx] = val.to(device=data.device, dtype=data.dtype).reshape(())
        return 1
    except Exception:
        return 0


def _apply_moe_scale_stash(model, stash) -> None:
    """Populate vLLM's FUSED MoE scale params from stashed per-expert HF scales.

    Online weight sync exports per-expert, per-projection scales
    (``experts.{E}.{gate,up,down}_proj.{input_scale,weight_scale_2}``) but vLLM's
    fused MoE loader does not route them into ``w13_input_scale (E,2)`` /
    ``w2_input_scale (E,)`` / ``w13_weight_scale_2 (E,2)`` / ``w2_weight_scale_2 (E,)``
    on reload, leaving them as uninitialized ``torch.empty`` garbage (signed ~1e-3).
    We re-apply them here, just before ``process_weights_after_loading`` runs the
    NVFP4 kernel-format convert. Gate->w13 col0, up->w13 col1, down->w2.
    """
    if os.environ.get("VERL_MOE_SCALE_STASH_FIX", "1").lower() in ("0", "false", "no"):
        return
    if not stash:
        return
    parsed: dict = {}
    for name, tensor in stash.items():
        m = _MOE_SCALE_NAME_RE.search(name)
        if m is None:
            continue
        layer_idx, expert_id, role, kind = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
        parsed.setdefault((layer_idx, expert_id), {})[(role, kind)] = tensor
    if not parsed:
        return

    applied = 0
    for mod_name, module in model.named_modules():
        if not hasattr(module, "w13_input_scale"):
            continue
        lm = _MOE_LAYER_IDX_RE.search(mod_name)
        if lm is None:
            continue
        layer_idx = int(lm.group(1))
        entries = {E: kv for (L, E), kv in parsed.items() if L == layer_idx}
        if not entries:
            continue
        map_fn = getattr(module, "_map_global_expert_id_to_local_expert_id", None)

        def _local(expert_id):
            if map_fn is None:
                return int(expert_id)
            try:
                v = map_fn(int(expert_id))
                return int(v.item()) if hasattr(v, "item") else int(v)
            except Exception:
                return -1

        w13_input = getattr(module, "w13_input_scale", None)
        w13_wscale2 = getattr(module, "w13_weight_scale_2", None)
        w2_input = getattr(module, "w2_input_scale", None)
        w2_wscale2 = getattr(module, "w2_weight_scale_2", None)
        for expert_id, kv in entries.items():
            local = _local(expert_id)
            if local < 0:
                continue
            for role, col in (("gate_proj", 0), ("up_proj", 1)):
                applied += _write_fused_scale(w13_input, (local, col), kv.get((role, "input_scale")))
                applied += _write_fused_scale(w13_wscale2, (local, col), kv.get((role, "weight_scale_2")))
            applied += _write_fused_scale(w2_input, local, kv.get(("down_proj", "input_scale")))
            applied += _write_fused_scale(w2_wscale2, local, kv.get(("down_proj", "weight_scale_2")))
    _qat_debug(
        f"_apply_moe_scale_stash: wrote {applied} fused MoE scale entries "
        f"from {len(parsed)} (layer,expert) pairs"
    )


class vLLMColocateWorkerExtension:
    """
    The class for vLLM's worker to inherit from, in the colocate setting.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.

    Feature support:
    1. LoRA
    2. Online FP8 quantization
    """

    def __new__(cls, **kwargs):
        set_death_signal()

        # 1. patch for Lora
        VLLMHijack.hijack()
        # 2. patch online fp8 quant
        if os.environ.get("VERL_VLLM_FP8_QUANT_ENABLED", "0") == "1":
            apply_vllm_fp8_patches()
        # 3. patch QAT (compressed-tensors NVFP4) for dynamic weight loading
        vllm_config = kwargs.get("vllm_config")
        quant_config = getattr(vllm_config, "quant_config", None) if vllm_config else None
        _qat_debug(
            "vLLMColocateWorkerExtension.__new__: "
            f"vllm_config={vllm_config is not None}, "
            f"quant_config type={type(quant_config).__name__ if quant_config else None}, "
            f"quant_config repr={repr(quant_config)[:300] if quant_config else None}"
        )
        if quant_config:
            _qat_debug(f"quant_config dir={[a for a in dir(quant_config) if not a.startswith('_')][:20]}")
            _qat_debug(f"quant_format={getattr(quant_config, 'quant_format', 'NO_ATTR')}")
            _qat_debug(f"quant_method={getattr(quant_config, 'quant_method', 'NO_ATTR')}")
        _is_qat_model = getattr(quant_config, "quant_format", None) == "nvfp4-pack-quantized"
        _is_modelopt_qat = type(quant_config).__name__ == "ModelOptNvFp4Config"
        _qat_debug(f"_is_qat_model={_is_qat_model}, _is_modelopt_qat={_is_modelopt_qat}")
        if _is_qat_model:
            from verl.utils.qat import apply_qat_patches

            apply_qat_patches()
            logger.info("Applied QAT (compressed-tensors) patches in vLLM worker subprocess")
        elif _is_modelopt_qat:
            from verl.utils.modelopt import apply_modelopt_nvfp4_patches

            _qat_mode = os.environ.get("VERL_QAT_MODE", "w4a4")
            apply_modelopt_nvfp4_patches(mode=_qat_mode)
            _qat_debug(f"apply_modelopt_nvfp4_patches(mode={_qat_mode}) called in worker subprocess")
            # Verify the patch was actually applied:
            from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4LinearMethod

            pwal_name = ModelOptNvFp4LinearMethod.process_weights_after_loading.__qualname__
            _qat_debug(f"ModelOptNvFp4LinearMethod.process_weights_after_loading is now: {pwal_name}")
            logger.info("Applied ModelOpt NVFP4 patches in vLLM worker subprocess")

        # TODO: For ascend NPU, when the corresponding vllm-ascend version is upgraded to v0.13.0,
        # please remove the VLLM_ASCEND_REQUIRED_ENV_VARS variable replacement action.
        # This is only a fix for vllm version < v0.13.0.
        if is_npu_available:
            for k in VLLM_ASCEND_REQUIRED_ENV_VARS:
                if k not in os.environ:
                    os.environ[k] = VLLM_ASCEND_REQUIRED_ENV_VARS[k]

        instance = super().__new__(cls)
        instance._is_qat_model = _is_qat_model
        instance._is_modelopt_qat = _is_modelopt_qat
        return instance

    def monkey_patch_model(self, vocab_size: int):
        # patch compute_logits to avoid sampling OOV token
        monkey_patch_compute_logits(self.model_runner.model, vocab_size)
        # patch weight loader to support MoE model
        patch_vllm_moe_model_weight_loader(self.model_runner.model)

    def update_weights_from_ipc(self, peft_config: dict = None, base_sync_done=False, use_shm: bool = False):
        """Update the weights of the rollout model."""
        from vllm.platforms import current_platform

        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

        if current_platform.device_type == "npu" and self.device is None:
            self.device = torch.device(f"npu:{self.local_rank}")

        # In async mode, make sure the old lora is removed before adding the new one
        if peft_config and base_sync_done:
            self.remove_lora(VLLM_LORA_INT_ID)

        use_standard_weight_load = not (peft_config and base_sync_done) and not is_fp8_model(
            self.model_runner.vllm_config
        )

        # Optional instance-attribute survival diagnostics.
        _qat_debug(
            f"update_weights_from_ipc: hasattr _is_modelopt_qat={hasattr(self, '_is_modelopt_qat')}, "
            f"_is_qat_model={getattr(self, '_is_qat_model', 'UNSET')}, "
            f"_is_modelopt_qat={getattr(self, '_is_modelopt_qat', 'UNSET')}, "
            f"use_standard_weight_load={use_standard_weight_load}"
        )

        if self._is_qat_model:
            # QAT (compressed-tensors): Prepare for weight loading BEFORE receiving any buckets
            from verl.utils.qat import prepare_qat_for_load_weights

            prepare_qat_for_load_weights(self.model_runner.model, device=self.device)
            logger.info("QAT: prepare_qat_for_load_weights completed")
        elif self._is_modelopt_qat:
            from verl.utils.modelopt.vllm_modelopt_patch import prepare_modelopt_for_weight_reload

            prepare_modelopt_for_weight_reload(self.model_runner.model, device=self.device)
            logger.info("ModelOpt: prepare_modelopt_for_weight_reload completed")
        elif use_standard_weight_load:
            # Re-apply here because async IPC weight sync can happen long after init and lose MoE weight_loader attrs.
            patch_vllm_moe_model_weight_loader(self.model_runner.model)

        assert self.device is not None
        receiver = BucketedWeightReceiver(
            zmq_handle=self._get_zmq_handle(),
            device=self.device,
            use_shm=use_shm,
        )
        receiver.receive_weights(
            on_bucket_received=lambda weights: self._update_weights(
                weights, peft_config=peft_config, base_sync_done=base_sync_done
            )
        )

        if self._is_qat_model:
            # QAT (compressed-tensors): call process_weights_after_loading AFTER all buckets are received
            from verl.utils.qat import manual_process_weights_after_loading

            manual_process_weights_after_loading(self.model_runner.model)
            logger.info("QAT: process_weights_after_loading completed")
        elif self._is_modelopt_qat:
            from verl.utils.modelopt.vllm_modelopt_patch import modelopt_process_weights_after_loading

            # Re-apply per-expert MoE scales into fused params before the
            # NVFP4 kernel-format convert reads them (vLLM's fused loader drops them on reload).
            _apply_moe_scale_stash(self.model_runner.model, getattr(self, "_verl_moe_scale_stash", None))
            modelopt_process_weights_after_loading(self.model_runner.model)
            logger.info("ModelOpt QAT: process_weights_after_loading completed")
        elif use_standard_weight_load:
            # Some post-load transforms are non-idempotent; run once after all buckets.
            from vllm.model_executor.model_loader.utils import process_weights_after_loading

            patch_vllm_unquantized_moe_process_weights_after_loading()
            model = self.model_runner.model
            model_config = self.model_runner.vllm_config.model_config
            process_weights_after_loading(model, model_config, self.device)

    def _update_weights(self, weights: list[tuple[str, torch.Tensor]], peft_config: dict, base_sync_done: bool):
        if peft_config and base_sync_done:
            weights = dict(weights)
            lora_request = TensorLoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=peft_config,
                lora_tensors=weights,
            )
            self.add_lora(lora_request)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
        else:
            # Add the FP8 related logic here as sharding manager has been deprecated.
            # Check if FP8 quantization is enabled and apply appropriate weight loading
            if is_fp8_model(self.model_runner.vllm_config):
                logger.info(f"FP8 model detected (async): {self.model_runner.vllm_config.quant_config}")
                # Convert bf16 weights to fp8 format before loading
                loaded_params = load_quanted_weights(weights, self.model_runner)
                logger.info(f"FP8 weights loaded (async), loaded_params: {len(loaded_params)}")
            else:
                if getattr(self, "_is_modelopt_qat", False):
                    # Stash per-expert MoE activation/global scales across buckets; vLLM's
                    # fused MoE loader drops them on reload (-> _apply_moe_scale_stash in finalize).
                    stash = getattr(self, "_verl_moe_scale_stash", None)
                    if stash is None:
                        stash = {}
                        self._verl_moe_scale_stash = stash
                    for _name, _t in weights:
                        if "experts." in _name and (_name.endswith(".input_scale") or _name.endswith(".weight_scale_2")):
                            try:
                                stash[_name] = _t.detach().to("cpu")
                            except Exception:
                                pass
                    log_count = getattr(self, "_modelopt_input_scale_bucket_log_count", 0)
                    if log_count < 16:
                        def _vstat(_t):
                            try:
                                _f = _t.detach().float()
                                return (f"shape={tuple(_t.shape)} min={_f.min().item():.6g} "
                                        f"max={_f.max().item():.6g} mean={_f.mean().item():.6g}")
                            except Exception as _e:  # pragma: no cover - debug only
                                return f"<{type(_t).__name__}:{_e}>"
                        input_scale_names = [name for name, _tensor in weights if "input_scale" in name]
                        # Optionally report exported MoE-expert scale values to distinguish
                        # incorrect export from scales dropped during loading.
                        moe_scale_samples = [
                            (name, t) for name, t in weights
                            if "experts." in name and ("input_scale" in name or "weight_scale_2" in name)
                        ]
                        if input_scale_names or log_count < 4:
                            shown = "; ".join(f"{n} {_vstat(t)}" for n, t in moe_scale_samples[:4])
                            _qat_debug(
                                f"ModelOpt reload bucket input_scale_count={len(input_scale_names)} "
                                f"moe_scale_count={len(moe_scale_samples)} sample_names={input_scale_names[:4]} "
                                f"moe_scale_values=[{shown}]"
                            )
                        setattr(self, "_modelopt_input_scale_bucket_log_count", log_count + 1)
                logger.info("Loading standard weights (non-FP8, async)")
                self.model_runner.model.load_weights(weights)

    def _get_zmq_handle(self) -> str:
        """Get ZMQ handle for communication."""
        if not hasattr(self, "device_uuid") or not self.device_uuid:
            self.device_uuid = get_device_uuid(self.device.index)
        return f"ipc:///tmp/rl-colocate-zmq-{self.device_uuid}.sock"


class vLLMOmniColocateWorkerExtension(_OmniWorkerBase):
    """
    The class for vLLM-Omni's worker to inherit from, in the colocate setting.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.

    Feature support:
    1. LoRA
    """

    def __new__(cls, **kwargs):
        assert _VLLM_OMNI_AVAILABLE, "vLLM-Omni is required to use vLLMOmniColocateWorkerExtension"
        set_death_signal()

        # 1. patch for Lora
        VLLMOmniHijack.hijack()

        return super().__new__(cls)

    def update_weights_from_ipc(self, peft_config: dict = None, base_sync_done=False, use_shm: bool = False):
        """Update the weights of the rollout model."""

        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

        # In async mode, make sure the old lora is removed before adding the new one
        if peft_config and base_sync_done:
            self.remove_lora(VLLM_LORA_INT_ID)

        assert self.device is not None
        receiver = BucketedWeightReceiver(
            zmq_handle=self._get_zmq_handle(),
            device=self.device,
            use_shm=use_shm,
        )
        receiver.receive_weights(
            on_bucket_received=lambda weights: self._update_weights(
                weights, peft_config=peft_config, base_sync_done=base_sync_done
            )
        )

    def _update_weights(self, weights: list[tuple[str, torch.Tensor]], peft_config: dict, base_sync_done: bool):
        if peft_config and base_sync_done:
            weights = dict(weights)
            lora_request = OmniTensorLoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=peft_config,
                lora_tensors=weights,
            )
            self.add_lora(lora_request)
            logger.info(f"vLLM-Omni load weights, loaded_params: {len(weights)}")
        else:
            logger.info("Loading standard weights (async)")
            self.load_weights(weights)

    def _get_zmq_handle(self) -> str:
        """Get ZMQ handle for communication."""
        if not hasattr(self, "device_uuid") or not self.device_uuid:
            self.device_uuid = get_device_uuid(self.device.index)
        return f"ipc:///tmp/rl-colocate-zmq-{self.device_uuid}.sock"


class SuppressSignalInThread:
    def __enter__(self):
        self.original_signal = signal.signal

        def no_op_signal(sig, action):
            if threading.current_thread() is not threading.main_thread():
                print(f"Ignored signal {sig} in thread {threading.current_thread().name}")
                return
            return self.original_signal(sig, action)

        signal.signal = no_op_signal
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal = self.original_signal


def build_cli_args_from_config(config: dict[str, Any]) -> list[str]:
    """
    Convert a config dictionary to CLI arguments for vLLM server.

    Handles different value types appropriately:
    - None: skipped
    - bool True: adds '--key'
    - bool False: skipped
    - list: expands to '--key item1 item2 ...'
    - empty list: skipped (vLLM uses nargs="+" which requires at least one value)
    - dict: JSON serialized
    - other: string converted

    Args:
        config: Dictionary of configuration key-value pairs

    Returns:
        List of CLI argument strings
    """
    cli_args = []
    for k, v in config.items():
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                cli_args.append(f"--{k}")
        elif isinstance(v, list):
            if not v:
                # Skip empty lists - vLLM uses nargs="+" which requires at least one value
                continue
            # Lists need to be expanded as multiple separate arguments
            # e.g., --cuda-graph-sizes 1 2 4 8 becomes ['--cuda-graph-sizes', '1', '2', '4', '8']
            cli_args.append(f"--{k}")
            cli_args.extend([str(item) for item in v])
        else:
            cli_args.append(f"--{k}")
            # Use json.dumps for dict to ensure valid JSON format
            cli_args.append(json.dumps(v) if isinstance(v, dict) else str(v))
    return cli_args


def extract_prompt_logprobs(output: RequestOutput, num_prompt_logprobs: Optional[int], result_dict: dict[str, list]):
    """Extract prompt log probabilities from generation output."""
    if num_prompt_logprobs is None:
        return

    prompt_logprobs_ls, prompt_ids_ls = [], []
    # NOTE: logprob of first prompt token is None.
    for logprobs_dict in output.prompt_logprobs[1:]:
        if num_prompt_logprobs == 0:
            token_id_str = list(logprobs_dict.keys())[0]
            logprob = logprobs_dict[token_id_str].logprob
            prompt_logprobs_ls.append([logprob])
            prompt_ids_ls.append([int(token_id_str)])
        else:
            prompt_ids = [None] * num_prompt_logprobs
            prompt_logprobs = [None] * num_prompt_logprobs
            # We get either top-k logprobs or top-k plus the sampled logprob (if sampled token is not in top-k)
            assert len(logprobs_dict) in [num_prompt_logprobs, num_prompt_logprobs + 1], len(logprobs_dict)
            for token_id_str, token_logprob in logprobs_dict.items():
                rank = token_logprob.rank
                if rank > num_prompt_logprobs:
                    continue  # the sampled token is not in the top-k
                logprob = token_logprob.logprob
                prompt_ids[rank - 1] = int(token_id_str)
                prompt_logprobs[rank - 1] = logprob
            prompt_logprobs_ls.append(prompt_logprobs)
            prompt_ids_ls.append(prompt_ids)

    # NOTE: pad a dummy prompt logprob for last prompt token.
    prompt_logprobs_ls.append([0.0] * max(num_prompt_logprobs, 1))
    prompt_ids_ls.append([0] * max(num_prompt_logprobs, 1))

    result_dict["prompt_ids"] = prompt_ids_ls
    result_dict["prompt_logprobs"] = prompt_logprobs_ls
