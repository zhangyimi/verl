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
import signal
import threading
from types import MethodType
from typing import Any, Literal, get_args

import torch

from verl.utils.device import is_npu_available
from verl.utils.vllm import TensorLoRARequest, VLLMHijack
from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader
from verl.utils.vllm.vllm_fp8_utils import apply_vllm_fp8_patches, is_fp8_model, load_quanted_weights

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# magic numbers that ensure we are using the same LoRA adapter during the rollout and training process
VLLM_LORA_INT_ID = 123
VLLM_LORA_NAME = "123"
VLLM_LORA_PATH = "simon_lora_path"

VLLM_ASCEND_REQUIRED_ENV_VARS = {"VLLM_ALL2ALL_BACKEND": "flashinfer_all2allv", "VLLM_ASCEND_ENABLE_NZ": "0"}

# vLLM declares this option with argparse.BooleanOptionalAction, and its
# default is version-dependent (True in the production 0.15 image).  False
# therefore must be serialized explicitly instead of being omitted.
_VLLM_BOOLEAN_OPTIONAL_FALSE_KEYS = {"enable_prefix_caching", "enable-prefix-caching"}


def route_capture_requested() -> bool:
    """Whether this vLLM worker must return per-token MoE routes.

    Production R3 deliberately uses a separate switch from ``VERL_ROUTE_DIAG``:
    the latter enables expensive counterfactual diagnostics, while R3 only
    needs the lightweight route callback and returned top-k expert ids.
    """

    return (
        os.environ.get("VERL_ROLLOUT_ROUTE_CAPTURE", "0") == "1"
        or os.environ.get("VERL_ROUTE_DIAG", "0") == "1"
    )


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


def _has_upstream_route_capture_binding() -> bool:
    """Whether vLLM contains the router-level binding from PR #33013.

    This is capability-gated instead of version-gated because the production
    runtime intentionally backports PR #33013 onto vLLM 0.15.0.
    """

    try:
        from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except (ImportError, AttributeError):
        return False

    return callable(getattr(BaseRouter, "set_capture_fn", None)) and callable(
        getattr(GPUModelRunner, "_bind_routed_experts_capturer", None)
    )


def _validate_upstream_route_capturers(model) -> None:
    """Validate PR #33013 binding without installing the legacy callback.

    PR #33013 captures logical expert ids in ``BaseRouter`` before EPLB maps
    them to physical ids.  Installing the old ``FusedMoE.capture`` callback on
    top would either be redundant or double-capture monolithic kernels.
    """

    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

    attached = []
    missing = []
    monolithic = []
    for module in model.modules():
        if not isinstance(module, FusedMoE):
            continue
        layer_id = int(module.layer_id)
        quant_method = module.quant_method
        if getattr(quant_method, "is_monolithic", False):
            monolithic.append(layer_id)
        if not isinstance(module.router, BaseRouter) or not callable(
            getattr(module.router, "capture_fn", None)
        ):
            missing.append(layer_id)
        else:
            attached.append(layer_id)

    if monolithic:
        raise RuntimeError(
            "PR #33013 route capture requires the modular MoE path; "
            f"monolithic layers={monolithic[:8]}"
        )
    if missing:
        raise RuntimeError(
            "PR #33013 is present but its router capture callback was not bound; "
            f"missing layers={missing[:8]}"
        )
    if not attached:
        raise RuntimeError("route capture requested but no vLLM FusedMoE layers were found")

    print(
        f"[route-capture] using upstream PR #33013 on {len(attached)} MoE layers; "
        f"layers={min(attached)}..{max(attached)}; "
        "require_upstream="
        f"{os.environ.get('VERL_REQUIRE_UPSTREAM_ROUTE_CAPTURE', '0')}",
        flush=True,
    )


def _attach_legacy_route_capturers(model) -> None:
    """Attach vLLM's pre-#33013 route capturer after initialization.

    vLLM 0.15 constructs ``FusedMoE`` modules before it creates the global
    ``RoutedExpertsCapturer``.  Consequently each module initializes its
    ``capture`` callback to ``None`` and the shared route buffer stays all
    zeros.  ``monkey_patch_model`` runs after engine initialization, which is
    the first point where both objects are guaranteed to exist.
    """

    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.fused_moe.routed_experts_capturer import RoutedExpertsCapturer

    capturer = RoutedExpertsCapturer.get_instance()
    if capturer is None:
        raise RuntimeError("route capture requested but vLLM RoutedExpertsCapturer is not initialized")

    attached = []
    for module in model.modules():
        if not isinstance(module, FusedMoE):
            continue
        layer_id = int(module.layer_id)
        module.capture = lambda topk_ids, layer_id=layer_id: capturer.capture(layer_id, topk_ids)

        # Most kernels expose top-k ids through FusedMoE's modular path.  A
        # monolithic kernel receives only router logits, so vLLM never calls
        # module.capture.  In diagnostic eager mode, derive the same top-k via
        # the layer's router immediately before invoking the original kernel.
        quant_method = module.quant_method
        if quant_method.is_monolithic and not getattr(quant_method, "_verl_route_capture_wrapped", False):
            original_apply_monolithic = quant_method.apply_monolithic

            def apply_monolithic_with_capture(
                _quant_method,
                layer,
                x,
                router_logits,
                *args,
                _original=original_apply_monolithic,
                **kwargs,
            ):
                _, topk_ids = layer.router.select_experts(hidden_states=x, router_logits=router_logits)
                layer.capture(topk_ids)
                return _original(layer, x, router_logits, *args, **kwargs)

            quant_method.apply_monolithic = MethodType(apply_monolithic_with_capture, quant_method)
            quant_method._verl_route_capture_wrapped = True
        attached.append(layer_id)

    if not attached:
        raise RuntimeError("route capture requested but no vLLM FusedMoE layers were found")
    print(
        f"[route-capture] attached vLLM route capturers to {len(attached)} MoE layers; "
        f"layers={min(attached)}..{max(attached)}",
        flush=True,
    )


def ensure_route_capturers(model) -> None:
    """Use PR #33013 when available, otherwise retain the legacy fallback."""

    if _has_upstream_route_capture_binding():
        _validate_upstream_route_capturers(model)
    elif os.environ.get("VERL_REQUIRE_UPSTREAM_ROUTE_CAPTURE", "0") == "1":
        raise RuntimeError(
            "VERL_REQUIRE_UPSTREAM_ROUTE_CAPTURE=1, but vLLM PR #33013 "
            "capabilities are missing; refusing to fall back to legacy capture"
        )
    else:
        _attach_legacy_route_capturers(model)


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
        _is_qat_model = getattr(quant_config, "quant_format", None) == "nvfp4-pack-quantized"
        if _is_qat_model:
            from verl.utils.qat import apply_qat_patches

            apply_qat_patches()
            logger.info("Applied QAT patches in vLLM worker subprocess")

        # TODO: For ascend NPU, when the corresponding vllm-ascend version is upgraded to v0.13.0,
        # please remove the VLLM_ASCEND_REQUIRED_ENV_VARS variable replacement action.
        # This is only a fix for vllm version < v0.13.0.
        if is_npu_available:
            for k in VLLM_ASCEND_REQUIRED_ENV_VARS:
                if k not in os.environ:
                    os.environ[k] = VLLM_ASCEND_REQUIRED_ENV_VARS[k]

        instance = super().__new__(cls)
        instance._is_qat_model = _is_qat_model
        return instance

    def monkey_patch_model(self, vocab_size: int):
        # patch compute_logits to avoid sampling OOV token
        monkey_patch_compute_logits(self.model_runner.model, vocab_size)
        # patch weight loader to support MoE model
        patch_vllm_moe_model_weight_loader(self.model_runner.model)
        if route_capture_requested():
            ensure_route_capturers(self.model_runner.model)

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

        if self._is_qat_model:
            # QAT: Prepare for weight loading BEFORE receiving any buckets
            from verl.utils.qat import prepare_qat_for_load_weights

            prepare_qat_for_load_weights(self.model_runner.model, device=self.device)
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
            # QAT: call process_weights_after_loading AFTER all buckets are received
            from verl.utils.qat import manual_process_weights_after_loading

            manual_process_weights_after_loading(self.model_runner.model)
            logger.info("QAT: process_weights_after_loading completed")
        elif use_standard_weight_load:
            # Some post-load transforms are non-idempotent; run once after all buckets.
            from vllm.model_executor.model_loader.utils import process_weights_after_loading

            model = self.model_runner.model
            model_config = self.model_runner.vllm_config.model_config
            process_weights_after_loading(model, model_config, self.device)

        if route_capture_requested():
            # On the PR #33013 path this only validates that router callbacks
            # survived QAT post-processing.  Legacy vLLM still re-attaches its
            # FusedMoE callback idempotently.
            ensure_route_capturers(self.model_runner.model)

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
                logger.info("Loading standard weights (non-FP8, async)")
                self.model_runner.model.load_weights(weights)

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
    - bool False: skipped, except known BooleanOptionalAction flags, which use
      their explicit ``--no-...`` spelling
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
            elif k in _VLLM_BOOLEAN_OPTIONAL_FALSE_KEYS:
                cli_args.append(f"--no-{k.replace('_', '-')}")
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
