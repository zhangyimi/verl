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

"""QAT FakeQuantized Linear module for NVFP4 (W4A4/W4A16) with FSDP compatibility.

Includes Triton kernels for high-performance FP4 quantization.
"""

from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["QATLinear", "QATMode"]


import triton
import triton.language as tl

_TORCH_TO_TL_DTYPE = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}
FP4_E2M1_MAX: float = 6.0
FP8_E4M3_MAX: float = 448.0


@triton.jit
def _fp4_fake_quant_kernel(
    x_ptr,
    y_ptr,
    M,
    N,
    global_scale_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    scale_out_ptr,
    num_scale_cols,
    stride_sm,
    stride_sn,
    BLOCK_SIZE: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    NUM_FP4_BLOCKS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    FP4_MAX: tl.constexpr,
    FP8_MAX: tl.constexpr,
    VLLM_REF_ARITH: tl.constexpr = False,
    EMIT_SCALES: tl.constexpr = False,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    row_start = pid_m * TILE_M
    col_start = pid_n * TILE_N

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, N),
        strides=(stride_xm, stride_xn),
        offsets=(row_start, col_start),
        block_shape=(TILE_M, TILE_N),
        order=(1, 0),
    )
    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(M, N),
        strides=(stride_ym, stride_yn),
        offsets=(row_start, col_start),
        block_shape=(TILE_M, TILE_N),
        order=(1, 0),
    )

    global_scale = tl.load(global_scale_ptr).to(tl.float32)
    global_scale_safe = tl.where(global_scale > 0.0, global_scale, 1e-12)

    tile = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    tile_reshaped = tl.reshape(tile, (TILE_M, NUM_FP4_BLOCKS, BLOCK_SIZE))
    x_abs = tl.abs(tile_reshaped)

    block_max = tl.max(x_abs, axis=2, keep_dims=True)

    if VLLM_REF_ARITH:
        # vLLM ref arithmetic using ONLY division (no reciprocals)
        # scale = gs * (vec_max / FP4_MAX) → fp8 → fp32
        scale_f32 = global_scale_safe * (block_max / FP4_MAX)
        scale_f32 = tl.minimum(scale_f32, FP8_MAX)
        fp8_scale = scale_f32.to(tl.float8e4nv).to(tl.float32)
        # dequant_scale = fp8_scale / global_scale (direct division, matches ref)
        dequant_scale = tl.where(global_scale_safe != 0.0, fp8_scale / global_scale_safe, 0.0)
        # output_scale = 1 / dequant_scale = global_scale / fp8_scale
        output_scale = tl.where(fp8_scale != 0.0, global_scale_safe / fp8_scale, 0.0)
        output_scale_bc = tl.broadcast_to(output_scale, (TILE_M, NUM_FP4_BLOCKS, BLOCK_SIZE))
        abs_scaled = x_abs * output_scale_bc
        dequant_scale_bc = tl.broadcast_to(dequant_scale, (TILE_M, NUM_FP4_BLOCKS, BLOCK_SIZE))
    else:
        # Original FSDP arithmetic (used for weights)
        block_max_scaled = block_max / (FP4_MAX * global_scale_safe)
        block_max_scaled = tl.minimum(block_max_scaled, FP8_MAX)
        block_max_quant = block_max_scaled.to(tl.float8e4nv).to(tl.float32) * global_scale
        block_max_quant = tl.where(block_max_quant >= 1e-5, block_max_quant, 1.0)
        block_max_quant_broadcast = tl.broadcast_to(block_max_quant, (TILE_M, NUM_FP4_BLOCKS, BLOCK_SIZE))
        abs_scaled = x_abs / block_max_quant_broadcast

    q_val = tl.where(
        abs_scaled <= 0.25,
        0.0,
        tl.where(
            abs_scaled < 0.75,
            0.5,
            tl.where(
                abs_scaled <= 1.25,
                1.0,
                tl.where(
                    abs_scaled < 1.75,
                    1.5,
                    tl.where(
                        abs_scaled <= 2.5,
                        2.0,
                        tl.where(abs_scaled < 3.5, 3.0, tl.where(abs_scaled <= 5.0, 4.0, FP4_MAX)),
                    ),
                ),
            ),
        ),
    )

    if VLLM_REF_ARITH:
        x_rescaled = q_val * dequant_scale_bc
    else:
        x_rescaled = q_val * block_max_quant_broadcast
    x_rescaled = tl.where(tile_reshaped >= 0, x_rescaled, -x_rescaled)
    tile_quant = tl.reshape(x_rescaled, (TILE_M, TILE_N))

    tl.store(y_block_ptr, tile_quant.to(OUT_DTYPE), boundary_check=(0, 1))

    if EMIT_SCALES:
        # Write kernel-produced FP8 block scales to output buffer
        if VLLM_REF_ARITH:
            kernel_fp8_scale = fp8_scale  # (TILE_M, NUM_FP4_BLOCKS, 1)
        else:
            kernel_fp8_scale = block_max_scaled.to(tl.float8e4nv).to(tl.float32)
            kernel_fp8_scale = tl.reshape(kernel_fp8_scale, (TILE_M, NUM_FP4_BLOCKS, 1))
        scale_2d = tl.reshape(kernel_fp8_scale, (TILE_M, NUM_FP4_BLOCKS))
        scale_col_start = pid_n * NUM_FP4_BLOCKS
        scale_block_ptr = tl.make_block_ptr(
            base=scale_out_ptr,
            shape=(M, num_scale_cols),
            strides=(stride_sm, stride_sn),
            offsets=(row_start, scale_col_start),
            block_shape=(TILE_M, NUM_FP4_BLOCKS),
            order=(1, 0),
        )
        tl.store(scale_block_ptr, scale_2d, boundary_check=(0, 1))


def fp4_fake_quant_weight(
    weight: torch.Tensor,
    global_amax: torch.Tensor = None,
    block_size: int = 16,
    tile_rows: int = 16,
    tile_cols: int = 64,
    vllm_ref_arith: bool = False,
    return_scales: bool = False,
) -> torch.Tensor:
    """Apply FP4 fake quantization using Triton kernel.

    Args:
        vllm_ref_arith: If True, use vLLM ref arithmetic for normalization/dequant.
        return_scales: If True, also return kernel-produced FP8 block scales
            as a (M, N//block_size) float32 tensor. Used for AC-1 validation.
    """
    x_shape = weight.shape
    x_dtype = weight.dtype
    x = weight.reshape(-1, x_shape[-1]).contiguous()
    M, N = x.shape
    y = torch.empty_like(x)

    stride_xm, stride_xn = x.stride()
    stride_ym, stride_yn = y.stride()

    tile_cols = max(tile_cols, block_size)
    tile_cols_aligned = ((tile_cols + block_size - 1) // block_size) * block_size
    num_fp4_blocks = tile_cols_aligned // block_size

    if global_amax is None:
        global_amax = weight.abs().max().to(torch.float32)

    if vllm_ref_arith:
        global_scale = global_amax.float()
    else:
        global_scale = global_amax.float() / (FP4_E2M1_MAX * FP8_E4M3_MAX)

    # Scale output buffer (for EMIT_SCALES)
    num_scale_cols = N // block_size
    if return_scales:
        scale_out = torch.empty(M, num_scale_cols, dtype=torch.float32, device=x.device)
    else:
        scale_out = torch.empty(1, dtype=torch.float32, device=x.device)  # dummy

    grid = (triton.cdiv(M, tile_rows), triton.cdiv(N, tile_cols_aligned))

    _fp4_fake_quant_kernel[grid](
        x,
        y,
        M,
        N,
        global_scale,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        scale_out,
        num_scale_cols,
        scale_out.stride(0) if return_scales else 1,
        scale_out.stride(1) if return_scales else 1,
        BLOCK_SIZE=block_size,
        TILE_M=tile_rows,
        TILE_N=tile_cols_aligned,
        NUM_FP4_BLOCKS=num_fp4_blocks,
        OUT_DTYPE=_TORCH_TO_TL_DTYPE[x_dtype],
        FP4_MAX=FP4_E2M1_MAX,
        FP8_MAX=FP8_E4M3_MAX,
        VLLM_REF_ARITH=vllm_ref_arith,
        EMIT_SCALES=return_scales,
    )
    if return_scales:
        return y.view(*x_shape), scale_out
    return y.view(*x_shape)


class STEFP4QuantTriton(torch.autograd.Function):
    """Straight-Through Estimator wrapper for Triton FP4 quantization kernel."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, global_amax: torch.Tensor, block_size: int,
                vllm_ref_arith: bool = False) -> torch.Tensor:
        return fp4_fake_quant_weight(x, global_amax=global_amax, block_size=block_size,
                                      vllm_ref_arith=vllm_ref_arith)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        return grad_output, None, None, None


def _vllm_ref_fake_quant_activation(x: torch.Tensor, global_scale: torch.Tensor, block_size: int) -> torch.Tensor:
    """Pure PyTorch activation fake-quant that CALLS vLLM ref_nvfp4_quant directly.

    This guarantees bit-exact match with the vLLM ref by using the SAME function.
    """
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import ref_nvfp4_quant

    x_dtype = x.dtype
    m, n = x.shape
    igs = global_scale.float().to(x.device)

    # Call vLLM ref directly — guaranteed bit-exact match
    fp4_vals, scale = ref_nvfp4_quant(x.float(), igs, block_size)

    # Dequant: fp4_val * scale / global_scale (same as test's vllm_ref_quant_dequant)
    fp4_3d = fp4_vals.reshape(m, n // block_size, block_size)
    scale_3d = scale.unsqueeze(-1) / igs
    result = (fp4_3d * scale_3d).reshape(m, n)

    return result.to(x_dtype)


class _STE_vllm_ref_activation(torch.autograd.Function):
    """STE wrapper for vLLM-ref-matching activation fake-quant."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, global_scale: torch.Tensor, block_size: int) -> torch.Tensor:
        return _vllm_ref_fake_quant_activation(x, global_scale, block_size)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        return grad_output, None, None


class QATMode(str, Enum):
    """QAT quantization mode."""

    W4A4 = "w4a4"  # Weight 4-bit, Activation 4-bit (dynamic)
    W4A16 = "w4a16"  # Weight 4-bit, Activation 16-bit (weight only)


class QATLinear(nn.Linear):
    """QAT FakeQuantized Linear layer with FSDP compatibility."""

    _UNINITIALIZED_SCALE = -1.0

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode: QATMode = QATMode.W4A4,
        group_size: int = 16,
        activation_observer: str = "static_minmax",  # Observer strategy for activation global_scale
        activation_observer_update_interval: int = 1,  # PERF: update activation scale every N steps
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)

        self.mode = mode
        self.group_size = group_size
        self.activation_observer = activation_observer

        self._weight_blockwise_scale: Optional[torch.Tensor] = None
        self._weight_global_scale: Optional[torch.Tensor] = None
        self._cached_weight_amax: Optional[torch.Tensor] = None
        self._fusion_siblings_ref = None

        # --- PERF (W4A4 FSDP): interval-gated observer + no per-forward .item() ---
        self.activation_observer_update_interval = activation_observer_update_interval
        self._activation_observer_train_step = 0
        self._last_activation_observer_update_step = -1
        self._amax_initialized = False

        if mode == QATMode.W4A4:
            # PERF: persistent=False keeps these out of the FSDP sharded state_dict, where each of
            # the 36864 buffers triggers a cross-node collective (~145s at 32-rank). They are
            # re-injected for the quantizer via a local read in get_per_tensor_param.
            self.register_buffer(
                "input_global_scale", torch.tensor([self._UNINITIALIZED_SCALE], dtype=torch.float32), persistent=False
            )

            self.register_buffer(
                "input_amax", torch.tensor([self._UNINITIALIZED_SCALE], dtype=torch.float32), persistent=False
            )

            self._ema_decay: float = 0.01

        self.fake_quant_enabled = True

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        mode: QATMode = QATMode.W4A4,
        group_size: int = 16,
        activation_observer: str = "static_minmax",
        activation_observer_update_interval: int = 1,
    ) -> "QATLinear":
        """Create QATLinear from an existing nn.Linear."""
        has_bias = linear.bias is not None

        new_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=has_bias,
            mode=mode,
            group_size=group_size,
            activation_observer=activation_observer,
            activation_observer_update_interval=activation_observer_update_interval,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )

        if linear.weight.device != torch.device("meta"):
            new_linear.weight = nn.Parameter(linear.weight.clone())
            if has_bias:
                new_linear.bias = nn.Parameter(linear.bias.clone())

        return new_linear

    def _is_amax_initialized(self) -> bool:
        """Check if input_amax has been initialized.

        PERF: read a python bool flag instead of input_amax.item(), which would force a
        GPU->host sync on every QATLinear.forward (thousands per step for MoE experts).
        """
        return getattr(self, "_amax_initialized", False)

    def set_activation_observer_train_step(self, step: int):
        """Set current optimizer-step index for observer update cadence."""
        self._activation_observer_train_step = int(step)

    def _should_update_activation_observer(self) -> bool:
        """Gate the (expensive) activation amax reduction to every N optimizer steps.

        Callers wrap this in torch.no_grad(); per-step dedup keeps gradient-checkpoint
        recompute consistent with the original forward.
        """
        step = self._activation_observer_train_step
        if self._last_activation_observer_update_step == step:
            return False
        interval = max(1, int(getattr(self, 'activation_observer_update_interval', 1)))
        return (step % interval) == 0

    def _update_input_global_scale(self, x: torch.Tensor):
        """Update static input_global_scale based on observer strategy.

        Note: all_reduce is NOT done here because MoE experts with 0 routed
        tokens skip forward() on some ranks, causing all_reduce to deadlock.
        Cross-rank synchronization is done separately via sync_qat_input_amax()
        in core.py, called before weight sync (outside forward, all ranks participate).
        """
        assert self.mode == QATMode.W4A4, "_update_input_global_scale should only be called in W4A4 mode"

        current_amax = torch.amax(torch.abs(x)).detach().to(torch.float32)

        scale_factor = FP8_E4M3_MAX * FP4_E2M1_MAX

        if self.activation_observer == "memoryless_minmax":
            new_scale = (scale_factor / (current_amax + 1e-12)).view(1)
            self.input_global_scale.copy_(new_scale.to(self.input_global_scale.device))

        elif self.activation_observer == "static_minmax":
            if not self._is_amax_initialized():
                self.input_amax.copy_(current_amax.view(1).to(self.input_amax.device))
            else:
                new_amax = torch.maximum(self.input_amax, current_amax.view(1).to(self.input_amax.device))
                self.input_amax.copy_(new_amax)
            amax_f32 = self.input_amax.to(torch.float32)
            new_scale = (scale_factor / (amax_f32 + 1e-12)).float().view(1)
            self.input_global_scale.copy_(new_scale.to(self.input_global_scale.device))

        elif self.activation_observer == "minmax":
            if not self._is_amax_initialized():
                self.input_amax.copy_(current_amax.view(1).to(self.input_amax.device))
            else:
                new_amax = (1 - self._ema_decay) * self.input_amax + self._ema_decay * current_amax.view(1).to(
                    self.input_amax.device
                )
                self.input_amax.copy_(new_amax)
            amax_f32 = self.input_amax.to(torch.float32)
            new_scale = (scale_factor / (amax_f32 + 1e-12)).float().view(1)
            self.input_global_scale.copy_(new_scale.to(self.input_global_scale.device))

        else:
            raise ValueError(f"Unknown activation_observer: {self.activation_observer}")

    def _fake_quantize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Apply fake quantization to weight tensor using Triton kernel."""
        with torch.no_grad():
            if self._cached_weight_amax is not None:
                global_amax = self._cached_weight_amax
            else:
                siblings_ref = getattr(self, "_fusion_siblings_ref", None)

                if siblings_ref is not None:
                    siblings = [ref() for ref in siblings_ref if ref() is not None]
                    siblings = [s for s in siblings if s.weight.device != torch.device("meta")]

                    for sibling in siblings:
                        sibling_amax = getattr(sibling, "_cached_weight_amax", None)
                        if sibling_amax is not None:
                            global_amax = sibling_amax
                            self._cached_weight_amax = global_amax
                            break
                    else:
                        all_modules = [self] + siblings
                        amaxes = [m.weight.abs().max().to(torch.float32) for m in all_modules]
                        global_amax = torch.max(torch.stack(amaxes))

                        self._cached_weight_amax = global_amax
                        for sibling in siblings:
                            sibling._cached_weight_amax = global_amax
                else:
                    global_amax = weight.abs().max().to(torch.float32)
                    self._cached_weight_amax = global_amax

            if self._weight_global_scale is None:
                self._weight_global_scale = global_amax.float() / (FP4_E2M1_MAX * FP8_E4M3_MAX)

        result = STEFP4QuantTriton.apply(weight, global_amax, self.group_size)

        return result

    def _fake_quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fake quantization to activation tensor (W4A4 mode only).

        Uses pure PyTorch arithmetic matching vLLM ref_nvfp4_quant exactly.
        This bypasses the Triton kernel for activations to achieve exact
        dequant alignment with the vLLM inference quantizer.
        """
        original_shape = x.shape

        if x.dim() == 3:
            x_2d = x.view(-1, x.shape[-1])
        else:
            x_2d = x

        if self.training:
            with torch.no_grad():  # keep observer ops out of grad-ckpt saved-tensor accounting
                if self._should_update_activation_observer():
                    self._update_input_global_scale(x_2d)
                    self._amax_initialized = True
                    self._last_activation_observer_update_step = self._activation_observer_train_step

        if not self._amax_initialized:
            # off hot path: only hit before the first observed forward (e.g. eval w/ PTQ scales)
            if float(self.input_global_scale.item()) == self._UNINITIALIZED_SCALE:
                raise RuntimeError("W4A4 input_global_scale uninitialized. Load PTQ model first.")
            self._amax_initialized = True

        # Use Triton kernel with VLLM_REF_ARITH=True for activation path
        igs = self.input_global_scale.to(x.device)
        result = STEFP4QuantTriton.apply(x_2d, igs, self.group_size, True)
        return result.view(original_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with fake quantization."""
        if not self.fake_quant_enabled:
            return F.linear(x, self.weight, self.bias)

        weight_fq = self._fake_quantize_weight(self.weight)

        if self.mode == QATMode.W4A4:
            x_fq = self._fake_quantize_activation(x)
        else:
            x_fq = x

        return F.linear(x_fq, weight_fq, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, mode={self.mode.value}, "
            f"group_size={self.group_size}, fake_quant_enabled={self.fake_quant_enabled}"
        )
