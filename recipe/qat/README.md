# NVFP4 QAT recipes (Megatron backend)

Quantization-Aware Training for dense Qwen3 models on the Megatron-LM backend with
vLLM 0.20.0 rollout. During training the actor uses fake-quantized NVFP4 linear layers;
on each weight sync the quantized weights (and, for W4A4, activation scales) are exported
to the vLLM engine so the rollout runs the deployment numerics.

## Recipes

| Script                                       | Mode  | Scope     |
|----------------------------------------------|-------|-----------|
| `run_qwen3_8b_bf16_megatron.sh`              | BF16  | reference |
| `run_qwen3_8b_w4a16_megatron_FFN_only.sh`    | W4A16 | FFN-only  |
| `run_qwen3_8b_w4a16_megatron_full.sh`        | W4A16 | full      |
| `run_qwen3_8b_w4a4_megatron_FFN_only.sh`     | W4A4  | FFN-only  |
| `run_qwen3_8b_w4a4_megatron_full.sh`         | W4A4  | full      |

- **FFN-only** adds `"*self_attn*"` to `ignore_patterns`, keeping attention in BF16.
- **full** quantizes all linear layers except `lm_head` and the MoE/router gate.

## Configuration keys

Set under `actor_rollout_ref.actor.megatron.qat` (and mirrored onto
`actor_rollout_ref.rollout.qat`):

| Key                        | Description                                              | Example |
|----------------------------|----------------------------------------------------------|---------|
| `enable`                   | Enable QAT.                                               | `True` |
| `mode`                     | Quantization mode.                                       | `w4a16` / `w4a4` |
| `quantization_config_path` | NVFP4 quantization config JSON (see `config/`).          | `config/nvfp4_w4a16_megatron.json` |
| `ignore_patterns`          | Layer-name globs to leave in BF16 (fnmatch globs).       | `["lm_head","*mlp.gate","*self_attn*"]` |
| `calib_data_path`          | W4A4 only: data file for input-activation calibration.   | `${TRAIN_FILE}` |

**`ignore_patterns` must be identical on `actor.megatron.qat` and `rollout.qat`.** The
Megatron side only quantizes layers passing this filter; vLLM decides per layer using the
mirrored list. If they disagree, the weight sync narrows incompatible shapes and fails
(BF16 `[out, in]` vs packed NVFP4 `[out, in/2]`).

## Quantization config files (`config/`)

| File                            | Mode  |
|---------------------------------|-------|
| `nvfp4_w4a16_megatron.json`     | W4A16 |
| `nvfp4_w4a4_megatron.json`      | W4A4  |

Each describes the NVFP4 scheme (group size, scale dtype, quantized/ignored modules) read
by both the Megatron quantizer and the vLLM modelopt loader.

## Notes

- W4A16 runs on the Marlin GEMM backend (`VLLM_NVFP4_GEMM_BACKEND=marlin`); W4A4 uses the
  flashinfer-cutlass path, which quantizes activations to FP4.
- The QAT linear layers live in `verl/utils/qat/`; the modelopt NVFP4 integration and the
  Megatron→vLLM weight exporter live in `verl/utils/modelopt/`.
