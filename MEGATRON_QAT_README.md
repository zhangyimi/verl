# NVFP4 QAT — Megatron backend (vLLM 0.20.0)

Quantization-Aware Training (QAT) for **dense Qwen3** models with the **Megatron-LM**
training backend and **vLLM 0.20.0** rollout. The actor trains with fake-quantized
NVFP4 weights/activations and synchronizes quantized weights to vLLM each step, so the
rollout engine runs the same NVFP4 numerics that will be used at deployment.

## Supported modes

- **W4A16** — NVFP4 4-bit weights, BF16 activations.
- **W4A4**  — NVFP4 4-bit weights and 4-bit activations (with input-activation calibration).
- **FFN-only** vs **full**: FFN-only quantizes the MLP blocks and leaves attention in
  BF16; full quantizes all linear layers except `lm_head` and the router gate.

## Recipes

| TASK            | recipe                                        | quantization              |
|-----------------|-----------------------------------------------|---------------------------|
| `8B_bf16`       | `recipe/qat/run_qwen3_8b_bf16_megatron.sh`            | none (BF16 reference)     |
| `8B_w4a16_FFN`  | `recipe/qat/run_qwen3_8b_w4a16_megatron_FFN_only.sh`  | W4A16, FFN-only           |
| `8B_w4a16_full` | `recipe/qat/run_qwen3_8b_w4a16_megatron_full.sh`      | W4A16, full               |
| `8B_w4a4_FFN`   | `recipe/qat/run_qwen3_8b_w4a4_megatron_FFN_only.sh`   | W4A4, FFN-only            |
| `8B_w4a4_full`  | `recipe/qat/run_qwen3_8b_w4a4_megatron_full.sh`       | W4A4, full                |

The training algorithm is standard DAPO/GRPO. See `recipe/qat/README.md` for the QAT
configuration keys and quantization config files.

## Running

`train_megatron.job` launches a multi-node run inside the training container and sets up
the Ray cluster. Set the environment paths at the top of the file (or via environment
variables) for your site:

- `MODEL_PATH`  — path to the base model (default: `Qwen3-8B-Base`).
- `TRAIN_FILE` / `TEST_FILE` — training / evaluation parquet files.
- container image and mount points (`CONTAINER`, `MOUNTS`).

```bash
# multi-node (default 8 nodes, 4 GPU/node):
sbatch train_megatron.job 8B_w4a16_FFN

# single-node smoke (cap the step count):
total_training_steps=3 sbatch --nodes=1 train_megatron.job 8B_bf16
```

W&B logging uses the `WANDB_API_KEY` environment variable or a mounted `~/.netrc`.

## Key configuration

QAT is configured under `actor_rollout_ref.actor.megatron.qat`:

| Key                          | Description                                   |
|------------------------------|-----------------------------------------------|
| `enable`                     | Enable QAT.                                    |
| `mode`                       | `w4a16` or `w4a4`.                             |
| `quantization_config_path`   | NVFP4 quantization config JSON.               |
| `ignore_patterns`            | Layer-name globs left unquantized (BF16).     |
| `calib_data_path`            | (W4A4) data file for input-activation calibration. |

`ignore_patterns` on `actor.megatron.qat` must match `rollout.qat.ignore_patterns`, so
that the layers Megatron quantizes are exactly the layers vLLM expects; otherwise the
Megatron→vLLM weight sync fails with a shape mismatch.
