# TRT-LLM companion patch for W4A8 NVFP4 rollout

This PR is verl-only. The rollout side of this fix lives in a TensorRT-LLM
branch that is ~7 months behind `NVIDIA/TensorRT-LLM:main`, so rebasing it
for a standalone PR (especially `linear.py`, which has changed heavily
upstream) is out of scope here. Instead, the two changed files are shipped
as a patch.

## Base

`w4a8-nvfp4-rollout.patch` is generated against commit `4517988cb` on
`NVIDIA/TensorRT-LLM:main`:

```
git diff 4517988cb39689f5462000f9251c5255a0492825..HEAD -- \
    tensorrt_llm/_torch/modules/linear.py \
    tensorrt_llm/llmapi/rlhf_utils.py
```

Note: `6f4ae8a` (the SHA recorded in this repo's `BASELINE-COMMITS.txt`) is
**not** a valid base for this patch. It is an abandoned sibling line, not an
ancestor of the branch this patch was cut from — the branch was rebuilt from
`4517988cb` with re-committed equivalents. Diffing against `6f4ae8a` produces
a misleading cross-branch diff.

## What the two files fix

- `tensorrt_llm/_torch/modules/linear.py` — `W4A8NVFP4FP8LinearMethod`:
  - Proper partial-loading scale path for W4A8 NVFP4 weights (per-tensor
    scales are accumulated across partial-load buckets and finalized once
    all shards have arrived, matching the pattern already used for
    `NVFP4LinearMethod`).
  - Accepts a `weight_packed` key in the vanilla/fused-gate-up weight
    loading helpers (verl's QAT export uses `output_format=trtllm`, which
    names the packed weight tensor `weight_packed`).
  - Falls back to dynamic activation quantization when the checkpoint omits
    a static `input_scale` — verl's QAT recipe quantizes activations
    online per-token rather than exporting a static scale, so
    `force_dynamic_quantization` is flipped on when no `input_scale` is
    present.
- `tensorrt_llm/llmapi/rlhf_utils.py` — `WorkerExtension` IPC weight sync:
  - Accepts 3-tuple IPC handle entries `(param_name, tensor_handle,
    dtype_tag)` in addition to the legacy 2-tuple form. verl bitcasts
    `float8_e4m3fn` weight/scale tensors to `uint8` around the CUDA IPC
    pickle (the legacy storage pickler can't handle `float8_e4m3fn`
    directly) and tags the dtype so the receiver can view it back.
  - Re-allows `torch.storage._load_from_bytes` in the restricted unpickler
    used for IPC handle deserialization — needed to reconstruct CUDA IPC
    tensors with non-standard storage (e.g. the uint8-viewed fp8
    `weight_scale` above). `torch.hub._load_local` and `torch.save` stay
    blocked.

## How this is applied today

The launcher applies these by copying the whole files over the installed
`tensorrt_llm` package at job submission time (see
`sbatch-config/config.yaml:68-75` in the `stable-pass-1.fix-precision`
snapshot):

```bash
cp "${TRTLLM_SRC}/llmapi/rlhf_utils.py" "${TRTLLM_PKG}/llmapi/rlhf_utils.py"
cp "${TRTLLM_SRC}/_torch/modules/linear.py" "${TRTLLM_PKG}/_torch/modules/linear.py"
```

This patch is the portable, reviewable form of that same change, for anyone
who wants to apply it against a clean TRT-LLM checkout instead of copying
whole files.

## Applying

```
git checkout 4517988cb39689f5462000f9251c5255a0492825
git apply /path/to/w4a8-nvfp4-rollout.patch
```
