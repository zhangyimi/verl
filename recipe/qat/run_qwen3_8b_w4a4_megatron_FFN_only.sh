#!/usr/bin/env bash
# 8B Qwen3 Dense - Megatron W4A4 QAT, FFN-only
set -xeuo pipefail
current_dir="$(dirname "$(readlink -f "$0")")"

project_name='DAPO-NVFP4-QAT'
exp_name=${exp_name:-"h100_8B_w4a4_megatron_FFN_$(date +%m%d)"}

adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=${max_prompt_length:-$((1024))}
max_response_length=${max_response_length:-$((1024 * 20))}
enable_overlong_buffer=${enable_overlong_buffer:-True}
overlong_buffer_len=${overlong_buffer_len:-512}
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

train_prompt_bsz=${train_prompt_bsz:-32}
n_resp_per_prompt=${n_resp_per_prompt:-16}
train_prompt_mini_bsz=${train_prompt_mini_bsz:-${train_prompt_bsz}}
gen_prompt_bsz=${gen_prompt_bsz:-$((train_prompt_bsz * 2))}
total_training_steps=${total_training_steps:-}

WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}

RAY_DATA_HOME=${RAY_DATA_HOME:-"${WORKING_DIR}"}
MODEL_PATH=${MODEL_PATH:?"set MODEL_PATH to Qwen3-8B-Base"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}

temperature=1.0
top_p=1.0
top_k=-1
val_top_p=1.0
enable_filter_groups=${enable_filter_groups:-True}
filter_groups_metric=acc
max_num_gen_batches=${max_num_gen_batches:-10}

use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=True
gen_tp=1

rollout_is=token
rollout_is_threshold=2.0
rollout_rs=null

# QAT Configuration — W4A4 (FFN-only, attention in BF16)
qat_enable=True
qat_mode=w4a4
qat_config_path="${qat_config_path:-"${WORKING_DIR}/recipe/qat/config/nvfp4_w4a4_megatron.json"}"

export VERL_LOGGING_LEVEL=INFO
export VLLM_CONFIGURE_LOGGING=1
export VLLM_USE_V1=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

DATA=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.prompt_key=prompt
    data.truncation='left'
    data.return_raw_chat=True
    data.filter_overlong_prompts=True
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.gen_batch_size=${gen_prompt_bsz}
    data.train_batch_size=${train_prompt_bsz}
)

ALGORITHM=(
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=${use_kl_in_reward}
    algorithm.kl_ctrl.kl_coef=${kl_coef}
    algorithm.filter_groups.enable=${enable_filter_groups}
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches}
    algorithm.filter_groups.metric=${filter_groups_metric}
    algorithm.rollout_correction.rollout_is=${rollout_is}
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold}
    algorithm.rollout_correction.rollout_rs=${rollout_rs}
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
)

ACTOR=(
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=0
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
    actor_rollout_ref.actor.megatron.param_offload=${offload}
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload}
    actor_rollout_ref.actor.megatron.grad_offload=${offload}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1
    actor_rollout_ref.actor.megatron.context_parallel_size=1
    actor_rollout_ref.actor.megatron.sequence_parallel=False
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}
)

# FFN-only: ignore lm_head + router (no-op on dense 8B) + entire attention block.
# Megatron uses fnmatch globs (NOT regex). See verl/utils/modelopt/quantize.py mapping.
# rollout.qat MUST use identical ignore_patterns — otherwise vLLM quantizes layers that
# Megatron sends as BF16, and weight sync hits shape mismatch (BF16 [out,in] ≠ NVFP4 [out,in/2]).
QAT=(
    actor_rollout_ref.actor.megatron.qat.enable=${qat_enable}
    actor_rollout_ref.actor.megatron.qat.mode=${qat_mode}
    actor_rollout_ref.actor.megatron.qat.quantization_config_path="${qat_config_path}"
    'actor_rollout_ref.actor.megatron.qat.ignore_patterns=["lm_head","*mlp.gate","*self_attn*"]'
    # W4A4 input-activation calibration data (Ray workers don't inherit TRAIN_FILE env).
    ++actor_rollout_ref.actor.megatron.qat.calib_data_path="${TRAIN_FILE}"
    # Explicitly mirror onto rollout.qat — the oc.select inheritance in rollout.yaml
    # doesn't propagate Hydra command-line overrides at runtime.
    ++actor_rollout_ref.rollout.qat.enable=${qat_enable}
    ++actor_rollout_ref.rollout.qat.mode=${qat_mode}
    ++actor_rollout_ref.rollout.qat.quantization_config_path="${qat_config_path}"
    '++actor_rollout_ref.rollout.qat.ignore_patterns=["lm_head","*mlp.gate","*self_attn*"]'
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.enforce_eager=${enforce_eager:-True}
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.gpu_memory_utilization=0.90
    actor_rollout_ref.rollout.max_model_len=$(( max_prompt_length + max_response_length ))
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=$(( 1024 * 32 ))
    actor_rollout_ref.rollout.max_num_seqs=256
    actor_rollout_ref.rollout.temperature=${temperature}
    actor_rollout_ref.rollout.top_p=${top_p}
    actor_rollout_ref.rollout.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
)

PERF_OPT=(
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_arbitrary_attention_mask=False
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
)

REWARD=(
    reward.reward_manager.name=dapo
    reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer}
    reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len}
    reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor}
    reward.reward_kwargs.max_resp_len=${max_response_length}
)

TRAINER=(
    trainer.logger='["console","wandb"]'
    trainer.project_name="${project_name}"
    trainer.experiment_name="${exp_name}"
    trainer.n_gpus_per_node=${n_gpus_per_node:-8}
    trainer.nnodes="${NNODES}"
    trainer.val_before_train=${val_before_train:-True}
    trainer.test_freq=${test_freq:-10}
    trainer.save_freq=${save_freq:-5}
    trainer.total_epochs=1
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.resume_mode=auto
    trainer.max_actor_ckpt_to_keep=3
    trainer.use_legacy_worker_impl=disable
    ${total_training_steps:+trainer.total_training_steps=${total_training_steps}}
)

FORWARD_ONLY_SETS=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=1
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1
    actor_rollout_ref.ref.megatron.context_parallel_size=1
    actor_rollout_ref.ref.megatron.sequence_parallel=False
)

python3 -m recipe.dapo.main_dapo \
    --config-path "${WORKING_DIR}/recipe/qat/config" \
    --config-name dapo_qat_megatron_trainer \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${QAT[@]}" \
    "${ROLLOUT[@]}" \
    "${PERF_OPT[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${FORWARD_ONLY_SETS[@]}"
