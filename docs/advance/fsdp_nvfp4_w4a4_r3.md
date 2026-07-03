# FSDP NVFP4 W4A4 研究分支说明

本分支整理自 30B MoE W4A4 的 FSDP 正式实验代码，基于
`zhangyimi/verl@233241ea`。它是可审阅、可复现实验配置的研究分支，未包含
checkpoint、日志、集群提交脚本或机器相关的 runtime 环境覆盖。

## 包含内容

- FSDP NVFP4 W4A4 QAT、量化 scale 同步与 vLLM 动态权重更新。
- R3 routing replay、路由一致性诊断及对应指标。
- rollout/training log-prob mismatch 与 TIS 统计。
- P70 等自适应长度参考、GRPO group recenter。
- segment gate、local segment align 和其他可选 mismatch guard。
- W&B resume identity 处理以及对应 CPU 测试。

所有新增训练策略默认关闭，只有显式配置后才生效。配套 DAPO trainer 和通用
启动脚本位于 `zhangyimi/verl-recipe` 的同名分支
`share/fsdp-w4a4-r3-advp70`；本仓库的 `recipe` submodule 指向该提交。

## R3 运行约束

当前验证过的 FSDP R3 组合要求：

- vLLM 包含 routed-expert replay 所需能力（实验环境使用 PR 33013 对应实现）；
- `enable_rollout_routing_replay=True`；
- `enable_prefix_caching=False`；
- FSDP `use_torch_compile=False`，避免 hook/recompute 语义被改变；
- rollout 侧 CUDA Graph 可以保持开启。

recipe runner 会在 R3 参数组合不一致时直接报错，避免静默退化为非 replay。

## 有意排除

- 实验目录中的 `runtime_env.yaml` 强制开关；
- `VERL_W4A4_GROUPED_MOE=1` 和 pre-routing activation fake-quant 原型；
- Slurm/chain 脚本、训练产物、缓存和本地路径；
- 仅用于定位性能问题的 `UW-TIMING` 打印。

其中 grouped-MoE 原型存在已知 backward 学习问题，因此不应作为默认实现传播。

## 验证

分享前执行了 Python 语法检查、`git diff --check` 和 recipe shell/JSON 静态检查。
完整 CPU pytest 需要 veRL 训练依赖；若环境具备依赖，可运行本分支新增和修改的
`tests/trainer/ppo`、`tests/utils/test_route_diagnostics.py`、
`tests/utils/test_tracking_wandb_identity.py` 与
`tests/workers/rollout/test_vllm_cli_args_on_cpu.py`。

Hydra generated config 未从实验快照直接复制；应以
`verl/trainer/config/actor/actor.yaml` 等源配置为准，并在安装 `hydra-core` 后运行
`scripts/generate_trainer_config.sh` 重新生成。
