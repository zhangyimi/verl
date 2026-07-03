# Megatron MoE NVFP4 v0.20 研究分支说明

本分支整理自 2026 年 6 月 30B MoE W4A4 正式实验使用的
`verl_megatron_qat_v020` 快照。为保留已验证行为，它基于
`volcengine/verl@bcb63864`，没有在本次分享中强行 rebase 到最新主线。

## 包含内容

- Megatron + ModelOpt 的 NVFP4 W4A4/W4A16 QAT。
- MoE expert 参数映射、量化 scale 导出与跨并行组同步。
- vLLM ModelOpt NVFP4 权重 reload、MoE kernel layout 和 CUDA Graph 稳定更新。
- activation calibration、动态 activation scale 与可选周期重校准。
- rollout/training mismatch 统计，以及可选的 gap、segment、align、advantage 策略。

主仓库变更只包含源码和配置，不包含 checkpoint、ray log、W&B 文件、缓存或
实验备份。配套 8B/30B recipe 位于 `zhangyimi/verl-recipe` 的同名分支
`share/megatron-moe-nvfp4-v020`；本仓库的 `recipe` submodule 指向该提交。

## 兼容性

这份快照使用 vLLM 0.20 系列 API；旧 recipe 文档中的 vLLM 0.15 说明不适用于
本分支。运行环境还需要与对应版本兼容的 Megatron-Core、Megatron-Bridge 和
NVIDIA ModelOpt。升级任一组件时，应先验证 MoE 参数名、W13 顺序、scale shape
以及二次 weight reload。

详细 QAT 诊断默认关闭；需要时设置 `VERL_QAT_DEBUG=1`。该开关只控制日志，
不改变量化、同步或 loss 语义。

## 验证边界

分享前执行了 Python 语法、shell、JSON 和 `git diff --check` 静态检查。该代码的
GPU 证据来自原 2026-06-13/14 正式实验；本次整理未重新运行多节点 GPU 训练，
因此不应把静态检查描述成新的端到端认证。

建议团队首次复现依次检查：模型构建、首轮 calibration、第一次 weight sync、
第二次 weight reload、CUDA Graph replay，最后再做至少一个 optimizer step。
