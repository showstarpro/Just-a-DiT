#!/bin/bash

# 单卡训练启动脚本
# 使用方法:
#   bash run_single_gpu.sh                                    # 使用默认参数
#   bash run_single_gpu.sh --num_register_tokens 4            # 设置 register tokens 数量
#   bash run_single_gpu.sh --num_register_tokens 8 --lr 2e-4  # 设置多个参数

# 设置要使用的 GPU ID
export CUDA_VISIBLE_DEVICES=6

# 使用 torchrun 启动单卡训练（也需要用 DDP，只是只有1个进程）
torchrun \
    --nproc_per_node=1 \
    --master_port=29500 \
    train.py "$@"

# 可用参数:
# --num_register_tokens: register tokens 数量 (默认: 0)
# --batch_size: 每个 GPU 的 batch size (默认: 128)
# --n_steps: 训练步数 (默认: 200000)
# --lr: 学习率 (默认: 1e-4)
