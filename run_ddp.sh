#!/bin/bash

# DDP 多卡训练启动脚本
# 使用方法:
#   bash run_ddp.sh                                    # 使用默认参数
#   bash run_ddp.sh --num_register_tokens 4            # 设置 register tokens 数量
#   bash run_ddp.sh --num_register_tokens 8 --lr 2e-4  # 设置多个参数

# 设置要使用的 GPU 数量
NUM_GPUS=4

# 设置要使用的 GPU ID（可选，如果不设置则使用前 NUM_GPUS 个 GPU）
# 例如: export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 启动 DDP 训练，传递所有命令行参数
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train.py "$@"

# 说明:
# --nproc_per_node: 每个节点使用的 GPU 数量
# --master_port: 主进程通信端口（如果默认端口被占用可以修改）
# "$@": 传递所有命令行参数到 train.py
#
# 可用参数:
# --num_register_tokens: register tokens 数量 (默认: 0)
# --batch_size: 每个 GPU 的 batch size (默认: 128)
# --n_steps: 训练步数 (默认: 200000)
# --lr: 学习率 (默认: 1e-4)
