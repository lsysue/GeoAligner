#!/bin/bash
# train.sh

# 定义虚拟环境路径，方便维护
VENV_PATH="/home/normaluser/lsy/envs/im2geo"

# 1. 物理屏蔽
export CUDA_VISIBLE_DEVICES=1

# 2. 启动训练
echo "Starting training on GPU: $CUDA_VISIBLE_DEVICES..."

# 直接运行，不使用 nohup，不进行日志重定向
$VENV_PATH/bin/torchrun --nproc_per_node=1 --master_port=29505 train_ddp.py --config configs/config.yaml

# 脚本会在这里阻塞，直到训练结束
echo "Training finished!"