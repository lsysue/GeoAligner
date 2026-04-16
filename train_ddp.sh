#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash train_ddp.sh [config_path] [extra train_ddp.py args...]
# Examples:
#   bash train_ddp.sh
#   bash train_ddp.sh configs/config_smallset_im2gps3k.yaml
#   CUDA_VISIBLE_DEVICES=0 bash train_ddp.sh configs/config_debug.yaml

VENV_PATH="${VENV_PATH:-/home/normaluser/lsy/envs/im2geo}"
CONFIG_PATH="${1:-configs/config.yaml}"
MASTER_PORT="${MASTER_PORT:-29505}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
EXTRA_ARGS=("${@:2}")

export CUDA_VISIBLE_DEVICES

echo "Starting training on GPU(s): ${CUDA_VISIBLE_DEVICES}"
echo "Config: ${CONFIG_PATH}"
echo "Master port: ${MASTER_PORT}"

"${VENV_PATH}/bin/torchrun" \
	--nproc_per_node=1 \
	--master_port="${MASTER_PORT}" \
	train_ddp.py \
	--config "${CONFIG_PATH}" \
	"${EXTRA_ARGS[@]}"

echo "Training finished"