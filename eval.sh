#!/bin/bash

# =========================================================
# GeoAligner Evaluation Pipeline (Hybrid: Tuning + Ablation)
# =========================================================

CONFIG=configs/config.yaml
RUN_DIR=mp16pro_20260416_233719
CKPT=checkpoint_best.pth

BASE_CMD="python eval.py --config ${CONFIG} --run_dir ${RUN_DIR} --checkpoint ${CKPT}"

echo "===== START EVAL PIPELINE ====="

# =========================================================
# Stage 0: Baseline（必须有）
# =========================================================
echo ">>> [Stage 0] Baseline (S-space only)"
$BASE_CMD \
  --set retrieval.use_rerank=False

# =========================================================
# Stage 1: Retrieval 调参（低成本 & 高收益）
# =========================================================
echo ">>> [Stage 1] Sweep Retrieval Params (top_k, nprobe)"

for TOPK in 50 100; do
for NPROBE in 16 32 64; do

echo ">>> top_k=$TOPK nprobe=$NPROBE"

$BASE_CMD \
  --set retrieval.use_rerank=False \
  --set retrieval.top_k=$TOPK \
  --set retrieval.nprobe=$NPROBE

done
done

# 👉 手动观察 CSV，选一个最优组合（记下来）
BEST_TOPK=100
BEST_NPROBE=64

# =========================================================
# Stage 2: Rerank 是否有效（关键 ablation）
# =========================================================
echo ">>> [Stage 2] Rerank On/Off"

for RERANK in False True; do

$BASE_CMD \
  --set retrieval.top_k=$BEST_TOPK \
  --set retrieval.nprobe=$BEST_NPROBE \
  --set retrieval.use_rerank=$RERANK

done

# =========================================================
# Stage 3: Fusion 权重（论文核心图：U-shape）
# =========================================================
echo ">>> [Stage 3] Fusion Weight Sweep"

for W in 1.0 0.8 0.6 0.5 0.4 0.2 0.0; do

$BASE_CMD \
  --set retrieval.use_rerank=True \
  --set retrieval.top_k=$BEST_TOPK \
  --set retrieval.nprobe=$BEST_NPROBE \
  --set retrieval.rerank_fusion_mode=weighted \
  --set retrieval.rerank_s_weight=$W

done

# =========================================================
# Stage 4: Dynamic Fusion（重点方法）
# =========================================================
echo ">>> [Stage 4] Dynamic Fusion Sweep (temperature + alpha)"

for TEMP in 0.05 0.07 0.1; do
for AMIN in 0.05 0.1; do
for AMAX in 0.9 0.95; do

echo ">>> temp=$TEMP alpha=[$AMIN,$AMAX]"

$BASE_CMD \
  --set retrieval.use_rerank=True \
  --set retrieval.top_k=$BEST_TOPK \
  --set retrieval.nprobe=$BEST_NPROBE \
  --set retrieval.rerank_fusion_mode=dynamic \
  --set retrieval.rerank_dynamic_temperature=$TEMP \
  --set retrieval.rerank_dynamic_alpha_min=$AMIN \
  --set retrieval.rerank_dynamic_alpha_max=$AMAX

done
done
done

# =========================================================
# Stage 5: OT 参数敏感性（论文分析）
# =========================================================
echo ">>> [Stage 5] OT Epsilon Sweep"

for EPS in 0.01 0.05 0.1; do

$BASE_CMD \
  --set retrieval.use_rerank=True \
  --set retrieval.top_k=$BEST_TOPK \
  --set retrieval.nprobe=$BEST_NPROBE \
  --set retrieval.ot_eps=$EPS

done

# =========================================================
# Stage 6: KDE 模块贡献（ablation）
# =========================================================
echo ">>> [Stage 6] KDE Ablation"

for KDE_W in 0.0 0.3 0.5; do

$BASE_CMD \
  --set retrieval.use_rerank=True \
  --set retrieval.top_k=$BEST_TOPK \
  --set retrieval.nprobe=$BEST_NPROBE \
  --set retrieval.kde_weight=$KDE_W \
  --set retrieval.kde_sigma_km=50

done

# =========================================================
# Stage 7: Rerank 深度（性能 vs 计算）
# =========================================================
echo ">>> [Stage 7] Rerank Depth Sweep"

for RTOPK in 10 50 100 200; do

$BASE_CMD \
  --set retrieval.use_rerank=True \
  --set retrieval.top_k=$BEST_TOPK \
  --set retrieval.nprobe=$BEST_NPROBE \
  --set retrieval.rerank_topk=$RTOPK

done

echo "===== ALL DONE ====="