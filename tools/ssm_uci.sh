#!/usr/bin/env bash
# UniChessSSM Stage A UCI 引擎入口（arena / cutechess 调用它）。
#
# 环境变量：
#   UNICHESS_SSM_CKPT  权重（默认 runs/stage_a_20260915/best.pt）
#   UNICHESS_SSM_DEVICE  cuda / cpu（默认 cuda）
#   UNICHESS_MCTS      >0 启用 MCTS，值为每步模拟次数；默认 400（与旧引擎同口径）
#   UNICHESS_MCTS_BATCH 推理批量（默认 64；多进程 arena 共享 16GB GPU 时防 OOM）
#                       只影响 virtual loss 收集粒度，不影响固定 sims 口径
#   UNICHESS_ROOT      旧项目路径（只读引用其 search/mcts.py 与 core/，默认 /home/jeefy/UniChess）
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unichess

export PYTHONPATH="$PWD:${UNICHESS_ROOT:-/home/jeefy/UniChess}:${PYTHONPATH}"
# 多进程 arena 共享单卡时减少碎片浪费（实测 4 worker 下防 OOM）
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec python tools/ssm_uci.py \
  --ckpt "${UNICHESS_SSM_CKPT:-runs/stage_a_20260915/best.pt}" \
  --device "${UNICHESS_SSM_DEVICE:-cuda}" "$@"
