#!/usr/bin/env bash
# run_arena_transformer.sh - Convenient runner for SSM vs Transformer / ResNet Arena evaluation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="/home/jeefy/miniconda3/envs/unichess/bin/python"
SSM_CKPT="${1:-runs/stage_b_training_fix500_cs01/best.pt}"
OPPONENT_TYPE="${2:-transformer}" # transformer or resnet
OPPONENT_CKPT="${3:-/home/jeefy/UniChess/Transformer/runs/stratified_middlegame_curriculum/best_model.pt}"
GAMES="${4:-64}"
WORKERS="${5:-2}"
OUT_DIR="${6:-runs/arena_ssm_vs_${OPPONENT_TYPE}}"

# Fallback checkpoint for resnet if opponent is resnet and default transformer path was kept
if [ "${OPPONENT_TYPE}" = "resnet" ] && [ "${OPPONENT_CKPT}" = "/home/jeefy/UniChess/Transformer/runs/stratified_middlegame_curriculum/best_model.pt" ]; then
    OPPONENT_CKPT="/home/jeefy/UniChess/ResNet/runs/autoloop/models/small-champion.pt"
fi

echo "============================================================"
echo "Starting SSM vs ${OPPONENT_TYPE^^} Arena Evaluation"
echo "  SSM Checkpoint:      ${SSM_CKPT}"
echo "  Opponent Checkpoint: ${OPPONENT_CKPT}"
echo "  Games:               ${GAMES}"
echo "  Workers:             ${WORKERS}"
echo "  Output Directory:    ${OUT_DIR}"
echo "============================================================"

${PYTHON} tools/ssm_eval_vs_transformer.py \
    --ssm-ckpt "${SSM_CKPT}" \
    --opponent-type "${OPPONENT_TYPE}" \
    --opponent-ckpt "${OPPONENT_CKPT}" \
    --games "${GAMES}" \
    --workers "${WORKERS}" \
    --n-sims 64 \
    --m0 16 \
    --c-scale 0.1 \
    --out "${OUT_DIR}"
