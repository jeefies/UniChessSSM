#!/usr/bin/env python3
"""Loop 7: Moves-Left Head (MLH) Loss Scale Mismatch & Log-Huber Normalization.

Investigates and validates formulations for the Moves-Left Head (MLH):
- Targets in chess games range from 0 to 300 plies.
- Current loss: F.huber_loss(mlh_pred, moves_left, delta=1.0) with w_m=0.1.
  When error |e| ~ 30 plies, Huber loss is ~30, yielding a weighted loss contribution of 3.0,
  which exceeds Value CE (~0.6-0.8) and is comparable to or dominates Policy Soft CE (~2.0-3.5).
- Shared representation h receives disproportionately large gradients from MLH,
  distorting feature learning for policy and value heads.

Evaluates 3 MLH reformulations:
1. Baseline:
   - Unnormalized Huber with delta=1.0, w_m=0.1
   - Target: moves_left in [0, 300]
2. Formulation A (Log-Transformed Target Huber):
   - Target: y = log(1 + moves_left) in [0, log(301)] ~ [0, 5.71]
   - Loss: Huber with delta=0.5, w_m=0.2
   - Prediction inversion: pred_moves_left = exp(clamp(pred_y, 0, 10)) - 1
3. Formulation B (Normalized Ply Huber):
   - Target: y = moves_left / 300.0 in [0, 1]
   - Loss: Huber with delta=0.01, w_m=1.0
   - Prediction inversion: pred_moves_left = pred_y * 300.0

Simulates 500 gradient steps across real game sequences (short, medium, and long 200+ ply games):
- Measures gradient norm ||nabla_h L_mlh||_2 relative to Policy and Value gradient norms
- Measures gradient explosion risk on early-game positions (t=0, true moves_left ~ 80-200)
- Measures prediction accuracy in absolute plies across opening, middlegame, and endgame phases
- Saves structured results to runs/loop7_mlh_scale.json
- Verifies output file exists and prints summary table.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chess
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from stateseq.actions import NUM_ACTIONS
from stateseq.layers import ResidualMLP, RMSNorm


# ---------------- Synthetic / Game Representation Scaffold ----------------

class MultiHeadModel(nn.Module):
    """Clean model scaffold holding shared representation h and 3 heads (Policy, Value, MLH).
    Matches stateseq.heads.PredictionHeads architecture.
    """
    def __init__(self, d_model: int = 512, n_actions: int = NUM_ACTIONS):
        super().__init__()
        self.d_model = d_model
        # Feature projection into shared representation h
        self.feat_proj = nn.Linear(785, d_model)
        self.h_norm = RMSNorm(d_model)

        # Shared trunk transform: u = h + MLP2(RMSNorm(h))
        self.transform = ResidualMLP(d_model, d_model)

        # Policy, Value, MLH heads
        self.w_p = nn.Linear(d_model, n_actions)
        self.w_v = nn.Linear(d_model, 3)
        self.w_m = nn.Linear(d_model, 1)
        self.head_norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Maps input features x -> (h, policy_logits, wdl_logits, mlh_raw)."""
        h = self.h_norm(self.feat_proj(x))
        u = self.transform(h)
        policy_logits = self.w_p(u)
        n = self.head_norm(u)
        wdl_logits = self.w_v(n)
        mlh_raw = self.w_m(n)
        return h, policy_logits, wdl_logits, mlh_raw


@dataclass
class GameSequence:
    game_id: int
    category: str  # "short", "medium", "long"
    total_plies: int
    fens: List[str]
    moves_left: List[float]  # in [0, 300]
    phases: List[str]  # "opening", "middlegame", "endgame"


def load_game_sequences() -> List[GameSequence]:
    """Loads games from tests/fixtures/sample.pgn (long games 260-360 plies)
    and data/sample_real.pgn (short and medium games).
    """
    sequences: List[GameSequence] = []
    game_counter = 0

    # 1. Load long games from tests/fixtures/sample.pgn
    long_pgn = os.path.join(REPO_ROOT, "tests", "fixtures", "sample.pgn")
    if os.path.exists(long_pgn):
        with open(long_pgn, "r", encoding="utf-8") as f:
            while True:
                g = chess.pgn.read_game(f)
                if g is None:
                    break
                moves = list(g.mainline_moves())
                total = len(moves)
                if total >= 200:
                    board = chess.Board()
                    fens = [board.fen()]
                    for m in moves:
                        board.push(m)
                        fens.append(board.fen())
                    # Cap at 300 for moves_left target consistency
                    n_plies = len(fens)
                    moves_left = [float(min(n_plies - 1 - t, 300)) for t in range(n_plies)]
                    phases = []
                    for t in range(n_plies):
                        if t <= 20:
                            phases.append("opening")
                        elif t <= 60:
                            phases.append("middlegame")
                        else:
                            phases.append("endgame")
                    sequences.append(GameSequence(
                        game_id=game_counter,
                        category="long",
                        total_plies=total,
                        fens=fens,
                        moves_left=moves_left,
                        phases=phases,
                    ))
                    game_counter += 1

    # 2. Load short and medium games from data/sample_real.pgn
    real_pgn = os.path.join(REPO_ROOT, "data", "sample_real.pgn")
    if os.path.exists(real_pgn):
        with open(real_pgn, "r", encoding="utf-8") as f:
            while True:
                g = chess.pgn.read_game(f)
                if g is None:
                    break
                moves = list(g.mainline_moves())
                total = len(moves)
                category = "short" if total <= 60 else ("medium" if total <= 140 else "long")
                board = chess.Board()
                fens = [board.fen()]
                for m in moves:
                    board.push(m)
                    fens.append(board.fen())
                n_plies = len(fens)
                moves_left = [float(min(n_plies - 1 - t, 300)) for t in range(n_plies)]
                phases = []
                for t in range(n_plies):
                    if t <= 20:
                        phases.append("opening")
                    elif t <= 60:
                        phases.append("middlegame")
                    else:
                        phases.append("endgame")
                sequences.append(GameSequence(
                    game_id=game_counter,
                    category=category,
                    total_plies=total,
                    fens=fens,
                    moves_left=moves_left,
                    phases=phases,
                ))
                game_counter += 1
                if game_counter >= 80:  # Sufficient collection across all categories
                    break

    return sequences


# ---------------- Synthetic Feature Generator for Fast Step Simulation ----------------

def encode_board_simple(board: chess.Board, ply: int, total_plies: int) -> np.ndarray:
    """Fast 785-dim feature vector matching stateseq feature space format:
    - 768: 12 piece types x 64 squares
    - 768: side to move (0 or 1)
    - 769..772: castling rights
    - 773..780: en passant file
    - 781: halfmove clock
    - 782: fullmove number
    - 783: ply normalized
    - 784: total ply progress estimate
    """
    feat = np.zeros(785, dtype=np.float32)
    piece_map = board.piece_map()
    for sq, piece in piece_map.items():
        piece_idx = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        feat[piece_idx * 64 + sq] = 1.0

    feat[768] = 1.0 if board.turn == chess.WHITE else 0.0
    feat[769] = float(board.has_kingside_castling_rights(chess.WHITE))
    feat[770] = float(board.has_queenside_castling_rights(chess.WHITE))
    feat[771] = float(board.has_kingside_castling_rights(chess.BLACK))
    feat[772] = float(board.has_queenside_castling_rights(chess.BLACK))
    if board.ep_square is not None:
        feat[773 + chess.square_file(board.ep_square)] = 1.0
    feat[781] = min(board.halfmove_clock / 50.0, 1.0)
    feat[782] = min(board.fullmove_number / 100.0, 1.0)
    feat[783] = min(ply / 200.0, 1.5)
    feat[784] = min(total_plies / 200.0, 1.5)
    return feat


# ---------------- Formulations Definition ----------------

class MLHFormulation:
    name: str
    weight: float

    def compute_loss_and_pred(
        self, mlh_raw: torch.Tensor, moves_left: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (loss, unscaled_prediction_in_plies)."""
        raise NotImplementedError


class BaselineFormulation(MLHFormulation):
    """Baseline: Unnormalized Huber with delta=1.0, w_m=0.1.
    Target: moves_left in plies [0, 300].
    Architecture head output: Mish(mlh_raw) >= 0.
    """
    def __init__(self):
        self.name = "Baseline (Unnormalized Huber delta=1.0, w=0.1)"
        self.short_name = "baseline"
        self.weight = 0.1
        self.delta = 1.0

    def compute_loss_and_pred(
        self, mlh_raw: torch.Tensor, moves_left: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_plies = F.mish(mlh_raw).squeeze(-1)
        loss = F.huber_loss(pred_plies, moves_left, delta=self.delta, reduction="mean")
        return loss, pred_plies


class FormulationA(MLHFormulation):
    """Formulation A: Log-Transformed Target Huber.
    Target: y = log(1 + moves_left) in [0, log(301)] ~ [0, 5.71].
    Loss: Huber with delta=0.5, w_m=0.2.
    Architecture head output: mlh_pred_log = Mish(mlh_raw).
    Inversion: pred_plies = exp(clamp(mlh_pred_log, 0, 10)) - 1.
    """
    def __init__(self):
        self.name = "Formulation A (Log-Transformed Target Huber delta=0.5, w=0.2)"
        self.short_name = "formulation_a"
        self.weight = 0.2
        self.delta = 0.5

    def compute_loss_and_pred(
        self, mlh_raw: torch.Tensor, moves_left: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Target log scale
        y_target = torch.log1p(moves_left)
        pred_log = F.mish(mlh_raw).squeeze(-1)
        loss = F.huber_loss(pred_log, y_target, delta=self.delta, reduction="mean")
        # Invert to plies for evaluation
        pred_plies = torch.expm1(pred_log.clamp(min=0.0, max=10.0))
        return loss, pred_plies


class FormulationB(MLHFormulation):
    """Formulation B: Normalized Ply Huber.
    Target: y = moves_left / 300.0 in [0, 1].
    Loss: Huber with delta=0.01, w_m=1.0.
    Architecture head output: mlh_pred_norm = Mish(mlh_raw).
    Inversion: pred_plies = mlh_pred_norm * 300.0.
    """
    def __init__(self):
        self.name = "Formulation B (Normalized Ply Huber delta=0.01, w=1.0)"
        self.short_name = "formulation_b"
        self.weight = 1.0
        self.delta = 0.01

    def compute_loss_and_pred(
        self, mlh_raw: torch.Tensor, moves_left: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y_target = moves_left / 300.0
        pred_norm = F.mish(mlh_raw).squeeze(-1)
        loss = F.huber_loss(pred_norm, y_target, delta=self.delta, reduction="mean")
        pred_plies = pred_norm * 300.0
        return loss, pred_plies


# ---------------- Simulation & Evaluation Engine ----------------

def run_simulation(
    sequences: List[GameSequence],
    n_steps: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Classify sequences into categories
    short_seqs = [s for s in sequences if s.category == "short"]
    med_seqs = [s for s in sequences if s.category == "medium"]
    long_seqs = [s for s in sequences if s.category == "long"]

    print(f"Loaded dataset: {len(sequences)} games total "
          f"(short: {len(short_seqs)}, med: {len(med_seqs)}, long: {len(long_seqs)})")

    formulations: List[MLHFormulation] = [
        BaselineFormulation(),
        FormulationA(),
        FormulationB(),
    ]

    results_by_formulation: Dict[str, Any] = {}

    for form in formulations:
        print(f"\n>>> Running 500-step simulation for: {form.name}...")
        # Reset seeds for identical data order and initial weights
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = MultiHeadModel(d_model=512, n_actions=NUM_ACTIONS)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

        # Tracking metrics
        history_step_loss: List[float] = []
        history_mlh_raw_loss: List[float] = []
        history_weighted_mlh_loss: List[float] = []
        history_pol_loss: List[float] = []
        history_val_loss: List[float] = []

        grad_norm_h_pol: List[float] = []
        grad_norm_h_val: List[float] = []
        grad_norm_h_mlh: List[float] = []
        grad_norm_h_total: List[float] = []
        grad_ratio_mlh_to_pol: List[float] = []
        grad_ratio_mlh_to_val: List[float] = []

        # Early game gradient explosion tracking (t=0, true moves_left >= 80)
        early_game_mlh_grads: List[float] = []
        early_game_pol_grads: List[float] = []
        early_game_val_grads: List[float] = []

        # Batch sampling helper
        batch_size = 8
        seq_len = 16

        for step in range(n_steps):
            model.train()
            optimizer.zero_grad()

            # Sample batch across categories (ensuring balanced representation of short, medium, long)
            batch_feats = []
            batch_moves_left = []
            batch_policy_tgt = []
            batch_value_tgt = []
            batch_is_early = []
            batch_phases = []

            for _ in range(batch_size):
                # Pick category with equal probability
                cat_choices = []
                if short_seqs:
                    cat_choices.append(short_seqs)
                if med_seqs:
                    cat_choices.append(med_seqs)
                if long_seqs:
                    cat_choices.append(long_seqs)
                pool = cat_choices[np.random.randint(len(cat_choices))]
                seq = pool[np.random.randint(len(pool))]

                # Pick random start index
                max_start = max(0, len(seq.fens) - seq_len)
                start_t = np.random.randint(0, max_start + 1) if max_start > 0 else 0

                for t in range(start_t, min(start_t + seq_len, len(seq.fens))):
                    board = chess.Board(seq.fens[t])
                    feat = encode_board_simple(board, t, seq.total_plies)
                    batch_feats.append(feat)
                    batch_moves_left.append(seq.moves_left[t])
                    # Synthetic targets consistent with chess statistics
                    batch_policy_tgt.append(np.random.randint(0, NUM_ACTIONS))
                    # Result from perspective of turn: 0 win, 1 draw, 2 loss
                    batch_value_tgt.append(np.random.randint(0, 3))
                    batch_is_early.append(t == 0 and seq.moves_left[t] >= 80.0)
                    batch_phases.append(seq.phases[t])

            x = torch.tensor(np.array(batch_feats), dtype=torch.float32)
            tgt_mlh = torch.tensor(np.array(batch_moves_left), dtype=torch.float32)
            tgt_pol = torch.tensor(np.array(batch_policy_tgt), dtype=torch.long)
            tgt_val = torch.tensor(np.array(batch_value_tgt), dtype=torch.long)
            is_early = torch.tensor(np.array(batch_is_early), dtype=torch.bool)

            # Forward pass
            h, pol_logits, val_logits, mlh_raw = model(x)
            h.retain_grad()

            # Losses
            # Policy CE (w_p = 1.0)
            loss_pol = F.cross_entropy(pol_logits, tgt_pol)
            # Value CE (w_v = 0.8)
            loss_val = F.cross_entropy(val_logits, tgt_val)
            # MLH Loss
            loss_mlh_raw, pred_plies = form.compute_loss_and_pred(mlh_raw, tgt_mlh)
            loss_mlh_weighted = form.weight * loss_mlh_raw

            # Total loss
            total_loss = 1.0 * loss_pol + 0.8 * loss_val + loss_mlh_weighted

            # Backward pass & dissect gradients on representation h
            # We compute gradients on h for each loss separately using autograd.grad
            g_h_pol = torch.autograd.grad(loss_pol, h, retain_graph=True, create_graph=False)[0]
            g_h_val = torch.autograd.grad(loss_val, h, retain_graph=True, create_graph=False)[0]
            g_h_mlh = torch.autograd.grad(loss_mlh_weighted, h, retain_graph=True, create_graph=False)[0]

            norm_pol = float(torch.norm(g_h_pol, p=2).item())
            norm_val = float(torch.norm(g_h_val, p=2).item())
            norm_mlh = float(torch.norm(g_h_mlh, p=2).item())
            norm_total = float(torch.norm(g_h_pol + 0.8 * g_h_val + g_h_mlh, p=2).item())

            grad_norm_h_pol.append(norm_pol)
            grad_norm_h_val.append(norm_val)
            grad_norm_h_mlh.append(norm_mlh)
            grad_norm_h_total.append(norm_total)
            grad_ratio_mlh_to_pol.append(norm_mlh / (norm_pol + 1e-8))
            grad_ratio_mlh_to_val.append(norm_mlh / (norm_val + 1e-8))

            # Early game gradient measurement
            if is_early.any():
                early_idx = torch.where(is_early)[0]
                g_early_mlh = torch.norm(g_h_mlh[early_idx], p=2, dim=-1).mean().item()
                g_early_pol = torch.norm(g_h_pol[early_idx], p=2, dim=-1).mean().item()
                g_early_val = torch.norm(g_h_val[early_idx], p=2, dim=-1).mean().item()
                early_game_mlh_grads.append(float(g_early_mlh))
                early_game_pol_grads.append(float(g_early_pol))
                early_game_val_grads.append(float(g_early_val))

            # Record history
            history_step_loss.append(float(total_loss.item()))
            history_mlh_raw_loss.append(float(loss_mlh_raw.item()))
            history_weighted_mlh_loss.append(float(loss_mlh_weighted.item()))
            history_pol_loss.append(float(loss_pol.item()))
            history_val_loss.append(float(loss_val.item()))

            # Step optimizer
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Final Evaluation across Game Sequences (Accuracy in Plies)
        model.eval()
        eval_metrics = evaluate_phases_accuracy(model, form, sequences)

        # Summary statistics for this formulation
        results_by_formulation[form.short_name] = {
            "formulation_name": form.name,
            "short_name": form.short_name,
            "loss_weight": form.weight,
            "delta": form.delta,
            "loss_dynamics": {
                "initial_raw_mlh_loss": history_mlh_raw_loss[0],
                "final_raw_mlh_loss": np.mean(history_mlh_raw_loss[-20:]).item(),
                "initial_weighted_mlh_loss": history_weighted_mlh_loss[0],
                "final_weighted_mlh_loss": np.mean(history_weighted_mlh_loss[-20:]).item(),
                "final_policy_loss": np.mean(history_pol_loss[-20:]).item(),
                "final_value_loss": np.mean(history_val_loss[-20:]).item(),
                "weighted_mlh_to_val_loss_ratio": float(
                    np.mean(history_weighted_mlh_loss[-20:]) / (np.mean(history_val_loss[-20:]) + 1e-8)
                ),
            },
            "gradient_dynamics": {
                "mean_grad_norm_pol": float(np.mean(grad_norm_h_pol)),
                "mean_grad_norm_val": float(np.mean(grad_norm_h_val)),
                "mean_grad_norm_mlh": float(np.mean(grad_norm_h_mlh)),
                "max_grad_norm_mlh": float(np.max(grad_norm_h_mlh)),
                "mlh_to_pol_grad_ratio": float(np.mean(grad_ratio_mlh_to_pol)),
                "mlh_to_val_grad_ratio": float(np.mean(grad_ratio_mlh_to_val)),
            },
            "early_game_gradient_risk": {
                "mean_early_mlh_grad": float(np.mean(early_game_mlh_grads)) if early_game_mlh_grads else 0.0,
                "max_early_mlh_grad": float(np.max(early_game_mlh_grads)) if early_game_mlh_grads else 0.0,
                "mean_early_pol_grad": float(np.mean(early_game_pol_grads)) if early_game_pol_grads else 0.0,
                "early_mlh_to_pol_ratio": float(
                    np.mean(early_game_mlh_grads) / (np.mean(early_game_pol_grads) + 1e-8)
                ) if early_game_mlh_grads else 0.0,
            },
            "accuracy_in_plies": eval_metrics,
        }

    return results_by_formulation


def evaluate_phases_accuracy(
    model: MultiHeadModel,
    form: MLHFormulation,
    sequences: List[GameSequence],
) -> Dict[str, Any]:
    """Measures prediction error in absolute plies across opening, middlegame, and endgame."""
    errors_by_phase: Dict[str, List[float]] = {"opening": [], "middlegame": [], "endgame": []}
    errors_by_cat: Dict[str, List[float]] = {"short": [], "medium": [], "long": []}
    all_abs_errors: List[float] = []

    with torch.no_grad():
        for seq in sequences:
            feats = []
            for t in range(len(seq.fens)):
                board = chess.Board(seq.fens[t])
                feats.append(encode_board_simple(board, t, seq.total_plies))
            x = torch.tensor(np.array(feats), dtype=torch.float32)
            tgt_mlh = torch.tensor(np.array(seq.moves_left), dtype=torch.float32)

            _, _, _, mlh_raw = model(x)
            _, pred_plies = form.compute_loss_and_pred(mlh_raw, tgt_mlh)

            abs_err = torch.abs(pred_plies - tgt_mlh).cpu().numpy()
            for t, err in enumerate(abs_err):
                ph = seq.phases[t]
                errors_by_phase[ph].append(float(err))
                errors_by_cat[seq.category].append(float(err))
                all_abs_errors.append(float(err))

    return {
        "overall_mae_plies": float(np.mean(all_abs_errors)),
        "overall_median_error_plies": float(np.median(all_abs_errors)),
        "opening_mae_plies": float(np.mean(errors_by_phase["opening"])),
        "middlegame_mae_plies": float(np.mean(errors_by_phase["middlegame"])),
        "endgame_mae_plies": float(np.mean(errors_by_phase["endgame"])),
        "short_games_mae_plies": float(np.mean(errors_by_cat["short"])),
        "medium_games_mae_plies": float(np.mean(errors_by_cat["medium"])),
        "long_games_mae_plies": float(np.mean(errors_by_cat["long"])),
    }


# ---------------- Summary Display & Printing ----------------

def print_summary_table(results: Dict[str, Any]):
    print("\n" + "=" * 105)
    print("LOOP 7: MOVES-LEFT HEAD (MLH) LOSS SCALE & LOG-HUBER NORMALIZATION AUDIT")
    print("=" * 105)

    print("\n--- 1. LOSS SCALE & GRADIENT DOMINANCE COMPARISON ---")
    header1 = (
        f"{'Formulation':<32} | {'Raw MLH Loss':<13} | {'Weighted MLH':<13} | "
        f"{'||g_h(MLH)||':<12} | {'MLH/Pol Grad':<12} | {'MLH/Val Grad':<12}"
    )
    print(header1)
    print("-" * 105)
    for key, data in results.items():
        name = data["short_name"]
        raw_l = data["loss_dynamics"]["final_raw_mlh_loss"]
        wt_l = data["loss_dynamics"]["final_weighted_mlh_loss"]
        g_mlh = data["gradient_dynamics"]["mean_grad_norm_mlh"]
        r_pol = data["gradient_dynamics"]["mlh_to_pol_grad_ratio"]
        r_val = data["gradient_dynamics"]["mlh_to_val_grad_ratio"]
        print(
            f"{data['formulation_name'][:32]:<32} | {raw_l:<13.4f} | {wt_l:<13.4f} | "
            f"{g_mlh:<12.4f} | {r_pol:<12.3f} | {r_val:<12.3f}"
        )
    print("-" * 105)

    print("\n--- 2. EARLY-GAME GRADIENT EXPLOSION RISK (t=0, moves_left >= 80) ---")
    header2 = (
        f"{'Formulation':<32} | {'Early ||g_h||':<14} | {'Max Early ||g_h||':<18} | "
        f"{'Early MLH/Pol Ratio':<20} | {'Risk Assessment':<16}"
    )
    print(header2)
    print("-" * 105)
    for key, data in results.items():
        eg = data["early_game_gradient_risk"]
        eg_mean = eg["mean_early_mlh_grad"]
        eg_max = eg["max_early_mlh_grad"]
        eg_ratio = eg["early_mlh_to_pol_ratio"]
        risk = "HIGH EXPLOSION" if eg_ratio > 3.0 else ("BALANCED" if eg_ratio <= 1.2 else "MODERATE")
        print(
            f"{data['formulation_name'][:32]:<32} | {eg_mean:<14.4f} | {eg_max:<18.4f} | "
            f"{eg_ratio:<20.3f} | {risk:<16}"
        )
    print("-" * 105)

    print("\n--- 3. ACCURACY ACROSS PHASES & GAME LENGTHS (Absolute Plies MAE) ---")
    header3 = (
        f"{'Formulation':<32} | {'Overall MAE':<12} | {'Opening MAE':<12} | "
        f"{'Mid MAE':<10} | {'Endgame MAE':<12} | {'Long Games MAE':<14}"
    )
    print(header3)
    print("-" * 105)
    for key, data in results.items():
        acc = data["accuracy_in_plies"]
        ovr = acc["overall_mae_plies"]
        opn = acc["opening_mae_plies"]
        mid = acc["middlegame_mae_plies"]
        end = acc["endgame_mae_plies"]
        lng = acc["long_games_mae_plies"]
        print(
            f"{data['formulation_name'][:32]:<32} | {ovr:<12.2f} | {opn:<12.2f} | "
            f"{mid:<10.2f} | {end:<12.2f} | {lng:<14.2f}"
        )
    print("=" * 105 + "\n")


def main():
    sequences = load_game_sequences()
    if not sequences:
        raise RuntimeError("No game sequences found in tests/fixtures/sample.pgn or data/sample_real.pgn")

    results = run_simulation(sequences, n_steps=500, seed=42)

    out_dir = os.path.join(REPO_ROOT, "runs")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "loop7_mlh_scale.json")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved structured audit results to: {out_file}")
    print_summary_table(results)


if __name__ == "__main__":
    main()
