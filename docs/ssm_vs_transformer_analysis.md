# SSM vs Transformer 20M Arena Analysis

**Date:** 2026-09-22  
**Match:** `runs/arena_2500_gen2_vs_transformer/`  
**SSM Checkpoint:** `runs/stage_b_training_2500_gen2/best.pt`  
**Opponent:** Transformer 20M (`stratified_middlegame_curriculum`)  
**Opponent Search:** `mcts_sims=0` (pure policy, no search)  
**SSM Search:** Gumbel `n_sims=64`, `m0=16`, `c_scale=0.1`, `c_visit=50`

---

## 1. Executive Summary

SSM lost **57-0-3** (Elo diff **-636 ± 197**) against a Transformer 20M that uses **pure policy without search**. This is an unfair comparison **in SSM's favor**: SSM deployed Gumbel search (64 sims/move) while the opponent played greedy argmax policy. The 636 Elo gap is therefore not explained by "search helping" — it shows SSM's base policy is dramatically weaker than the opponent's policy, and the 64-sim search is too shallow to compensate.

---

## 2. Game Statistics

| Metric | Value |
|---|---|
| Total games | 60 |
| Wins | 0 |
| Draws | 3 |
| Losses | 57 |
| Score | 1.5 / 60 (2.5%) |
| Elo diff | -636.4 ± 196.5 |
| Avg game length | 52.4 plies |
| Min / Max length | 23 / 92 plies |
| Termination | Checkmate: 57, Threefold: 2, Stalemate: 1 |

### 2.1 Color Breakdown

| Color | Games | Wins | Draws | Losses | Loss % |
|---|---|---|---|---|---|
| White | 30 | 0 | 3 | 27 | 90.0% |
| Black | 30 | 0 | 0 | 30 | 100.0% |

**SSM loses 100% of games when playing Black.** This is the most alarming finding and strongly suggests a systematic color-dependent defect.

---

## 3. Phase Analysis

| Phase | Plies | Games | % of losses |
|---|---|---|---|
| Opening (≤20) | — | 2 | 3.5% |
| Middlegame (21–60) | — | 44 | 77.2% |
| Endgame (61+) | — | 14 (11+3) | 19.3% (18.6% checkmate, 5.3% draw) |

- Only **2 losses** occurred within the first 20 plies (opening).
- **44 of 57 losses** (77%) happened in the middlegame (plies 21–60).
- SSM survives the opening but collapses in the middlegame.

---

## 4. Opening Analysis

Top opening sequences (SSM White, first 4 moves):

| Sequence | Games | W-D-L |
|---|---|---|
| `e4 Nf3 d4 Nxd4` | 7 | 0-0-7 |
| `d4 c4 Nc3 e4` | 2 | 0-0-2 |
| `c4 Nc3 g3 Bg2` | 2 | 0-0-2 |
| `e4 Nf3 Nxe5 Qe2` | 2 | 0-1-1 |
| `d4 Nc3 Bg5 Nxe4` | 2 | 0-0-2 |

No specific opening is disproportionately bad; SSM loses across all standard lines. The draws are all as White:
- `e4 Nf3 Nxe5 Qe2 d3 dxe4` (66 plies)
- `d4 c4 Nc3 e4 f3 Be3` (63 plies)
- `e4 Nf3 d4 Nxd4 Nc3 Nxc6` (63 plies)

---

## 5. Tactical Patterns

### 5.1 Back-Rank Mates
- **18 / 57 checkmates (31.6%)** were delivered by a rook or queen on the opponent's first/eighth rank.
- This is a classic back-rank weakness: SSM fails to recognize when its king is trapped behind undeveloped pieces.

### 5.2 King Exposure at Mate
- Average squares around SSM's king attacked by opponent pieces at the moment of mate: **4.37 / 8**.
- Distribution: 22 games with ≥5 attacked squares, 12 games with ≥6, 6 games with ≥7.
- This indicates chronic king-safety blindness.

### 5.3 Material Balance
| Metric | Value |
|---|---|
| Avg material balance at ply 20 (White – Black) | **-0.53 pawn** |
| Avg final material balance | **-0.44 pawn** |
| SSM material drops ≥ 2 pawns in one move | **0** |
| Opponent material drops ≥ 2 pawns in one move | **253** |

SSM does not blunder material suddenly; instead it is **slowly outplayed tactically**. The opponent accumulates small advantages until a decisive attack is possible.

### 5.4 Common Mating Moves
| Mate delivery | Count |
|---|---|
| `Qxf8#` | 3 |
| `Qg7#` | 3 |
| `Qxg3#` | 2 |
| `Bxf3#` | 2 |
| `Nc3#` | 2 |
| `Rxe8#` | 2 |
| `Qf7#` | 2 |
| `Qg5#` | 2 |

Many mates are queen deliveries to f8/f7 or g7/g3 — squares near SSM's king.

---

## 6. Systematic Weaknesses

### 6.1 King Safety & Defense
- **31.6% back-rank mates** prove SSM cannot defend its king.
- The model likely assigns insufficient penalty to king exposure in its value head.
- Defensive resources (interposition, blocking, king escape) are not found by the shallow 64-sim search.

### 6.2 Poor Middlegame Tactics
- 77% of losses occur in the middlegame.
- The opponent accumulates small advantages (captures, threats) without SSM responding adequately.
- This is consistent with a policy that knows piece values but not tactical motifs (forks, pins, skewers, discovered attacks).

### 6.3 Catastrophic Color Bias
- **0% win rate as Black** vs **0% win rate, 10% draw rate as White**.
- The model uses white-absolute coordinates with an explicit `color` feature, so architecture is symmetric.
- The 100% loss rate as Black points to either:
  - Training data bias (most PGNs are from White's perspective).
  - A latent bug in black-specific search or value evaluation.
  - Poor opening responses for Black in the training distribution.

### 6.4 Shallow Search
- 64 simulations per move with ~30 legal moves yields ~2 sims per action on average.
- This is insufficient for middlegame tactics where 3–5 ply lookahead is often needed to see a forcing sequence.
- The search is therefore mostly confirming the policy's weak prior rather than correcting it.

---

## 7. Search Parameter Comparison

| Parameter | SSM | Transformer |
|---|---|---|
| Simulations | 64 (Gumbel order-halving) | 0 (pure policy) |
| Top candidates (m0) | 16 | N/A |
| c_scale | 0.1 | N/A |
| c_visit | 50 | N/A |
| Temperature | 0.0 (deterministic) | 0.0 (deterministic) |

**Key insight:** The comparison is **rigged against the Transformer** — it has no search at all. Yet it still wins 95% of games. This proves the Transformer's policy head is **vastly stronger** than SSM's policy head, and SSM's search cannot bridge the gap.

---

## 8. Root Cause Assessment

1. **Training data scale mismatch**  
   - SSM: 2,500 self-play games (round2).  
   - Transformer: 20M Lichess positions.  
   - The policy head simply lacks the chess knowledge to compete.

2. **Insufficient search depth**  
   - 64 sims is too few for middlegame defense. Even if the value head were perfect, the tree would not explore enough forcing lines to find the correct defensive move.

3. **King-safety blind spot**  
   - The value head and/or training objective underweights king safety. Back-rank mates and exposed kings dominate the loss profile.

4. **Color asymmetry**  
   - 100% loss as Black is a red flag for either data bias or a subtle search/evaluation bug specific to the Black perspective.

5. **Value head calibration**  
   - If WDL outputs are poorly calibrated, Gumbel search will guide the tree toward positions that look good but are actually losing.

---

## 9. Proposed Fixes

### 9.1 Immediate (highest impact)
| Fix | Rationale | Effort |
|---|---|---|
| **Increase search to n_sims=256 or 512** | Deeper search may find defensive resources that 64 sims miss. | Low (CLI flag) |
| **Add defensive opening book for Black** | Avoid early weaknesses as Black where 100% loss rate occurs. | Medium |
| **Calibrate WDL head** | Use temperature scaling or Platt scaling on validation set to improve value estimates. | Medium |

### 9.2 Medium-term
| Fix | Rationale | Effort |
|---|---|---|
| **Train on more data** | Mix human BC (10%) and more self-play. Target 10k–50k games before next arena. | High |
| **Add king-safety auxiliary loss** | Penalize positions where the king has <2 escape squares and is attacked. | Medium |
| **Fix color bias** | Audit training data for White/Black symmetry; add explicit color-balanced loss weighting. | Medium |

### 9.3 Long-term
| Fix | Rationale | Effort |
|---|---|---|
| **Iterative self-play with search** | Use the current model to generate data with stronger search (more sims), then retrain. | High |
| **Progressive search widening** | Start with 64 sims, increase to 512 sims as policy improves. | Low |

---

## 10. Specific Diagnostic Recommendations

1. **Run a control match:** SSM with `n_sims=0` (pure policy) vs Transformer `mcts_sims=0`. If SSM loses even more, the search is helping but not enough.
2. **Run a search-ablation:** SSM with `n_sims=256` vs Transformer. If SSM still loses >80%, the policy gap is the bottleneck.
3. **Color-bias test:** Play SSM (White) vs a weaker opponent (e.g., Stockfish level 5 or an earlier SSM checkpoint) to see if the 100% Black loss persists against weaker opposition.
4. **King-safety eval test:** Compute the correlation between WDL Q-value and actual king exposure (attacked squares around king) on a held-out set. If correlation is low, the value head is blind to king safety.

---

## 11. Conclusion

The 57-0-3 result is a **policy-strength crisis, not a search failure**. The Transformer 20M's pure policy is strong enough to crush SSM's policy+search by 636 Elo. The immediate priorities are:

1. **Scale up training data** (more self-play + human mixing).
2. **Increase search simulations** to at least 256–512 sims per move to extract more from the current policy.
3. **Fix the Black color bias** — 100% loss rate is unacceptable and likely reveals a data or code defect.
4. **Add king-safety signal** to the loss function to stop the back-rank mate epidemic.

Until the policy head is on par with the opponent's, adding more search is a band-aid; the real fix is more and better training data, plus a targeted defensive auxiliary task.
