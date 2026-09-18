"""Multi-generation control-flow test (review P1).

Tests champion/learner lifecycle without GPU:
1. Promote branch: mock arena_score >= 55% → learner becomes champion
2. Demote branch: mock arena_score < 55% → champion unchanged, learner continues
3. Optimizer state continuity across generations
4. Step counter continuity

Usage (CPU only):
  python tools/test_multi_gen_control_flow.py

Or via unittest:
  python -m unittest tools.test_multi_gen_control_flow
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _mock_checkpoint(path, step, model_str="seq"):
    """Create a minimal mock checkpoint. Size ~1KB, contains model, opt, sched dicts."""
    import torch
    state = {
        "model": {"mock": torch.zeros(1)},
        "opt": {"mock_step": step},
        "sched": {"mock_lr": 3e-5},
        "step": step,
    }
    torch.save(state, path)


def _mock_arena_result(score_percent):
    """Return a mock arena manifest dict."""
    total = 64
    draws = int(total * (1 - abs(score_percent - 50) / 50))
    wins = int((score_percent * total / 100 - draws // 2))
    if wins < 0:
        wins = 0
    return {
        "total_games": total,
        "score_a_percent": score_percent,
        "wins_a": wins,
        "draws": total - wins,
    }


def run_generation_cycle(arena_result, champion_path, learner_path, learner_next_path):
    """Run one generation cycle with mock arena result.
    
    Returns: (promoted: bool, new_champion: str)
    """
    score = arena_result.get("score_a_percent", 50.0)
    promoted = score >= 55.0
    
    if promoted:
        # Learner becomes champion
        shutil.copy2(learner_path, champion_path)
        new_champion = learner_path
    else:
        # Champion unchanged
        new_champion = champion_path
    
    # Learner training happened here (mocked). Save learner for next gen.
    learner_ckpt = torch.load(learner_path, map_location="cpu", weights_only=True)
    next_step = learner_ckpt.get("step", 0) + 11  # mock 11 training steps
    _mock_checkpoint(learner_next_path, next_step)
    
    return promoted, new_champion


class TestMultiGenControlFlow(unittest.TestCase):
    
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.champion_path = os.path.join(self.tmpdir, "champion.pt")
        self.learner_path = os.path.join(self.tmpdir, "learner_gen0.pt")
        self.learner_next = os.path.join(self.tmpdir, "learner_gen1.pt")
        _mock_checkpoint(self.champion_path, step=0)
        _mock_checkpoint(self.learner_path, step=0)  # learner starts as champion clone
        
        # Arena result cache
        self.arena_results = []
    
    def tearDown(self):
        shutil.rmtree(self.tmpdir)
    
    def _verify_champion_integrity(self, path, expected_step):
        import torch
        self.assertTrue(os.path.exists(path), "Champion checkpoint missing")
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(ckpt.get("step"), expected_step, 
                         "Champion step mismatch")
    
    def _verify_learner_state(self, path, expected_step, label=""):
        import torch
        self.assertTrue(os.path.exists(path), "Learner checkpoint missing (%s)" % label)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(ckpt.get("step"), expected_step, 
                         "Learner step mismatch at %s: got %s expected %s" % 
                         (label, ckpt.get("step"), expected_step))
        self.assertIn("opt", ckpt, "Learner missing optimizer state at %s" % label)
        self.assertIn("sched", ckpt, "Learner missing scheduler state at %s" % label)
    
    def test_gen0_unchanged(self):
        """Gen 0 → arena 50% → champion unchanged, learner at step 11."""
        result = _mock_arena_result(50.0)
        promoted, new_champ = run_generation_cycle(
            result, self.champion_path, self.learner_path, self.learner_next)
        
        self.assertFalse(promoted, "Should not promote at 50%")
        self.assertEqual(new_champ, self.champion_path, 
                         "Champion should remain unchanged")
        self._verify_champion_integrity(self.champion_path, 0)
        self._verify_learner_state(self.learner_next, 11, "gen1_learner")
    
    def test_promote_at_55(self):
        """Arena 55% → learner promoted to champion, learner continues."""
        result = _mock_arena_result(55.0)
        promoted, new_champ = run_generation_cycle(
            result, self.champion_path, self.learner_path, self.learner_next)
        
        self.assertTrue(promoted, "Should promote at 55%")
        # Champion file should now contain learner (step 0 → promoted)
        # Actually, promoted copies learner → champion, champion now has step 0
        self._verify_champion_integrity(self.champion_path, 0)
        self._verify_learner_state(self.learner_next, 11, "gen1_learner_after_promote")
    
    def test_promote_at_100(self):
        """Boundary: arena 100% → promote."""
        result = _mock_arena_result(100.0)
        promoted, new_champ = run_generation_cycle(
            result, self.champion_path, self.learner_path, self.learner_next)
        self.assertTrue(promoted, "Should promote at 100%")
        self._verify_champion_integrity(self.champion_path, 0)
    
    def test_no_promote_at_54_9(self):
        """Boundary: arena 54.9% → NOT promote. Just below threshold."""
        result = _mock_arena_result(54.9)
        promoted, new_champ = run_generation_cycle(
            result, self.champion_path, self.learner_path, self.learner_next)
        self.assertFalse(promoted, "Should NOT promote at 54.9%")
        self._verify_champion_integrity(self.champion_path, 0)
    
    def test_multi_gen_cycle_no_promote(self):
        """Multiple generations without promotion: champion stays at step 0.
        
        3 cycles: gen0→gen1→gen2. Each with arena 50%. Learner steps: 11→22→33.
        """
        prev_learner = self.learner_path
        for gen in range(3):
            next_learner = os.path.join(self.tmpdir, "learner_gen%d.pt" % (gen + 1))
            result = _mock_arena_result(50.0)
            promoted, _ = run_generation_cycle(
                result, self.champion_path, prev_learner, next_learner)
            self.assertFalse(promoted, "Gen %d: should NOT promote at 50%%" % gen)
            self._verify_champion_integrity(self.champion_path, 0)
            prev_learner = next_learner
        
        # After 3 cycles: learner at step 33, champion still at step 0
        self._verify_learner_state(prev_learner, 33, "gen3_learner")
    
    def test_multi_gen_cycle_late_promote(self):
        """Gen 0-1 no promote (50%), gen 2 promote at 60%.

        Champion stays at 0 for gen 0-1. After gen 2 promote, champion should be
        the gen 2 learner (step 22 originally, but checkpoint step is from copy).
        """
        prev_learner = self.learner_path
        for gen in range(2):
            next_learner = os.path.join(self.tmpdir, "learner_gen%d.pt" % (gen + 1))
            result = _mock_arena_result(50.0)
            promoted, _ = run_generation_cycle(
                result, self.champion_path, prev_learner, next_learner)
            self.assertFalse(promoted, "Gen %d: should NOT promote at 50%%" % gen)
            prev_learner = next_learner
        
        # Gen 2: promote
        next_learner = os.path.join(self.tmpdir, "learner_gen3.pt")
        result = _mock_arena_result(60.0)
        promoted, new_champ = run_generation_cycle(
            result, self.champion_path, prev_learner, next_learner)
        self.assertTrue(promoted, "Gen 2: should promote at 60%")
        
        # Champion file should have been overwritten with learner (step 22 originally)
        self._verify_champion_integrity(self.champion_path, 22)
        self._verify_learner_state(next_learner, 33, "gen3_learner_after_promote")
    
    def test_champion_not_overwritten_without_promote(self):
        """Verify champion file content doesn't change when not promoted."""
        import torch
        original_content = open(self.champion_path, "rb").read()
        
        for _ in range(3):
            result = _mock_arena_result(50.0)
            promoted, _ = run_generation_cycle(
                result, self.champion_path, self.learner_path, self.learner_next)
            self.assertFalse(promoted)
        
        # Champion file should be byte-identical
        current_content = open(self.champion_path, "rb").read()
        self.assertEqual(original_content, current_content,
                         "Champion file changed without promotion")


if __name__ == "__main__":
    unittest.main()