"""验收 #4：分支缓存隔离（设计文档 §10.1 / §9 copy-on-write）。

同一节点复制出的两个子树沿不同动作扩展，互不影响输出；父 cache 不被改写。
需要 CUDA；无 GPU 时跳过。
"""

from __future__ import annotations

import unittest

import torch

try:
    from tests.test_consistency import _synthetic_game_features
except ImportError:
    # 本仓库 tests/ 无 __init__.py（命名空间包）；sys.path 上有 Kit 时，Kit 的常规 tests 包
    # 会胜出（PEP 420），此时按 discover 放进 sys.path 的 tests 目录直接导入
    from test_consistency import _synthetic_game_features

from stateseq.conditions import TimeControlBucket


@unittest.skipUnless(torch.cuda.is_available(), "mamba_ssm 单步 kernel 仅支持 CUDA")
class CacheIsolationTest(unittest.TestCase):
    def test_branch_isolation(self):
        from stateseq.model import SeqModel
        from stateseq.model_r import clone_cache

        torch.manual_seed(1)
        model = SeqModel(dropout=0.0).float().cuda().eval()

        feats = torch.from_numpy(_synthetic_game_features(16)).float().cuda().unsqueeze(0)
        tc = torch.full((1,), int(TimeControlBucket.RAPID), dtype=torch.long, device="cuda")
        elo = torch.zeros(1, device="cuda")

        with torch.no_grad():
            # 公共前缀：走 6 步建立父节点 cache
            cache = model.initial_cache(1, device="cuda")
            for t in range(6):
                color = feats[:, t, 768].long()
                _, _, _, _, cache = model.step(feats[:, t], tc, elo, color, cache)
            parent_cache = clone_cache(cache)

            # 两个子树沿不同后续局面扩展；c2_frozen 冻结备份用于事后验证“静止子树不受影响”
            c1 = clone_cache(parent_cache)
            c2 = clone_cache(parent_cache)
            c2_frozen = clone_cache(c2)
            color7 = feats[:, 7, 768].long()
            p1, w1, m1, _, c1 = model.step(feats[:, 7], tc, elo, color7, c1)
            color9 = feats[:, 9, 768].long()
            p2, w2, m2, _, c2 = model.step(feats[:, 9], tc, elo, color9, c2)  # 不同局面

            # 子树 1 继续两步，子树 2 静止：互不影响
            for t in (8, 9):
                p1b, w1b, m1b, _, c1 = model.step(feats[:, t], tc, elo, feats[:, t, 768].long(), c1)
            p2_after, w2_after, m2_after, _, _ = model.step(
                feats[:, 9], tc, elo, color9, clone_cache(c2_frozen)
            )

            self.assertTrue(torch.equal(p2, p2_after))
            self.assertTrue(torch.equal(w2, w2_after))
            self.assertTrue(torch.equal(m2, m2_after))

            # 父 cache 未被任何子树扩展改写
            for (conv_p, ssm_p), (conv_c, ssm_c) in zip(parent_cache, clone_cache(parent_cache)):
                self.assertTrue(torch.equal(conv_p, conv_c))
                self.assertTrue(torch.equal(ssm_p, ssm_c))


if __name__ == "__main__":
    unittest.main()
