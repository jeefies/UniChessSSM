"""路径解析口径（2026-10-11 修）：相对路径 import 根优先、cwd 兜底。

背景：``_resolve`` 原本把一切相对路径拼到 import 根上，配置里这么写是对的，但命令行 /
脚本里手写的相对路径（cwd 常是 SSM 仓库根）被静默改解释，报错指向八竿子打不着的路径——
本会话连踩三次（champion 加载、worker config.json、raw-policy 评测）。
"""

from __future__ import annotations
import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_IMPORT_ROOT = _os.path.dirname(_HERE)   # import 根：~/UniChess
if _IMPORT_ROOT not in _sys.path:
    _sys.path.insert(0, _IMPORT_ROOT)


import shutil
import tempfile
import unittest
from pathlib import Path

from SSM import kit


class ResolvePathTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.cwd = Path(tempfile.mkdtemp())
        self._old_root = kit.ROOT
        self._old_cwd = _os.getcwd()
        kit.ROOT = self.root
        _os.chdir(self.cwd)

    def tearDown(self):
        kit.ROOT = self._old_root
        _os.chdir(self._old_cwd)
        shutil.rmtree(self.root, ignore_errors=True)
        shutil.rmtree(self.cwd, ignore_errors=True)

    def _touch(self, base: Path, rel: str) -> Path:
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x", encoding="utf-8")
        return p

    def test_absolute_unchanged(self):
        p = self._touch(self.cwd, "abs.pt")
        self.assertEqual(kit._resolve(str(p)), p)

    def test_relative_prefers_import_root(self):
        """两边都存在时 import 根赢——配置口径（SSM/runs/...）不变。"""
        a = self._touch(self.root, "runs/a.pt")
        self._touch(self.cwd, "runs/a.pt")          # 同名干扰项必须被忽略
        self.assertEqual(kit._resolve("runs/a.pt"), a)

    def test_relative_falls_back_to_cwd(self):
        """import 根没有、cwd 有时用 cwd 的——这正是踩坑的那条。"""
        b = self._touch(self.cwd, "runs/b.pt")
        self.assertEqual(kit._resolve("runs/b.pt"), b)

    def test_missing_returns_root_candidate(self):
        """两边都没有 ⇒ 仍报 import 根口径（既有错误信息不变）。"""
        self.assertEqual(kit._resolve("runs/c.pt"), self.root / "runs" / "c.pt")

    def test_nested_relative(self):
        d = self._touch(self.cwd, "a/b/c/d.pt")
        self.assertEqual(kit._resolve("a/b/c/d.pt"), d)

    def test_missing_path_error_lists_both_candidates(self):
        err = kit._missing_path(self.root / "x.pt", "权重")
        text = str(err)
        self.assertIn("import 根", text)
        self.assertIn("cwd", text)
        self.assertIn("绝对路径", text)
        self.assertIsInstance(err, FileNotFoundError)


class RealWorldScenarioTest(unittest.TestCase):
    """真实布局下的两条口径（不 monkeypatch）：配置式路径走 import 根，仓库内路径也能解析。"""

    def test_config_style_path_uses_import_root(self):
        # 配置里的 SSM/runs/... 必须解析到 import 根下的 SSM 仓库
        p = kit._resolve("SSM/runs/champion.pt")
        self.assertTrue(p.is_absolute())
        self.assertEqual(p, kit.ROOT / "SSM" / "runs" / "champion.pt")

    def test_repo_relative_path_resolves_inside_repo(self):
        # cwd 若是 SSM 仓库根，仓库内相对路径也要能解析（import 根下没有这一层）
        if Path.cwd() != kit.SSM_ROOT:
            self.skipTest("cwd 不是 SSM 仓库根")
        p = kit._resolve("runs/champion.pt")
        self.assertTrue(p.is_absolute())
        self.assertTrue(p.exists() or (kit.ROOT / "runs" / "champion.pt").exists())


if __name__ == "__main__":
    unittest.main()
