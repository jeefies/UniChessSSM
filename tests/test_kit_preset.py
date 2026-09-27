"""SSM 侧 kit 接入契约：SPI 版本、config.json 预设、工厂参数校验。

2026-09-27 出的两件事都该有测试兜底：

1. 扁平化把 Kit 的 ``SPI_VERSION`` 提到 2 时没同步 SSM，``KIT_SPI_VERSION`` 停在 1，
   Server 竞技场一启动就 ``RegistryError``（普通对局走 engine.py 不经过 registry，线上没暴露）。
2. ``make_player_factory`` 原先不认 ``preset=``，而 Server 观战/批量对弈正是按
   ``{"preset": <arg>}`` 调工厂的，修了 SPI 也会立刻 TypeError。

这两类测试都不加载权重、不碰 GPU：校验参数与查表就够了。
"""
from __future__ import annotations
import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_IMPORT_ROOT = _os.path.dirname(_HERE)   # import 根：~/UniChess，SSM 与 Kit 都是它的顶层包
if _IMPORT_ROOT not in _sys.path:
    _sys.path.insert(0, _IMPORT_ROOT)


import unittest


class TestSpiVersion(unittest.TestCase):
    def test_matches_kit(self):
        import Kit
        from SSM import kit
        self.assertEqual(kit.KIT_SPI_VERSION, Kit.SPI_VERSION,
                         "SSM.kit 声明的 SPI 版本与 Kit 不一致：registry 会拒绝加载，"
                         "Server 竞技场/观战、Kit match/selfplay/loop 全部起不来")


class TestLoadPreset(unittest.TestCase):
    def test_champion_preset(self):
        from SSM.kit import load_preset
        d = load_preset("champion")
        self.assertEqual(d["engine"], "fast")
        self.assertIn("checkpoint", d)
        self.assertGreater(int(d["simulations"]), 0)
        self.assertNotIn("description", d)      # 给 UI 看的说明，不进参数表

    def test_unknown_preset(self):
        from SSM.kit import load_preset
        with self.assertRaises(KeyError):
            load_preset("no_such_preset")


class TestFactoryArgs(unittest.TestCase):
    def test_unknown_kwarg_rejected(self):
        from SSM.kit import make_player_factory
        with self.assertRaises(TypeError):
            make_player_factory(preset="champion", bogus_param=1)

    def test_requires_checkpoint_or_preset(self):
        from SSM.kit import make_player_factory
        with self.assertRaises(FileNotFoundError):
            make_player_factory()

    def test_missing_weight_file(self):
        from SSM.kit import make_player_factory
        with self.assertRaises(FileNotFoundError):
            make_player_factory("/nonexistent/champion.pt")


if __name__ == "__main__":
    unittest.main()
