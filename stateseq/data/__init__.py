"""数据管线（Stage A 规格见设计文档 §4）。

- pgns.py：小样本 PGN 获取（lichess 公开接口）与解析过滤；
- sequences.py：PGN → 每半回合记录（特征/动作/合法mask/结果/剩余ply/条件）；
- shards.py：分片二进制存储（memmap）+ 按 game_id 哈希划分 train/val（同局不跨集）。
"""

try:
    from .sequences import StepRecord, game_to_sequence
    __all__ = ["StepRecord", "game_to_sequence"]
except (ImportError, ModuleNotFoundError):
    __all__ = []
