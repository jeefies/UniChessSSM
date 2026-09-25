"""PGN 小样本获取与过滤（设计文档 §4.1；当前阶段：小样本验证，不下全量）。

全量阶段将改用 database.lichess.org 月度 .pgn.zst；此处只提供：
- fetch_sample_pgns()：经 lichess 公开接口取一个已完成竞技场的棋局（匿名可用）；
- iter_games()：解析并过滤（排除变体、<10 ply、无结果局）。
元数据：双方 Elo、TimeControl、结果。
"""

from __future__ import annotations

import io
import json
import urllib.request

import chess.pgn

LICHESS_API = "https://lichess.org"
MIN_PLIES = 10


def _http_get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "UniChessSSM/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_sample_pgns(out_path: str, max_games: int = 50) -> int:
    """取一个已完成竞技场的 PGN 存入 out_path；返回写入局数。接口不可用则抛异常（调用方兜底）。"""
    data = json.loads(_http_get(f"{LICHESS_API}/api/tournament"))
    arena_id = None
    for section in ("finished", "started"):
        for ev in data.get(section, []):
            if ev.get("id"):
                arena_id = ev["id"]
                break
        if arena_id:
            break
    if not arena_id:
        raise RuntimeError("lichess 锦标赛列表为空，无法取样")
    pgn_bytes = _http_get(f"{LICHESS_API}/api/tournament/{arena_id}/games?format=pgn&max={max_games}",
                          timeout=120)
    text = pgn_bytes.decode("utf-8", errors="replace")
    games = text.count("\n\n\n")
    if games == 0 and text.strip():
        games = 1
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return games


def _parse_int(value: str | None) -> int | None:
    try:
        return int(value) if value not in (None, "?", "-") else None
    except ValueError:
        return None


def game_metadata(game: chess.pgn.Game) -> dict:
    """提取元数据：双方 Elo、time control、结果（1-0/0-1/1/2-1/2 或 None）。"""
    heads = game.headers
    w_elo = _parse_int(heads.get("WhiteElo"))
    b_elo = _parse_int(heads.get("BlackElo"))
    elos = [e for e in (w_elo, b_elo) if e is not None]
    result = heads.get("Result", "*")
    return {
        "white_elo": w_elo,
        "black_elo": b_elo,
        "elo_mean": float(sum(elos) / len(elos)) if elos else None,
        "time_control": heads.get("TimeControl"),
        "result": result if result in ("1-0", "0-1", "1/2-1/2") else None,
        "variant": heads.get("Variant", "Standard"),
    }


def keep_game(game: chess.pgn.Game, meta: dict) -> bool:
    """过滤规则（§4.1）：变体排除、<10 ply 排除、无结果局排除（abandoned 保留但结果按实际计）。"""
    if meta["variant"] != "Standard":
        return False
    if meta["result"] is None:
        return False
    if len(list(game.mainline_moves())) < MIN_PLIES:
        return False
    return True


def iter_games(path: str):
    """惰性产出 (game, meta)（已过滤）。"""
    with open(path, encoding="utf-8", errors="replace") as fh:
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                return
            meta = game_metadata(game)
            if keep_game(game, meta):
                yield game, meta
