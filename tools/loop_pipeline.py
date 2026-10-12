"""UniChessSSM 多轮自动化强化学习迭代循环 (Autonomous Multi-Gen RL Loop)

调度流程（每代闭环）：
1. 自博弈生成 (Selfplay)：
   - 采用 8 Workers x 20 Concurrency (黄金甜点配置)
   - 现役 Champion vs 历史对手池，TF32 张量核加速
   - KataGo 式自适应模拟预算 (战术硬触发 256 sims, 开局 15% 探索深潜)
   - 四重防和棋铁壁 (twofold 1.0, stalemate 1.0, insufficient 1.0, contempt 0.5)
2. 混合强化训练 (Train)：
   - 85% 自博弈 + 15% 人类谱混合
   - warmup_then_cosine 学习率退火调度
   - 基于留出验证集最低 Loss 自动选优导出 best.pt (候选模型)
   - 自动清理中间 step_*.pt 检查点释放磁盘
3. 双轨竞技场裁决 (Arena)：
   - Track 1: 纯网络策略 (0-sim, 100局) 快速直觉基准
   - Track 2: 系统级成对换先 MCTS 裁决 + 五项式 GSPRT 动态早停 (H0: 0, H1: 35)
4. 换代晋级与归档 (Promotion & Archiving)：
   - 若 SPRT 判定 H1 胜出或得分率 >= 55%：加冕新 Champion，归档旧霸主
   - 若未达标：现役 Champion 守擂
   - 自动更新 loop_state.json，全流程可断点续跑
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

IMPORT_ROOT = os.environ.get("UNICHESS_IMPORT_ROOT", str(Path(__file__).resolve().parents[2]))
if IMPORT_ROOT not in sys.path:
    sys.path.insert(0, IMPORT_ROOT)

from Kit.rules.openings import OpeningBook  # noqa: E402


def log(msg: str, log_file: Path | None = None) -> None:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] [SSMLoop] {msg}"
    print(line, flush=True)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


#: Arena 每代开对数。每对同一开局打 2 局（A 执白 / B 执白），所以总局数 = pairs × 2。
#: 600 对 = 1200 局：配合 4000 条开局库（openings_4000.txt）零重复，CI 半宽约 ±20.6 Elo，
#: 是单代分辨 +14 Elo 量级边际增益的实用下限。
ARENA_PAIRS = 600

#: 换代门限的 Elo CI 下界阈值：候选"没有显著变弱"即可晋级。
ELO_CI_FLOOR = -5.0

#: CI 门限要求的最少对局数。SPRT 早停可能只剩几十局，那种样本的 CI 是纯噪声
#: （gen_0026 仅 59 局）；1000 局是 1200 局预算下 CI 仍有意义的下限。
MIN_GAMES_FOR_CI = 1000

#: 漂移护栏：上一代候选的 Elo 点估（CI 中点）低于此值，就不从它继续链式训练，回霸主重训。
#: 连续 Learner 演化的前提是"上一代候选至少没变差"；否则 init 与数据生成模型（恒为霸主）
#: 脱钩，Arena 比的就是累计漂移而不是本轮增量（Gen 34→37 实测 49.3%→45.9%→38.3%→48.4%）。
#: 2026-10-12 根因取证：自对弈 --ckpt 每代都是 champion.pt，数据分布全程持平，
#: 漂移只来自链式 init——所以判据应该直接看"候选有没有变差"，而不是"CI 下界跌没跌到 -20"。
DRIFT_GUARD_ELO_FLOOR = 0.0

#: 开局库（自对弈与 Arena 共用）。确定性对局下重复开局 = 逐字节重复棋谱，
#: 而 match.py 的 elo_ci95 按原始局数算、不按 duplicate_rate 校正 ⇒ 库不够时 CI 虚窄。
#: 4000 条 = 旧 curated 200 条 + 随机合法 6-12 ply 线路（seed 20261012，可复现）。
#: 放在 Kit/data/ 是因为 Kit 拥有 OpeningBook，且该目录的 .gitignore 已放行 .txt
#: （SSM 的 /data/ 是全禁，开局书放那边不会入 git ⇒ 干净克隆跑不了循环）。
OPENINGS_FILE = os.path.join(IMPORT_ROOT, "Kit/data/openings_4000.txt")


def promotion_gate(sprt_verdict, elo_ci95, games=None,
                   elo_ci_floor: float = ELO_CI_FLOOR,
                   min_games: int = MIN_GAMES_FOR_CI):
    """换代门限 → (是否晋升, 依据)。以 Elo CI 为主通道，SPRT H1 为快速通道。

    1. ``SPRT H1``——序贯检验判"显著变强"。大跨步改进会在远不到局数上限时就越界，
       这条让它不必等满 1200 局（gen 33 的 +45 Elo 就是 220 局判出的）。
    2. ``Elo 95% CI 下界 ≥ elo_ci_floor`` **且对局数 ≥ min_games**——主通道。

    为什么不再看"得分率 ≥ 55%"：55% 对应 +45 Elo，在 1000 局下 CI 下界必然远超 −5，
    被第 2 条完全覆盖；而它要求的效应量远高于当前边际增益（+14 Elo），留着只会让人
    误以为门禁有这么宽的通道。2026-10-12 用真实 CI 回测 32 代：旧三选一门禁与
    纯 CI 门禁判决完全一致（零翻转），删掉它不改变任何历史判决。

    为什么 elo1=35 的 SPRT 平时判不出 H1：真实边际增益只有 +14 Elo 时，LLR 的理论
    漂移 D(+14‖0) − D(+14‖35) 是**负值**，H1 边界结构上够不到（实测 5 代 llr 均值
    −0.616，理论 −0.504，吻合）。所以 SPRT 只在跨大步时起作用；它的 H0 早停仍然有价值
    ——候选明显变差时几十局就收工，不浪费 1200 局。
    """
    lo = None
    if elo_ci95 and len(elo_ci95) > 0 and elo_ci95[0] is not None:
        lo = float(elo_ci95[0])
    if sprt_verdict == "H1":
        return True, "SPRT H1"
    if lo is not None and lo >= float(elo_ci_floor):
        if games is None or int(games) >= int(min_games):
            return True, "Elo CI 下界 %+.1f ≥ %g（%s 局）" % (lo, elo_ci_floor, games)
        return False, "CI 达标但局数不足（%s < %d）" % (games, min_games)
    return False, "未达门限"


class SsmLoopOrchestrator:
    def __init__(self, out_dir: Path, total_gens: int, games_per_gen: int,
                 initial_champion: str, opp_checkpoint: str,
                 workers: int = 8, concurrency: int = 20,
                 python_bin: str = sys.executable,
                 value_target: str = "mc", value_lambda_end: float = 1.0,
                 value_anchor_weight: float = 0.0,
                 record_qsearch: bool = False,
                 arena_pairs: int = ARENA_PAIRS,
                 openings: str = OPENINGS_FILE):
        # out_dir 收 str 也收 Path（单测/外部调用方常传 str，早先会在这里 AttributeError）
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.total_gens = total_gens
        self.games_per_gen = games_per_gen
        self.initial_champion = os.path.abspath(initial_champion)
        self.opp_checkpoint = os.path.abspath(opp_checkpoint) if opp_checkpoint else None
        self.workers = workers
        self.concurrency = concurrency
        self.python_bin = python_bin
        # 价值头改造开关（默认全部关闭 ⇒ 与改造前逐位一致；开启时见 SSM/tasks.py D2/D4/D5）
        if value_target not in ("mc", "lambda"):
            raise ValueError(f"value_target 只能是 'mc' / 'lambda'，得到 {value_target!r}")
        self.value_target = str(value_target)
        self.value_lambda_end = float(value_lambda_end)
        self.value_anchor_weight = float(value_anchor_weight)
        self.record_qsearch = bool(record_qsearch)
        # Arena 开对数（×2 = 总局数）与开局库。两者必须配套：库规模 < pairs 时
        # OpeningBook.plan 会循环复用开局，而 match.py 的 elo_ci95 按原始局数算、
        # 不按 duplicate_rate 校正 ⇒ CI 虚窄、门禁被误导。
        self.arena_pairs = int(arena_pairs)
        if self.arena_pairs < 1:
            raise ValueError(f"arena_pairs 必须 ≥ 1，得到 {arena_pairs}")
        self.openings = str(openings)
        if not os.path.exists(self.openings):
            raise FileNotFoundError(f"开局库不存在：{self.openings}")
        # 不变式：开局库规模必须 ≥ arena_pairs。否则 OpeningBook.plan 会 i % n 循环复用
        # 开局，而 match.py 的 elo_ci95 按原始局数算、不按 duplicate_rate 校正
        # ⇒ 报告 CI 虚窄、门禁被误导（2026-10-12：库 200 / pairs 250 时 duplicate_rate
        # 已达 22.4%）。宁可在这里响亮报错，也不静默产出重复局。
        book = OpeningBook.from_file(self.openings)
        if len(book) < self.arena_pairs:
            raise ValueError(
                f"开局库只有 {len(book)} 条，撑不起 arena_pairs={self.arena_pairs}"
                f"（{self.arena_pairs * 2} 局）——会循环复用开局产生逐字节重复棋谱，"
                f"而 elo_ci95 不按 duplicate_rate 校正。请扩容开局库或调小 --arena-pairs。")

        self.log_file = self.out_dir / "loop.log"
        self.state_file = self.out_dir / "loop_state.json"
        self.state = self.load_state()

    def load_state(self) -> dict:
        if self.state_file.exists():
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "current_gen": 6,  # 紧随 Round 5 (gen 5) 之后
            "target_gens": self.total_gens,
            "champion": self.initial_champion,
            "opp_champion": self.opp_checkpoint,
            "history": [],
            "status": "ready"
        }

    def save_state(self) -> None:
        write_json(self.state_file, self.state)

    def run_command(self, cmd: list[str], phase_log: Path) -> int:
        phase_log.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["UNICHESS_IMPORT_ROOT"] = IMPORT_ROOT
        py_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{IMPORT_ROOT}:{py_path}" if py_path else IMPORT_ROOT

        with open(phase_log, "a", encoding="utf-8") as lf:
            lf.write(f"\n$ {' '.join(cmd)}\n")
            lf.flush()
            res = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=IMPORT_ROOT)
        return res.returncode

    def step_selfplay(self, gen_id: int, gen_dir: Path, champion_pt: str, opp_pt: str | None) -> Path:
        sp_out = gen_dir / "selfplay"
        manifest_path = sp_out / "manifest.json"
        phase_log = gen_dir / "selfplay.log"

        if manifest_path.exists():
            log(f"Gen {gen_id} 自对弈分片已存在 ({manifest_path})，跳过生成阶段", self.log_file)
            return sp_out

        log(f"Gen {gen_id} 开始自对弈生成 ({self.games_per_gen} 局, {self.workers}x{self.concurrency} 并发)...", self.log_file)
        cmd = [
            self.python_bin, "-u",
            os.path.join(IMPORT_ROOT, "SSM/tools/run_gpu_server_selfplay.py"),
            "--ckpt", champion_pt,
            "--out", str(sp_out),
            "--total-games", str(self.games_per_gen),
            "--workers", str(self.workers),
            "--concurrency", str(self.concurrency),
            "--simulations", "64",
            "--m0", "16",
            "--c-scale", "0.02",
            "--twofold-penalty", "1.0",
            "--stalemate-penalty", "1.0",
            "--insufficient-penalty", "1.0",
            "--contempt", "0.5",
            "--temp-plies", "15",
            "--temperature", "1.0",
            "--openings", self.openings,
            "--book-plies", "8",
            "--min-book-plies", "6",
            "--pcr-rate", "0.5",
            "--pcr-fast-sims", "16",
            "--adaptive-sims",
            "--deep-sims", "256",
            "--p-explore", "0.15",
            "--precision", "tf32",
            "--seed", str(int(time.time() * 1000) & 0x7FFFFFFF),
            "--first-game", str((gen_id - 1) * self.games_per_gen)
        ]
        if self.record_qsearch:
            # 逐 ply 搜索价值落盘（零额外搜索成本；训练侧 value_target='lambda' 才消费）
            cmd.append("--record-qsearch")
        # 多模型博弈联盟机制 (League Play: 50% 当前自博弈 + 25% 历史最强 + 25% 历史联盟轮换)
        league_ckpts = []
        for g_id in [4, 3, 2, 1]:
            p = os.path.join(IMPORT_ROOT, f"SSM/runs/champion_gen{g_id}.pt")
            if os.path.exists(p) and os.path.abspath(p) != os.path.abspath(champion_pt):
                league_ckpts.append(p)
        if opp_pt and os.path.exists(opp_pt) and os.path.abspath(opp_pt) not in league_ckpts:
            league_ckpts.insert(0, os.path.abspath(opp_pt))

        if league_ckpts:
            cmd.append("--league")
            cmd.extend(["--league-opps", ",".join(league_ckpts)])
        elif opp_pt and os.path.exists(opp_pt):
            cmd.extend(["--opp-ckpt", opp_pt])

        rc = self.run_command(cmd, phase_log)
        if rc != 0 or not manifest_path.exists():
            raise RuntimeError(f"Gen {gen_id} 自对弈生成失败，退出码={rc}，详见 {phase_log}")

        log(f"Gen {gen_id} 自对弈生成完毕！", self.log_file)
        return sp_out

    def step_train(self, gen_id: int, gen_dir: Path, sp_dir: Path, base_pt: str) -> Path:
        train_out = gen_dir / "train"
        best_pt = train_out / "best.pt"
        phase_log = gen_dir / "train.log"

        if best_pt.exists():
            log(f"Gen {gen_id} 训练产物已存在 ({best_pt})，跳过训练阶段", self.log_file)
            return best_pt

        log(f"Gen {gen_id} 开始混合强化训练 (85% 自博弈多代经验池 + 15% 人类谱, 基座={os.path.basename(base_pt)})...", self.log_file)
        # 构建滑动窗口经验回放池 (Sliding Window Replay Buffer, 最近 4 代)
        window_size = 4
        past_sp_dirs = []
        for g in range(max(1, gen_id - window_size + 1), gen_id + 1):
            d = self.out_dir / f"gen_{g:04d}" / "selfplay"
            if (d / "manifest.json").exists():
                past_sp_dirs.append(str(d))
        if not past_sp_dirs:
            past_sp_dirs = [str(sp_dir)]
        if self.record_qsearch:
            # 只收带 q̂ 的代。混入无 q̂ 的老代时，那些局的价值目标会退化成纯 MC
            # （整局无 q̂ ⇒ lambda_return 自动回落 z），而 λ=1 实测比 λ=0.5 差 8.5 分，
            # 混合等价于把 λ 往 1 拉——收益被稀释。开 q̂ 记录时才过滤，不影响原行为。
            keep = []
            for d in past_sp_dirs:
                try:
                    with open(os.path.join(d, "manifest.json"), encoding="utf-8") as fh:
                        if json.load(fh).get("qsearch"):
                            keep.append(d)
                except Exception:
                    pass
            dropped = len(past_sp_dirs) - len(keep)
            if dropped:
                log(f"Gen {gen_id} 回放池剔除 {dropped} 个无 q̂ 的旧代（价值目标口径不一致）",
                    self.log_file)
            if keep:
                past_sp_dirs = keep

        cfg_path = gen_dir / "train_config.json"
        # 价值目标开关：关闭时一个键都不加 ⇒ 生成的 train_config 与改造前逐位相同
        value_kwargs = {}
        if self.value_target != "mc":
            value_kwargs = {
                "value_target": self.value_target,
                "value_lambda_start": 1.0,
                "value_lambda_end": self.value_lambda_end,
                "value_lambda_anneal_frac": 0.25,
                # 终局锚的常数权重：递改自带的 λ^k 锚在百步棋里指数衰减到 0，目标退化成纯自举，
                # 实测两代内价值尺度坍缩（mean|q| −24%、|q|>0.9 占比腰斩）、Arena −33 Elo。
                "value_anchor_weight": self.value_anchor_weight,
                "material_weight": 0.1,
            }
        cfg_data = {
            "task": {
                "factory": "SSM.tasks:make_task",
                "kwargs": {
                    "kind": "stage_b2",
                    "data": {"dir": os.path.join(IMPORT_ROOT, "SSM/data/shards"), "t_max": 300},
                    "selfplay": {"dirs": past_sp_dirs},
                    "ckpt": base_pt,
                    "microbatch": 4,
                    "weights": {"w_v": 1.5, "w_r_start": 0.1, "w_r_end": 0.1},
                    "workers": 4,
                    "val_batches": 8,
                    "w_selfplay": 0.85,
                    "w_human": 0.15,
                    "mlh_log": True,
                    "kl_reweight": True,
                    "t_max": 300,
                    **value_kwargs,
                }
            },
            "out": str(train_out),
            "steps": 0,
            "accum": 16,
            "seed": int(time.time() * 1000) & 0x7FFFFFFF,
            "device": "cuda",
            "precision": "bf16",
            "grad_scaler": False,
            "clip": 1.0,
            "nonfinite": "raise",
            "optimizer": {"lr": 2.5e-05, "weight_decay": 0.1, "betas": [0.9, 0.999], "fused": True},
            "schedule": {"kind": "warmup_then_cosine", "warmup_frac": 0.1, "warmup_max": 30, "floor": 0.1, "scale": 0.9},
            "torch": {"tf32": True, "num_threads": 4, "cuda_mem_fraction": 0.65},
            "log_every": 10,
            "save_every": 20,
            "validate_every": 20,
            "export": {"best": "best.pt", "final": "final.pt", "every": 20}
        }
        write_json(cfg_path, cfg_data)

        cmd = [self.python_bin, "-u", "-m", "Kit", "train", str(cfg_path)]
        rc = self.run_command(cmd, phase_log)
        if rc != 0 or not best_pt.exists():
            raise RuntimeError(f"Gen {gen_id} 训练失败，退出码={rc}，详见 {phase_log}")

        # 清理多余的中间步检查点
        for p in train_out.glob("step_*.pt"):
            try: p.unlink()
            except Exception: pass

        log(f"Gen {gen_id} 训练完毕，成功导出最优泛化检查点 best.pt", self.log_file)
        return best_pt

    def step_eval_raw_policy(self, gen_id: int, gen_dir: Path, cand_pt: str, champion_pt: str) -> dict:
        phase_log = gen_dir / "eval_raw_policy.log"
        log(f"Gen {gen_id} 进行 Track 1: 纯网络策略 (0-sim) 对决...", self.log_file)

        cmd = [
            self.python_bin, "-u",
            os.path.join(IMPORT_ROOT, "SSM/tools/evaluate_raw_policy.py"),
            "--a-ckpt", cand_pt,
            "--b-ckpt", champion_pt,
            "--pairs", "50"
        ]
        rc = self.run_command(cmd, phase_log)
        summary = {"raw_policy_score": None}
        if rc == 0 and phase_log.exists():
            text = phase_log.read_text(encoding="utf-8")
            for line in text.splitlines():
                if "得分率:" in line:
                    summary["raw_policy_line"] = line.strip()
                    break
        return summary

    def step_arena(self, gen_id: int, gen_dir: Path, cand_pt: str, champion_pt: str) -> dict:
        results_jsonl = gen_dir / "arena.jsonl"
        summary_json = gen_dir / "arena.jsonl.summary.json"
        phase_log = gen_dir / "arena.log"

        if summary_json.exists():
            log(f"Gen {gen_id} Arena 裁决已存在 ({summary_json})，直接读取", self.log_file)
            return json.loads(summary_json.read_text(encoding="utf-8"))

        log(f"Gen {gen_id} 开始 Track 2: 系统级成对换先 MCTS 竞技场裁决 (SPRT 动态早停)...", self.log_file)
        cfg_path = gen_dir / "arena_config.json"
        match_cfg = {
            "a": {
                "factory": "SSM.kit:make_player_factory",
                "root": IMPORT_ROOT,
                "label": f"Candidate_gen{gen_id}",
                "kwargs": {
                    "checkpoint": cand_pt,
                    "simulations": 64,
                    "m0": 16,
                    "c_scale": 0.02,
                    "c_visit": 50.0,
                    "g": 0.0,
                    "engine": "fast",
                    "pool_slots": 1024,
                    "contempt": 0.5,
                    "stalemate_penalty": 1.0,
                    "insufficient_penalty": 0.25,
                    "twofold_penalty": 1.0,
                    "c_scale_schedule": True
                }
            },
            "b": {
                "factory": "SSM.kit:make_player_factory",
                "root": IMPORT_ROOT,
                "label": f"Champion_prev",
                "kwargs": {
                    "checkpoint": champion_pt,
                    "simulations": 64,
                    "m0": 16,
                    "c_scale": 0.02,
                    "c_visit": 50.0,
                    "g": 0.0,
                    "engine": "fast",
                    "pool_slots": 1024,
                    "contempt": 0.5,
                    "stalemate_penalty": 1.0,
                    "insufficient_penalty": 0.25,
                    "twofold_penalty": 1.0,
                    "c_scale_schedule": True
                }
            },
            "match": {
                "pairs": self.arena_pairs,
                "concurrency": 4,
                "max_plies": 300,
                "openings": self.openings,
                "seed": int(time.time() * 1000) & 0x7FFFFFFF,
                "workers": 4,
                "sprt": {
                    "elo0": 0.0,
                    "elo1": 35.0,
                    "alpha": 0.05,
                    "beta": 0.05,
                    "min_pairs": 20
                }
            }
        }
        write_json(cfg_path, match_cfg)

        cmd = [self.python_bin, "-u", "-m", "Kit", "match", str(cfg_path), "--out", str(results_jsonl)]
        rc = self.run_command(cmd, phase_log)
        if rc != 0 or not summary_json.exists():
            raise RuntimeError(f"Gen {gen_id} Arena 运行失败，退出码={rc}，详见 {phase_log}")

        summary = json.loads(summary_json.read_text(encoding="utf-8"))
        log(f"Gen {gen_id} Arena 裁决完成！战绩: {summary.get('a_wins')}胜/{summary.get('b_wins')}负/{summary.get('draws')}和, 得分率={summary.get('score_a'):.4f}, SPRT={summary.get('sprt', {}).get('verdict')}", self.log_file)
        return summary

    def run(self) -> None:
        start_gen = self.state["current_gen"]
        end_gen = start_gen + self.total_gens

        log("=" * 65, self.log_file)
        log(f"【SSM 自动化强化学习循环启动】目标迭代代数: {self.total_gens} 代 (Gen {start_gen} -> Gen {end_gen - 1})", self.log_file)
        log(f"当前 Champion: {self.state['champion']}", self.log_file)
        log("=" * 65, self.log_file)

        for gen_id in range(start_gen, end_gen):
            gen_dir = self.out_dir / f"gen_{gen_id:04d}"
            gen_dir.mkdir(parents=True, exist_ok=True)
            log(f"\n>>> [进入代际循环 Gen {gen_id} / {end_gen - 1}] <<<", self.log_file)

            current_champ = self.state["champion"]
            opp_champ = self.state.get("opp_champion")

            # 1. 自对弈
            t0 = time.time()
            sp_dir = self.step_selfplay(gen_id, gen_dir, current_champ, opp_champ)
            t_sp = time.time() - t0

            # 2. 确定本代训练起点 (Learner 连续演化 vs 5 代回滚安全熔断)
            # 策略：五代内有晋升则不回滚（从上一代最优 Candidate 连续推进）；
            # 连续 5 代冲击 Champion 失败则触发安全熔断，回滚至 Champion。
            consecutive_fails = 0
            for h in reversed(self.state.get("history", [])):
                if h.get("promoted", False):
                    break
                consecutive_fails += 1

            window_phase = consecutive_fails % 5  # 0: 周期第1代(新王/回滚重置), 1..4: 连续推进代
            prev_cand = None
            prev_ci_lo = None
            if gen_id > 1:
                p_cand = self.out_dir / f"gen_{gen_id - 1:04d}" / "train" / "best.pt"
                if p_cand.exists():
                    prev_cand = str(p_cand)
                # 上一代候选的 Arena Elo 点估（CI 中点）：候选没变差才值得继续链。
                # 优先读 history 记录；老记录没这个字段时退回读 arena summary
                # （2026-10-12 实测：gen 6–37 全缺该字段，护栏会静默失效）。
                prev_elo = None
                for h in reversed(self.state.get("history", [])):
                    if h.get("generation") == gen_id - 1:
                        ci = h.get("elo_ci95") or [None, None]
                        if ci and ci[0] is not None and len(ci) > 1 and ci[1] is not None:
                            prev_elo = (float(ci[0]) + float(ci[1])) / 2.0
                        elif h.get("elo") is not None:
                            prev_elo = float(h["elo"])
                        break
                if prev_elo is None:
                    sp = self.out_dir / f"gen_{gen_id - 1:04d}" / "arena.jsonl.summary.json"
                    if sp.exists():
                        try:
                            s = json.load(open(sp, encoding="utf-8"))
                            if s.get("elo") is not None:
                                prev_elo = float(s["elo"])
                            else:
                                ci = s.get("elo_ci95") or [None, None]
                                if ci[0] is not None and len(ci) > 1 and ci[1] is not None:
                                    prev_elo = (float(ci[0]) + float(ci[1])) / 2.0
                        except Exception:
                            prev_elo = None

            # 连续演化只在"上一代候选至少没变差"时才有意义：数据每代都由霸主生成
            # （自对弈 --ckpt 恒为 champion.pt，实测数据分布全程持平），链式 init 越链
            # 离数据生成模型越远，Arena 比的就是累计漂移而不是本轮增量
            # （实测 Gen 34→37 链式下滑 49.3%→45.9%→38.3%→48.4%）。
            drift_guard = prev_elo is not None and prev_elo < DRIFT_GUARD_ELO_FLOOR

            if window_phase != 0 and prev_cand and not drift_guard:
                base_pt = prev_cand
                log(f"🔄【连续 Learner 演化】五代非回滚窗口 (周期第 {window_phase + 1}/5 代, 累计未晋升: {consecutive_fails})，本代从上一代最优检查点连续推进: {base_pt}", self.log_file)
            elif window_phase != 0 and prev_cand and drift_guard:
                base_pt = current_champ
                log(f"🛑【漂移护栏】上一代候选 Elo 点估 {prev_elo:+.1f} < {DRIFT_GUARD_ELO_FLOOR:+.0f}（已变差），"
                    f"init 与数据生成模型脱钩，本代回霸主重训: {base_pt}", self.log_file)
            elif consecutive_fails > 0 and window_phase == 0:
                base_pt = current_champ
                log(f"⚠️【安全熔断回滚】前一个 5 代窗口冲击 Champion 未果 (累计未晋升: {consecutive_fails})，触发安全回滚，本代重置从 Champion 重新热启: {base_pt}", self.log_file)
            else:
                base_pt = current_champ
                log(f"🚀【首代/新王启动】本代从现役 Champion 启动: {base_pt}", self.log_file)

            # 混合训练
            t0 = time.time()
            cand_pt = self.step_train(gen_id, gen_dir, sp_dir, base_pt)
            t_train = time.time() - t0

            # 3. 双轨评测
            t0 = time.time()
            raw_summary = self.step_eval_raw_policy(gen_id, gen_dir, str(cand_pt), current_champ)
            arena_summary = self.step_arena(gen_id, gen_dir, str(cand_pt), current_champ)
            t_arena = time.time() - t0

            # 4. 裁决与换代
            # 门限见 promotion_gate：主通道是 Elo 95% CI 下界 ≥ -5 且局数 ≥ 1000；
            # SPRT H1 是快速通道（大跨步时不必等满局数）。不再看得分率 55%——它对应
            # +45 Elo，被 CI 通道完全覆盖，而当前边际增益只有 +14 Elo，留着会误导。
            sprt_verdict = arena_summary.get("sprt", {}).get("verdict")
            score_a = arena_summary.get("score_a", 0.0)
            elo_ci = arena_summary.get("elo_ci95") or [None, None]
            promoted, gate_reason = promotion_gate(sprt_verdict, elo_ci,
                                                   games=arena_summary.get("games"))
            elo_ci_lo = elo_ci[0] if elo_ci and elo_ci[0] is not None else None
            elo_ci_hi = elo_ci[1] if len(elo_ci) > 1 and elo_ci[1] is not None else None

            champion_before = current_champ
            if promoted:
                # 归档备份原 champion
                backup_champ = self.out_dir / f"champion_gen{gen_id - 1}_backup.pt"
                shutil.copy2(current_champ, backup_champ)
                # 晋升候选模型
                shutil.copy2(cand_pt, current_champ)
                self.state["opp_champion"] = str(backup_champ)
                log(f"🎉【加冕新霸主】Gen {gen_id} 满足换代门限（依据：{gate_reason}；"
                    f"SPRT={sprt_verdict}, 得分率={score_a:.2%}, Elo CI=[{elo_ci_lo}, {elo_ci_hi}]），"
                    f"晋升为现役 Champion！", self.log_file)
            else:
                log(f"🛡️【前任守擂】Gen {gen_id} 未突破换代门限（依据：{gate_reason}；"
                    f"SPRT={sprt_verdict}, 得分率={score_a:.2%}, Elo CI=[{elo_ci_lo}, {elo_ci_hi}]）。"
                    f"(连续冲击窗口: {consecutive_fails + 1}/5 代)", self.log_file)

            record = {
                "generation": gen_id,
                "promoted": promoted,
                "gate_reason": gate_reason,
                "elo_ci95": [elo_ci_lo, elo_ci_hi],
                "candidate": str(cand_pt),
                "train_base": base_pt,
                "champion_before": champion_before,
                "champion_after": self.state["champion"],
                "score_a": score_a,
                "elo": arena_summary.get("elo"),
                "sprt": sprt_verdict,
                "games_arena": arena_summary.get("games"),
                "raw_policy": raw_summary,
                "value_target": self.value_target,
                "value_lambda_end": self.value_lambda_end if self.value_target == "lambda" else None,
                "value_anchor_weight": self.value_anchor_weight if self.value_target == "lambda" else None,
                "record_qsearch": self.record_qsearch,
                "elapsed_sec": {
                    "selfplay": round(t_sp, 1),
                    "train": round(t_train, 1),
                    "arena": round(t_arena, 1)
                }
            }
            self.state["history"].append(record)
            self.state["current_gen"] = gen_id + 1
            self.save_state()

            log(f"[LOOP GEN {gen_id} FINISHED] 本代完成，耗时={round(t_sp+t_train+t_arena, 1)}s\n", self.log_file)

        log("=" * 65, self.log_file)
        log(f"[LOOP ALL {self.total_gens} GENERATIONS COMPLETED] 全部 {self.total_gens} 代迭代完毕！", self.log_file)
        log("=" * 65, self.log_file)


def main():
    parser = argparse.ArgumentParser(description="Autonomous Multi-Gen RL Loop for SSM")
    parser.add_argument("--generations", type=int, default=4, help="迭代代数 (默认 4 代)")
    parser.add_argument("--games", type=int, default=3000, help="每代自对弈局数 (默认 3000)")
    parser.add_argument("--out", default=os.path.join(IMPORT_ROOT, "SSM/runs/loop_stage_b"), help="产物输出主目录")
    parser.add_argument("--champion", default=os.path.join(IMPORT_ROOT, "SSM/runs/champion.pt"), help="初始冠军检查点")
    parser.add_argument("--opp", default=os.path.join(IMPORT_ROOT, "SSM/runs/champion_gen3.pt"), help="历史对手池检查点")
    parser.add_argument("--workers", type=int, default=8, help="自对弈 Worker 数量 (默认 8)")
    parser.add_argument("--concurrency", type=int, default=20, help="单 Worker 并发 (默认 20)")
    parser.add_argument("--arena-pairs", type=int, default=ARENA_PAIRS,
                        help="Arena 开对数（×2 = 总局数，默认 %(default)s = 1200 局）。"
                             "必须 ≤ 开局库规模，否则开局循环复用会产生逐字节重复棋谱，"
                             "而 elo_ci95 不按 duplicate_rate 校正 ⇒ CI 虚窄")
    parser.add_argument("--openings", default=OPENINGS_FILE,
                        help="开局库（自对弈与 Arena 共用，默认 %(default)s）")
    parser.add_argument("--value-target", default="mc", choices=["mc", "lambda"],
                        help="价值学习目标：mc=终局 z 硬标签（原口径）；lambda=TD(λ) 软目标（D2/D4）")
    parser.add_argument("--value-lambda-end", type=float, default=1.0,
                        help="lambda 目标的退火终点（默认 1.0 = 与 mc 等价；建议 0.5）")
    parser.add_argument("--value-anchor-weight", type=float, default=0.0,
                        help="终局锚的常数权重 β（默认 0 = 旧行为；防价值尺度代际坍缩，建议 0.5）")
    parser.add_argument("--record-qsearch", action="store_true", default=False,
                        help="自对弈额外落逐 ply 搜索价值 q̂（D3；lambda 目标的前置条件）")
    args = parser.parse_args()

    orchestrator = SsmLoopOrchestrator(
        out_dir=Path(args.out).resolve(),
        total_gens=args.generations,
        games_per_gen=args.games,
        initial_champion=args.champion,
        opp_checkpoint=args.opp,
        workers=args.workers,
        concurrency=args.concurrency,
        arena_pairs=args.arena_pairs,
        openings=args.openings,
        value_target=args.value_target,
        value_lambda_end=args.value_lambda_end,
        value_anchor_weight=args.value_anchor_weight,
        record_qsearch=args.record_qsearch,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()
