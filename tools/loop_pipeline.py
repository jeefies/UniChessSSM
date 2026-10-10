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


class SsmLoopOrchestrator:
    def __init__(self, out_dir: Path, total_gens: int, games_per_gen: int,
                 initial_champion: str, opp_checkpoint: str,
                 workers: int = 8, concurrency: int = 20,
                 python_bin: str = sys.executable):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.total_gens = total_gens
        self.games_per_gen = games_per_gen
        self.initial_champion = os.path.abspath(initial_champion)
        self.opp_checkpoint = os.path.abspath(opp_checkpoint) if opp_checkpoint else None
        self.workers = workers
        self.concurrency = concurrency
        self.python_bin = python_bin

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
            "--first-game", "0",
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
            "--min-book-plies", "6",
            "--pcr-rate", "0.5",
            "--pcr-fast-sims", "16",
            "--adaptive-sims",
            "--deep-sims", "256",
            "--p-explore", "0.15",
            "--precision", "tf32"
        ]
        if opp_pt and os.path.exists(opp_pt):
            cmd.extend(["--opp-ckpt", opp_pt])

        rc = self.run_command(cmd, phase_log)
        if rc != 0 or not manifest_path.exists():
            raise RuntimeError(f"Gen {gen_id} 自对弈生成失败，退出码={rc}，详见 {phase_log}")

        log(f"Gen {gen_id} 自对弈生成完毕！", self.log_file)
        return sp_out

    def step_train(self, gen_id: int, gen_dir: Path, sp_dir: Path, champion_pt: str) -> Path:
        train_out = gen_dir / "train"
        best_pt = train_out / "best.pt"
        phase_log = gen_dir / "train.log"

        if best_pt.exists():
            log(f"Gen {gen_id} 训练产物已存在 ({best_pt})，跳过训练阶段", self.log_file)
            return best_pt

        log(f"Gen {gen_id} 开始混合强化训练 (85% 自博弈 + 15% 人类谱)...", self.log_file)
        cfg_path = gen_dir / "train_config.json"
        cfg_data = {
            "task": {
                "factory": "SSM.tasks:make_task",
                "kwargs": {
                    "kind": "stage_b2",
                    "data": {"dir": os.path.join(IMPORT_ROOT, "SSM/data/shards"), "t_max": 300},
                    "selfplay": {"dir": str(sp_dir)},
                    "ckpt": champion_pt,
                    "microbatch": 4,
                    "weights": {"w_v": 1.0, "w_r_start": 0.1, "w_r_end": 0.1},
                    "workers": 4,
                    "val_batches": 8,
                    "w_selfplay": 0.85,
                    "w_human": 0.15,
                    "mlh_log": True,
                    "kl_reweight": True,
                    "t_max": 300
                }
            },
            "out": str(train_out),
            "steps": 0,
            "accum": 16,
            "seed": gen_id,
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
                "pairs": 250,
                "concurrency": 4,
                "max_plies": 300,
                "openings": os.path.join(IMPORT_ROOT, "SSM/data/openings_200.txt"),
                "seed": 20261000 + gen_id * 100,
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

            # 2. 混合训练
            t0 = time.time()
            cand_pt = self.step_train(gen_id, gen_dir, sp_dir, current_champ)
            t_train = time.time() - t0

            # 3. 双轨评测
            t0 = time.time()
            raw_summary = self.step_eval_raw_policy(gen_id, gen_dir, str(cand_pt), current_champ)
            arena_summary = self.step_arena(gen_id, gen_dir, str(cand_pt), current_champ)
            t_arena = time.time() - t0

            # 4. 裁决与换代
            sprt_verdict = arena_summary.get("sprt", {}).get("verdict")
            score_a = arena_summary.get("score_a", 0.0)
            promoted = (sprt_verdict == "H1") or (score_a >= 0.55)

            champion_before = current_champ
            if promoted:
                # 归档备份原 champion
                backup_champ = self.out_dir / f"champion_gen{gen_id - 1}_backup.pt"
                shutil.copy2(current_champ, backup_champ)
                # 晋升候选模型
                shutil.copy2(cand_pt, current_champ)
                self.state["opp_champion"] = str(backup_champ)
                log(f"🎉【加冕新霸主】Gen {gen_id} 满足换代门限 (SPRT={sprt_verdict}, 得分率={score_a:.2%})，晋升为现役 Champion！", self.log_file)
            else:
                log(f"🛡️【前任守擂】Gen {gen_id} 未突破换代门限 (SPRT={sprt_verdict}, 得分率={score_a:.2%})，现任 Champion 守擂成功。", self.log_file)

            record = {
                "generation": gen_id,
                "promoted": promoted,
                "candidate": str(cand_pt),
                "champion_before": champion_before,
                "champion_after": self.state["champion"],
                "score_a": score_a,
                "elo": arena_summary.get("elo"),
                "sprt": sprt_verdict,
                "games_arena": arena_summary.get("games"),
                "raw_policy": raw_summary,
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
    args = parser.parse_args()

    orchestrator = SsmLoopOrchestrator(
        out_dir=Path(args.out),
        total_gens=args.generations,
        games_per_gen=args.games,
        initial_champion=args.champion,
        opp_checkpoint=args.opp,
        workers=args.workers,
        concurrency=args.concurrency
    )
    orchestrator.run()


if __name__ == "__main__":
    main()
