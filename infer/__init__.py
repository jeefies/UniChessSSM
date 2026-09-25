"""推理路径：GPU 状态槽 / GPU 服务进程 / 原地单步。

- ``fast_eval.py``：推理黄金口径（GPU 状态槽 + 原地单步 + CUDA graph）；
- ``gpu_server.py``：/dev/shm + FIFO 跨进程拼批的 GPU 服务进程；
- ``ssm_update.py``：mamba 内核读父槽写子槽。

arena 与自对弈都走这里；逐位可复现的口径见 ``SSM/AGENTS.md``。注意 ``fast_eval`` /
``gpu_server`` 都 import torch 与 mamba-ssm，只想拿常量就别碰本包。
"""
