"""UniChess Server 适配层：把 S 包装成服务端要求的 GameEngine。

把整个仓库目录（或指向它的符号链接）放进 ``Server/models/<name>/``，
服务端发现后按本文件契约驱动它。**实现就是 ``Kit.serving.make_game_engine``**：
S 的 Player（``SsmPlayer``，Gumbel 搜索）在服务端本来就是 kit Player 的一层壳，
6 个方法（终局判定、白方视角 eval、悔棋重放）都在 kit 里实现。

与 T / R 的差别只有一处：S 自带两套推理后端（``engine="fast"`` 同进程 GPU 槽池 +
CUDA graph；``engine="server"`` 连共享 GPU 服务进程跨进程拼批）。默认 ``fast``
单进程够用；批量对弈 / 多进程 arena 要压吞吐时，服务端用 preset 覆盖成 ``server``。

加载方式带来的两条约束（`Server/models/__init__.py`）：

* 服务端用 ``spec_from_file_location`` + ``exec_module`` 加载本文件，模型目录**不会**进
  ``sys.path``，所以这里自己挂上**仓库根目录**，之后才能 ``import SSM``——包名与目录名
  相同，远端是 ``~/UniChess`` 下的一个普通顶层包，没有同名冲突。
* `/api/models` 只为上报状态就会 import 本模块，因此 torch / numpy 一律延迟到
  ``_factory`` 被第一次调用时。
"""
from __future__ import annotations

import sys
from pathlib import Path

SSM_ROOT = Path(__file__).resolve().parent
# import 根是仓库根目录的父目录（~/UniChess）；只追加，不插到最前——
# 插到最前会遮蔽服务端自己的包（Server 的 models 也是这么被发现的）。
_IMPORT_ROOT = str(SSM_ROOT.parent)
if _IMPORT_ROOT not in sys.path:
    sys.path.append(_IMPORT_ROOT)

from Kit.serving import make_game_engine  # noqa: E402


def factory(**kwargs):
    """延迟导入：服务端上报模型状态时只 import 本文件，不该吃下 torch 的导入耗时。"""
    from SSM.kit import make_player_factory
    return make_player_factory(**kwargs)


# 批量对弈 / 观战走 kit 原生 Player（跨局攒批），以 preset=<config.json 预设名> 调用
KIT_FACTORY = "SSM.kit:make_player_factory"

GameEngine = make_game_engine(factory, name="SsmEngine", kit_factory=KIT_FACTORY)
