#!/usr/bin/env bash
# SSM UCI 屏蔽 value 模式启动器：照常 MCTS 搜索，但叶节点 wdl 改为均匀 (1/3,1/3,1/3)。
# 其余配置（remote/socket/mcts 环境）与 tools/ssm_uci.sh 完全一致。
export UNICHESS_NEUTRAL_WDL=1
exec "$(dirname "$0")/ssm_uci.sh" "$@"
