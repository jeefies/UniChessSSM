#!/usr/bin/env bash
# SSM UCI 纯 policy 模式启动器：不搜索，go 直接选根局面 policy 概率最高的合法着。
# 其余配置（remote/socket/mcts 环境）与 tools/ssm_uci.sh 完全一致。
export UNICHESS_PURE_POLICY=1
exec "$(dirname "$0")/ssm_uci.sh" "$@"
