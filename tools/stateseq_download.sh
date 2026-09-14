#!/usr/bin/env bash
# Stage A 数据下载：三个月度文件各取前 4 GB 前缀（zstd 截断流可直接解压）。
# 双路径：8 路直连 + 8 路经 Windows 代理（172.16.1.55:7897），实测聚合 ~305 KB/s。
# 用法：nohup bash tools/stateseq_download.sh > data/download.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
mkdir -p data/raw
PROXY="http://172.16.1.55:7897"
CHUNK_BYTES=$((256 * 1024 * 1024))   # 256 MB × 16 = 4 GB/月
NCHUNKS=16
MONTHS="2026-08 2026-07 2026-06"

fetch_chunk() { # $1=month $2=idx
  local month=$1 i=$2
  local s=$((i * CHUNK_BYTES)) e=$((s + CHUNK_BYTES - 1))
  local out="data/raw/${month}.chunk$(printf %02d "$i")"
  local tries=0
  while [ $tries -lt 30 ]; do
    if [ $i -lt 8 ]; then
      curl -s --max-time 7200 -r "${s}-${e}" -o "$out" \
        "https://database.lichess.org/standard/lichess_db_standard_rated_${month}.pgn.zst" && true
    else
      curl -s --max-time 7200 -x "$PROXY" -r "${s}-${e}" -o "$out" \
        "https://database.lichess.org/standard/lichess_db_standard_rated_${month}.pgn.zst" && true
    fi
    local sz
    sz=$(stat -c%s "$out" 2>/dev/null || echo 0)
    if [ "$sz" -eq "$CHUNK_BYTES" ]; then return 0; fi
    tries=$((tries + 1))
    sleep 5
  done
  echo "CHUNK_FAIL $month $i (size=$sz)" >&2
  return 1
}

for month in $MONTHS; do
  final="data/raw/lichess_standard_${month}.pgn.zst"
  if [ -f "$final.done" ]; then echo "SKIP $month (done)"; continue; fi
  echo "== $month 开始 $(date)"
  pids=""
  for i in $(seq 0 $((NCHUNKS - 1))); do
    fetch_chunk "$month" "$i" &
    pids="$pids $!"
  done
  fail=0
  for p in $pids; do wait "$p" || fail=1; done
  if [ $fail -ne 0 ]; then echo "MONTH_FAIL $month"; exit 1; fi
  cat $(for i in $(seq 0 $((NCHUNKS - 1))); do echo "data/raw/${month}.chunk$(printf %02d "$i")"; done) > "$final"
  rm -f data/raw/"${month}".chunk*
  touch "$final.done"
  echo "== $month 完成 $(date) size=$(stat -c%s "$final")"
done
echo "ALL_MONTHS_DONE"
