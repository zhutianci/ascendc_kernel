#!/usr/bin/env bash
# 一键跑完三个算子的官方评测。
#
#   bash run_benchmark.sh                 # 三个算子各跑一轮
#   bash run_benchmark.sh 3               # 三个算子各跑三轮（µs 级差异需要多轮取中位）
#   KS_BENCH=auto_bench.py bash run_benchmark.sh   # 指定官方评测脚本路径
#
# 首次运行时每个算子的 __init__ 会用 bisheng 编译 kernel（约 1-2 分钟），
# 编译产物按源码哈希缓存在 ~/.cache/ks_kernels，之后就直接命中。
# 编译发生在 __init__ 内，不计入评测的计时区间。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

ROUNDS="${1:-1}"
BENCH="${KS_BENCH:-auto_bench.py}"

if [ ! -f "$BENCH" ]; then
  echo "找不到官方评测脚本 $BENCH"
  echo "请把 auto_bench.py 放到本目录，或用 KS_BENCH=/path/to/auto_bench.py 指定。"
  echo "来源：https://github.com/DeepLink-org/DLBlas/blob/main/benchmarks/ks/auto_bench.py"
  exit 1
fi

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

# tools/run_eval.py 只做一件事：先 import torch_npu 和 transfer_to_npu，再以 __main__
# 执行官方 auto_bench.py。赛题 v0/v1 的 get_inputs 里有 .cuda()，需要这个映射才能跑起来。
run_one () {
  local name="$1" v0="$2" v1="$3"
  printf '%-14s ' "$name"
  python3 tools/run_eval.py "$BENCH" --v0_file "$v0" --v1_file "$v1" 2>&1 \
    | grep -E "PASS|FAIL" | head -1
}

echo "=== KernelSwift 三算子评测（$ROUNDS 轮）==="
echo "评测脚本: $BENCH"
python3 - <<'PY'
import torch, torch_npu
print("torch %s / torch_npu %s" % (torch.__version__, torch_npu.__version__))
PY
echo ""
for r in $(seq 1 "$ROUNDS"); do
  echo "-- round $r --"
  run_one "Task1 attn"  reference/task1_v0_ref.py src/task1_sparse_attn.py
  run_one "Task2 index"  reference/task2_v0_ref.py src/task2_indexer.py
  run_one "Task3 sink"   reference/task3_v0_ref.py src/task3_sinkhorn.py
done

echo ""
echo "done."
