# -*- coding: utf-8 -*-
"""在昇腾机上执行官方 auto_bench.py 的薄包装。

只做一件事：先 import torch_npu 和 transfer_to_npu，再以 __main__ 方式执行官方评测脚本。
赛题 v0/v1 的 get_inputs() 里有 .cuda() 调用，而 auto_bench.py 自身只 import torch；
如果评测机的 torch 不自动加载 torch_npu（或没启用 .cuda -> npu 映射），
直接跑会在 get_inputs 处就失败。这个包装消除该环境差异，不改动任何评测逻辑。

用法（除第一个参数外与 auto_bench.py 完全一致）：
    python3 run_eval.py <auto_bench.py 路径> --v0_file <参考>.py --v1_file <我方>.py
"""
import runpy
import sys

try:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except Exception as e:                                    # pragma: no cover
    print("[run_eval] 警告: torch_npu / transfer_to_npu 不可用: %s" % e)

if len(sys.argv) < 2:
    raise SystemExit("用法: python3 run_eval.py <auto_bench.py> [auto_bench 的参数...]")

bench = sys.argv[1]
sys.argv = ["auto_bench.py"] + sys.argv[2:]
runpy.run_path(bench, run_name="__main__")
