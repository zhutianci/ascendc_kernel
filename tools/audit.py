# -*- coding: utf-8 -*-
"""Static audit of the submission sources.  Everything here is checkable without an NPU,
so it runs before any device time is spent.

Checks, per file:
  1. it parses as Python;
  2. nothing at module level would be silently discarded by the evaluator's AST filter
     (it keeps imports / defs / classes / literal assignments and throws the rest away —
     a dropped assignment becomes a NameError at run time, which is how the official
     reference file itself fails to load);
  3. every self._ext.<name> the model calls is actually bound in PYBIND11_MODULE, and
     nothing is bound that is never called;
  4. the embedded kernel copy is byte-identical to the standalone .asc next to it;
  5. no leftover experiment tags, and the required entry points exist.
"""
import ast
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "src"))

FILES = {
    "task1_sparse_attn.py": {"_KERNEL_MM_SRC_EMBED": "sparse_attn_kernel_mm.asc"},
    "task2_indexer.py": {"_KERNEL_MM_SRC_EMBED": "indexer_kernel_mm.asc",
                         "_KERNEL_AIV_SRC_EMBED": "indexer_kernel_aiv.asc"},
    "task3_sinkhorn.py": {"_KERNEL_SRC_EMBED": "sinkhorn_kernel.asc"},
}
LITERAL = (ast.Constant, ast.Tuple, ast.List, ast.Dict, ast.Set, ast.UnaryOp)


def audit(py, embeds):
    ok = True
    path = os.path.join(SRC, py)
    text = io.open(path, encoding="utf-8").read()
    print("=" * 78)
    print("%s  (%d bytes)" % (py, len(text)))

    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        print("  FAIL syntax: %s" % e)
        return False
    print("  ok   parses as python")

    dropped = []
    for n in tree.body:
        if isinstance(n, ast.Assign) and not isinstance(n.value, LITERAL):
            dropped.append(getattr(n.targets[0], "id", "?"))
    if dropped:
        print("  FAIL module-level assigns the AST filter would DROP: %s" % dropped)
        ok = False
    else:
        print("  ok   no module-level assign would be dropped by the AST filter")

    called = set(re.findall(r"self\._ext\.(\w+)", text))
    bound = set(re.findall(r'm\.def\("(\w+)"', text)) | set(re.findall(r'm\.attr\("(\w+)"\)', text))
    missing = called - bound
    unused = bound - called
    if missing:
        print("  FAIL called but not bound: %s" % sorted(missing))
        ok = False
    else:
        print("  ok   all %d called entry points are bound" % len(called))
    if unused:
        print("  WARN bound but never called from python: %s" % sorted(unused))

    for var, asc in embeds.items():
        head = var + " = r'''"
        i = text.index(head) + len(head)
        j = text.index("'''", i)
        disk = io.open(os.path.join(SRC, asc), encoding="utf-8").read()
        if text[i:j] == disk:
            print("  ok   %s matches %s byte for byte" % (var, asc))
        else:
            print("  FAIL %s differs from %s" % (var, asc))
            ok = False

    tags = re.findall(r"\[E\d+|\[v\d+", text)
    if tags:
        print("  FAIL %d leftover experiment tags: %s" % (len(tags), sorted(set(tags))[:8]))
        ok = False
    else:
        print("  ok   no leftover experiment tags")

    for need in ("class ModelNew", "def get_inputs", "def get_init_inputs"):
        if need not in text:
            print("  FAIL missing %s" % need)
            ok = False
    print("  ok   ModelNew / get_inputs / get_init_inputs present")
    return ok


def main():
    good = all([audit(py, e) for py, e in FILES.items()])
    print("=" * 78)
    print("AUDIT %s" % ("PASS" if good else "FAIL"))
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
