# -*- coding: utf-8 -*-
"""Splice each .asc kernel source into its submission .py as the embedded fallback copy,
then assert the two are byte-identical.

Why this exists: `_load_asc()` prefers the standalone .asc when it sits next to the .py and
falls back to the embedded string otherwise.  Editing only one of the two is silent — the
build just keeps using the other one.  So the embedded copy is never hand-maintained; it is
always generated from the .asc by this script, and the round-trip is checked.

Usage:  python sync_embed.py            # write + verify
        python sync_embed.py --check    # verify only, non-zero exit on drift
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "src"))

# (python file, [(placeholder-or-embed-var, asc file)])
TARGETS = [
    ("task1_sparse_attn.py", [("_KERNEL_MM_SRC_EMBED", "sparse_attn_kernel_mm.asc")]),
    ("task2_indexer.py", [("_KERNEL_MM_SRC_EMBED", "indexer_kernel_mm.asc"),
                          ("_KERNEL_AIV_SRC_EMBED", "indexer_kernel_aiv.asc")]),
    ("task3_sinkhorn.py", [("_KERNEL_SRC_EMBED", "sinkhorn_kernel.asc")]),
]


def embed_span(text, var):
    """Return (start, end) of the raw-string body for `var = r'''...'''`."""
    head = var + " = r'''"
    i = text.index(head) + len(head)
    j = text.index("'''", i)
    return i, j


def main():
    check_only = "--check" in sys.argv
    bad = 0
    for py, pairs in TARGETS:
        ppath = os.path.join(SRC, py)
        if not os.path.isfile(ppath):
            print("SKIP  %s (not built yet)" % py)
            continue
        text = io.open(ppath, encoding="utf-8").read()
        for var, asc in pairs:
            apath = os.path.join(SRC, asc)
            src = io.open(apath, encoding="utf-8").read()
            assert "'''" not in src, "%s contains a triple quote and cannot be embedded" % asc
            i, j = embed_span(text, var)
            cur = text[i:j]
            if cur == src:
                print("OK    %-22s <- %s (%d bytes, identical)" % (var, asc, len(src)))
                continue
            if check_only:
                print("DRIFT %-22s != %s" % (var, asc))
                bad = 1
                continue
            text = text[:i] + src + text[j:]
            print("SYNC  %-22s <- %s (%d bytes)" % (var, asc, len(src)))
        if not check_only:
            io.open(ppath, "w", encoding="utf-8", newline="\n").write(text)
            # round-trip check
            text2 = io.open(ppath, encoding="utf-8").read()
            for var, asc in pairs:
                i, j = embed_span(text2, var)
                src = io.open(os.path.join(SRC, asc), encoding="utf-8").read()
                if text2[i:j] != src:
                    print("FAIL  round-trip mismatch for %s in %s" % (var, py))
                    bad = 1
    return bad


if __name__ == "__main__":
    sys.exit(main())
