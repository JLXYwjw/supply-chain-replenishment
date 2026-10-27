#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
仓库质量检查门 —— 基础可读版（quality_check_simple.py）
=====================================================
这个文件做的事，和原版 quality_check.py 完全一样：
  1. 英文名文档（.md/.html）里不能有任何中文（CJK）字符
  2. 中文名文档（文件名含中文或 .zh-CN 后缀）里必须有中文
  3. README/index 里引用的 assets/ 图片文件必须真实存在
  4. 双语版 README + index 四个文件都必须存在

全部通过打印 QUALITY OK 并退出 0；任何不通过打印 QUALITY FAIL 和错误列表，退出 1。

区别只是：原版用正则表达式做 CJK 检测和 assets 引用提取，
这里改成了普通循环（逐字符判断 Unicode 范围、逐段扫描文件名），
行为完全一致。

运行方式：
    python scripts/quality_check_simple.py [仓库根目录]
    不带参数则检查当前目录。
"""

import subprocess
import sys
from pathlib import Path

# 合法的图片扩展名
ASSET_EXTS = ["png", "jpg", "jpeg", "gif", "svg"]

# 必须存在的四个双语文件
REQUIRED = ["README.md", "README.zh-CN.md", "index.html", "index.zh-CN.html"]


def is_cjk_char(ch):
    """判断一个字符是不是中文（CJK 统一表意文字，U+4E00~U+9FFF）。"""
    code = ord(ch)
    return 0x4E00 <= code <= 0x9FFF


def count_cjk(s):
    """数字符串里有多少个中文字符。"""
    n = 0
    for ch in s:
        if is_cjk_char(ch):
            n = n + 1
    return n


def is_zh_name(name):
    """判断文件名是不是"中文名"：文件名里有中文，或带 .zh-CN 后缀。"""
    b = Path(name).name
    # 逐字符看文件名里有没有中文
    for ch in b:
        if is_cjk_char(ch):
            return True
    # 没有中文但有 .zh-CN 后缀也算
    if ".zh-CN" in b:
        return True
    return False


def find_asset_refs(text):
    r"""从文本里找出所有 assets/xxx.png|jpg|... 形式的引用。

    原版用正则 assets/[word/dash/dot]+.(png|jpg|jpeg|gif|svg)，
    这里改成普通扫描：找 "assets/" 开头，往后读连续的文件名字符
    （字母、数字、下划线、点、横线），再看末尾扩展名是否合法。
    """
    refs = []
    pos = 0
    while True:
        idx = text.find("assets/", pos)
        if idx == -1:
            break
        # 从 "assets/" 之后开始读文件名
        start = idx + len("assets/")
        end = start
        while end < len(text) and (text[end].isalnum() or text[end] in "._-"):
            end = end + 1
        fname = text[start:end]
        # 看文件名最后一个点后面是不是合法图片扩展名
        dot = fname.rfind(".")
        if dot > 0:
            ext = fname[dot + 1:]
            if ext in ASSET_EXTS:
                refs.append(text[idx:end])
        # 从下一个位置继续找
        pos = idx + 1
    return refs


def main():
    # 仓库根目录：命令行第一个参数，或默认当前目录
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

    errors = []

    # ---- 1/2. 文档语言一致性检查 ----
    # 只扫描 git 跟踪的 .md/.html 文件（git ls-files），
    # 这样检查的正好是会发布出去的文件。
    r = subprocess.run(
        ["git", "ls-files", "*.md", "*.html"],
        cwd=str(root), capture_output=True, text=True, encoding="utf-8",
    )
    doc_files = [root / f for f in r.stdout.splitlines() if f]

    seen = set()
    for fp in doc_files:
        p = fp.resolve()
        # 跳过重复文件和隐藏文件
        if p in seen or fp.name.startswith("."):
            continue
        seen.add(p)
        # 读文件内容（读不了就跳过）
        try:
            text = fp.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        n = count_cjk(text)
        if is_zh_name(fp.name):
            # 中文名文档：必须有中文
            if n == 0:
                errors.append(f"[{fp.name}] Chinese-named doc has no Chinese content")
        else:
            # 英文名文档：不能有中文
            if n > 0:
                errors.append(f"[{fp.name}] English-named doc contains {n} CJK chars")

    # ---- 3. README/index 里引用的 assets 文件必须存在 ----
    for fname in REQUIRED:
        fp = root / fname
        if not fp.is_file():
            errors.append(f"missing {fname}")
            continue
        text = fp.read_text(encoding="utf-8", errors="replace")
        for ref in find_asset_refs(text):
            if not (root / ref).is_file():
                errors.append(f"missing asset referenced in {fname}: {ref}")

    # ---- 4. 四个双语文件都必须存在 ----
    for fname in REQUIRED:
        if not (root / fname).is_file():
            errors.append(f"missing {fname}")

    # ---- 输出结果 ----
    if errors:
        print("QUALITY FAIL")
        for e in sorted(set(errors)):
            print(" -", e)
        return 1
    print("QUALITY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
