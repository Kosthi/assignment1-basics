"""
运行期通用工具（device / dtype / 数据加载）。

这个模块刻意保持轻量，便于被训练、生成、数据管线复用，避免循环依赖。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch


def auto_device(requested: str | None) -> str:
    """把用户输入的 device 规范化为可用的运行设备。"""

    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_numpy_dtype(name: str) -> np.dtype:
    """解析 dtype 名称并给出统一的错误信息。"""

    try:
        return np.dtype(name)
    except TypeError as e:
        raise ValueError(f"无效 dtype: {name}") from e


def open_tokens(path: str | os.PathLike, dtype: np.dtype) -> np.ndarray:
    """以 memmap 的方式打开 token 数据文件（.bin / .npy）。"""

    p = Path(path)
    if p.suffix == ".bin":
        return np.memmap(p, mode="r", dtype=dtype)
    if p.suffix == ".npy":
        arr = np.load(p, mmap_mode="r")
        if arr.dtype != dtype:
            arr = arr.astype(dtype, copy=False)
        return arr
    raise ValueError(f"不支持的数据文件后缀: {p.suffix}（仅支持 .bin / .npy）")
