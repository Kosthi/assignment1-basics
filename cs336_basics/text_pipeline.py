"""
从原始文本到训练数据的管线（BPE 训练 + 编码）。

职责：
1) 使用 C++ 扩展训练 BPE tokenizer；
2) 把文本编码为 token id 流并落盘为 .bin；
3) 为 train-from-text 子命令组织上述步骤并转交训练循环。
"""

from __future__ import annotations

import argparse
import importlib
import multiprocessing
import os
import queue
import sys
import time
from collections.abc import Iterable
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import numpy as np

from .BPETokenizer import BPETokenizer
from .pretokenization import get_word_counts_parallel
from .runtime import parse_numpy_dtype

_ENCODE_WORKER_TOKENIZER: BPETokenizer | None = None


def _encode_worker_init(tokenizer: BPETokenizer) -> None:
    global _ENCODE_WORKER_TOKENIZER
    _ENCODE_WORKER_TOKENIZER = tokenizer


def _encode_worker(text: str) -> list[int]:
    if _ENCODE_WORKER_TOKENIZER is None:
        raise RuntimeError("worker tokenizer 未初始化")
    return _ENCODE_WORKER_TOKENIZER.encode(text)


def _encode_shm_worker_loop(
    task_q: Any,
    done_q: Any,
    *,
    tokenizer: BPETokenizer,
    shm_name: str,
    dtype_str: str,
    slot_tokens: int,
    num_slots: int,
    max_value: int | None,
) -> None:
    # 连接到共享内存（不是创建，是连接已存在的）
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        dtype = np.dtype(dtype_str)
        # 映射为numpy数组，与主进程看到的是同一块内存
        slots = np.ndarray((num_slots, slot_tokens), dtype=dtype, buffer=shm.buf)
        while True:
            # 从任务队列获取任务
            item = task_q.get()
            # None是停止信号
            if item is None:
                return
            seq, text, slot_id = item
            try:
                token_ids = tokenizer.encode(text)
                if max_value is not None:
                    bad_tok = None
                    for tok in token_ids:
                        if tok > max_value:
                            bad_tok = tok
                            break
                    if bad_tok is not None:
                        done_q.put(
                            (seq, slot_id, 0, f"token id {bad_tok} 超出 dtype={dtype} 可表示范围（max={max_value}）")
                        )
                        continue

                length = len(token_ids)
                if length > slot_tokens:
                    done_q.put((seq, slot_id, 0, f"片段 token 数 {length} 超过 slot_tokens={slot_tokens}"))
                    continue
                slots[slot_id, :length] = np.asarray(token_ids, dtype=dtype)
                done_q.put((seq, slot_id, length, None))
            except Exception as e:
                done_q.put((seq, slot_id, 0, str(e)))
    finally:
        shm.close()


def load_bpe_tokenizer(vocab_path: str | os.PathLike, merges_path: str | os.PathLike, special_tokens: list[str]):
    """从训练产物（vocab.json / merges.txt）重建 BPETokenizer。"""

    import json

    with open(vocab_path, encoding="utf-8") as f:
        vocab_json: dict[str, str] = json.load(f)
    vocab: dict[int, bytes] = {int(i): bytes.fromhex(hex_str) for i, hex_str in vocab_json.items()}
    merges: list[tuple[bytes, bytes]] = []
    with open(merges_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a_hex, b_hex = line.split()
            merges.append((bytes.fromhex(a_hex), bytes.fromhex(b_hex)))
    return BPETokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def _import_cpp_bpe() -> Any:
    project_root = Path(__file__).resolve().parents[1]
    cpp_build_dir = project_root / "cpp" / "build"
    if cpp_build_dir.exists():
        build_dir_str = str(cpp_build_dir)
        if build_dir_str not in sys.path:
            sys.path.append(build_dir_str)
    try:
        return importlib.import_module("bpe")
    except ImportError as e:
        raise RuntimeError(
            "bpe 模块不可用；请先构建 cpp 扩展（cmake -S cpp -B cpp/build && cmake --build cpp/build）"
        ) from e


def train_bpe_tokenizer(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> BPETokenizer:
    """把原始文本训练成一个 BPETokenizer（核心计算在 C++ 扩展中）。"""

    bpe = _import_cpp_bpe()
    train_fn = getattr(bpe, "train", None)
    if train_fn is None:
        raise RuntimeError("bpe.train 不可用；请先构建 cpp 扩展")

    cpu_cores = min(multiprocessing.cpu_count(), 24)
    total_word_counts = get_word_counts_parallel(str(input_path), special_tokens, cpu_cores)
    sorted_items = sorted(total_word_counts.items(), key=lambda x: (-x[1], x[0]))
    distinct_words = [w for w, _ in sorted_items]
    counts = [c for _, c in sorted_items]

    result = train_fn(distinct_words, counts, int(vocab_size), list(special_tokens))
    return BPETokenizer(vocab=result.vocab, merges=result.merges, special_tokens=special_tokens)


def _iter_text_chunks(path: str | os.PathLike, chunk_bytes: int) -> Iterable[str]:
    with open(path, encoding="utf-8", errors="ignore") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                return
            yield chunk


def _dtype_max_value(dtype: np.dtype) -> int | None:
    if dtype.kind not in {"u", "i"}:
        return None
    info = np.iinfo(dtype)
    return int(info.max)


def encode_text_to_bin(
    *,
    tokenizer: BPETokenizer,
    input_path: str | os.PathLike,
    output_path: str | os.PathLike,
    dtype: np.dtype,
    overwrite: bool,
    chunk_bytes: int = 1 << 22,
    buffer_tokens: int = 50_000_000,
    encode_workers: int = 8,
    encode_backend: str = "pool",
    encode_slot_tokens: int = 0,
    encode_num_slots: int = 0,
    log_interval_sec: float = 5.0,
) -> None:
    """把文本编码成 token ids 并写入 .bin。"""

    out_path = Path(output_path)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"输出文件已存在: {out_path}（传 --overwrite 以覆盖）")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    max_value = _dtype_max_value(dtype)
    total_tokens = 0
    start_time = time.time()
    last_log_time = start_time
    last_log_tokens = 0

    def iter_complete_texts() -> Iterable[str]:
        buffer = ""
        for chunk in _iter_text_chunks(input_path, chunk_bytes=chunk_bytes):
            buffer += chunk
            boundary = tokenizer._find_last_boundary(buffer)
            if boundary > 0:
                yield buffer[:boundary]
                buffer = buffer[boundary:]
        if buffer:
            yield buffer

    with open(out_path, "wb") as out_f:

        def log_progress() -> None:
            nonlocal last_log_time, last_log_tokens
            now = time.time()
            dt = now - last_log_time
            if dt >= log_interval_sec:
                d_tokens = total_tokens - last_log_tokens
                tok_per_s = d_tokens / dt if dt > 0 else float("nan")
                elapsed = now - start_time
                print(
                    f"encode {out_path.name}: tokens={total_tokens} tok/s={tok_per_s:.0f} time={elapsed / 60:.1f}m",
                    flush=True,
                )
                last_log_time = now
                last_log_tokens = total_tokens

        if encode_workers and encode_workers > 1 and encode_backend == "shm":
            ctx = multiprocessing.get_context("fork")
            slot_tokens = encode_slot_tokens if encode_slot_tokens > 0 else max(1_000_000, int(chunk_bytes) * 2)
            num_slots = encode_num_slots if encode_num_slots > 0 else int(encode_workers) * 2
            shm_size = int(num_slots) * int(slot_tokens) * int(dtype.itemsize)
            shm = shared_memory.SharedMemory(create=True, size=shm_size)
            slots = np.ndarray((num_slots, slot_tokens), dtype=dtype, buffer=shm.buf)
            task_q = ctx.Queue(maxsize=num_slots * 2)
            done_q = ctx.Queue(maxsize=num_slots * 2)
            free_q = ctx.Queue()
            for i in range(num_slots):
                free_q.put(i)

            workers: list[multiprocessing.Process] = []
            try:
                for _ in range(int(encode_workers)):
                    p = ctx.Process(
                        target=_encode_shm_worker_loop,
                        args=(task_q, done_q),
                        kwargs={
                            "tokenizer": tokenizer,
                            "shm_name": shm.name,
                            "dtype_str": dtype.str,
                            "slot_tokens": slot_tokens,
                            "num_slots": num_slots,
                            "max_value": max_value,
                        },
                    )
                    p.start()
                    workers.append(p)

                pending: dict[int, tuple[int, int]] = {}
                next_seq = 0
                submitted = 0
                completed = 0

                def process_one_done(*, block: bool) -> None:
                    nonlocal completed, next_seq, total_tokens
                    done_seq, slot_id, length, err = done_q.get(block=block)
                    if err is not None:
                        raise RuntimeError(err)
                    pending[int(done_seq)] = (int(slot_id), int(length))
                    completed += 1
                    while next_seq in pending:
                        slot_id2, length2 = pending.pop(next_seq)
                        out_f.write(slots[slot_id2, :length2].tobytes())
                        total_tokens += length2
                        free_q.put(slot_id2)
                        if (total_tokens & 0x3FFFF) == 0:
                            log_progress()
                        next_seq += 1

                for text in iter_complete_texts():
                    while True:
                        try:
                            slot_id = free_q.get_nowait()
                            break
                        except queue.Empty:
                            process_one_done(block=True)
                    task_q.put((submitted, text, slot_id))
                    submitted += 1

                    while True:
                        try:
                            process_one_done(block=False)
                        except queue.Empty:
                            break

                for _ in workers:
                    task_q.put(None)

                while completed < submitted:
                    process_one_done(block=True)
            finally:
                for p in workers:
                    p.join(timeout=1)
                for p in workers:
                    if p.is_alive():
                        print(f"进程 {p.pid} 仍在运行，强制终止")
                        p.terminate()
                for p in workers:
                    if p.is_alive():
                        print(f"进程 {p.pid} 在 terminate() 后仍未退出")
                        p.join(timeout=1)
                shm.close()
                shm.unlink()
        else:
            buf = np.empty((buffer_tokens,), dtype=dtype)
            n = 0

            def write_tok(tok: int) -> None:
                nonlocal n, total_tokens
                if max_value is not None and tok > max_value:
                    raise ValueError(f"token id {tok} 超出 dtype={dtype} 可表示范围（max={max_value}）")
                buf[n] = tok
                n += 1
                total_tokens += 1
                if n == buf.shape[0]:
                    out_f.write(buf.tobytes())
                    n = 0
                if (total_tokens & 0x3FFFF) == 0:
                    log_progress()

            if encode_workers and encode_workers > 1:
                ctx = multiprocessing.get_context("fork")
                with ctx.Pool(
                    processes=int(encode_workers), initializer=_encode_worker_init, initargs=(tokenizer,)
                ) as pool:
                    for token_ids in pool.imap(_encode_worker, iter_complete_texts(), chunksize=1):
                        for tok in token_ids:
                            write_tok(tok)
            else:
                for tok in tokenizer.encode_iterable(_iter_text_chunks(input_path, chunk_bytes=chunk_bytes)):
                    write_tok(tok)
            if n:
                out_f.write(buf[:n].tobytes())
    elapsed = time.time() - start_time
    tok_per_s = total_tokens / elapsed if elapsed > 0 else float("nan")
    print(
        f"encode done {out_path.name}: tokens={total_tokens} tok/s={tok_per_s:.0f} time={elapsed / 60:.1f}m", flush=True
    )


def train_from_text(args: argparse.Namespace) -> None:
    """train-from-text 子命令：训练 BPE，编码数据，然后调用训练循环。"""

    from .train import train

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = out_dir / "vocab.json"
    merges_path = out_dir / "merges.txt"
    train_bin = out_dir / "train.bin"
    valid_bin = out_dir / "valid.bin"

    if (vocab_path.exists() or merges_path.exists()) and not args.overwrite:
        raise FileExistsError(f"{vocab_path} 或 {merges_path} 已存在（传 --overwrite 以覆盖）")

    dtype = parse_numpy_dtype(args.data_dtype)

    print(f"stage=bpe_train vocab_size={args.vocab_size}", flush=True)
    t0 = time.time()
    tokenizer = train_bpe_tokenizer(
        input_path=args.train_text,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
    )
    t1 = time.time()
    print(f"stage=bpe_train_done time={t1 - t0:.1f}s", flush=True)

    tokenizer.save(str(vocab_path), str(merges_path))
    print(f"stage=tokenizer_saved vocab={vocab_path.name} merges={merges_path.name}", flush=True)
    actual_vocab_size = tokenizer.get_vocab_size()

    print(f"stage=encode_train path={Path(args.train_text).name}", flush=True)
    encode_text_to_bin(
        tokenizer=tokenizer,
        input_path=args.train_text,
        output_path=train_bin,
        dtype=dtype,
        overwrite=args.overwrite,
        encode_workers=args.encode_workers,
        encode_backend=args.encode_backend,
        encode_slot_tokens=args.encode_slot_tokens,
        encode_num_slots=args.encode_num_slots,
    )
    print(f"stage=encode_valid path={Path(args.valid_text).name}", flush=True)
    encode_text_to_bin(
        tokenizer=tokenizer,
        input_path=args.valid_text,
        output_path=valid_bin,
        dtype=dtype,
        overwrite=args.overwrite,
        encode_workers=args.encode_workers,
        encode_backend=args.encode_backend,
        encode_slot_tokens=args.encode_slot_tokens,
        encode_num_slots=args.encode_num_slots,
    )

    print("stage=train", flush=True)
    train_args = argparse.Namespace(**vars(args))
    train_args.vocab_size = actual_vocab_size
    train_args.train_data = str(train_bin)
    train_args.valid_data = str(valid_bin)
    train(train_args)
