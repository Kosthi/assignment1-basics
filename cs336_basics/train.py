"""
训练循环（train 子命令）。

顺序阅读建议：
1) 数据读取与 batch 采样：get_batch / eval_loss
2) 优化器与学习率：build_optimizer / set_lr
3) 训练主循环：train
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .cross_entropy import cross_entropy
from .optimizer.adamw import AdamW
from .optimizer.lr_cosine_schedule import lr_cosine_schedule
from .optimizer.sgd import SGD
from .runtime import auto_device, open_tokens, parse_numpy_dtype
from .transformer_lm import TransformerLM
from .utils.checkpointing import load_checkpoint, save_checkpoint
from .utils.dataloader import DataLoader
from .utils.gradient_clipping import gradient_clipping


def loss_per_token(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """把 (B,T,V) logits 与 (B,T) targets 转成单 token 的 CE loss。"""

    vocab_size = logits.shape[-1]
    loss = cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
    return loss


@torch.no_grad()
def eval_loss(
    model: torch.nn.Module,
    dataloader: DataLoader,
    eval_iters: int,
    rng: np.random.Generator,
) -> float:
    """多次采样评估 loss，返回均值。"""

    model.eval()
    losses: list[float] = []
    for _ in range(eval_iters):
        seed = int(rng.integers(0, 2**31 - 1))
        x, y = dataloader.get_batch(seed=seed)
        logits = model(x)
        loss = loss_per_token(logits, y)
        losses.append(loss.detach().float().cpu().item())
    return float(np.mean(losses)) if losses else float("nan")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """把一条结构化记录追加到 jsonl。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def format_duration(seconds: float) -> str:
    """把秒数格式化为易读的 ETA。"""

    if not np.isfinite(seconds) or seconds < 0:
        return "?"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """原地更新 optimizer 的 learning rate。"""

    for group in optimizer.param_groups:
        group["lr"] = lr


def build_optimizer(name: str, params: Iterable[torch.nn.Parameter], args: argparse.Namespace) -> torch.optim.Optimizer:
    """根据 args 创建 optimizer。"""

    if name == "adamw":
        return AdamW(
            params,
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
        )
    if name == "sgd":
        return SGD(params, lr=args.learning_rate)
    raise ValueError(f"未知 optimizer: {name}")


def train(args: argparse.Namespace) -> None:
    """train 子命令入口：构建模型、优化器与训练循环。"""

    device = auto_device(args.device)
    if device.startswith("cuda") and args.matmul_precision:
        torch.set_float32_matmul_precision(args.matmul_precision)

    if args.cosine_cycle_iters is None:
        args.cosine_cycle_iters = max(args.max_steps - 1, args.warmup_iters + 1)
    elif args.cosine_cycle_iters < args.warmup_iters + 1:
        args.cosine_cycle_iters = args.warmup_iters + 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    resolved_config_path = out_dir / "config_resolved.yaml"
    with open(resolved_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"cmd": getattr(args, "cmd", None), "config_path": getattr(args, "config", None), "args": vars(args)},
            f,
            sort_keys=True,
            allow_unicode=True,
        )

    dtype = parse_numpy_dtype(args.data_dtype)
    train_tokens = open_tokens(args.train_data, dtype=dtype)
    valid_tokens = open_tokens(args.valid_data, dtype=dtype)
    train_loader = DataLoader(
        train_tokens,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=device,
        dtype=train_tokens.dtype,
    )
    valid_loader = DataLoader(
        valid_tokens,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=device,
        dtype=valid_tokens.dtype,
    )

    if args.d_ff is None:
        args.d_ff = 4 * args.d_model if args.ffn_type == "silu" else 1344

    ckpt_config = vars(args).copy()
    base_model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        use_rmsnorm=not args.no_rmsnorm,
        norm_style=args.norm_style,
        use_rope=not args.no_rope,
        ffn_type=args.ffn_type,
    ).to(device)

    model: torch.nn.Module = base_model
    if args.compile:
        preferred_backend = args.compile_backend
        if preferred_backend is None and device.startswith("mps"):
            preferred_backend = "aot_eager"
        try:
            model = torch.compile(model, backend=preferred_backend) if preferred_backend else torch.compile(model)
        except Exception as e:
            print(f"compile_failed backend={preferred_backend or 'inductor'} err={type(e).__name__}: {e}", flush=True)
            if preferred_backend != "aot_eager":
                try:
                    model = torch.compile(model, backend="aot_eager")
                    print("compile_fallback=success backend=aot_eager", flush=True)
                except Exception as e2:
                    print(f"compile_fallback=failed backend=aot_eager err={type(e2).__name__}: {e2}", flush=True)
                    print("compile_disabled=true", flush=True)
                    model = base_model
            else:
                print("compile_disabled=true", flush=True)
                model = base_model

    optimizer = build_optimizer(args.optimizer, base_model.parameters(), args)

    it0 = 0
    if args.resume_from:
        it0 = int(load_checkpoint(args.resume_from, model=model, optimizer=optimizer, map_location=device))
        print(f"resume_from={args.resume_from} step={it0}", flush=True)
        if it0 >= args.max_steps:
            print(f"nothing_to_do: checkpoint_step={it0} >= max_steps={args.max_steps}", flush=True)
            return

    rng = np.random.default_rng(args.seed)
    eval_rng = np.random.default_rng(args.seed + 1)

    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                "vocab_size": args.vocab_size,
                "context_length": args.context_length,
                "d_model": args.d_model,
                "d_ff": args.d_ff,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "rope_theta": args.rope_theta,
                "optimizer": args.optimizer,
                "learning_rate": args.learning_rate,
                "min_learning_rate": args.min_learning_rate,
                "warmup_iters": args.warmup_iters,
                "cosine_cycle_iters": args.cosine_cycle_iters,
                "beta1": args.beta1,
                "beta2": args.beta2,
                "eps": args.eps,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "max_steps": args.max_steps,
                "grad_clip": args.grad_clip,
                "no_rmsnorm": args.no_rmsnorm,
                "norm_style": args.norm_style,
                "no_rope": args.no_rope,
                "ffn_type": args.ffn_type,
                "device": device,
            },
        )

    start_time = time.time()
    last_log_time = start_time
    last_log_tokens = 0

    for it in range(it0, args.max_steps):
        model.train()
        lr = lr_cosine_schedule(
            it=it,
            max_learning_rate=args.learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_cycle_iters,
        )
        set_lr(optimizer, lr)

        seed = int(rng.integers(0, 2**31 - 1))
        x, y = train_loader.get_batch(seed=seed)
        try:
            logits = model(x)
        except Exception as e:
            backend_failed = type(e).__name__ == "BackendCompilerFailed"
            triton_missing = "Cannot find a working triton installation" in str(e)
            if backend_failed or triton_missing:
                print(f"compile_runtime_failed err={type(e).__name__}: {e}", flush=True)
                print("compile_disabled=true", flush=True)
                model = base_model
                logits = model(x)
            else:
                raise
        loss = loss_per_token(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        wall_time = time.time() - start_time
        tokens_seen = (it + 1) * args.batch_size * args.context_length

        if (it + 1) % args.log_interval == 0:
            now = time.time()
            dt = now - last_log_time
            d_tokens = tokens_seen - last_log_tokens
            tok_per_s = d_tokens / dt if dt > 0 else float("nan")
            last_log_time = now
            last_log_tokens = tokens_seen

            train_loss = float(loss.detach().float().cpu().item())
            record = {
                "split": "train",
                "step": it + 1,
                "loss": train_loss,
                "lr": lr,
                "wall_time_sec": wall_time,
                "tokens_seen": tokens_seen,
                "tokens_per_sec": tok_per_s,
            }
            pct = 100.0 * (it + 1) / args.max_steps
            total_tokens = args.max_steps * args.batch_size * args.context_length
            remaining_tokens = max(total_tokens - tokens_seen, 0)
            eta = remaining_tokens / tok_per_s if np.isfinite(tok_per_s) and tok_per_s > 0 else float("nan")
            print(
                f"step={it + 1}/{args.max_steps} ({pct:.1f}%) loss={train_loss:.4f} lr={lr:.3e} "
                f"tok/s={tok_per_s:.0f} eta={format_duration(eta)} time={wall_time / 60:.1f}m",
                flush=True,
            )
            append_jsonl(metrics_path, record)
            if wandb_run:
                wandb_run.log(record, step=it + 1)

        if (it + 1) % args.eval_interval == 0:
            val_loss = eval_loss(
                model=model,
                dataloader=valid_loader,
                eval_iters=args.eval_iters,
                rng=eval_rng,
            )
            record = {
                "split": "valid",
                "step": it + 1,
                "loss": val_loss,
                "lr": lr,
                "wall_time_sec": wall_time,
                "tokens_seen": tokens_seen,
            }
            print(f"valid step={it + 1} loss={val_loss:.4f}", flush=True)
            append_jsonl(metrics_path, record)
            if wandb_run:
                wandb_run.log(record, step=it + 1)

        if args.ckpt_interval and (it + 1) % args.ckpt_interval == 0:
            ckpt_path = out_dir / f"checkpoint_step_{it + 1}.pt"
            save_checkpoint(model, optimizer, iteration=it + 1, out=ckpt_path, config=ckpt_config)
            print(f"ckpt_saved={ckpt_path.name}", flush=True)

    final_ckpt = out_dir / "checkpoint_final.pt"
    save_checkpoint(model, optimizer, iteration=args.max_steps, out=final_ckpt, config=ckpt_config)
    print(f"ckpt_saved={final_ckpt.name}", flush=True)
    if wandb_run:
        wandb_run.finish()
