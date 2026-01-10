"""
文本生成（generate 子命令）。

这里实现的是一个简单、可复现的采样器：
1) 从 checkpoint 读取训练时保存的 config（优先）；
2) 构建同结构的 TransformerLM 并加载权重；
3) 逐 token 采样，支持 temperature / top-k / top-p / repetition penalty / stop strings。
"""

from __future__ import annotations

import argparse

import torch

from .runtime import auto_device
from .text_pipeline import load_bpe_tokenizer
from .transformer_lm import TransformerLM


@torch.no_grad()
def generate(args: argparse.Namespace) -> None:
    """generate 子命令入口。"""

    device = auto_device(args.device)
    if device.startswith("cuda") and args.matmul_precision:
        torch.set_float32_matmul_precision(args.matmul_precision)

    tokenizer = load_bpe_tokenizer(args.vocab_path, args.merges_path, special_tokens=args.special_tokens)
    vocab_size_from_tokenizer = tokenizer.get_vocab_size()

    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    config = ckpt_config if isinstance(ckpt_config, dict) else {}

    vocab_size = int(config.get("vocab_size", vocab_size_from_tokenizer))
    if "vocab_size" in config and vocab_size_from_tokenizer != vocab_size:
        raise ValueError(
            f"checkpoint vocab_size={vocab_size} 与 tokenizer vocab_size={vocab_size_from_tokenizer} 不一致"
        )

    context_length = int(config.get("context_length", args.context_length))
    d_model = int(config.get("d_model", args.d_model))
    num_layers = int(config.get("num_layers", args.num_layers))
    num_heads = int(config.get("num_heads", args.num_heads))
    rope_theta = float(config.get("rope_theta", args.rope_theta))
    no_rmsnorm = bool(config.get("no_rmsnorm", args.no_rmsnorm))
    norm_style = str(config.get("norm_style", args.norm_style))
    no_rope = bool(config.get("no_rope", args.no_rope))
    ffn_type = str(config.get("ffn_type", args.ffn_type))
    d_ff = config.get("d_ff", args.d_ff)
    if d_ff is None:
        d_ff = 4 * d_model if ffn_type == "silu" else 1344
    d_ff = int(d_ff)

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=device,
        use_rmsnorm=not no_rmsnorm,
        norm_style=norm_style,
        use_rope=not no_rope,
        ffn_type=ffn_type,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if args.compile:
        if device.startswith("mps") and args.compile_backend is None:
            model = torch.compile(model, backend="aot_eager")
        else:
            model = torch.compile(model, backend=args.compile_backend) if args.compile_backend else torch.compile(model)
        model.eval()

    eos_id = None
    if args.eos_token:
        eos_bytes = args.eos_token.encode("utf-8")
        eos_id = tokenizer.token_to_id.get(eos_bytes, None)

    ids = tokenizer.encode(args.prompt)
    generated: list[int] = list(ids)
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)

    for _ in range(args.max_new_tokens):
        x = torch.tensor([generated[-context_length:]], dtype=torch.long, device=device)
        logits = model(x)[:, -1, :]
        logits = logits / max(args.temperature, 1e-8)

        if args.repetition_penalty and args.repetition_penalty != 1.0 and generated:
            penalty = float(args.repetition_penalty)
            unique_ids = torch.unique(torch.tensor(generated, dtype=torch.long, device=device))
            logits[:, unique_ids] = logits[:, unique_ids] / penalty

        if args.top_k and args.top_k > 0:
            v, _ = torch.topk(logits, k=min(args.top_k, logits.shape[-1]))
            cutoff = v[:, -1].unsqueeze(-1)
            logits = torch.where(logits < cutoff, torch.tensor(float("-inf"), device=device), logits)

        if args.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(probs, dim=-1)
            to_remove = cumprobs > args.top_p
            to_remove[:, 1:] = to_remove[:, :-1].clone()
            to_remove[:, 0] = False
            remove_mask = to_remove.scatter(1, sorted_idx, to_remove)
            logits = logits.masked_fill(remove_mask, float("-inf"))

        probs = torch.softmax(logits, dim=-1)
        next_id = int(torch.multinomial(probs, num_samples=1, generator=rng).item())
        generated.append(next_id)

        if eos_id is not None and next_id == eos_id:
            break

        if args.stop_strings:
            text_so_far = tokenizer.decode(generated)
            for s in args.stop_strings:
                if not s:
                    continue
                stop_at = text_so_far.find(s)
                if stop_at != -1:
                    print(text_so_far[:stop_at], flush=True)
                    return

    text = tokenizer.decode(generated)
    print(text, flush=True)
