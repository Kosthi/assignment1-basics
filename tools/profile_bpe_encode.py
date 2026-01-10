from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tests.adapters import get_tokenizer
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode


def _load_gpt2_tokenizer(*, vocab_path: Path, merges_path: Path, special_tokens: list[str] | None) -> object:
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges: list[tuple[str, str]] = []
    with open(merges_path) as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))

    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }

    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return get_tokenizer(vocab, merges, special_tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        type=str,
        default=str(FIXTURES_PATH / "tinystories_sample.txt"),
        help="Path to a UTF-8 text file.",
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=0)
    parser.add_argument("--special", type=str, action="append", default=[])
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    text = corpus_path.read_text(encoding="utf-8", errors="ignore")
    if args.max_chars and args.max_chars > 0:
        text = text[: args.max_chars]

    tokenizer = _load_gpt2_tokenizer(
        vocab_path=FIXTURES_PATH / "gpt2_vocab.json",
        merges_path=FIXTURES_PATH / "gpt2_merges.txt",
        special_tokens=args.special if args.special else None,
    )

    encode = getattr(tokenizer, "encode")
    total = 0
    for _ in range(args.repeat):
        ids = encode(text)
        total += len(ids)

    print(f"pid={os.getpid()} chars={len(text)} repeat={args.repeat} total_ids={total}")


if __name__ == "__main__":
    main()
