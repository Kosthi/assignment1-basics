"""
YAML 配置注入到 argparse 的实现。

目标：
1) 让 YAML 提供“默认值”，不覆盖命令行显式传入的参数；
2) 支持把原本 required 的参数放入 YAML；
3) 对未知字段给出明确报错，减少实验配置的隐性 bug。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def find_subparser(parser: argparse.ArgumentParser, cmd: str) -> argparse.ArgumentParser | None:
    """拿到某个子命令的 subparser。"""

    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices.get(cmd)
    return None


def extract_config_path(argv_after_cmd: list[str]) -> str | None:
    """从子命令后的 argv 中解析 --config 路径（支持 --config X 与 --config=X）。"""

    for i, tok in enumerate(argv_after_cmd):
        if tok == "--config" and i + 1 < len(argv_after_cmd):
            return argv_after_cmd[i + 1]
        if tok.startswith("--config="):
            return tok.split("=", 1)[1]
    return None


def load_yaml_config(path: str) -> dict[str, Any]:
    """读取 YAML 并保证顶层为 mapping。"""

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config 文件不存在: {p}")
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("config 顶层必须是 YAML mapping（key: value）")
    return data


def select_cmd_config(full_config: dict[str, Any], cmd: str) -> dict[str, Any]:
    """从 YAML 中挑选当前子命令的配置块。"""

    cmd_alt = cmd.replace("-", "_")
    if cmd in full_config and isinstance(full_config[cmd], dict):
        return dict(full_config[cmd])
    if cmd_alt in full_config and isinstance(full_config[cmd_alt], dict):
        return dict(full_config[cmd_alt])
    if "args" in full_config and isinstance(full_config["args"], dict):
        return dict(full_config["args"])
    return dict(full_config)


def normalize_config_keys(config: dict[str, Any]) -> dict[str, Any]:
    """把 YAML key 统一映射到 argparse 的 dest 命名风格（- -> _）。"""

    normalized: dict[str, Any] = {}
    for k, v in config.items():
        if not isinstance(k, str):
            raise ValueError("config key 必须是字符串")
        kk = k.strip().replace("-", "_")
        if kk in {"cmd", "args"}:
            continue
        normalized[kk] = v
    return normalized


def collect_overridden_dests(subparser: argparse.ArgumentParser, argv_after_cmd: list[str]) -> set[str]:
    """收集命令行显式传入的 dest，避免被 YAML 默认值覆盖。"""

    option_to_dest: dict[str, str] = {}
    for action in subparser._actions:
        for opt in action.option_strings:
            option_to_dest[opt] = action.dest

    overridden: set[str] = set()
    for tok in argv_after_cmd:
        if not tok.startswith("-"):
            continue
        opt = tok.split("=", 1)[0]
        dest = option_to_dest.get(opt)
        if dest:
            overridden.add(dest)
    return overridden


def coerce_action_value(action: argparse.Action, value: Any) -> Any:
    """把 YAML 值转换成 argparse 期望的类型/形态。"""

    if value is None:
        return None

    if action.nargs in {"*", "+"}:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise ValueError(f"config 参数 {action.dest} 需要是 list（nargs={action.nargs}）")
        return [str(x) for x in value]

    if isinstance(action, argparse._StoreTrueAction):
        if isinstance(value, bool):
            return value
        raise ValueError(f"config 参数 {action.dest} 需要是 bool")

    if action.choices is not None:
        if value not in action.choices:
            raise ValueError(f"config 参数 {action.dest}={value!r} 不在 choices={sorted(action.choices)!r}")
        return value

    if action.type is not None:
        if isinstance(value, bool) and action.type in {int, float}:
            raise ValueError(f"config 参数 {action.dest} 需要是 {action.type.__name__}")
        try:
            return action.type(value)
        except Exception as e:
            raise ValueError(f"config 参数 {action.dest} 无法转换为 {action.type.__name__}: {value!r}") from e

    return value


def apply_config_defaults(
    subparser: argparse.ArgumentParser, config: dict[str, Any], overridden_dests: set[str]
) -> tuple[list[str], list[str]]:
    """把 config 注入为默认值，同时收集 required 字段并临时取消 required。"""

    required_dests: list[str] = []
    known_dests: set[str] = set()
    for action in subparser._actions:
        if action.dest == "help":
            continue
        known_dests.add(action.dest)
        if getattr(action, "required", False):
            required_dests.append(action.dest)

    unknown_keys = sorted([k for k in config.keys() if k not in known_dests])
    if unknown_keys:
        return required_dests, unknown_keys

    for action in subparser._actions:
        if action.dest == "help":
            continue
        if action.dest in overridden_dests:
            continue
        if action.dest not in config:
            continue
        action.default = coerce_action_value(action, config[action.dest])

    for action in subparser._actions:
        if getattr(action, "required", False):
            action.required = False

    return required_dests, []


def enforce_required(required_dests: list[str], args: argparse.Namespace) -> None:
    """在 parse_args 之后检查 required 是否齐全。"""

    missing: list[str] = []
    for dest in required_dests:
        val = getattr(args, dest, None)
        if val is None:
            missing.append(dest)
        elif isinstance(val, str) and not val.strip():
            missing.append(dest)
    if missing:
        formatted = ", ".join(sorted(missing))
        raise SystemExit(f"缺少必填参数: {formatted}（可用 CLI 传入，或写入 --config）")
