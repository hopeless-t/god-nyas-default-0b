from __future__ import annotations

import argparse
import json
from pathlib import Path
from .model import load_default_0b


def dump(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(prog="godnyas-0b", description="God Nyazu default strict-0B public asset")
    parser.add_argument("--state-dir", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info")
    sub.add_parser("capabilities")
    sub.add_parser("selftest")
    chat = sub.add_parser("chat"); chat.add_argument("text")
    arithmetic = sub.add_parser("arithmetic"); arithmetic.add_argument("expr")
    linear = sub.add_parser("linear"); linear.add_argument("a", type=int); linear.add_argument("b", type=int); linear.add_argument("c", type=int)
    args = parser.parse_args()
    model = load_default_0b(args.state_dir)
    if args.command == "info": payload = model.info()
    elif args.command == "capabilities": payload = {"capabilities": model.capabilities(), "trainable_parameters": 0, "model_calls": 0}
    elif args.command == "selftest": payload = model.self_test()
    elif args.command == "chat": payload = model.respond(args.text)
    elif args.command == "arithmetic": payload = model.reason("arithmetic", {"expr": args.expr})
    elif args.command == "linear": payload = model.reason("linear", {"a": args.a, "b": args.b, "c": args.c})
    else: raise AssertionError(args.command)
    dump(payload)
    return 0 if payload.get("status", "DONE") not in {"FAILED"} and payload.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
