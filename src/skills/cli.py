"""
act-skill CLI — operator-facing skill runner.

Usage:
    python -m src.skills.cli list
    python -m src.skills.cli list --tag diagnostic
    python -m src.skills.cli run readiness
    python -m src.skills.cli run regime-check
    python -m src.skills.cli run emergency-flatten confirm=true reason="paper soak halt"
    python -m src.skills.cli describe emergency-flatten
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

from src.skills.registry import SkillResult, get_registry


def _parse_kv(tokens: List[str]) -> Dict[str, Any]:
    """Parse k=v pairs from argv tail. Values are JSON if possible, else string.

    `confirm=true` → {"confirm": True}, `reason="x y"` → {"reason": "x y"}.
    """
    out: Dict[str, Any] = {}
    for tok in tokens:
        if "=" not in tok:
            continue
        k, _, v = tok.partition("=")
        try:
            out[k] = json.loads(v)
        except Exception:
            # Strip surrounding quotes if any.
            if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
                v = v[1:-1]
            out[k] = v
    return out


def _cmd_list(args: argparse.Namespace) -> int:
    reg = get_registry()
    skills = reg.list_skills(tag=args.tag)
    if not skills:
        print("no skills found" + (f" with tag {args.tag!r}" if args.tag else ""))
        return 1
    for s in skills:
        badge = " [CONFIRM]" if s.requires_confirmation else ""
        flag = " [NON-REVERSIBLE]" if not s.reversible else ""
        tags = f"  tags: {', '.join(s.tags)}" if s.tags else ""
        print(f"/{s.name}{badge}{flag}")
        print(f"  {s.description.strip()}")
        if tags:
            print(tags)
    return 0


def _cmd_describe(args: argparse.Namespace) -> int:
    reg = get_registry()
    skill = reg.get(args.name)
    if skill is None:
        print(f"unknown skill: {args.name}", file=sys.stderr)
        return 1
    print(json.dumps(skill.to_dict(), indent=2))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    reg = get_registry()
    kv = _parse_kv(args.args or [])
    result: SkillResult = reg.dispatch(args.name, kv, invoker="operator")
    print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0 if result.ok else 2


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(prog="act-skill", description="ACT skills runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List registered skills")
    p_list.add_argument("--tag", default=None, help="Filter by tag")
    p_list.set_defaults(func=_cmd_list)

    p_describe = sub.add_parser("describe", help="Print one skill's metadata")
    p_describe.add_argument("name")
    p_describe.set_defaults(func=_cmd_describe)

    p_run = sub.add_parser("run", help="Invoke one skill with k=v args")
    p_run.add_argument("name")
    p_run.add_argument("args", nargs=argparse.REMAINDER)
    p_run.set_defaults(func=_cmd_run)

    ns = parser.parse_args(argv)
    return ns.func(ns)


if __name__ == "__main__":
    sys.exit(main())
