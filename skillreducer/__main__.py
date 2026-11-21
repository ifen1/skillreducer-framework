"""SkillReducer command-line interface."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

from .bundle import compare_bundles, inspect_bundle
from .pipeline import optimize_bundle


def _show_inventory(report: dict) -> None:
    print(f"Skill: {report['name']}")
    print(f"Description: {report['description_bytes']} bytes")
    print(f"Body: {report['body_bytes']} bytes")
    for item in report["files"]:
        print(f"  {item['path']} ({item['kind']}, {item['bytes']} bytes)")


def _show_difference(report: dict) -> None:
    before, after = report["before"], report["after"]
    print(f"Skill: {before['name']} → {after['name']}")
    for field in ("description", "body"):
        key = f"{field}_bytes"
        print(f"{field.capitalize()}: {before[key]} → {after[key]} bytes"
              f" ({'changed' if report[f'{field}_changed'] else 'unchanged'})")
    for kind in ("added", "removed", "changed"):
        for filename in report[f"{kind}_files"]:
            print(f"  {kind.capitalize()}: {filename}")


def _offline(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog=f"skillreducer {argv[0]}")
    parser.add_argument("skill_dir", type=Path)
    if argv[0] == "compare":
        parser.add_argument("after", type=Path)
    parser.add_argument("--json", action="store_true", help="Print machine-readable output")
    args = parser.parse_args(argv[1:])
    if argv[0] == "inspect":
        report = inspect_bundle(args.skill_dir)
        display = _show_inventory
    else:
        report = compare_bundles(args.skill_dir, args.after)
        display = _show_difference
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        display(report)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        # Keep the original `skillreducer PATH -o OUTPUT` syntax.
        if argv and argv[0] in {"inspect", "compare"}:
            return _offline(argv)
        parser = argparse.ArgumentParser(
            prog="skillreducer", description="Optimize a local SKILL.md bundle.",
            epilog="Offline tools: skillreducer inspect PATH; skillreducer compare BEFORE AFTER.",
        )
        parser.add_argument("skill_dir", type=Path, help="Directory containing SKILL.md")
        parser.add_argument("-o", "--output", type=Path, required=True,
                            help="New output directory outside the input bundle")
        parser.add_argument("--stage", choices=["routing", "body", "both"], default="both")
        parser.add_argument("--competitor", action="append", default=[], type=Path,
                            help="Optional competitor directory containing SKILL.md; repeat as needed")
        parser.add_argument("--query", action="append", default=[],
                            help="Optional positive routing query; repeat as needed")
        parser.add_argument("--validate-routing", action="store_true",
                            help="Validate routing in a temporary project using Claude Code")
        parser.add_argument("--task-eval", action="store_true",
                            help="Enable rubric-based Gate 2 and feedback (requires DASHSCOPE_API_KEY)")
        args = parser.parse_args(argv)
        result = optimize_bundle(
            args.skill_dir, args.output, stage=args.stage,
            competitors=args.competitor, queries=args.query,
            validate_routing=args.validate_routing, task_eval=args.task_eval,
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"skillreducer: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {result.output_dir / 'SKILL.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
