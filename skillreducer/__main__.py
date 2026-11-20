"""Apply SkillReducer to a local skill directory."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import yaml


def _read_skill(file: Path) -> tuple[dict, str]:
    text = file.read_text(encoding="utf-8")
    match = re.match(r"\A---[ \t]*\n(.*?)^---[ \t]*(?:\n|\Z)", text, re.DOTALL | re.MULTILINE)
    if text.splitlines()[:1] == ["---"] and not match:
        raise ValueError(f"{file}: unclosed YAML frontmatter")
    metadata = yaml.safe_load(match[1]) if match else {}
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError(f"{file}: YAML frontmatter must be a mapping")
    body = text[match.end():] if match else text
    if not body.strip():
        raise ValueError(f"{file}: missing skill body")
    return metadata, body


def _files(root: Path):
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if any((root.joinpath(*rel.parts[:i])).is_symlink()
               for i in range(1, len(rel.parts) + 1)):
            continue
        if path.is_file() and rel.as_posix() != "SKILL.md":
            yield path, rel


def _name(metadata: dict, root: Path) -> str:
    name = metadata.get("name") or root.name
    if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", name):
        raise ValueError("Skill names must use letters, digits, hyphens, or underscores")
    return name


def _description(metadata: dict) -> str:
    return str(metadata.get("description") or "")


def _write(root: Path, output: Path, metadata: dict, result: dict,
           original_refs: set[str]) -> None:
    # Stage all files before moving the output into place.
    from tempfile import TemporaryDirectory

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="skillreducer-", dir=output.parent) as tmp:
        staged = Path(tmp) / "skill"
        staged.mkdir()
        for source, rel in _files(root):
            if rel.as_posix() not in original_refs:
                target = staged / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        for ref in result["references"]:
            rel = Path(ref["name"])
            if (rel.is_absolute() or ".." in rel.parts or not rel.parts
                    or rel.as_posix() == "SKILL.md"):
                raise ValueError(f"Invalid reference path: {rel}")
            target = staged / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(ref["content"], encoding="utf-8")
        frontmatter = yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False)
        (staged / "SKILL.md").write_text(
            f"---\n{frontmatter}---\n\n{result['core_body'].strip()}\n", encoding="utf-8")
        staged.rename(output)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="skillreducer", description="Optimize a local SKILL.md bundle with SkillReducer.")
    parser.add_argument("skill_dir", type=Path, help="Directory containing SKILL.md")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="New output directory outside the input bundle")
    parser.add_argument("--stage", choices=["routing", "body", "both"], default="both")
    parser.add_argument("--competitor", action="append", default=[], type=Path,
                        help="Optional competitor directory containing SKILL.md; repeat as needed")
    parser.add_argument("--query", action="append", default=[],
                        help="Optional positive routing query; repeat as needed")
    parser.add_argument("--validate-routing", action="store_true",
                        help="Validate routing in an isolated temporary project using Claude Code")
    parser.add_argument("--task-eval", action="store_true",
                        help="Enable rubric-based Gate 2 and feedback (requires DASHSCOPE_API_KEY)")
    args = parser.parse_args(argv)

    root, output = args.skill_dir.resolve(), args.output.resolve()
    if output == root or output.is_relative_to(root) or output.exists():
        parser.error("--output must be a new directory outside the input bundle")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        parser.error("Set DEEPSEEK_API_KEY")
    if args.task_eval and args.stage == "routing":
        parser.error("--task-eval requires the body stage")
    if args.task_eval and not os.environ.get("DASHSCOPE_API_KEY"):
        parser.error("--task-eval requires DASHSCOPE_API_KEY")
    if args.validate_routing and (args.stage == "body" or not shutil.which("claude")):
        parser.error("--validate-routing requires the routing stage and Claude Code on PATH")

    try:
        metadata, body = _read_skill(root / "SKILL.md")
        name = _name(metadata, root)
        if args.stage in {"routing", "both"}:
            from .stage1 import optimize_skill

            skills = {name: _description(metadata)}
            for directory in args.competitor:
                other, _ = _read_skill(directory / "SKILL.md")
                other_name = _name(other, directory)
                if other_name in skills:
                    raise ValueError(f"Duplicate skill name: {other_name}")
                skills[other_name] = _description(other)
            queries = {name: [{"query": q} for q in args.query]} if args.query else None
            routing = optimize_skill(
                {"name": name, "description": skills[name], "body": body},
                skills, queries, skip_real_validation=not args.validate_routing)
            metadata["description"] = routing["optimized_desc"]

        result = {"core_body": body, "references": []}
        refs = []
        if args.stage in {"body", "both"}:
            from .stage2 import count_tokens, restructure_body

            for path, rel in _files(root):
                if rel.parts[0] != "references" or path.suffix.lower() != ".md":
                    continue
                content = path.read_text(encoding="utf-8")
                refs.append({"name": rel.as_posix(), "content": content,
                             "tokens": count_tokens(content)})
            result = restructure_body(
                name, body, original_references=refs, task_eval=args.task_eval)

        metadata.setdefault("name", name)
        _write(root, output, metadata, result, {ref["name"] for ref in refs})
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"skillreducer: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {output / 'SKILL.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
