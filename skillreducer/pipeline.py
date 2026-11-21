"""Standalone file workflow shared by the Python API and the CLI."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import yaml

from .bundle import bundle_files, load_bundle, markdown_references


@dataclass(frozen=True)
class OptimizationResult:
    output_dir: Path
    stage: str
    description_changed: bool
    body_changed: bool
    routing_validation: str | None
    feedback_rounds: int
    body_fallback: bool


def _write(root: Path, output: Path, metadata: dict, result: dict,
           original_refs: set[str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="skillreducer-", dir=output.parent) as tmp:
        staged = Path(tmp) / "skill"
        staged.mkdir()
        for source, rel in bundle_files(root):
            if rel.as_posix() not in original_refs:
                target = staged / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        for ref in result["references"]:
            rel = Path(ref["name"])
            if (rel.is_absolute() or ".." in rel.parts or len(rel.parts) < 2
                    or rel.parts[0] != "references"):
                raise ValueError(f"Invalid reference path: {rel}")
            target = staged / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8") as stream:
                stream.write(ref["content"])
        frontmatter = yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False)
        (staged / "SKILL.md").write_text(
            f"---\n{frontmatter}---\n\n{result['core_body'].strip()}\n", encoding="utf-8"
        )
        if output.exists() or output.is_symlink():
            raise ValueError(f"Output already exists: {output}")
        staged.rename(output)


def optimize_bundle(
    source: str | Path,
    output_dir: str | Path,
    *,
    stage: str = "both",
    competitors: Sequence[str | Path] = (),
    queries: Sequence[str] = (),
    validate_routing: bool = False,
    task_eval: bool = False,
) -> OptimizationResult:
    """Optimize one bundle, writing to a new directory outside the input."""
    if stage not in {"routing", "body", "both"}:
        raise ValueError("stage must be routing, body, or both")
    if stage == "body" and (competitors or queries or validate_routing):
        raise ValueError("Competitors, queries, and routing validation require the routing stage")
    if stage == "routing" and task_eval:
        raise ValueError("Task evaluation requires the body stage")
    if validate_routing and not shutil.which("claude"):
        raise ValueError("Routing validation requires Claude Code on PATH")

    bundle = load_bundle(source)
    destination = Path(output_dir).absolute()
    if destination.is_symlink():
        raise ValueError("Output must be a new directory")
    output = destination.resolve()
    if output.is_relative_to(bundle.root) or output.exists():
        raise ValueError("Output must be a new directory outside the input bundle")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("Set DEEPSEEK_API_KEY")
    if task_eval and not os.environ.get("DASHSCOPE_API_KEY"):
        raise ValueError("Task evaluation requires DASHSCOPE_API_KEY")

    metadata = dict(bundle.metadata)
    metadata.setdefault("name", bundle.name)
    routing = {}
    if stage in {"routing", "both"}:
        from .stage1 import optimize_skill

        pool = {bundle.name: bundle.description}
        for path in competitors:
            other = load_bundle(path)
            if other.name in pool:
                raise ValueError(f"Duplicate skill name: {other.name}")
            pool[other.name] = other.description
        test_queries = {bundle.name: [{"query": q} for q in queries]} if queries else None
        routing = optimize_skill(
            {"name": bundle.name, "description": bundle.description, "body": bundle.body},
            pool, test_queries, skip_real_validation=not validate_routing,
        )
        metadata["description"] = routing["optimized_desc"]

    refs = []
    result = {"core_body": bundle.body, "references": []}
    if stage in {"body", "both"}:
        from .stage2 import count_tokens, restructure_body

        refs = [
            {"name": name, "content": text, "tokens": count_tokens(text)}
            for name, text in markdown_references(bundle)
        ]
        result = restructure_body(
            bundle.name, bundle.body, original_references=refs, task_eval=task_eval
        )

    _write(bundle.root, output, metadata, result, {ref["name"] for ref in refs})
    return OptimizationResult(
        output_dir=output,
        stage=stage,
        description_changed=metadata.get("description", bundle.description) != bundle.description,
        body_changed=result["core_body"].strip() != bundle.body.strip(),
        routing_validation=routing.get("validation", {}).get("outcome"),
        feedback_rounds=result.get("feedback_iters", 0),
        body_fallback=bool(result.get("fallback")),
    )
