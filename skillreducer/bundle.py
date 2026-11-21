"""Read, inspect, and compare local skill bundles without contacting a model."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from typing import Iterator

import yaml

_FRONTMATTER = re.compile(
    r"\A---[ \t]*\n(.*?)^---[ \t]*(?:\n|\Z)", re.DOTALL | re.MULTILINE
)
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


@dataclass(frozen=True)
class Bundle:
    root: Path
    name: str
    description: str
    body: str
    metadata: dict


def load_bundle(path: str | Path) -> Bundle:
    """Load a UTF-8 SKILL.md and its optional YAML frontmatter."""
    root = Path(path).resolve()
    file = root / "SKILL.md"
    if file.is_symlink():
        raise ValueError(f"{file}: SKILL.md must be a regular file, not a symlink")
    text = file.read_text(encoding="utf-8")
    match = _FRONTMATTER.match(text)
    if text.splitlines() and re.fullmatch(r"---[ \t]*", text.splitlines()[0]) and not match:
        raise ValueError(f"{file}: unclosed YAML frontmatter")
    metadata = yaml.safe_load(match[1]) if match else {}
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError(f"{file}: YAML frontmatter must be a mapping")
    body = text[match.end():] if match else text
    if not body.strip():
        raise ValueError(f"{file}: missing skill body")
    name = metadata.get("name") or root.name
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError(f"{file}: invalid skill name")
    description = metadata.get("description")
    if description is None:
        description = ""
    if not isinstance(description, str):
        raise ValueError(f"{file}: description must be text")
    return Bundle(root, name, description, body, metadata)


def bundle_files(root: Path) -> Iterator[tuple[Path, Path]]:
    """Yield regular bundle files other than SKILL.md, with relative paths."""
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if any(root.joinpath(*rel.parts[:i]).is_symlink() for i in range(1, len(rel.parts) + 1)):
            continue
        if rel.as_posix() != "SKILL.md" and path.is_file():
            yield path, rel


def _kind(rel: Path) -> str:
    if rel.parts[0] == "references" and rel.suffix.lower() == ".md":
        return "markdown_reference"
    if rel.parts[0] == "scripts":
        return "script"
    return "asset"


def markdown_references(bundle: Bundle) -> Iterator[tuple[str, str]]:
    """Yield optimizable reference paths and text."""
    for path, rel in bundle_files(bundle.root):
        if _kind(rel) == "markdown_reference":
            yield rel.as_posix(), path.read_text(encoding="utf-8")


def _size(text: str) -> int:
    return len(text.encode("utf-8"))


def _inspect(bundle: Bundle) -> dict:
    files = [
        {"path": rel.as_posix(), "kind": _kind(rel), "bytes": path.stat().st_size}
        for path, rel in bundle_files(bundle.root)
    ]
    return {
        "name": bundle.name,
        "description": bundle.description,
        "description_bytes": _size(bundle.description),
        "body_bytes": _size(bundle.body),
        "files": files,
    }


def inspect_bundle(path: str | Path) -> dict:
    """Return a JSON-ready inventory and UTF-8 sizes, with no model calls."""
    return _inspect(load_bundle(path))


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_bundles(before: str | Path, after: str | Path) -> dict:
    """Compare descriptions, bodies, and assets without sending either bundle."""
    old, new = load_bundle(before), load_bundle(after)
    a = {rel.as_posix(): _hash_file(p) for p, rel in bundle_files(old.root)}
    b = {rel.as_posix(): _hash_file(p) for p, rel in bundle_files(new.root)}
    return {
        "before": _inspect(old),
        "after": _inspect(new),
        "description_changed": old.description != new.description,
        "body_changed": old.body.strip() != new.body.strip(),
        "added_files": sorted(b.keys() - a.keys()),
        "removed_files": sorted(a.keys() - b.keys()),
        "changed_files": sorted(p for p in a.keys() & b.keys() if a[p] != b[p]),
    }
