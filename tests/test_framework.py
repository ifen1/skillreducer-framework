"""Exercise local bundle handling and the quality checks without API calls."""

import contextlib
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from skillreducer import compare_bundles, inspect_bundle, optimize_bundle, stage2
from skillreducer.__main__ import main


def _bundle(root, *, body="## Rules\nKeep all rows.", name="csv-report"):
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Describe the CSV table.\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return root


class BundleTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_cli_inspects_and_compares_without_credentials(self):
        first, second = _bundle(self.root / "a"), _bundle(self.root / "b")
        for root, data in ((first, b"same length A"), (second, b"same length B")):
            (root / "scripts").mkdir()
            (root / "scripts/check.sh").write_bytes(data)
        (first / ".env").write_text("private local configuration")
        (second / "references").mkdir()
        (second / "references/new.md").write_text("A reference")
        (first / "old.txt").write_text("Old asset")

        with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(["inspect", str(first), "--json"]), 0)
        files = {item["path"]: item["kind"] for item in json.loads(out.getvalue())["files"]}
        self.assertEqual(files, {"old.txt": "asset", "scripts/check.sh": "script"})
        changes = compare_bundles(first, second)
        self.assertEqual(changes["added_files"], ["references/new.md"])
        self.assertEqual(changes["removed_files"], ["old.txt"])
        self.assertEqual(changes["changed_files"], ["scripts/check.sh"])
        self.assertFalse(changes["body_changed"])

    def test_inspection_rejects_malformed_frontmatter(self):
        root = _bundle(self.root / "bad")
        (root / "SKILL.md").write_text("---\nname: missing-end\n")
        with self.assertRaisesRegex(ValueError, "unclosed YAML"):
            inspect_bundle(root)

    def test_python_api_passes_references_and_preserves_scripts(self):
        source = _bundle(self.root / "source")
        output = self.root / "output"
        (source / "references").mkdir()
        old_ref = source / "references/original.md"
        old_ref.write_text("Reference text")
        (source / "scripts").mkdir()
        (source / "scripts/check.sh").write_text("Do not execute")
        supplied = {
            "core_body": "Retain every row.",
            "references": [
                {"name": "references/skillreducer/examples.md", "content": "New example"},
            ],
            "feedback_iters": 1,
        }
        with (
            patch.dict(os.environ, {"DEEPSEEK_API_KEY": "mock"}, clear=True),
            patch.object(stage2, "count_tokens", return_value=3),
            patch.object(stage2, "restructure_body", return_value=supplied) as body,
        ):
            result = optimize_bundle(source, output, stage="body")
            with self.assertRaisesRegex(ValueError, "new directory"):
                optimize_bundle(source, output, stage="body")
            self.assertEqual(body.call_count, 1)
            self.assertEqual(body.call_args.kwargs["original_references"], [
                {"name": "references/original.md", "content": "Reference text", "tokens": 3},
            ])
        self.assertTrue(result.body_changed)
        self.assertFalse(result.description_changed)
        self.assertEqual(result.feedback_rounds, 1)
        self.assertTrue(old_ref.exists())
        self.assertFalse((output / "references/original.md").exists())
        self.assertEqual((output / "scripts/check.sh").read_text(), "Do not execute")
        self.assertEqual((output / "references/skillreducer/examples.md").read_text(), "New example")


class Stage2Tests(unittest.TestCase):
    def test_markdown_headings_inside_fences_do_not_split_examples(self):
        body = "## Rules\nKeep rows.\n## Example\n~~~text\n## a literal heading\n~~~\n## End\nDone."
        sections = stage2.segment_body(body)
        self.assertEqual([s["title"] for s in sections], ["Rules", "Example", "End"])
        self.assertIn("## a literal heading", sections[1]["content"])

    def test_invalid_classification_retries_then_preserves_original(self):
        with (
            patch.object(stage2, "_ds_client"),
            patch.object(stage2, "_call_with_retry", return_value=_response('{"items":[]}')) as model,
            patch("time.sleep"),
        ):
            items = stage2.classify_section("csv", "Rules", "Keep all rows.")
        self.assertEqual(model.call_count, 3)
        self.assertEqual([(i["text"], i["type"]) for i in items], [("Keep all rows.", "core_rule")])

    def test_core_kept_when_faithfulness_cannot_be_verified(self):
        raw = "Keep every original row, including rows with missing values."
        responses = [_response("Remove blanks"), _response('{"unexpected":true}')]
        with (
            patch.object(stage2, "_ds_client"),
            patch.object(stage2, "_call_with_retry", side_effect=responses),
            patch.object(stage2, "count_tokens", return_value=60),
        ):
            core, faithful = stage2._build_core_body("csv", [{"section": "intro", "text": raw}])
        self.assertEqual(core, raw)
        self.assertFalse(faithful)


def _response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


if __name__ == "__main__":
    unittest.main()
