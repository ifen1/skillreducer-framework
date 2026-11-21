# SkillReducer

SkillReducer optimizes Markdown skill bundles for LLM agents. It refines the description that the agent uses to select a skill, then separates the body into core instructions and reference files that the agent can read on demand.

This repository accompanies **SkillReducer: Structure-aware Debloating of LLM Agent Skills**, accepted at ICSE 2027. It provides a standalone implementation for optimizing one skill bundle at a time.

## Quick start

Install with Python 3.10 or later. The optimizer uses DeepSeek; set `DEEPSEEK_API_KEY` before running it. The help command does not need credentials.

```sh
git clone https://github.com/ifen1/skillreducer-framework.git
cd skillreducer-framework
python -m venv .venv
source .venv/bin/activate
python -m pip install .
skillreducer --help
skillreducer inspect ./examples/csv-report
```

Once the key is set, run:

```sh
skillreducer ./my-skill --output ./my-skill-optimized
```

The input is a directory containing a UTF-8 `SKILL.md`. YAML frontmatter is optional. The `name` field, or the directory name when omitted, must begin with a letter or digit and can contain letters, digits, hyphens, and underscores. The output directory must be a new path outside the input bundle. The optimizer writes a separate bundle so you can compare it with the original.

## Inspect and compare locally

These commands run without API credentials. `inspect` lists the description and body sizes, references, scripts, and other assets. `compare` reports changes to the description and body and lists the files added, removed, or changed. Both commands accept `--json`.

```sh
skillreducer inspect ./examples/csv-report --json
skillreducer compare ./my-skill ./my-skill-optimized
```

## What the optimizer does

| Stage | How it works | Output |
| --- | --- | --- |
| **1. Routing description** | Generate a description when it is missing or at most 40 tokens. Split the description into semantic clauses and test shorter candidates with a simulated router, a generated shadow skill, and up to four similar competitors from the supplied bundles. Optional Claude Code validation checks real triggers and tries restoring removed clauses if triggers regress. | Updated `description` in the `SKILL.md` frontmatter. |
| **2. Body and references** | Classify content as core rules, templates, examples, background, or redundant text. Compress by type and check faithfulness. Move supplementary text to references and annotate their intended use. Check existing Markdown references for overlap with the original body. | A compact core plus references under `references/skillreducer/`, when needed. |

With `--task-eval`, the task gate compares the original and optimized bodies, each with on-demand access to its own references. It scores responses with an LLM rubric and uses failed tasks to identify content to restore to the core. It runs up to two feedback rounds and restores the original body and references if the optimized mean score trails the original by more than 0.10.

## Input and output

The included [CSV-report bundle](examples/csv-report/SKILL.md) has a reference template. A minimal `SKILL.md` has this form:

```markdown
---
name: csv-report
description: Summarize CSV tables and flag missing values.
---

# CSV report

## Rules

- Report the row count and missing values per column.
- Keep rows with missing values in the total.

## Example

For 12 rows with two missing dates, report 12 rows and two missing dates.
```

The input may also contain Markdown references under `references/` and scripts or other assets. The output has this shape, depending on the input:

```text
my-skill-optimized/
├── SKILL.md                        # Optimized instructions and reference pointers
├── references/
│   └── skillreducer/
│       ├── examples.md             # Created if examples are moved out
│       ├── templates.md            # Created if templates are moved out
│       └── background.md           # Created if background is moved out
└── scripts/                        # Copied if present in the input
```

Existing Markdown references that survive deduplication keep their paths. Other regular bundle files are copied; scripts are not executed by the optimizer. The optimizer skips hidden files and symlinks. Review the optimized bundle before replacing a deployed skill.

## Run a stage on its own

Optimize routing with a representative query and a competing skill:

```sh
skillreducer ./my-skill -o ./my-skill-routing --stage routing \
  --query "Summarize the missing values in this CSV" \
  --competitor ./another-skill
```

Optimize the body and enable the task gate after setting `DASHSCOPE_API_KEY`:

```sh
skillreducer ./my-skill -o ./my-skill-with-feedback --stage body --task-eval
```

| Option | Purpose |
| --- | --- |
| `-o`, `--output PATH` | Required. Write the optimized bundle to a new directory. |
| `--stage {routing,body,both}` | Select the optimizer stage. The default is `both`. |
| `--competitor PATH` | Repeat to add competing skill bundles for Stage 1. |
| `--query TEXT` | Repeat to supply positive routing queries. Otherwise, Stage 1 generates queries. |
| `--validate-routing` | Enable real trigger checks. Requires Claude Code on `PATH`, configured for use on macOS or Linux. |
| `--task-eval` | Enable Stage 2 rubric evaluation, feedback, and fallback. Requires `DASHSCOPE_API_KEY`. |

## Python API

```python
from skillreducer import compare_bundles, inspect_bundle, optimize_bundle

inventory = inspect_bundle('examples/csv-report')  # local only
result = optimize_bundle('examples/csv-report', 'csv-report-optimized', stage='body')
changes = compare_bundles('examples/csv-report', result.output_dir)  # local only
print(result.body_fallback, changes['added_files'])
```

`optimize_bundle` uses the same settings and credentials as the CLI. It returns the output path, selected stage, whether the description and body changed, the routing validation outcome, the number of feedback rounds, and whether the original body was restored.

## Models and code map

| Use | Model IDs in the code |
| --- | --- |
| Description generation, splitting, and rewriting; body and reference optimization | DeepSeek `deepseek-chat` |
| Simulated routing | DeepSeek `deepseek-reasoner` |
| Task generation and rubric scoring with `--task-eval` | DashScope `qwen3.5-plus` |
| Task responses with `--task-eval` | DashScope `qwen3-max-2026-01-23` |

Skill and query text may be sent to the providers used by the enabled stages.

- [`skillreducer/stage1.py`](skillreducer/stage1.py): description optimization, simulated routing, and optional real trigger checks.
- [`skillreducer/stage2.py`](skillreducer/stage2.py): content classification, compression, reference generation and deduplication, and the optional task gate.
- [`skillreducer/pipeline.py`](skillreducer/pipeline.py): Python API, orchestration, and bundle writing.
- [`skillreducer/bundle.py`](skillreducer/bundle.py): Markdown and YAML loading, local inspection, and comparison.
- [`skillreducer/__main__.py`](skillreducer/__main__.py): command-line entry point.

Both stage modules keep their prompts next to the functions that use them.

## Development

Run the tests from the repository root after installing the package:

```sh
python -m unittest discover -s tests -v
```

The tests use temporary bundles and simulated model responses, so no service credentials or study assets are needed.

## Cite

If you use SkillReducer in research, cite *SkillReducer: Structure-aware Debloating of LLM Agent Skills* (ICSE 2027).

## License

MIT. See [LICENSE](LICENSE).
