# SkillReducer

Source code for **SkillReducer: Structure-aware Debloating of LLM Agent Skills** (ICSE 2027). This release contains the framework and its prompts. It uses your own skill bundle as input.

## Install and run

Use Python 3.10 or newer and set `DEEPSEEK_API_KEY` in your environment.

    python -m pip install .
    skillreducer /path/to/my-skill --output /path/to/optimized-skill

The input directory must contain a UTF-8 `SKILL.md`. Supply an output directory that does not yet exist and is outside the input bundle.

- **Routing:** Stage 1 generates missing or short descriptions and reduces routing descriptions with delta debugging. Use `--competitor /path/to/another-skill` to supply competing descriptions and `--query "your typical request"` to supply positive queries. Both options can be repeated. Add `--validate-routing` to check real triggers with Claude Code on macOS or Linux.
- **Body:** Stage 2 classifies and compresses body content, checks faithfulness, and moves supplementary material into references. Markdown files under `references/` are eligible for deduplication. Other bundle files, including scripts, are copied without execution. Add `--task-eval` and set `DASHSCOPE_API_KEY` to enable rubric-based task evaluation, feedback, and fallback. This release does not run generated Python.

Use `--stage routing` or `--stage body` to run a single stage. The default runs both, with live routing validation and task evaluation available as opt-in steps. Skill and query text is sent to the model providers used by the enabled stages.

This standalone release supports individual skill bundles. It excludes study datasets, task transcripts, experiment runners, and precomputed results.

## License

MIT. See [LICENSE](LICENSE).
