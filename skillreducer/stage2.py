#!/usr/bin/env python3
"""
Stage 2: Progressive Disclosure Restructuring (with Task Evaluation Gate)
=========================================================================
Fine-grained content classification of Skill body, reorganized by type into a layered structure:

  Before: SKILL.md (fully loaded)
  After:  SKILL.md (core only) + references/*.md (loaded on demand via read_file)

Two quality gates:
  Gate 1 - Faithfulness check (structural): whether key concepts are preserved after compression
  Gate 2 - Task evaluation (functional): whether the Agent can complete real tasks using compressed content

    Condition A: original full body (baseline)
    Condition C: optimized core
    A and C have on-demand access to their respective references.

Feedback loop:
  Condition C failure -> locate missing entries -> reclassify (example/background -> core_rule) -> recompress -> retest

Content types:
  core_rule   Operational rules the model would not know on its own (high uniqueness)
  template    Reusable code blocks/config snippets, directly copy-paste usable
  example     Code snippets illustrating concepts, for understanding, not directly reusable
  background  Background explanations, motivation, performance data
  redundant   Content duplicated elsewhere in the body
"""

from __future__ import annotations

import os
import re
import json
import tiktoken
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from openai import OpenAI

# ============================================================
# Config - DeepSeek (classify / compress / faithfulness)
# ============================================================
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
COMPRESS_MODEL = "deepseek-chat"


# ============================================================
# Config - Qwen (task generation / Agent execution / evaluation judge)
# ============================================================
QWEN_BASE_URL = "https://coding.dashscope.aliyuncs.com/v1"
TASK_GEN_MODEL = "qwen3.5-plus"          # Task generation + judge
AGENT_MODEL    = "qwen3-max-2026-01-23"  # Agent execution (strongest tool-calling model)

# ============================================================
# Flow control
# ============================================================
TASKS_PER_SKILL        = 5
MAX_TOOL_CALLS         = 6     # Max reference reads per run
MAX_FEEDBACK_ITERS     = 2     # Max feedback reclassification iterations
FALLBACK_TOLERANCE    = 0.10

# ============================================================
# Clients
# ============================================================
@lru_cache(maxsize=1)
def _ds_client():
    return OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=DEEPSEEK_BASE_URL,
                  timeout=120.0, max_retries=2)


@lru_cache(maxsize=1)
def _qwen_client():
    return OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url=QWEN_BASE_URL,
                  timeout=180.0, max_retries=3)


@lru_cache(maxsize=1)
def _encoder():
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text, disallowed_special=())) if text else 0


import time as _time

def _call_with_retry(fn, *args, **kwargs):
    """Call fn with automatic backoff retry on 429/network errors."""
    _RETRYABLE_NET = ("connection error", "connection reset", "timeout",
                      "timed out", "apitimeouterror",
                      "apiconnectionerror", "remote end closed",
                      "server disconnected", "eof occurred")
    _RETRYABLE_429 = ("429", "throttling")
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            err_lower = str(e).lower()
            is_429 = any(tok in err_lower for tok in _RETRYABLE_429)
            is_net = any(tok in err_lower for tok in _RETRYABLE_NET)
            if is_429:
                delay = [60, 180, 300][min(attempt, 2)]
                print(f"      [429 rate limit] retry {attempt+1}, waiting {delay}s ... ({type(e).__name__})")
            elif is_net:
                delay = [5, 10, 30, 60, 60][min(attempt, 4)]
                print(f"      [network error] retry {attempt+1}, waiting {delay}s ... ({type(e).__name__})")
            else:
                raise
            _time.sleep(delay)
    return fn(*args, **kwargs)


# ============================================================
# Prompts - Gate 1 (classify / compress / faithfulness) [existing, unchanged]
# ============================================================

CLASSIFY_PROMPT = """You are analyzing a section from an AI agent skill's instruction body.
Classify each bullet point or sentence into ONE of these types:

- core_rule: A specific operational rule or constraint that the model would NOT know from general training.
  Must be something unique to this skill (specific threshold, required step, tool-specific flag, etc.)
  Do NOT classify code/commands here unless they are a mandatory fixed pattern (exact flag combination,
  required invocation format) that the agent must follow verbatim every time.
- template: A reusable code block, config snippet, or boilerplate that the agent should copy/adapt directly.
  Distinct from examples: templates are meant to be used as-is, not just to illustrate a concept.
  Examples: starter scripts, config file templates, prompt templates, scaffold code.
- example: An illustrative code snippet or concrete use case that demonstrates a concept.
  The agent reads it to understand, not to copy verbatim.
- background: Explanation of WHY a rule exists, motivation, performance data, or general context.
- redundant: Content that repeats something already stated elsewhere in this body (cross-section duplication).

Skill Name: {name}
Section Title: {title}
Section Content:
{content}

Output ONLY valid JSON:
{{
  "items": [
    {{"text": "exact text of bullet/sentence", "type": "core_rule|template|example|background|redundant", "reason": "one sentence"}},
    ...
  ]
}}

Rules:
- Preserve the exact text of each item
- Every bullet and sentence must appear exactly once
- If a bullet has multiple sentences mixing types, classify by the dominant type
- Be conservative with "redundant" — only mark if clearly repeated verbatim or nearly verbatim"""


REWRITE_CORE_PROMPT = """Rewrite the following core rules into a minimal instruction block.
Your output MUST be shorter than the input. Remove redundancy, verbose phrasing, and filler.
Do NOT add new information, new headers, or new structure unless essential.
Keep ALL rules — do not drop any rule.
Use tight bullet points. Merge related rules into one bullet where possible.

Rules to include:
{rules}

Output ONLY the rewritten content, no title."""


DEDUPLICATE_EXAMPLES_PROMPT = """You have a list of examples/code snippets from an AI skill.
Many illustrate the same concept. Your task: group by concept, keep ONE best example per concept,
then minimize each kept example to its essential form.

Examples:
{examples}

Output ONLY valid JSON:
{{
  "kept": [
    {{"text": "minimized example text", "concept": "one-phrase concept name"}}
  ],
  "removed_count": 0
}}

Rules:
- Keep the most clear, representative example per concept
- If an example is unique (no similar ones), always keep it
- MINIMIZE each kept example: remove redundant comments, boilerplate, unnecessary variables.
  Keep only the lines that demonstrate the core pattern. Result must still be valid/runnable.
- Do NOT merge different concepts into one example"""


COMPRESS_BACKGROUND_PROMPT = """Summarize the following background/context items into the most concise form.
Preserve ALL unique factual claims (numbers, thresholds, specific reasons).
Remove verbose explanations and repetitive justifications.

Items:
{items}

Output ONLY the summarized markdown text, no title."""


FAITHFULNESS_CHECK_PROMPT = """Compare the ORIGINAL content and the COMPRESSED version.
Check if all key operational concepts and rules from the original are present in the compressed version.

Original:
{original}

Compressed:
{compressed}

Output ONLY valid JSON:
{{
  "faithful": true,
  "missing_concepts": ["concept that was lost (if any)"],
  "verdict": "one sentence summary"
}}"""


# ============================================================
# Prompts - Gate 2 (task generation / evaluation / reclassification) [new]
# ============================================================

TASK_GEN_PROMPT = """You are a QA engineer building a test suite for an AI agent skill.

Generate {n} realistic, diverse user task requests that exercise DIFFERENT parts of this skill.
Design tasks at two levels:
- "core_only": answerable from the core rules alone, no examples or templates needed
- "needs_reference": requires concrete examples, templates, or background details to answer WELL
  (not just to know the rule exists, but to produce a complete, accurate response)

CRITICAL CONSTRAINTS — every task MUST satisfy ALL of the following:
1. The agent answers in PLAIN TEXT or self-contained code only — no external files, no user-provided
   data, no scripts that must be run externally. The agent cannot open, read, or modify actual files.
2. The task is fully answerable from the skill content (core + references). Do NOT ask the agent
   to do something the skill does not cover.
3. For "needs_reference" tasks: the agent genuinely needs specific details from a reference file
   (a code template, a concrete example, a specific threshold or formula) — not just general guidance.
   Frame as: "explain how to...", "write code that...", "what are the specific steps for...",
   "what exact rules apply when..." — never "update this file" or "run this script".

Skill Name: {skill_name}

=== CORE CONTENT (always loaded by agent) ===
{core_body}

=== REFERENCE FILES (agent may load on demand) ===
{refs_summary}

=== ORIGINAL FULL BODY (your context only) ===
{original_snippet}

Output ONLY valid JSON:
{{
  "tasks": [
    {{
      "query": "exact user request string",
      "eval_type": "rubric",
      "reference_answer": ["key concept the response must address", "another key concept"],
      "rubric": [
        {{"criterion": "criterion anchored to a reference_answer key concept", "required": true}}
      ],
      "expected_references": ["examples.md"],
      "difficulty": "core_only"
    }}
  ]
}}

Rules:
- reference_answer: 3-5 key concepts/points a correct response MUST address, derived from the
  ORIGINAL FULL BODY above. Be specific — use concrete concepts, not vague descriptions.
  Good: ["Refuse to fix without root cause", "Ask for error messages and stack trace",
         "Outline Phase 1 evidence-gathering steps"]
  Bad:  ["Be helpful", "Follow best practices"]
- rubric: each criterion must correspond to one reference_answer key concept.
  Criteria test for PRESENCE OF CONCEPT, not exact phrasing.
  3-5 criteria; mark at most 3 as required.
- Use "rubric" for every task. Criteria must be assessable from the response text.
- Generate exactly {n} tasks, mix of both difficulty levels
- CRITICAL: Do NOT reference specific file names in queries or criteria. Describe functional intent only."""


JUDGE_PROMPT = """You are evaluating whether an AI agent's response covers the required concepts.

Task: {query}

Reference Key Points (what a correct answer must address — INTENT matters, exact wording does NOT):
{reference_points}

Agent Response:
{response}

Rubric (each criterion corresponds to one reference key point):
{rubric_text}

Output ONLY valid JSON:
{{
  "results": [
    {{"criterion": "...", "satisfied": true, "note": "brief explanation of whether the concept is present"}}
  ],
  "overall": true,
  "summary": "one sentence"
}}

Rules:
- satisfied: true if the response addresses the underlying concept from the reference key points.
  Paraphrasing, synonyms, different examples, and alternative phrasing ALL count as satisfied.
  Only mark false if the key concept is clearly absent or contradicted.
- Do NOT penalize for style, length, ordering, or tone differences.
- overall: true ONLY IF every [REQUIRED] criterion is satisfied."""


RECLASSIFY_PROMPT = """An AI agent failed to complete tasks using only the skill's core content.
Identify which non-core items from the original classification should be promoted to core_rule.

Failed tasks and what went wrong:
{failed_summary}

Non-core items available for reclassification (index | type | first 80 chars of text):
{non_core_items}

Output ONLY valid JSON:
{{
  "items_to_promote": [
    {{
      "index": 0,
      "reason": "why this must be in core (one sentence)"
    }}
  ]
}}

Rules:
- Return the exact integer "index" shown in the list above
- Only promote items that are DIRECTLY needed for the failed tasks
- Do NOT promote items just because they seem useful — must be clearly required
- If no items clearly caused the failure, return empty list"""


WHEN_GEN_PROMPT = """You are writing the routing metadata for a reference file used by an AI agent.

The agent already has the CORE content loaded. It should only load this reference file when
the core content is INSUFFICIENT for the task.

Skill: {skill_name}
Reference type: {ref_type}  (templates | examples | background)

=== CORE CONTENT (agent already has this) ===
{core_preview}

=== THIS REFERENCE FILE (agent must decide whether to load) ===
{content_preview}

Output ONLY valid JSON:
{{
  "when": "one-sentence description starting with 'you need', describing the GAP this file fills beyond core",
  "topics": "3-5 specific topic keywords or phrases from the file, comma-separated"
}}

Rules:
- "when" must describe what this file adds that core does NOT already cover
- Type-specific guidance:
  * templates: trigger only when agent must PRODUCE/WRITE code, YAML, config, or scripts — NOT for factual questions
  * examples: trigger only when agent needs to see HOW to do something in code — NOT for conceptual questions
  * background: trigger only when agent needs to understand WHY or needs context/rationale — NOT for operational steps
- "topics" must be concrete terms from THIS file (function names, command flags, format names)
- The "when" should help the agent decide in <5 seconds: "do I need this file for my specific task?"
- Bad: "you need GitHub Actions information" (too broad, overlaps with core)
- Good: "you need to WRITE a workflow YAML file or need exact RUV-Swarm command flags not in core" """


COMPRESS_REF_PROMPT = """Compress this skill reference file by removing content that overlaps with the body.
The agent already has the BODY loaded. Keep ALL unique information not already covered.

Skill: {skill_name}
File: {ref_name}

=== BODY (already loaded) ===
{body_preview}

=== FILE TO COMPRESS ===
{content}

Rules:
- Remove content already stated in the body above
- Keep all unique rules, examples, data, formulas, thresholds, code patterns
- Be concise but preserve all unique factual content
- If mostly unique content, keep most of it
- Output ONLY the compressed content, no wrapper or title"""


# ============================================================
# Utility functions
# ============================================================

def segment_body(body: str) -> list[dict]:
    """Split body into sections by ## headings."""
    sections, current_title, current_lines = [], "intro", []
    for line in body.split('\n'):
        if line.startswith('## '):
            if current_lines:
                sections.append({'title': current_title,
                                  'content': '\n'.join(current_lines).strip()})
            current_title, current_lines = line[3:].strip(), []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append({'title': current_title,
                          'content': '\n'.join(current_lines).strip()})
    return [s for s in sections if s['content'].strip()]


def classify_section(skill_name: str, title: str, content: str,
                     max_retries: int = 3) -> list[dict]:
    """Classify section content item by item. Retry up to max_retries times on failure; fallback only if all fail."""
    import time
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = _call_with_retry(
                _ds_client().chat.completions.create,
                model=COMPRESS_MODEL,
                messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(
                    name=skill_name, title=title, content=content)}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            return json.loads(resp.choices[0].message.content).get("items", [])
        except Exception as e:
            last_err = e
            wait = 2 ** attempt  # 1s, 2s, 4s
            print(f"    [classify error attempt {attempt+1}/{max_retries}] {e}, retrying in {wait}s...")
            time.sleep(wait)
    # All retries failed: keep entire content as core_rule to prevent information loss
    print(f"    [classify failed, keeping as core_rule] {last_err}")
    return [{"text": content, "type": "core_rule", "reason": "classification failed, kept as core"}]


def _check_faithfulness(original: str, compressed: str) -> tuple:
    if count_tokens(original) < 50:
        return True, []
    try:
        resp = _call_with_retry(
            _ds_client().chat.completions.create,
            model=COMPRESS_MODEL,
            messages=[{"role": "user", "content": FAITHFULNESS_CHECK_PROMPT.format(
                original=original[:3000], compressed=compressed)}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        data = json.loads(resp.choices[0].message.content)
        faithful = data.get("faithful", True)
        missing  = data.get("missing_concepts", [])
        print(f"    faithfulness: {'✓' if faithful else '✗'} {data.get('verdict','')}")
        return faithful, missing
    except Exception as e:
        print(f"    [faithfulness check error] {e}")
        return True, []


def _build_reference_file(title: str, text: str) -> str:
    return f"# {title}\n\n{text}" if text.strip() else ""


def _generate_ref_when(skill_name: str, ref_type: str, content: str,
                       core_body: str = "") -> dict:
    """Use LLM to generate precise 'when' descriptions and topic keywords. Falls back to defaults on failure."""
    defaults = {
        "templates":  {"when": "you need to WRITE or PRODUCE code, YAML, or config — not for factual questions",   "topics": ""},
        "examples":   {"when": "you need to see HOW to implement something in code — not for conceptual questions",  "topics": ""},
        "background": {"when": "you need to understand WHY a rule exists or need broader context/rationale", "topics": ""},
        "reference":  {"when": "you need specific details from this reference not covered in core instructions", "topics": ""},
    }
    try:
        resp = _call_with_retry(
            _ds_client().chat.completions.create,
            model=COMPRESS_MODEL,
            messages=[{"role": "user", "content": WHEN_GEN_PROMPT.format(
                skill_name=skill_name,
                ref_type=ref_type,
                core_preview=core_body[:600],
                content_preview=content[:800],
            )}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        data = json.loads(resp.choices[0].message.content)
        return {
            "when":   data.get("when",   defaults.get(ref_type, defaults["background"])["when"]),
            "topics": data.get("topics", ""),
        }
    except Exception as e:
        print(f"    [when generation error] {e}")
        return defaults.get(ref_type, defaults["background"])


def _build_references_section(refs: list[dict]) -> str:
    if not refs:
        return ""
    lines = ["", "## References (read_file when needed)"]
    for ref in refs:
        topics = ref.get("topics", "").strip()
        topic_hint = f"  [{topics}]" if topics else ""
        lines.append(f"- {ref['name']} — {ref['when']}{topic_hint}")
    return "\n".join(lines)


# ============================================================
# Original reference files: load + compress (Gate 1)
# ============================================================


def _compress_original_refs(skill_name: str, body: str, raw_refs: list[dict],
                            core_body: str) -> list[dict]:
    """Compress original reference files (Gate 1): deduplicate body overlap -> compress -> faithfulness check."""
    if not raw_refs:
        return []

    def _compress_one(ref: dict) -> dict | None:
        name, content, tokens = ref["name"], ref["content"], ref["tokens"]

        # Small files are kept as-is
        if tokens < 30:
            return ref

        # Only compress .md files; keep other files as-is
        if not name.endswith(".md"):
            return ref

        try:
            resp = _call_with_retry(
            _ds_client().chat.completions.create,
                model=COMPRESS_MODEL,
                messages=[{"role": "user", "content": COMPRESS_REF_PROMPT.format(
                    skill_name=skill_name, ref_name=name,
                    body_preview=body[:2000], content=content)}],
                temperature=0.0,
            )
            comp_text = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"    [ref compress error] {name}: {e}")
            return ref

        comp_tokens = count_tokens(comp_text)

        # Compressed to nearly empty -> content fully overlaps with body, skip
        if comp_tokens < 30:
            print(f"    [skip overlap] {name}: {tokens}→{comp_tokens} tokens")
            return None

        # Expanded -> LLM added content, use original text
        if comp_tokens >= tokens:
            print(f"    [ref no gain] {name}: {tokens}→{comp_tokens}, keeping original")
            return ref

        # Faithfulness check
        faithful, missing = _check_faithfulness(content, comp_text)
        if not faithful:
            print(f"    [ref faithfulness ✗] {name}: {missing}, keeping original")
            return ref

        pct = (1 - comp_tokens / max(tokens, 1)) * 100
        print(f"    [ref compressed] {name}: {tokens}→{comp_tokens} (-{pct:.0f}%)")
        return {"name": name, "content": comp_text, "tokens": comp_tokens,
                "original_tokens": tokens}

    # Parallel compression
    with ThreadPoolExecutor(max_workers=min(len(raw_refs), 6)) as pool:
        results = list(pool.map(_compress_one, raw_refs))

    valid = [r for r in results if r is not None]

    # Parallel when/topics generation
    def _gen_when(ref: dict) -> dict:
        when_data = _generate_ref_when(skill_name, "reference", ref["content"], core_body)
        return {**ref, "when": when_data["when"], "topics": when_data["topics"],
                "source": "original"}

    if not valid:
        return []

    with ThreadPoolExecutor(max_workers=min(len(valid), 6)) as pool:
        final = list(pool.map(_gen_when, valid))

    return final


# ============================================================
# Compressor (Gate 1 internal, existing logic unchanged)
# ============================================================

def _build_core_body(skill_name: str, items: list[dict]) -> tuple:
    if not items:
        return "", True
    by_section = {}
    for item in items:
        by_section.setdefault(item['section'], []).append(item['text'])
    parts = [
        '\n'.join(texts) if sec == 'intro' else f"## {sec}\n" + '\n'.join(texts)
        for sec, texts in by_section.items()
    ]
    raw = '\n\n'.join(parts)
    try:
        resp = _call_with_retry(
            _ds_client().chat.completions.create,
            model=COMPRESS_MODEL,
            messages=[{"role": "user", "content": REWRITE_CORE_PROMPT.format(rules=raw)}],
            temperature=0.0,
        )
        compressed = resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"    [rewrite error] {e}, using original text")
        return raw, True
    faithful, missing = _check_faithfulness(raw, compressed)
    if not faithful:
        print(f"    [faithfulness warning] core compression lost: {missing} -> falling back to original")
        return raw, False
    return compressed, True


def _compress_examples(items: list[dict]) -> tuple:
    if not items:
        return "", 0
    raw_text = '\n---\n'.join(i['text'] for i in items)
    try:
        resp = _call_with_retry(
            _ds_client().chat.completions.create,
            model=COMPRESS_MODEL,
            messages=[{"role": "user", "content": DEDUPLICATE_EXAMPLES_PROMPT.format(
                examples=raw_text)}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        data  = json.loads(resp.choices[0].message.content)
        kept  = data.get("kept", [])
        removed = data.get("removed_count", len(items) - len(kept))
        if not kept:
            return raw_text, 0
        result = '\n\n'.join(f"**{k['concept']}**\n{k['text']}" for k in kept)
        print(f"    examples: {len(items)} -> {len(kept)} items")
        faithful, missing = _check_faithfulness(raw_text, result)
        if not faithful:
            print(f"    [faithfulness warning] examples compression lost: {missing} -> falling back to original")
            return raw_text, 0
        return result, removed
    except Exception as e:
        print(f"    [example dedup error] {e}")
        return raw_text, 0


def _collect_templates(items: list[dict]) -> tuple:
    if not items:
        return "", 0
    raw_text = '\n---\n'.join(i['text'] for i in items)
    try:
        resp = _call_with_retry(
            _ds_client().chat.completions.create,
            model=COMPRESS_MODEL,
            messages=[{"role": "user", "content": DEDUPLICATE_EXAMPLES_PROMPT.format(
                examples=raw_text)}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        data    = json.loads(resp.choices[0].message.content)
        kept    = data.get("kept", [])
        removed = data.get("removed_count", len(items) - len(kept))
        if not kept:
            return raw_text, 0
        result = '\n\n'.join(f"**{k['concept']}**\n{k['text']}" for k in kept)
        print(f"    templates: {len(items)} -> {len(kept)} items")
        faithful, _ = _check_faithfulness(raw_text, result)
        if not faithful:
            print(f"    [faithfulness warning] templates compression lost -> falling back to original")
            return raw_text, 0
        return result, removed
    except Exception as e:
        print(f"    [template dedup error] {e}")
        return raw_text, 0


def _compress_background(items: list[dict]) -> tuple:
    if not items:
        return "", True
    raw_text = '\n'.join(i['text'] for i in items)
    try:
        resp = _call_with_retry(
            _ds_client().chat.completions.create,
            model=COMPRESS_MODEL,
            messages=[{"role": "user", "content": COMPRESS_BACKGROUND_PROMPT.format(
                items=raw_text)}],
            temperature=0.0,
        )
        compressed = resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"    [background compress error] {e}")
        return raw_text, True
    faithful, missing = _check_faithfulness(raw_text, compressed)
    if not faithful:
        print(f"    [faithfulness warning] background compression lost: {missing} -> falling back to original")
        return raw_text, False
    return compressed, True


# ============================================================
# Gate 2 - Task generation
# ============================================================

def generate_tasks(skill_name: str, core_body: str, references: list,
                   original_body: str) -> list[dict]:
    refs_summary = "\n".join(
        f"- {r['name']} ({count_tokens(r['content'])} tokens): {r.get('when','')}"
        for r in references
    ) or "(no reference files)"
    prompt = TASK_GEN_PROMPT.format(
        n=TASKS_PER_SKILL,
        skill_name=skill_name,
        core_body=core_body[:2000],
        refs_summary=refs_summary,
        original_snippet=original_body[:3000],
    )
    try:
        resp = _call_with_retry(
            _qwen_client().chat.completions.create,
            model=TASK_GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        tasks = json.loads(resp.choices[0].message.content).get("tasks", [])
        print(f"  generated {len(tasks)} tasks")
        for t in tasks:
            print(f"    [{t.get('difficulty','?')}] {t['query'][:65]}...")
        return tasks
    except Exception as e:
        print(f"  [task generation error] {e}")
        return []


# ============================================================
# Gate 2 - Agent execution (three conditions)
# ============================================================

READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Read a skill reference file for detailed examples, templates, or background context. "
            "Use only when the core instructions are insufficient for the task."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Reference filename, e.g. 'examples.md', 'templates.md', 'background.md'"
                }
            },
            "required": ["path"],
        },
    },
}


def _handle_read_file(path: str, references: list) -> str:
    for ref in references:
        if ref["name"] == path:
            return ref["content"]
    return f"[File not found: {path}. Available: {[r['name'] for r in references]}]"


def run_agent(condition: str, body: str, query: str, references: list) -> dict:
    """
    Run agent under given condition.
    condition A: original body and references
    condition C: optimized core and references
    condition D: no skill content (lower bound)
    """
    if condition == "D":
        system = "You are a helpful AI agent."
    else:
        system = (
            "You are a helpful AI agent. Use the following skill instructions to complete the task.\n\n"
            "=== SKILL INSTRUCTIONS ===\n" + body
        )
    tools = [READ_FILE_TOOL] if condition in {"A", "C"} else None
    if tools:
        names = ", ".join(ref["name"] for ref in references) or "(none)"
        system += (
            "\n\n=== NOTE ===\n"
            f"Available reference files: {names}. "
            "Use read_file if you need more detail."
        )
    messages      = [{"role": "system", "content": system},
                     {"role": "user",   "content": query}]
    files_loaded  = []

    for turn in range(MAX_TOOL_CALLS + 1):
        kwargs = dict(model=AGENT_MODEL, messages=messages, temperature=0.0)
        if tools and turn < MAX_TOOL_CALLS and len(files_loaded) < MAX_TOOL_CALLS:
            kwargs["tools"]       = tools
            kwargs["tool_choice"] = "auto"

        resp = _call_with_retry(
            _qwen_client().chat.completions.create, **kwargs)
        msg  = resp.choices[0].message

        if not msg.tool_calls:
            raw = msg.content or ""
            # qwen3 extended-thinking: strip <think>...</think> block to get the actual answer
            final = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            messages.append({"role": "assistant", "content": final})
            return {"response": final, "files_loaded": files_loaded}

        tool_results = []
        for tc in msg.tool_calls:
            if len(files_loaded) < MAX_TOOL_CALLS:
                path = json.loads(tc.function.arguments).get("path", "")
                content = _handle_read_file(path, references)
                files_loaded.append(path)
            else:
                content = "[Reference read limit reached]"
            tool_results.append({
                "tool_call_id": tc.id,
                "role": "tool",
                "name": "read_file",
                "content": content,
            })
        messages.append(msg)
        messages.extend(tool_results)

    last_raw = next((m.get("content","") for m in reversed(messages)
                     if isinstance(m, dict) and m.get("role") == "assistant"), "")
    last = re.sub(r"<think>.*?</think>", "", last_raw, flags=re.DOTALL).strip()
    return {"response": last or "[Tool call limit reached]", "files_loaded": files_loaded}


# ============================================================
# Gate 2 — evaluate
# ============================================================

def eval_rubric(response: str, task: dict) -> dict:
    rubric_text = "\n".join(
        f"- {'[REQUIRED] ' if r.get('required') else '[OPTIONAL] '}{r['criterion']}"
        for r in task.get("rubric", [])
    )
    ref_points = task.get("reference_answer", [])
    reference_points = "\n".join(f"- {p}" for p in ref_points) if ref_points else "(none provided)"
    try:
        resp = _call_with_retry(
            _qwen_client().chat.completions.create,
            model=TASK_GEN_MODEL,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                query=task["query"],
                reference_points=reference_points,
                response=response[:6000],
                rubric_text=rubric_text)}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        data    = json.loads(resp.choices[0].message.content)
        results = data.get("results", [])

        # The judge may echo [REQUIRED]/[OPTIONAL] prefix in returned criterion names.
        # Strip it before matching so "Response X" == "[REQUIRED] Response X".
        def _strip_prefix(s: str) -> str:
            return re.sub(r"^\[(REQUIRED|OPTIONAL)\]\s*", "", s or "").strip()

        # Overall is strictly computed by code: pass only if all required criteria pass
        # Does not rely on the LLM's own overall field
        required = [r for r in task.get("rubric", []) if r.get("required")]
        required_criteria = {r["criterion"] for r in required}
        if required_criteria:
            satisfied_required = sum(
                1 for r in results
                if _strip_prefix(r.get("criterion")) in required_criteria and r.get("satisfied")
            )
            score   = satisfied_required / len(required_criteria)
            overall = satisfied_required == len(required_criteria)
        else:
            satisfied = sum(1 for r in results if r.get("satisfied"))
            score     = satisfied / max(len(results), 1)
            overall   = score >= 0.5

        return {"overall": overall, "score": round(score, 3),
                "rubric_results": results,
                "summary": data.get("summary", "")}
    except Exception as e:
        print(f"    [judge error] {e}")
        return {"overall": False, "rubric_results": [], "summary": str(e)}


def evaluate(response: str, task: dict) -> dict:
    return eval_rubric(response, task)


def _run_gate2(skill_name: str, core_body: str, original_body: str,
               references: list, tasks: list, original_references: list) -> dict:
    """
    Run Condition A (original full body) and Condition C (core + read_file).
    Both scored against reference_answer GT.
    Per-task retention ratio = min(score_C / score_A, 1.0).
    If score_A == 0: ratio = 1.0 (A also failed; C is not worse).
    Pass condition: score_C >= score_A for all tasks (retention_mean == 1.0).
    """
    if not tasks:
        return {"retention_mean": None, "task_results_A": [], "task_results_C": []}

    def run_condition(cond: str, body: str, task_list: list) -> list:
        results = []
        for i, task in enumerate(task_list):
            print(f"    [Cond {cond}] Task {i+1}/{len(tasks)}: {task['query'][:50]}...")
            refs = original_references if cond == "A" else references if cond == "C" else []
            run    = run_agent(cond, body, task["query"], refs)
            result = evaluate(run["response"], task)
            score  = result.get("score", 1.0 if result["overall"] else 0.0)
            status = f"{'✅' if result['overall'] else '❌'} score={score:.2f}"
            loaded = f" loaded={run['files_loaded']}" if run["files_loaded"] else ""
            print(f"      {status}{loaded}")
            if score == 0.0:
                # Debug: show which criteria failed
                req_set = {rb["criterion"] for rb in task.get("rubric", []) if rb.get("required")}
                for r in result.get("rubric_results", []):
                    sat  = "✓" if r.get("satisfied") else "✗"
                    # Strip [REQUIRED]/[OPTIONAL] prefix the judge may have echoed back
                    crit = re.sub(r"^\[(REQUIRED|OPTIONAL)\]\s*", "", r.get("criterion", "")).strip()
                    req  = "[REQUIRED]" if crit in req_set else "[OPT]"
                    print(f"        {sat} {req} {crit[:70]}")
            results.append({
                "task_index":       i,
                "query":            task["query"],
                "difficulty":       task.get("difficulty", "?"),
                "condition":        cond,
                "files_loaded":     run["files_loaded"],
                "eval":             result,
                "score":            score,
                "response_snippet": run["response"][:300],
            })
        return results

    print(f"\n  [Gate 2] Condition D — no skill content (lower bound)")
    res_D = run_condition("D", "", tasks)

    print(f"\n  [Gate 2] Condition A — original full body (vs reference_answer GT)")
    res_A = run_condition("A", original_body, tasks)

    print(f"\n  [Gate 2] Condition C — core + read_file (vs reference_answer GT)")
    res_C = run_condition("C", core_body, tasks)

    # per-task retention ratio: min(score_C / score_A, 1.0)
    per_task_ratios = []
    for a, c in zip(res_A, res_C):
        score_a = a["score"]
        score_c = c["score"]
        ratio = 1.0 if score_a == 0 else min(score_c / score_a, 1.0)
        per_task_ratios.append(ratio)

    score_D_mean   = round(sum(r["score"] for r in res_D) / max(len(res_D), 1), 3)
    score_A_mean   = round(sum(r["score"] for r in res_A) / max(len(res_A), 1), 3)
    score_C_mean   = round(sum(r["score"] for r in res_C) / max(len(res_C), 1), 3)
    retention_mean = round(sum(per_task_ratios) / max(len(per_task_ratios), 1), 3)
    # skill_lift: how much A improves over no-skill baseline
    skill_lift     = round(score_A_mean - score_D_mean, 3)

    all_ref_names = [r["name"] for r in references]
    loading_stats = {
        name: sum(1 for r in res_C if name in r["files_loaded"])
        for name in all_ref_names
    }

    print(f"\n  Gate 2 result:")
    print(f"    score_D (no skill):     {score_D_mean:.2f}")
    print(f"    score_A (full body):    {score_A_mean:.2f}")
    print(f"    score_C (core+tool):    {score_C_mean:.2f}")
    print(f"    skill_lift (A-D):       {skill_lift:+.2f}")
    print(f"    retention (C/A ratio):  {retention_mean:.2f}")
    print(f"    Reference loading: {loading_stats}")
    if skill_lift < 0.1:
        print(f"    Warning: skill_lift near 0, tasks may not depend on skill content")

    return {
        "score_D_mean":    score_D_mean,
        "score_A_mean":    score_A_mean,
        "score_C_mean":    score_C_mean,
        "skill_lift":      skill_lift,
        "retention_mean":  retention_mean,
        "per_task_ratios": per_task_ratios,
        "loading_stats":   loading_stats,
        "task_results_D":  res_D,
        "task_results_A":  res_A,
        "task_results_C":  res_C,
    }


# ============================================================
# Feedback loop - reclassification
# ============================================================

def analyze_for_reclassification(failed_results: list, all_items: list) -> list[int]:
    """
    Given failed Condition C tasks, have the LLM identify which non-core entries need to be promoted to core_rule.
    Returns the global all_items indices of entries to promote from the non_core sublist.
    """
    if not failed_results:
        return []

    failed_summary = "\n".join(
        f"- Task: {r['query'][:60]}...\n"
        f"  Wrong: {'; '.join(c['criterion'] for c in r['eval'].get('rubric_results', []) if not c.get('satisfied', True))}"
        for r in failed_results
    )

    # List only non-core entries, tracking their actual indices in all_items
    non_core_with_idx = [
        (idx, item) for idx, item in enumerate(all_items)
        if item.get("type") in ("example", "background", "template")
    ]
    if not non_core_with_idx:
        return []

    # List shown to LLM: indices start at 0 (corresponding to positions in non_core_with_idx)
    non_core_lines = "\n".join(
        f"{pos} | [{item['type']}] {item['text'][:80]}"
        for pos, (_, item) in enumerate(non_core_with_idx)
    )

    try:
        resp = _call_with_retry(
            _qwen_client().chat.completions.create,
            model=TASK_GEN_MODEL,
            messages=[{"role": "user", "content": RECLASSIFY_PROMPT.format(
                failed_summary=failed_summary,
                non_core_items=non_core_lines,
            )}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        data = json.loads(resp.choices[0].message.content)
        # LLM returns positions in non_core_with_idx (pos); convert back to actual all_items indices
        promoted_positions = [p["index"] for p in data.get("items_to_promote", [])
                              if isinstance(p.get("index"), int)]
        all_items_indices = [
            non_core_with_idx[pos][0]
            for pos in promoted_positions
            if 0 <= pos < len(non_core_with_idx)
        ]
        print(f"    -> identified {len(all_items_indices)} entries to promote")
        return all_items_indices
    except Exception as e:
        print(f"    [reclassification analysis error] {e}")
        return []


def apply_reclassification(all_items: list, indices: list[int]) -> int:
    """
    Change the type of entries at specified indices to core_rule.
    Returns the number of entries actually modified.
    """
    promoted = 0
    for idx in indices:
        if 0 <= idx < len(all_items):
            item = all_items[idx]
            if item.get("type") in ("example", "background", "template"):
                item["type"] = "core_rule"
                item["reason"] = item.get("reason", "") + " [promoted by task feedback]"
                promoted += 1
    return promoted


# ============================================================
# Main pipeline orchestration
# ============================================================

def _classify_body(skill_name: str, body: str) -> tuple:
    """Step 1: classify (done once). Returns (all_items, type_counts)."""
    sections = segment_body(body)
    print(f"  segment: {len(sections)} sections")

    # Classify all sections in parallel (no data dependencies between sections)
    section_results: dict[int, list] = {}
    with ThreadPoolExecutor(max_workers=min(len(sections), 8)) as pool:
        futures = {
            pool.submit(classify_section, skill_name, sec['title'], sec['content']): i
            for i, sec in enumerate(sections)
        }
        for future in futures:
            idx = futures[future]
            sec = sections[idx]
            print(f"    classify [{sec['title']}] ({count_tokens(sec['content'])} tokens)...")
            section_results[idx] = future.result()

    # Assemble results in original order
    all_items   = []
    type_counts = {"core_rule": 0, "template": 0, "example": 0, "background": 0, "redundant": 0}
    for i, sec in enumerate(sections):
        items = section_results[i]
        for item in items:
            item['section'] = sec['title']
            type_counts[item.get('type', 'core_rule')] = \
                type_counts.get(item.get('type', 'core_rule'), 0) + 1
            all_items.append(item)

    print(f"  classification result: " + ", ".join(f"{k}={v}" for k, v in type_counts.items()))
    return all_items, type_counts


def _compress_from_classification(skill_name: str, all_items: list) -> dict:
    """Step 2+3: compress based on current classification + faithfulness check (Gate 1)."""
    core_items       = [i for i in all_items if i['type'] == 'core_rule']
    template_items   = [i for i in all_items if i['type'] == 'template']
    example_items    = [i for i in all_items if i['type'] == 'example']
    background_items = [i for i in all_items if i['type'] == 'background']
    redundant_items  = [i for i in all_items if i['type'] == 'redundant']

    type_counts = {
        "core_rule": len(core_items), "template": len(template_items),
        "example": len(example_items), "background": len(background_items),
        "redundant": len(redundant_items),
    }

    print(f"\n  [compress] core={len(core_items)}, template={len(template_items)}, "
          f"example={len(example_items)}, background={len(background_items)}")

    # Phase 1: 4 compression functions are fully independent, run in parallel
    with ThreadPoolExecutor(max_workers=4) as pool:
        f_core = pool.submit(_build_core_body, skill_name, core_items)
        f_tmpl = pool.submit(_collect_templates, template_items)
        f_exam = pool.submit(_compress_examples, example_items)
        f_bg   = pool.submit(_compress_background, background_items)

    core_text,       core_faithful    = f_core.result()
    templates_text,  templates_removed = f_tmpl.result()
    examples_text,   examples_removed  = f_exam.result()
    background_text, bg_faithful       = f_bg.result()

    # Phase 2: ref_when calls are mutually independent (all depend on Phase 1's core_text), run in parallel
    ref_when_futures: dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        if templates_text.strip():
            ref_when_futures["templates"] = pool.submit(
                _generate_ref_when, skill_name, "templates", templates_text, core_text)
        if examples_text.strip():
            ref_when_futures["examples"] = pool.submit(
                _generate_ref_when, skill_name, "examples", examples_text, core_text)
        if background_text.strip():
            ref_when_futures["background"] = pool.submit(
                _generate_ref_when, skill_name, "background", background_text, core_text)

    references = []
    if "templates" in ref_when_futures:
        when_tmpl = ref_when_futures["templates"].result()
        references.append({"name": "templates.md",
                            "when": when_tmpl["when"], "topics": when_tmpl["topics"],
                            "content": _build_reference_file("Templates", templates_text),
                            "removed_count": templates_removed})
    if "examples" in ref_when_futures:
        when_ex = ref_when_futures["examples"].result()
        references.append({"name": "examples.md",
                            "when": when_ex["when"], "topics": when_ex["topics"],
                            "content": _build_reference_file("Examples", examples_text),
                            "removed_count": examples_removed})
    if "background" in ref_when_futures:
        when_bg = ref_when_futures["background"].result()
        references.append({"name": "background.md",
                            "when": when_bg["when"], "topics": when_bg["topics"],
                            "content": _build_reference_file("Background & Context", background_text)})

    core_body_final = core_text + _build_references_section(references)

    return {
        "core_body":          core_body_final,
        "core_text":          core_text,       # Without References section, used for merging original refs
        "core_faithful":      core_faithful,
        "background_faithful": bg_faithful,
        "references":         references,      # generated refs only (from body classification)
        "examples_removed":   examples_removed,
        "templates_removed":  templates_removed,
        "type_counts":        type_counts,
        "core_items":         core_items,
        "example_items":      example_items,
        "background_items":   background_items,
        "redundant_items":    redundant_items,
    }


def restructure_body(skill_name: str, body: str, *,
                     original_references: list | None = None,
                     task_eval: bool = False) -> dict:
    """
    Full pipeline: classify -> compress -> Gate 1 -> Gate 2 -> feedback reclassification (loop).
    Optional Gate 2 scores responses with a rubric and never executes generated code.
    """
    # Step 1: classify (done once; classification results are dynamically modified in the feedback loop)
    print("\n  [Step 1] Content classification...")
    all_items, _ = _classify_body(skill_name, body)

    # Step 1.5: load original reference files
    raw_refs = original_references or []
    if raw_refs:
        raw_ref_tokens = sum(f["tokens"] for f in raw_refs)
        print(f"  original refs: {len(raw_refs)} files ({raw_ref_tokens} tokens)")

    # Task generation is outside the loop (tasks contain reference_answer, generated from original body, independent of compression)
    tasks = []
    compressed_original = []  # Cached after first-round compression, reused in subsequent iterations

    best = None
    for iteration in range(MAX_FEEDBACK_ITERS + 1):
        if iteration > 0:
            print(f"\n  -- Feedback iteration {iteration} ----------------------------------")

        # Step 2: compress body (Gate 1)
        print(f"\n  [Step 2] Compressing (current classification)...")
        comp = _compress_from_classification(skill_name, all_items)

        # Step 2.5: compress original refs (Gate 1, first round only; refs do not participate in feedback reclassification)
        if iteration == 0 and raw_refs:
            print(f"\n  [Step 2.5] Compressing {len(raw_refs)} original reference files...")
            compressed_original = _compress_original_refs(
                skill_name, body, raw_refs, comp["core_text"])
            if compressed_original:
                orig_tokens = sum(r.get("original_tokens", r["tokens"]) for r in compressed_original)
                comp_tokens = sum(r["tokens"] for r in compressed_original)
                print(f"    original refs: {orig_tokens}→{comp_tokens} tokens "
                      f"(-{(1-comp_tokens/max(orig_tokens,1))*100:.0f}%)")

        # Merge all refs: generated (from body) + compressed original
        existing_names = {ref["name"] for ref in raw_refs}
        generated = []
        for ref in comp["references"]:
            name = f"references/skillreducer/{ref['name']}"
            while name in existing_names:
                name = name.replace("/skillreducer/", "/skillreducer/generated/", 1)
            generated.append({**ref, "name": name})
        all_refs = generated + compressed_original
        core_body_final = comp["core_text"] + _build_references_section(all_refs)

        if not task_eval:
            best = {**comp, "core_body": core_body_final, "references": all_refs,
                    "all_items": all_items, "gate2": None, "feedback_iters": 0,
                    "original_refs": compressed_original, "raw_refs": raw_refs}
            break

        # Task generation (generated on first iteration, reused afterwards; contains reference_answer)
        if iteration == 0:
            print(f"\n  [Step 3] Gate 2 - Task generation (with reference_answer)...")
            tasks = generate_tasks(skill_name, core_body_final,
                                   all_refs, body)
            if not tasks:
                print("  Task generation failed; keeping the original body and references")
                best = {**comp, "core_body": body, "references": raw_refs,
                        "all_items": all_items, "gate2": None, "feedback_iters": 0,
                        "original_refs": raw_refs, "raw_refs": raw_refs, "fallback": True}
                break

        # Step 4: Gate 2 - Condition A vs Condition C, compared against reference_answer GT
        print(f"\n  [Step 4] Gate 2 - Task evaluation (iteration {iteration})...")
        gate2 = _run_gate2(skill_name, core_body_final, body, all_refs, tasks, raw_refs)

        best = {**comp, "core_body": core_body_final, "references": all_refs,
                "all_items": all_items, "gate2": gate2,
                "feedback_iters": iteration, "tasks": tasks,
                "original_refs": compressed_original, "raw_refs": raw_refs}

        retention      = gate2["retention_mean"]
        per_task_ratios = gate2.get("per_task_ratios", [])

        # Find tasks where score_C < score_A (ratio < 1.0)
        weak_tasks = [
            c for c, ratio in zip(gate2["task_results_C"], per_task_ratios)
            if ratio < 1.0
        ]

        # All tasks pass if score_C >= score_A (retention == 1.0)
        if not weak_tasks:
            print(f"  Gate 2 passed (retention: {retention:.2f})" if retention else "  Gate 2 passed: no tasks")
            break

        if iteration == MAX_FEEDBACK_ITERS:
            print(f"  Reached the feedback limit ({MAX_FEEDBACK_ITERS})")
            break

        # Step 5: Feedback - tasks with score_C < score_A trigger reclassification
        print(f"\n  [Step 5] Feedback analysis (retention: {retention:.2f}, {len(weak_tasks)} tasks with score_C < score_A)...")
        failed = weak_tasks

        indices = analyze_for_reclassification(failed, all_items)

        if not indices:
            print("  Warning: unable to identify entries to promote, stopping iteration")
            break

        promoted = apply_reclassification(all_items, indices)
        print(f"  -> promoted {promoted} entries to core_rule, recompressing...")

    gate2 = best.get("gate2")
    if gate2 and gate2["score_A_mean"] - gate2["score_C_mean"] > FALLBACK_TOLERANCE:
        best.update(core_body=body, references=raw_refs, fallback=True)
    return best


# ============================================================
# Single skill entry point
# ============================================================
