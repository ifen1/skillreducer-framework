#!/usr/bin/env python3
"""
Stage 1: Routing Layer Optimizer
=================================
Bidirectional optimization of skill description routing quality:
- Long/redundant description -> compress (Adversarial Test-Driven Delta Debugging)
- Short/missing description -> generate effective routing description from body

Evaluation: use routing accuracy to measure effect before and after optimization.

"""

import math
import os
import re
import json
import random
from collections import Counter
from functools import lru_cache
import tiktoken
from openai import OpenAI


# ============================================================
# Config
# ============================================================
BASE_URL = "https://api.deepseek.com"
COMPRESS_MODEL = "deepseek-chat"
EVAL_MODEL = "deepseek-reasoner"


# Description classification threshold (tokens)
SHORT_THRESHOLD = 40   # <= 40 tokens: too short, need to generate from body first

@lru_cache(maxsize=1)
def get_client():
    return OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=BASE_URL)


@lru_cache(maxsize=1)
def _encoder():
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text, disallowed_special=())) if text else 0


# ============================================================
# 1. Description Quality Classifier
# ============================================================
def classify_description(name: str, description: str, body: str) -> dict:
    """Classify description quality.

    Returns:
        {
            "category": "missing" | "too_short" | "adequate" | "verbose",
            "desc_tokens": int,
            "body_tokens": int,
            "needs_generation": bool,   # missing / too_short
            "needs_compression": bool,  # verbose
        }
    """
    desc_tokens = count_tokens(description)
    body_tokens = count_tokens(body)

    if not description or desc_tokens == 0:
        category = "missing"
    elif desc_tokens <= SHORT_THRESHOLD:
        category = "too_short"
    else:
        category = "has_desc"

    return {
        "name": name,
        "category": category,
        "desc_tokens": desc_tokens,
        "body_tokens": body_tokens,
        "needs_generation": category in ("missing", "too_short"),
    }


# ============================================================
# 2. Description Generator (from body)
# ============================================================
GENERATE_DESC_PROMPT = """You are writing a routing description for an AI agent skill.
The description's ONLY purpose is to help a router decide WHEN to activate this skill.

Skill name: {name}
Skill body (full instructions):
{body}

Write a concise routing description (20-40 tokens) that captures:
1. The skill's PRIMARY capability (what it does)
2. The TRIGGER condition (when to use it)
3. Any UNIQUE identifiers that distinguish it from similar skills

Output ONLY the description, nothing else. No quotes, no prefix."""

IMPROVE_SHORT_DESC_PROMPT = """You are improving a routing description for an AI agent skill.
The current description is too short to reliably route to this skill.

Skill name: {name}
Current description: {current_desc}
Skill body (full instructions):
{body}

Rewrite the description to be 20-40 tokens. Keep any useful information from the current description.
Add the skill's primary capability and trigger condition from the body.

Output ONLY the improved description, nothing else. No quotes, no prefix."""


def generate_description(name: str, body: str, current_desc: str = "") -> str:
    """Generate or improve description from body."""
    # Truncate body to avoid excessive length (take first 3000 tokens)
    body_truncated = body
    if count_tokens(body) > 3000:
        words = body.split()
        while count_tokens(" ".join(words)) > 3000:
            words = words[:int(len(words) * 0.8)]
        body_truncated = " ".join(words)

    if current_desc and count_tokens(current_desc) > 0:
        prompt = IMPROVE_SHORT_DESC_PROMPT.format(
            name=name, current_desc=current_desc, body=body_truncated
        )
    else:
        prompt = GENERATE_DESC_PROMPT.format(name=name, body=body_truncated)

    try:
        resp = get_client().chat.completions.create(
            model=COMPRESS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [generate error] {e}")
        return current_desc


# ============================================================
# 3. Description Compressor (delta debugging, from good_skill.py)
# ============================================================
ADVERSARIAL_SKILL_PROMPT = """You are an adversarial AI. Generate a competitor skill description that is hard to distinguish from the target skill, but still clearly different.

Target Skill Name: {name}
Target Skill Description: {description}

The adversarial skill must:
1. Operate in the same domain and share key vocabulary with the target
2. Describe a RELATED but DISTINCT capability
3. Be realistic and plausible as a standalone AI agent skill
4. Be confusable with the target when the target's description is vague
5. CRITICAL: Do NOT copy the target's exact trigger phrases or unique identifiers

Output ONLY the adversarial skill description, nothing else."""

SPLIT_PROMPT = """Split the following skill description into semantic units (phrases or clauses that each carry distinct routing-relevant information).
For each unit, also provide a maximally compressed version that preserves its core meaning.

Description: {description}

Output ONLY valid JSON:
{{
  "units": [
    {{"original": "...", "compressed": "..."}},
    {{"original": "...", "compressed": "..."}}
  ]
}}"""

POLISH_PROMPT = """Rewrite the following skill description fragment into a single complete, grammatically correct sentence or short paragraph.
Do NOT add new information. Only fix grammar and make it self-contained.

Fragment: {fragment}

Output ONLY the rewritten description, nothing else."""


def _split_and_compress(text: str) -> list:
    try:
        resp = get_client().chat.completions.create(
            model=COMPRESS_MODEL,
            messages=[{"role": "user", "content": SPLIT_PROMPT.format(description=text)}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        data = json.loads(resp.choices[0].message.content)
        units = data.get("units", [])
        return [(u["original"], u["compressed"]) for u in units if u.get("original")]
    except Exception as e:
        print(f"  [split error] {e}")
        return [(text, text)]


def _polish(fragment: str) -> str:
    try:
        resp = get_client().chat.completions.create(
            model=COMPRESS_MODEL,
            messages=[{"role": "user", "content": POLISH_PROMPT.format(fragment=fragment)}],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return fragment


def _generate_adversarial(name: str, desc: str) -> str:
    resp = get_client().chat.completions.create(
        model=COMPRESS_MODEL,
        messages=[{"role": "user", "content": ADVERSARIAL_SKILL_PROMPT.format(
            name=name, description=desc
        )}],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


def _build_desc(units, choices):
    parts = []
    for (orig, comp), choice in zip(units, choices):
        if choice == "delete":
            continue
        elif choice == "compress":
            parts.append(comp)
        else:
            parts.append(orig)
    return " ".join(parts)


def _partition(lst, n):
    size = max(1, len(lst) // n)
    return [lst[i:i + size] for i in range(0, len(lst), size)]


# ============================================================
# 4. Routing Evaluator
# ============================================================
def test_routing(candidate_desc: str, target_name: str,
                 distractors: list, skill_dict: dict) -> bool:
    """Test whether candidate_desc can be correctly routed to target_name among distractors.
    Uses a generic query for testing."""
    candidates = [target_name] + distractors
    random.shuffle(candidates)

    skills_list = "\n".join(
        f"- {name} [{name}/SKILL.md]: {candidate_desc if name == target_name else skill_dict.get(name, '')}"
        for name in candidates
    )

    system_prompt = f"""AVAILABLE SKILLS

Skills are pre-packaged instructions for specific tasks.
When a user request matches a skill description, read the full
SKILL.md file to get detailed instructions.

- These are the currently available skills for "evaluation" mode:
{skills_list}

To use a skill:
1. Identify which skill matches the user's request based on the description
2. Use read_file to load the full SKILL.md file from the path shown in brackets
3. Follow the instructions in the skill file

Reply with ONLY the exact skill name from the list above, nothing else."""

    return system_prompt  # Return the constructed prompt; actual testing is done externally


def routing_test_with_queries(candidate_desc: str, target_name: str,
                              queries: list, distractors: list,
                              skill_dict: dict) -> tuple:
    """Test routing accuracy with multiple queries.
    Returns: (correct_count, total_count)
    """
    candidates = [target_name] + distractors
    correct = 0

    for query in queries:
        shuffled = candidates[:]
        random.shuffle(shuffled)

        skills_list = "\n".join(
            f"- {name} [{name}/SKILL.md]: "
            f"{candidate_desc if name == target_name else skill_dict.get(name, '')}"
            for name in shuffled
        )

        system_prompt = f"""AVAILABLE SKILLS

Skills are pre-packaged instructions for specific tasks.
When a user request matches a skill description, read the full
SKILL.md file to get detailed instructions.

- These are the currently available skills for "evaluation" mode:
{skills_list}

To use a skill:
1. Identify which skill matches the user's request based on the description
2. Use read_file to load the full SKILL.md file from the path shown in brackets
3. Follow the instructions in the skill file

Reply with ONLY the exact skill name from the list above, nothing else."""

        try:
            resp = get_client().chat.completions.create(
                model=EVAL_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
            )
            content = resp.choices[0].message.content
            if "</think>" in content:
                content = content.split("</think>")[-1]
            selected = content.strip()
            target_safe = "".join(
                c if c.isalnum() or c in ['-', '_'] else '_' for c in target_name
            )
            if target_safe.casefold() == selected.casefold():
                correct += 1
        except Exception as e:
            print(f"    [routing error] {e}")

    return correct, len(queries)


# ============================================================
# 5. Query Generator (for skills without benchmark queries)
# ============================================================
GENERATE_QUERIES_PROMPT = """Generate {n} realistic user queries that should trigger the following AI agent skill.
Each query should be a natural request that a user might type.

Skill name: {name}
Skill description: {description}
Skill body excerpt: {body_excerpt}

Output ONLY valid JSON:
{{
  "queries": ["query1", "query2", ...]
}}"""


def generate_test_queries(name: str, description: str, body: str, n: int = 3) -> list:
    """Generate test queries for skills without benchmark queries."""
    body_excerpt = body[:1500] if body else ""

    # If there's no description either, use the first few lines of body
    desc_for_prompt = description if description else body[:300]

    try:
        resp = get_client().chat.completions.create(
            model=COMPRESS_MODEL,
            messages=[{"role": "user", "content": GENERATE_QUERIES_PROMPT.format(
                n=n, name=name, description=desc_for_prompt, body_excerpt=body_excerpt
            )}],
            response_format={"type": "json_object"},
            temperature=0.5,
        )
        data = json.loads(resp.choices[0].message.content)
        return data.get("queries", [])
    except Exception as e:
        print(f"  [query gen error] {e}")
        return []


# ============================================================
# 6. Distractor Selection
# ============================================================
def select_distractors(target_name: str, all_skills: dict, n: int = 4) -> list:
    """Rank other descriptions by TF-IDF cosine similarity to the target."""
    candidates = [name for name in all_skills if name != target_name and all_skills[name]]
    names = [target_name, *candidates]
    docs = {
        name: Counter(re.findall(r"\w+", f"{name} {all_skills.get(name, '')}".casefold()))
        for name in names
    }
    df = Counter(word for doc in docs.values() for word in doc)

    def weights(doc):
        return {word: tf * (1 + math.log((1 + len(names)) / (1 + df[word])))
                for word, tf in doc.items()}

    target = weights(docs[target_name])
    target_norm = math.sqrt(sum(v * v for v in target.values()))

    def score(name):
        vec = weights(docs[name])
        norm = target_norm * math.sqrt(sum(v * v for v in vec.values()))
        return sum(target.get(w, 0) * v for w, v in vec.items()) / norm if norm else 0.0

    return sorted(candidates, key=lambda name: (-score(name), name))[:n]


# ============================================================
# 7. Delta Debugging Compressor (adapted from good_skill.py)
# ============================================================
def compress_description(target_name: str, original_desc: str,
                         queries: list, distractors: list,
                         skill_dict: dict, return_details: bool = False):
    """Perform delta debugging compression on longer descriptions.

    If return_details=True, returns (compressed_desc, details_dict) where
    details_dict contains units, active_indices, all_indices for Phase 2.
    """
    print(f"  [compress] starting delta debugging for [{target_name}]")

    # Generate adversarial skill
    adv_name = "shadow-skill"
    MAX_ADV_RETRIES = 3
    augmented_dict = None

    for attempt in range(1, MAX_ADV_RETRIES + 1):
        adv_desc = _generate_adversarial(target_name, original_desc)
        augmented_dict = {**skill_dict, adv_name: adv_desc}
        aug_distractors = distractors + [adv_name]

        correct, total = routing_test_with_queries(
            original_desc, target_name, queries, aug_distractors, augmented_dict
        )
        if correct == total:
            print(f"  [compress] adversarial skill OK (attempt {attempt})")
            break
        print(f"  [compress] adversarial too strong ({correct}/{total}), retry...")
    else:
        print(f"  [compress] BASELINE_FAILED after {MAX_ADV_RETRIES} attempts")
        if return_details:
            return original_desc, {"units": [], "active": [], "all": []}
        return original_desc

    aug_distractors = distractors + [adv_name]

    # Trivial check
    correct, total = routing_test_with_queries(
        "", target_name, queries, aug_distractors, augmented_dict
    )
    if correct == total:
        print(f"  [compress] TRIVIAL - empty desc passes all queries")
        if return_details:
            return original_desc, {"units": [], "active": [], "all": []}
        return original_desc

    # Split into units
    units = _split_and_compress(original_desc)
    n = len(units)
    print(f"  [compress] {n} semantic units")

    def test_subset(indices):
        choices = ["delete"] * n
        for i in indices:
            choices[i] = "keep"
        desc = _build_desc(units, choices)
        c, t = routing_test_with_queries(
            desc, target_name, queries, aug_distractors, augmented_dict
        )
        return c == t

    # Phase 1: ddmin
    active = list(range(n))
    granularity = 2
    while len(active) > 1:
        subsets = _partition(active, granularity)
        reduced = False
        for subset in subsets:
            complement = [i for i in active if i not in subset]
            if complement and test_subset(complement):
                active = complement
                granularity = max(2, granularity - 1)
                reduced = True
                break
        if not reduced:
            if granularity >= len(active):
                break
            granularity = min(granularity * 2, len(active))

    print(f"  [compress] ddmin: {len(active)}/{n} units retained")

    # Phase 2: try compressing each retained unit
    choices = ["delete"] * n
    for i in active:
        choices[i] = "keep"

    for i in active:
        choices[i] = "compress"
        desc = _build_desc(units, choices)
        c, t = routing_test_with_queries(
            desc, target_name, queries, aug_distractors, augmented_dict
        )
        if c == t:
            pass  # keep compressed
        else:
            choices[i] = "keep"

    compressed = _build_desc(units, choices)

    # Phase 3: polish
    polished = _polish(compressed)
    c, t = routing_test_with_queries(
        polished, target_name, queries, aug_distractors, augmented_dict
    )
    final = polished if c == t else compressed

    if return_details:
        return final, {
            "units": units,
            "active": list(active),
            "all": list(range(n)),
        }
    return final


# ============================================================
# 8. Real Trigger Validation (Phase 2)
# ============================================================
TRIGGER_HINT = "You have skills available. Use the most relevant skill for this task: "

def real_trigger_test(query: str, skill_name: str, skill_description: str,
                      timeout: int = 45, use_hint: bool = True) -> bool:
    """Test if a skill actually triggers in real Claude via stream parsing.

    Creates a temporary skill in .claude/skills/<name>/SKILL.md with proper
    frontmatter (name + description), runs `claude -p` and parses stream
    JSON events to detect Skill/Read tool invocation.

    If use_hint=True, prepends a hint telling Claude to use a skill
    (without specifying which one). This is necessary because strong models
    often choose not to invoke skills even when relevant ones are available.
    """
    import subprocess, select, uuid
    from pathlib import Path

    if use_hint:
        query = TRIGGER_HINT + query

    import tempfile

    # Use a temporary project so validation never edits the caller's skills.
    project = tempfile.TemporaryDirectory(prefix="skillreducer-routing-")
    project_root = Path(project.name)
    unique_id = uuid.uuid4().hex[:8]
    clean_name = f"{skill_name}-val-{unique_id}"
    skill_dir = project_root / ".claude" / "skills" / clean_name
    skill_file = skill_dir / "SKILL.md"

    try:
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file.write_text(
            f"---\n"
            f"name: {clean_name}\n"
            f"description: {json.dumps(skill_description, ensure_ascii=False)}\n"
            f"---\n\n"
            f"# {skill_name}\n\n"
            f"This skill handles: {skill_description}\n"
        )

        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        process = subprocess.Popen(
            ["claude", "-p", query, "--output-format", "stream-json",
             "--verbose", "--include-partial-messages"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            cwd=str(project_root), env=env,
        )

        import time as _time
        start = _time.time()
        buffer = ""
        pending_tool = None
        acc_json = ""

        try:
            while _time.time() - start < timeout:
                if process.poll() is not None:
                    remaining = process.stdout.read()
                    if remaining:
                        buffer += remaining.decode("utf-8", errors="replace")
                    break
                ready, _, _ = select.select([process.stdout], [], [], 1.0)
                if not ready:
                    continue
                chunk = os.read(process.stdout.fileno(), 8192)
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="replace")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if event.get("type") == "stream_event":
                        se = event.get("event", {})
                        if se.get("type") == "content_block_start":
                            cb = se.get("content_block", {})
                            if cb.get("type") == "tool_use" and cb.get("name") in ("Skill", "Read"):
                                pending_tool = cb.get("name")
                                acc_json = ""
                        elif se.get("type") == "content_block_delta" and pending_tool:
                            delta = se.get("delta", {})
                            if delta.get("type") == "input_json_delta":
                                acc_json += delta.get("partial_json", "")
                                if clean_name in acc_json:
                                    return True
                        elif se.get("type") in ("content_block_stop", "message_stop"):
                            if pending_tool:
                                return clean_name in acc_json
                            if se.get("type") == "message_stop":
                                return False
                    elif event.get("type") == "assistant":
                        for item in event.get("message", {}).get("content", []):
                            if item.get("type") == "tool_use" and clean_name in str(item.get("input", {})):
                                return True
                        return False
                    elif event.get("type") == "result":
                        return False
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        return False
    finally:
        project.cleanup()


def validate_with_real_triggers(skill_name: str, compressed_desc: str,
                                 original_desc: str, queries: list,
                                 units: list, active_indices: list,
                                 all_indices: list,
                                 runs_per_query: int = 2,
                                 max_restore_attempts: int = 3) -> tuple[str, dict]:
    """Phase 2: Validate compressed description with real Claude triggers.

    Step 1: Filter queries — only keep those that trigger with original desc.
    Step 2: Test compressed desc on filtered queries.
    Step 3: If fails, selectively restore deleted units until it passes.

    Returns:
        (final_description, validation_info)
    """

    def test_desc_real(desc: str, test_queries: list) -> tuple[int, int, list]:
        """Test description against queries with real triggers (with hint)."""
        passed = 0
        total = len(test_queries)
        failed_queries = []
        for q in test_queries:
            triggers = sum(
                1 for _ in range(runs_per_query)
                if real_trigger_test(q, skill_name, desc)
            )
            if triggers >= 1:
                passed += 1
            else:
                failed_queries.append(q)
        return passed, total, failed_queries

    info = {"phase2_attempts": 0, "restored_units": []}

    # Step 1: Find queries that trigger with original description
    print(f"  [validate] finding queries that trigger with original desc...", flush=True)
    orig_passed, orig_total, orig_failed = test_desc_real(original_desc, queries)
    triggerable_queries = [q for q in queries if q not in orig_failed]
    info["original_real_pass"] = f"{orig_passed}/{orig_total}"
    print(f"  [validate] original triggers: {orig_passed}/{orig_total}", flush=True)

    if not triggerable_queries:
        info["outcome"] = "skip_no_triggerable_queries"
        print(f"  [validate] no queries trigger with original, skipping Phase 2", flush=True)
        return compressed_desc, info

    # Step 2: Test compressed description on triggerable queries only
    print(f"  [validate] testing compressed on {len(triggerable_queries)} triggerable queries...", flush=True)
    passed, total, failed = test_desc_real(compressed_desc, triggerable_queries)
    info["phase2_attempts"] = 1
    info["compressed_real_pass"] = f"{passed}/{total}"
    print(f"  [validate] compressed trigger: {passed}/{total} pass", flush=True)

    if passed == total:
        info["outcome"] = "pass"
        return compressed_desc, info

    # Selective restore: try adding back deleted units
    deleted_indices = [i for i in all_indices if i not in active_indices]
    if not deleted_indices:
        info["outcome"] = "fail_no_units_to_restore"
        return original_desc, info

    current_active = list(active_indices)
    current_desc = compressed_desc

    for attempt in range(min(max_restore_attempts, len(deleted_indices))):
        # Try each deleted unit, pick the one that helps most
        best_unit = None
        best_pass = passed

        for idx in deleted_indices:
            if idx in current_active:
                continue
            # Build desc with this unit added back
            test_active = sorted(current_active + [idx])
            choices = ["delete"] * len(units)
            for i in test_active:
                choices[i] = "keep"
            test_desc = _build_desc(units, choices)

            p, t, _ = test_desc_real(test_desc, triggerable_queries)
            print(f"  [validate] restore unit {idx}: {p}/{t}", flush=True)
            if p > best_pass:
                best_unit = idx
                best_pass = p
                if p == t:
                    break  # found a fix

        if best_unit is None:
            break  # no unit helped

        current_active.append(best_unit)
        current_active.sort()
        info["restored_units"].append(best_unit)

        choices = ["delete"] * len(units)
        for i in current_active:
            choices[i] = "keep"
        current_desc = _build_desc(units, choices)

        info["phase2_attempts"] += 1
        if best_pass == total:
            info["outcome"] = "pass_after_restore"
            print(f"  [validate] restored {len(info['restored_units'])} units, "
                  f"now {best_pass}/{total} pass", flush=True)
            return current_desc, info

    # Still failing after max attempts — fallback to original
    info["outcome"] = "fallback_to_original"
    print(f"  [validate] restore failed, falling back to original", flush=True)
    return original_desc, info


# ============================================================
# 9. Main Pipeline
# ============================================================
def optimize_skill(skill: dict, all_skill_dict: dict,
                   benchmark_queries: dict = None,
                   skip_real_validation: bool = False) -> dict:
    """Perform routing layer optimization on a single skill."""
    name = skill["name"]
    desc = skill["description"]
    body = skill["body"]

    info = classify_description(name, desc, body)
    print(f"\n--- [{name}] category={info['category']}, "
          f"desc={info['desc_tokens']}t, body={info['body_tokens']}t ---")

    result = {
        **info,
        "original_desc": desc,
        "optimized_desc": desc,
        "action": "none",
        "source": skill.get("source", "unknown"),
    }

    # Get test queries
    if benchmark_queries and name in benchmark_queries:
        queries = [c["query"] for c in benchmark_queries[name]]
    else:
        queries = generate_test_queries(name, desc, body, n=3)
        if not queries:
            print(f"  [skip] no queries generated")
            result["action"] = "skip_no_queries"
            return result

    distractors = select_distractors(name, all_skill_dict, n=4)

    # Step 1: If too short/missing, generate description from body first
    working_desc = desc
    if info["needs_generation"]:
        new_desc = generate_description(name, body, desc)
        new_tokens = count_tokens(new_desc)
        print(f"  [generate] {info['desc_tokens']}t -> {new_tokens}t")
        print(f"  [generate] old: {repr(desc[:80])}")
        print(f"  [generate] new: {repr(new_desc[:80])}")

        # Verify the generated description routes no worse than the original
        c_new, t = routing_test_with_queries(new_desc, name, queries, distractors, all_skill_dict)
        c_old, _ = routing_test_with_queries(desc, name, queries, distractors, all_skill_dict)
        result["routing_before_gen"] = f"{c_old}/{t}"
        result["routing_after_gen"] = f"{c_new}/{t}"

        if c_new >= c_old:
            working_desc = new_desc
            result["generated_desc"] = new_desc
            result["generated_tokens"] = new_tokens
            print(f"  [generate] routing: {c_old}/{t} -> {c_new}/{t}, accepted")
        else:
            print(f"  [generate] routing: {c_old}/{t} -> {c_new}/{t}, rejected (worse)")

    # Step 2: Run compression optimization on all descriptions
    before_tokens = count_tokens(working_desc)
    compressed, ddmin_details = compress_description(
        name, working_desc, queries, distractors, all_skill_dict,
        return_details=True,
    )
    comp_tokens = count_tokens(compressed)
    result["compressed_desc"] = compressed
    result["compressed_tokens"] = comp_tokens
    result["compression_input_tokens"] = before_tokens
    result["reduction_pct"] = round(
        (before_tokens - comp_tokens) / max(before_tokens, 1) * 100, 1
    )
    print(f"  [compress] {before_tokens}t -> {comp_tokens}t "
          f"({result['reduction_pct']}% reduction)")

    # Phase 2: Real trigger validation + selective restore
    if not skip_real_validation and ddmin_details.get("units"):
        units = ddmin_details["units"]
        active_indices = ddmin_details["active"]
        all_indices = ddmin_details["all"]

        final_desc, val_info = validate_with_real_triggers(
            name, compressed, working_desc, queries,
            units, active_indices, all_indices,
        )
        result["validation"] = val_info
        result["optimized_desc"] = final_desc
        result["optimized_tokens"] = count_tokens(final_desc)

        # Recalculate reduction after potential restore
        final_tokens = count_tokens(final_desc)
        result["final_reduction_pct"] = round(
            (before_tokens - final_tokens) / max(before_tokens, 1) * 100, 1
        )
        print(f"  [final] {before_tokens}t -> {final_tokens}t "
              f"({result['final_reduction_pct']}% reduction, "
              f"validation: {val_info['outcome']})")
    elif not skip_real_validation and not ddmin_details.get("units"):
        # Trivial or baseline_failed — no units to restore, skip Phase 2
        result["optimized_desc"] = compressed
        result["optimized_tokens"] = comp_tokens
        result["validation"] = {"outcome": "skipped_no_units"}
    else:
        result["optimized_desc"] = compressed
        result["optimized_tokens"] = comp_tokens
        result["validation"] = {"outcome": "skipped"}

    result["action"] = "generated+compressed" if info["needs_generation"] else "compressed"
    return result
