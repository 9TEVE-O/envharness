# Copyright 2026 The EnvHarness Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ReasoningBank induction prompts -- SWE-bench domain wrapper.

Sibling of envharness/reasoning_bank/induce.py. Same RB recipe (SUCCESSFUL_SI /
FAILED_SI, up to 3 items per trajectory, parse via the canonical Markdown
schema), but the domain word "text-based household environment (ALFWorld)" is
substituted with "software-engineering tasks (SWE-bench)" so the lessons read
naturally for bug-fix work.

Output schema matches the canonical RB MemoryItem ({title, description,
content}); embedding is done downstream with `embed_texts` over
`f"{title}: {description}"`.
"""
from __future__ import annotations

# Verbatim ReasoningBank prompts, kept with the rest of the third-party
# material. The paired-diff prompts below are ours.
from envharness.third_party.reasoning_bank.swebench_prompts import (  # noqa: F401
    FAILED_SI, SUCCESSFUL_SI,
)

from envharness.infra.model import completion_kwargs
import re
import time
import litellm







# Shared with the other benchmarks on purpose. The local copy this replaced
# accepted only the strict "## Title <text>" spelling, so a model that writes
# the title straight after "##" -- which Gemini does -- parsed to nothing and
# the bank came out empty with no error anywhere.
from envharness.reasoning_bank.induce import parse_memory_items


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_ACTION_RE = re.compile(r"<action>.*?</action>", re.DOTALL)


def format_trajectory_swebench(steps: list[dict],
                                max_cmd_chars: int = 250,
                                max_obs_chars: int = 400,
                                max_steps_shown: int = 18,
                                max_think_chars: int = 600) -> str:
    """Render a SWE-bench Policy trajectory for the induction prompt.

    When the Policy was run with action_format=think_action, the runner
    captures the full raw response (<think>...</think><action>cmd</action>)
    into Step.policy_raw_response. We render each step as RB's canonical
    recipe expects:

        Step N:
        <think>...reasoning...</think>
        <action>cmd</action>
        Observation: <truncated stdout+stderr>
        rc=<code>  [SUBMITTED]?

    Falls back to action-only rendering for traces without
    policy_raw_response (function_calling traces from the earlier SWE-bench
    Lite run -- those don't carry Policy reasoning).
    """
    n_total = len(steps)
    shown = steps[-max_steps_shown:] if n_total > max_steps_shown else steps
    start = n_total - len(shown)
    out = []
    for i, s in enumerate(shown):
        idx = start + i + 1
        ra = s.get("raw_action") or {}
        kwargs = ra.get("kwargs") or {}
        cmd = (kwargs.get("command") or "").strip()
        if len(cmd) > max_cmd_chars:
            cmd = cmd[:max_cmd_chars] + " ...[truncated]"
        fo = s.get("filtered_observation") or s.get("raw_observation") or {}
        obs = (fo.get("text") if isinstance(fo, dict) else "") or ""
        obs = obs.strip()
        if len(obs) > max_obs_chars:
            obs = obs[:max_obs_chars] + " ...[truncated]"
        info = s.get("info") or {}
        rc = info.get("returncode")
        sub = " [SUBMITTED]" if info.get("submitted") else ""

        raw = s.get("policy_raw_response") or ""
        if raw:
            think_m = _THINK_RE.search(raw)
            act_m = _ACTION_RE.search(raw)
            if think_m or act_m:
                pieces = [f"Step {idx}:"]
                if think_m:
                    think_block = think_m.group(0)
                    if len(think_block) > max_think_chars:
                        think_block = think_block[:max_think_chars] + " ...[truncated]</think>"
                    pieces.append(think_block)
                pieces.append(act_m.group(0) if act_m else f"<action>{cmd}</action>")
                if obs: pieces.append(f"Observation: {obs}")
                pieces.append(f"rc={rc}{sub}")
                out.append("\n".join(pieces))
                continue
            # raw response present but no tags -- include it as reasoning
            short_raw = raw.strip()[:max_think_chars]
            block = [f"Step {idx}:", f"Reasoning: {short_raw}",
                     f"<action>{cmd}</action>"]
            if obs: block.append(f"Observation: {obs}")
            block.append(f"rc={rc}{sub}")
            out.append("\n".join(block))
            continue

        # Legacy fallback: no policy_raw_response (FC-format traces)
        block = [f"Step {idx}:", f"$ {cmd}", f"rc={rc}{sub}"]
        if obs: block.append(obs)
        out.append("\n".join(block))
    return "\n\n".join(out)


PAIRED_DIFF_ATOMIC_SI = """
You are extracting ATOMIC GENERAL recovery skills by COMPARING two trajectories on the same SWE-bench task.

The two trajectories:
  - BASELINE: the policy was given the standard task and FAILED.
  - MUTATED:  the env was perturbed by a Mutator to test a specific weakness; the policy SUCCEEDED.

The Mutator's hypothesized policy weakness:
  {failure_label}

The Mutator's perturbation (summary):
  {mutation_summary}

## Your job
COMPARE the two trajectories. Identify the SPECIFIC TECHNIQUE used in the MUTATED trajectory that the BASELINE trajectory did not use. That technique is the recovery skill. If the only difference is luck or unrelated repo knowledge, emit nothing.

## Hard constraints -- every item must satisfy ALL of:
1. ATOMIC: one specific bash/Python/git primitive (a flag, a builtin, a stdlib idiom). NOT a meta-recommendation.
2. GENERAL-AT-CORE: the primitive itself must work across any Python project. Domain context is allowed (e.g. "when the bug is in an ORM serializer" or "for a numerics codebase that uses NaN-aware ops") AS LONG AS the actual technique is portable. Do not name a SPECIFIC repo/file/test.
3. STRUCTURE: phrasable as "When X happens (X may include a domain context), use Y." Y must name a specific tool / flag / primitive.
4. TITLE: <= 10 words, imperative ("For ORM serialization tests, use `model_to_dict` for round-trip equality").
5. DESCRIPTION: <= 1 sentence.
6. CONTENT: <= 250 characters; must name a specific bash/python/git primitive AND may mention the domain context where it applies.

## FORBIDDEN -- DO NOT emit:
- "Verify problem statement alignment" / "Be careful about scope" / "Cross-reference the issue" -- process paranoia
- "Run tests/check_framework" / "Run system checks" -- repo-specific workflow
- "Always write regression tests" / "Verify your fix" -- engineering platitude
- "Use targeted tests" -- vague unless you name a flag like `pytest -k <pattern>`
- Anything containing a Django/SymPy/astropy/specific repo/file/test name

If you cannot find a clean technique-level difference between the two trajectories, output NOTHING. Better empty than noisy.

## Output format (Markdown):
```
# Memory Item i
## Title <<= 8 words, imperative>
## Description <when to apply, <= 1 sentence>
## Content <<= 200 chars; must name a specific primitive (e.g. `subprocess.run(..., capture_output=True)`, `git diff --stat`, `pytest -x --tb=short`, `sed -n '1,80p'`)>
```

Emit at most 3 items per pair. Quality > quantity, but if you can identify multiple distinct atomic techniques the mutated trajectory used and the baseline did not, emit one item per technique. Empty output is acceptable.
""".strip()


def induce_paired_diff_atomic(
    *, problem_statement: str, baseline_trajectory_text: str,
    mutated_trajectory_text: str,
    failure_label: str | None = None, mutation_rationale: str | None = None,
    existing_lesson_titles: list[str] | None = None,
    extra_note: str | None = None,
    llm_model: str = "openai/gpt-4.1-mini",
    max_items: int = 3, retries: int = 4,
    gemini_api_key: str | None = None,
) -> list[dict]:
    """Paired-diff atomic-skill RB induction.

    Compares a BASELINE (failed, unmutated) trajectory against a MUTATED
    (succeeded, perturbed) trajectory on the same SWE-bench task. The Mutator's
    diagnosis is threaded into the system prompt so induction focuses on
    recovery techniques, not surface bug-fix tricks.

    `extra_note`: an OPTIONAL soft addendum injected into the user message
    before the final "Compare." instruction. Use for context the LLM should
    weigh ALONGSIDE the existing atomic/general/structure constraints (e.g.
    "this trajectory is a LINKED long-horizon episode" -- callers can mention
    additional skill TYPES that are valid AS LONG AS they still satisfy the
    base constraints). NOT used to OVERRIDE the prompt's hard constraints --
    the LLM is instructed to still apply them.
    """
    label = (failure_label or "unspecified").strip()
    summary = (mutation_rationale or "unspecified").strip()[:400]
    system = PAIRED_DIFF_ATOMIC_SI.format(
        failure_label=label, mutation_summary=summary
    )
    existing = ""
    if existing_lesson_titles:
        existing_str = "\n".join(f"- {t}" for t in existing_lesson_titles)
        existing = (
            f"\n\n## EXISTING LESSONS (from prior rounds) — DO NOT RE-EMIT THESE\n"
            f"The following lessons are ALREADY in the bank. Emit ONLY techniques "
            f"NOT covered by these. If the trajectory difference is fully captured "
            f"by an existing lesson, emit NOTHING.\n{existing_str}\n"
        )
    note_block = ""
    if extra_note:
        note_block = (
            f"\n\n## ADDITIONAL CONTEXT (soft -- existing constraints still apply)\n"
            f"{extra_note.strip()}\n"
        )
    user = (
        f"Problem statement:\n{problem_statement[:3000]}\n\n"
        f"BASELINE trajectory (FAILED, env unmutated):\n"
        f"{baseline_trajectory_text[:7000]}\n\n"
        f"MUTATED trajectory (SUCCEEDED, env perturbed by Mutator):\n"
        f"{mutated_trajectory_text[:7000]}\n"
        f"{existing}"
        f"{note_block}"
        f"\nCompare. Extract atomic recovery skills per the constraints. "
        f"If no clean technique-level difference, emit NOTHING. "
        f"If a difference exists but is already covered by an existing lesson "
        f"above, also emit NOTHING."
    )
    for attempt in range(retries):
        try:
            kwargs = dict(
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                **completion_kwargs(llm_model, temperature=0.0,
                                    reasoning_effort="minimal",
                                    api_key=gemini_api_key),
            )
            r = litellm.completion(**kwargs)
            txt = r.choices[0].message.content or ""
            items = parse_memory_items(txt)
            return items[:max_items]
        except Exception:
            if attempt == retries - 1:
                return []
            time.sleep(2 ** attempt)
    return []


PAIRED_DIFF_DUAL_SI = """
You are extracting ATOMIC GENERAL skills by ANALYZING two trajectories on the same SWE-bench task.

The two trajectories:
  - BASELINE: standard task, the policy FAILED.
  - MUTATED:  env was perturbed by a Mutator, the policy SUCCEEDED.

Mutator's hypothesized policy weakness:
  {failure_label}

Mutator's perturbation (summary):
  {mutation_summary}

## Your job
Extract atomic GENERAL skills demonstrated by the MUTATED (successful) trajectory. Pull from BOTH of these sources of insight:

  TYPE A — ENV / WORKFLOW RECOVERY: a specific bash/git/shell PRIMITIVE the MUTATED
  trajectory used that BASELINE did not. Examples:
    - `patch -p1 < change.patch` to apply edits when inline scripting is blocked
    - `cat << 'EOF' > script.py` then `python script.py` to avoid shell escaping
    - `conda run -n testbed python ...` to run in the right env without activating
    - `git show <hash>` to inspect when a feature was added
    - `/opt/miniconda3/envs/testbed/bin/pytest` when default pytest is missing

  TYPE B — ALGORITHMIC / SEMANTIC PYTHON: a specific Python/stdlib/library IDIOM or
  CODE PATTERN the MUTATED trajectory's successful patch demonstrates. This type may
  ALSO appear in baseline — what matters is the technique is general and the MUTATED
  trajectory's solve PROVES it works for the bug class. Examples:
    - `list(dict.fromkeys(seq))` for order-preserving dedup
    - `dataclasses.make_dataclass("X", ["a","b"])` for dynamic container classes
    - `if x.is_real is False:` (NOT `not x.is_real`) for ternary symbolic booleans
    - `collections.Mapping = collections.abc.Mapping` monkey-patch for legacy 3.10+
    - Avoid `map(str, items)` when items may be safe-string wrappers — strips subclass
    - Use `isinstance(x, SafeString)` to preserve wrapper-aware code paths
    - `Q.deconstruct()` / `__repr__` round-trip for ORM serialization tests
    - Read traceback bottom-up for the actual `raise` site

A high-quality bank contains BOTH types — they target different failure modes (TYPE A
helps when the env is friction-heavy; TYPE B helps when the bug is a Python semantics
trap).

## Hard constraints -- every item must satisfy ALL of:
1. ATOMIC: one specific bash/Python/git primitive (a flag, a builtin, a stdlib idiom).
   NOT a meta-recommendation.
2. GENERAL-AT-CORE: the primitive itself must work across any Python project. Domain
   context (e.g. "for ORM queries", "for numeric ops") is allowed. NO specific
   repo/file/test names.
3. STRUCTURE: phrasable as "When X happens (X may include domain context), use Y."
4. TITLE: <= 10 words, imperative.
5. DESCRIPTION: <= 1 sentence.
6. CONTENT: <= 250 characters; must name a specific bash/python/git primitive.

## FORBIDDEN — DO NOT emit:
- "Verify problem statement alignment", "Be careful about scope" — process paranoia
- "Run tests / check_framework" — repo-specific workflow without a specific flag
- "Always write regression tests" — engineering platitude
- "Use targeted tests" — vague unless you name a flag like `pytest -k <pattern>`
- Anything containing a Django/SymPy/astropy/specific repo/file/test name

Emit at most 3 items per pair. PREFER a mix of TYPE A and TYPE B when both clearly apply.
TYPE B items are valid EVEN IF baseline used the same idiom — the bank's job is to provide
generalizable patterns, not just diff-only. Empty output is acceptable if neither type
yields a clean GENERAL technique.

## Output format (Markdown):
```
# Memory Item i
## Title <<= 10 words, imperative>
## Description <when to apply, <= 1 sentence>
## Content <<= 250 chars; must name a specific primitive>
```
""".strip()


def induce_paired_diff_dual(
    *, problem_statement: str, baseline_trajectory_text: str,
    mutated_trajectory_text: str,
    failure_label: str | None = None, mutation_rationale: str | None = None,
    llm_model: str = "openai/gpt-4.1-mini",
    max_items: int = 3, retries: int = 4,
    gemini_api_key: str | None = None,
) -> list[dict]:
    """Paired-diff DUAL-axis induction.

    Same paired (baseline_fail, mutated_succ) signal as
    induce_paired_diff_atomic, but the prompt explicitly asks for both
    env-recovery (TYPE A) and algorithmic/semantic Python (TYPE B) lessons.
    TYPE B may overlap baseline — that's allowed, because the bank's job is
    generalizable patterns, not strict diff.
    """
    label = (failure_label or "unspecified").strip()
    summary = (mutation_rationale or "unspecified").strip()[:400]
    system = PAIRED_DIFF_DUAL_SI.format(
        failure_label=label, mutation_summary=summary
    )
    user = (
        f"Problem statement:\n{problem_statement[:3000]}\n\n"
        f"BASELINE trajectory (FAILED, env unmutated):\n"
        f"{baseline_trajectory_text[:7000]}\n\n"
        f"MUTATED trajectory (SUCCEEDED, env perturbed):\n"
        f"{mutated_trajectory_text[:7000]}\n\n"
        f"Extract up to 3 atomic GENERAL skills per the dual-type instructions. "
        f"Prefer a mix of TYPE A and TYPE B when both apply."
    )
    for attempt in range(retries):
        try:
            kwargs = dict(
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                **completion_kwargs(llm_model, temperature=0.0,
                                    reasoning_effort="minimal",
                                    api_key=gemini_api_key),
            )
            r = litellm.completion(**kwargs)
            txt = r.choices[0].message.content or ""
            items = parse_memory_items(txt)
            return items[:max_items]
        except Exception:
            if attempt == retries - 1:
                return []
            time.sleep(2 ** attempt)
    return []


def induce_memory_items_swebench(
    *, problem_statement: str, trajectory_text: str, success: bool,
    extra_note: str | None = None,
    llm_model: str = "openai/gpt-4.1-mini",
    max_items: int = 3, retries: int = 4,
    gemini_api_key: str | None = None,
) -> list[dict]:
    """RB-style induction adapted for SWE-bench domain.

    Returns at most `max_items` parsed memory items {title, description, content}.

    `extra_note`: OPTIONAL soft addendum injected into the user message after
    the trajectory. Same role as `induce_paired_diff_atomic.extra_note`: gives
    additional context the LLM should weigh ALONGSIDE the base prompt's
    constraints, not as an override (the system prompt's hard rules still apply).
    """
    system = SUCCESSFUL_SI if success else FAILED_SI
    note_block = ""
    if extra_note:
        note_block = (
            f"\n\n## ADDITIONAL CONTEXT (soft -- base guidelines still apply)\n"
            f"{extra_note.strip()}\n"
        )
    user = (f"Problem statement:\n{problem_statement[:6000]}\n\n"
            f"Trajectory ({'SUCCESSFUL' if success else 'FAILED'}):\n"
            f"{trajectory_text[:12000]}"
            f"{note_block}")
    for attempt in range(retries):
        try:
            kwargs = dict(
                messages=[{"role": "system", "content": system},
                           {"role": "user",   "content": user}],
                **completion_kwargs(llm_model, temperature=0.0,
                                    reasoning_effort="minimal",
                                    api_key=gemini_api_key),
            )
            r = litellm.completion(**kwargs)
            txt = r.choices[0].message.content or ""
            items = parse_memory_items(txt)
            return items[:max_items]
        except Exception:
            if attempt == retries - 1:
                return []
            time.sleep(2 ** attempt)
    return []
