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

"""Prompts + helpers for the SpreadsheetBench RB experiment.

Ported from Qwen-Applications/Trace2Skill (the spreadsheet-agent system
prompt + skill-injection protocol) and adapted to EnvHarness's `run_python` +
`submit` tools and the ReasoningBank `MemoryItem` representation. Three pieces:

  1. AGENT_SYSTEM / build_agent_system(skill_block)
        the policy's system prompt; with retrieved skills injected as a
        "Relevant skills" block (Trace2Skill's cli_skill_preloaded protocol).

  2. format_trajectory_sb(steps)
        render a SpreadsheetBench trajectory (reasoning + run_python code +
        execution output) into the text the induction prompts consume.

  3. SUCCESS_SI / FAILED_SI / PAIRED_DIFF_SI + induce_*()
        Trace2Skill-style skill induction: extract atomic, transferable
        spreadsheet-manipulation techniques from success / failure / paired
        trajectories, parsed into {title, description, content} memory items.

Skill text uses ReasoningBank's MemoryItem.text rendering at retrieval time, so
the induction output format matches `parse_memory_items` (## Title / ##
Description / ## Content).
"""
from __future__ import annotations

from envharness.infra.model import completion_kwargs

import time

from envharness.reasoning_bank.induce import parse_memory_items
# Trace2Skill's system prompt and injection protocol, and ReasoningBank's
# per-step gate, kept with the rest of the third-party material.
from envharness.third_party.reasoning_bank.prompts import PER_STEP_GATE  # noqa: F401
from envharness.third_party.trace2skill.prompts import (  # noqa: F401
    AGENT_SYSTEM, SKILL_BLOCK_HEADER, SKILL_DIR_HEADER, SKILL_DOC_HEADER,
)


# ---------------------------------------------------------------------------
# 1. Policy system prompt (run_python + submit; function calling)
# ---------------------------------------------------------------------------



# RB-canonical per-step "discuss gate" (verbatim intent from
# alfworld_skill_prompt.build_memory_block, style="soft"): make the agent, at every
# step, explicitly decide whether a skill applies BEFORE acting. In RB/SkillOS
# this line is re-injected each turn (stateless per-turn template); here we put
# it in the skill header AND re-inject it on every observation (see
# per_step_gate() + reasoning_bank_eval.run_episode).


def per_step_gate() -> str:
    """The one-line per-step gate to append to each observation when skills are
    active. Empty string means no skills -> no gate (base condition)."""
    return PER_STEP_GATE




def build_agent_system(skill_block: str | None) -> str:
    """Return the policy system prompt, with retrieved skills injected when
    `skill_block` is non-empty (the `ours` / `orig` conditions). Empty/None
    reproduces the no-bank baseline verbatim."""
    if not skill_block or not skill_block.strip():
        return AGENT_SYSTEM
    return AGENT_SYSTEM + "\n\n" + SKILL_BLOCK_HEADER.format(
        skill_block=skill_block.strip())


# Trace2Skill cli_skill_preloaded-style WHOLE-DOC injection: a single
# consolidated skill library, mandatory-use framing, NO retrieval.


def build_agent_system_doc(skill_doc: str | None) -> str:
    """Inject a WHOLE consolidated skill library (Trace2Skill cli_skill_preloaded
    protocol) into the system prompt -- no per-task retrieval. Empty/None
    reproduces the no-bank baseline."""
    if not skill_doc or not skill_doc.strip():
        return AGENT_SYSTEM
    return AGENT_SYSTEM + "\n\n" + SKILL_DOC_HEADER.format(skill_doc=skill_doc.strip())


def build_agent_system_skillopt(skill_doc: str | None) -> str:
    """SkillOpt-FAITHFUL 'direct-chat' injection: append the whole skill as a
    bare `## Skill` section to the system prompt, once -- no usage protocol, no
    per-step gate. Verbatim shape of SkillOpt's spreadsheetbench
    codegen_agent._build_system (`base + '\\n\\n## Skill\\n' + skill_content`)."""
    if not skill_doc or not skill_doc.strip():
        return AGENT_SYSTEM
    return AGENT_SYSTEM + "\n\n## Skill\n" + skill_doc.strip()


# Trace2Skill-FAITHFUL on-disk injection (cli_skill_preloaded, verbatim intent):
# only the concise SKILL.md is preloaded into the prompt; the full skill folder
# (incl. references/*.md) sits ON DISK, and the agent reads the references on
# demand via run_python. This is progressive disclosure -- the always-in-context
# part stays small; depth is pulled only when relevant.


def build_agent_system_skilldir(skill_md: str | None, skill_dir: str | None) -> str:
    """Trace2Skill on-disk injection: preload SKILL.md, point the agent at the
    on-disk folder to read references/ on demand via run_python."""
    if not skill_md or not skill_md.strip() or not skill_dir:
        return AGENT_SYSTEM
    return AGENT_SYSTEM + "\n\n" + SKILL_DIR_HEADER.format(
        skill_dir=skill_dir, skill_md=skill_md.strip())


def task_query(instruction: str) -> str:
    """Retrieval query for a task = its instruction (truncated)."""
    return (instruction or "").strip()[:2000]


# ---------------------------------------------------------------------------
# 2. Trajectory formatting
# ---------------------------------------------------------------------------

def _step_code(step: dict) -> str:
    ra = step.get("raw_action") or {}
    kw = ra.get("kwargs") or {}
    if ra.get("name") == "submit":
        return "<submit>"
    return str(kw.get("code") or kw.get("text") or "")


def _step_output(step: dict, max_chars: int) -> str:
    obs = step.get("filtered_observation") or step.get("raw_observation") or {}
    text = obs.get("text") if isinstance(obs, dict) else ""
    text = text or ""
    # The bridge appends "last run_python output:\n<...>" -- keep the tail,
    # which is the execution result, dropping the static task context.
    marker = "last run_python output:"
    if marker in text:
        text = text.split(marker, 1)[1].strip()
    return text[:max_chars]


def format_trajectory_sb(steps: list[dict], max_code_chars: int = 1200,
                         max_obs_chars: int = 600,
                         max_think_chars: int = 500) -> str:
    """Render a SpreadsheetBench trajectory as reasoning + code + output per
    step, for the induction prompt."""
    lines: list[str] = []
    for i, step in enumerate(steps, 1):
        raw = (step.get("policy_raw_response") or "").strip()
        code = _step_code(step)
        out = _step_output(step, max_obs_chars)
        block = [f"Step {i}:"]
        if raw:
            block.append(f"  reasoning: {raw[:max_think_chars]}")
        if code == "<submit>":
            block.append("  action: submit()")
        else:
            block.append(f"  run_python:\n{code[:max_code_chars]}")
        if out:
            block.append(f"  output: {out}")
        lines.append("\n".join(block))
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Skill induction (Trace2Skill-style, spreadsheet domain)
# ---------------------------------------------------------------------------

_OUTPUT_FORMAT = """## Output Format
Your output must strictly follow this Markdown format:

```
# Memory Item i
## Title <imperative, <= 8 words, names the tactic>
## Description <one sentence shaped "When <situation>, <do Y>">
## Content <= 250 chars. MUST name a SPECIFIC primitive: a concrete openpyxl/pandas call or idiom, a way to locate a column by header, a way to write into answer_position, or a concrete verification step (reopen + read back the graded cells). Worked examples of the required specificity: "find the target column by matching its header string with df.columns.get_loc(...) rather than assuming a fixed letter -- column order varies"; "after writing, wb.save then reopen with openpyxl and assert the answer_position cells equal the intended values before submit"; "read with pandas.read_excel(dtype=str) when the values are codes/IDs so leading zeros are not dropped".>
```
"""

# Quality gate mirroring SWE-bench's FORBIDDEN enumeration. Without this the
# inducer emits generic/crude skills.
_FORBIDDEN = """## FORBIDDEN -- reject these; better EMPTY than noisy
Emit NOTHING rather than any item that is:
  - Process paranoia restating the task: "verify your output", "make sure the
    values are correct", "double-check the result matches the instruction".
  - A platitude: "be careful with the data", "read the spreadsheet carefully",
    "handle edge cases", "test your code", "understand the structure first".
  - A non-atomic tool mention: "use openpyxl", "use pandas to read the file",
    "run_python to inspect" -- names a library/tool but no specific call, idiom,
    or column/answer_position/verification heuristic.
  - Anything naming a specific sheet name, cell coordinate, header, or data value.
Every item must name a concrete openpyxl/pandas primitive or verification step.
If you cannot find a clean technique-level difference, output NOTHING.
"""

SUCCESS_SI = (
    """You are a spreadsheet-automation expert (openpyxl / pandas). You are given a """
    """spreadsheet task and a trajectory in which an agent SOLVED it with Python code.

Extract at most 3 reusable, transferable spreadsheet-manipulation PRIMITIVES that
would help solve SIMILAR tasks. Each must name a specific openpyxl/pandas call or
idiom, a header-based column-location tactic, a way to write into answer_position,
or a concrete verification step -- not a general strategy. Prefer EMPTY over generic.

"""
    + _FORBIDDEN + "\n"
    + _OUTPUT_FORMAT
)

FAILED_SI = (
    """You are a spreadsheet-automation expert (openpyxl / pandas). You are given a """
    """spreadsheet task and a trajectory in which an agent FAILED to solve it.

State the root cause in one line, then extract at most 3 preventative PRIMITIVES --
each naming a specific openpyxl/pandas technique or verification step that would
have avoided THIS failure (e.g. "reopen and read back answer_position before
submit", "write to output_path not the input path"). Prefer EMPTY over generic.

"""
    + _FORBIDDEN + "\n"
    + _OUTPUT_FORMAT
)

PAIRED_DIFF_SI = (
    """You are extracting atomic, transferable spreadsheet-automation PRIMITIVES by """
    """COMPARING two trajectories on the SAME task in the SAME environment """
    """(run_python with openpyxl / pandas, then submit).

  - FAILURE: the agent attempted the task and FAILED.
  - SUCCESS: the agent attempted the same task and SUCCEEDED.

Identify the SPECIFIC PRIMITIVE the SUCCESS used that the FAILURE did not -- a
better openpyxl/pandas call, locating a column by header, writing into the exact
answer_position, reopening to verify the graded cells. If the only difference is
luck or unrelated knowledge, emit NOTHING.
{diagnosis}
## Constraints (every item must satisfy ALL)
  1. ATOMIC: one specific openpyxl/pandas primitive or verification step -- NOT a
     meta-recommendation, NOT "handle edge cases".
  2. GENERAL-AT-CORE: the primitive works across spreadsheet tasks; the situation
     it triggers on may mention domain context, but never a specific sheet name,
     cell coordinate, header, or data value.
  3. STRUCTURE: phrasable as "When <situation>, <do specific-Y>" where Y names an
     openpyxl/pandas call, a header-lookup, an answer_position write, or a readback.
  4. At most 3 items; EMPTY output is correct if there is no clean primitive-level
     difference.

"""
    + _FORBIDDEN + "\n"
    + _OUTPUT_FORMAT
)


def _induce(system: str, user: str, llm_model: str, max_items: int,
            temperature: float, retries: int = 6,
            gemini_api_key: str | None = None) -> list[dict]:
    import litellm
    for attempt in range(retries):
        try:
            r = litellm.completion(
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                **completion_kwargs(llm_model, temperature=temperature,
                                    api_key=gemini_api_key),
            )
            txt = r.choices[0].message.content or ""
            return parse_memory_items(txt)[:max_items]
        except Exception:
            if attempt == retries - 1:
                return []
            time.sleep(min(2 ** attempt, 30))
    return []


def induce_success(instruction: str, traj_text: str,
                   llm_model: str = "openai/gpt-4.1-mini",
                   max_items: int = 3, gemini_api_key: str | None = None) -> list[dict]:
    user = f"Task instruction: {instruction[:2000]}\n\nSuccessful trajectory:\n{traj_text[:9000]}"
    return _induce(SUCCESS_SI, user, llm_model, max_items, 0.0, gemini_api_key=gemini_api_key)


def induce_failed(instruction: str, traj_text: str,
                  llm_model: str = "openai/gpt-4.1-mini",
                  max_items: int = 3, gemini_api_key: str | None = None) -> list[dict]:
    user = f"Task instruction: {instruction[:2000]}\n\nFailed trajectory:\n{traj_text[:9000]}"
    return _induce(FAILED_SI, user, llm_model, max_items, 0.0, gemini_api_key=gemini_api_key)


def induce_paired_diff_sb(instruction: str, fail_text: str, succ_text: str,
                          llm_model: str = "openai/gpt-4.1-mini",
                          max_items: int = 3, gemini_api_key: str | None = None,
                          diagnosis: str = "") -> list[dict]:
    # Optional (toggle): thread the Mutator's diagnosis (SWE-bench technique #3).
    # Off by default -- only ours traces carry a diagnosis, so enabling it gives
    # ours an induction signal orig lacks.
    diag_block = (f"\nThe env designer diagnosed the Policy's weakness as: "
                  f"{diagnosis.strip()[:400]}\nFavour a primitive that directly "
                  f"addresses that weakness.\n" if diagnosis.strip() else "")
    system = PAIRED_DIFF_SI.replace("{diagnosis}", diag_block)
    user = (f"Task: {instruction[:2000]}\n\n"
            f"FAILURE trajectory:\n{fail_text[:7000]}\n\n"
            f"SUCCESS trajectory:\n{succ_text[:7000]}\n\n"
            "Compare. Extract atomic primitives. If no clean difference, emit nothing.")
    return _induce(system, user, llm_model, max_items, 0.0, gemini_api_key=gemini_api_key)


# ---------------------------------------------------------------------------
# 4. Consolidation: many per-task skill items -> ONE compact general library
#    (Trace2Skill-style: merge/dedupe/generalize, hierarchical for large sets)
# ---------------------------------------------------------------------------

CONSOLIDATE_SI = """You are curating a SKILL LIBRARY for an agent that manipulates spreadsheets with Python (openpyxl / pandas).

You are given many skill snippets, each extracted from a single past task. Many overlap, repeat, or are too task-specific.

Consolidate them into a COMPACT, GENERAL, NON-OVERLAPPING skill library: merge duplicates, drop task-specific trivia, and keep the most broadly useful techniques, conventions, and failure-recovery procedures for solving spreadsheet-manipulation tasks.

## Rules
  - Output AT MOST {max_skills} skills, ordered from most to least broadly useful.
  - Each skill must be GENERAL (no specific sheet names, cell coordinates, or data values) and ACTIONABLE.
  - Prefer concrete openpyxl/pandas patterns, verification checks, and recovery steps over platitudes.

## Output Format (markdown; one block per skill, nothing else)
### <imperative title, <= 10 words>
<one sentence: when to apply>
<1-3 sentences: the concrete technique>
"""


def _items_to_text(items: list[dict]) -> str:
    """Render a list of {title,description,content} skill items as input text."""
    lines = []
    for i, it in enumerate(items, 1):
        lines.append(f"[{i}] {it.get('title','').strip()}\n"
                     f"    when: {it.get('description','').strip()}\n"
                     f"    how:  {it.get('content','').strip()}")
    return "\n".join(lines)


def _consolidate_once(items_text: str, max_skills: int, llm_model: str,
                      gemini_api_key: str | None) -> str:
    import litellm
    system = CONSOLIDATE_SI.format(max_skills=max_skills)
    for attempt in range(6):
        try:
            r = litellm.completion(
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": items_text[:120000]}],
                **completion_kwargs(llm_model, temperature=0.0, api_key=gemini_api_key),
            )
            return (r.choices[0].message.content or "").strip()
        except Exception:
            if attempt == 5:
                return ""
            time.sleep(min(2 ** attempt, 30))
    return ""


def consolidate_skills(items: list[dict], max_skills: int = 24,
                       llm_model: str = "openai/gpt-4.1-mini",
                       gemini_api_key: str | None = None,
                       batch_size: int = 60) -> str:
    """Consolidate per-task skill items into one markdown skill library.

    Hierarchical when there are many items: chunk into batches, consolidate each
    batch into ~max_skills, then a final merge pass over the batch outputs. This
    keeps each LLM call's input bounded and de-duplicates across the whole set.
    """
    if not items:
        return ""
    if len(items) <= batch_size:
        return _consolidate_once(_items_to_text(items), max_skills, llm_model, gemini_api_key)
    # Stage 1: per-batch consolidation
    partials = []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        out = _consolidate_once(_items_to_text(chunk), max_skills, llm_model, gemini_api_key)
        if out:
            partials.append(out)
    # Stage 2: merge the batch outputs (they're already markdown skill blocks)
    merged_input = "\n\n".join(partials)
    return _consolidate_once(merged_input, max_skills, llm_model, gemini_api_key)
