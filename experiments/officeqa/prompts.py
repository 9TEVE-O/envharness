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

"""Prompts + helpers for the OfficeQA RB experiment.

OfficeQA clone of `experiments/spreadsheetbench/prompts.py`. Same public
surface (reasoning_bank_eval / induce import these names), retargeted from the spreadsheet
run_python/submit domain to OfficeQA's read-only document navigation
(grep/read/glob/answer over parsed U.S. Treasury Bulletin tables). Three pieces:

  1. AGENT_SYSTEM / build_agent_system(skill_block)
        the policy's system prompt (identical to the corpus.yaml policy
        task_description), with retrieved skills injected as a "Relevant
        skills" block + a per-step gate.

  2. format_trajectory(steps)
        render an OfficeQA trajectory (reasoning + glob/read/grep/answer tool
        call + tool output) into the text the induction prompts consume.

  3. SUCCESS_SI / FAILED_SI / PAIRED_DIFF_SI + induce_*()
        skill induction: extract atomic, transferable document-navigation
        techniques from success / failure / paired trajectories, parsed into
        {title, description, content} memory items via parse_memory_items.

Skill text uses ReasoningBank's MemoryItem.text rendering at retrieval time, so
the induction output format matches `parse_memory_items` (# Memory Item i /
## Title / ## Description / ## Content).
"""
from __future__ import annotations

from envharness.infra.model import completion_kwargs

import time

from envharness.reasoning_bank.induce import parse_memory_items
# ReasoningBank's per-step gate, kept with the rest of the third-party
# material rather than restated per benchmark.
from envharness.third_party.reasoning_bank.prompts import PER_STEP_GATE  # noqa: F401


# ---------------------------------------------------------------------------
# 1. Policy system prompt (grep/read/glob/answer; function calling)
# ---------------------------------------------------------------------------

# Kept in sync with the corpus.yaml policy task_description so the corpus
# policy and the eval no-bank base condition share one prompt.
AGENT_SYSTEM = """You are an expert research assistant answering factual questions about \
U.S. Treasury Bulletin documents (dense financial tables parsed to text).

You are given a QUESTION and a source_document under docs_root. Find the exact \
figure/fact the question asks for and return it.

Tools:
- grep(pattern, path): case-insensitive substring search; returns matching lines \
with line numbers. Use it FIRST to locate the relevant row/heading (try the \
metric name, a year, or a distinctive noun from the question).
- read(path, start, limit): read a line window around a match to see the full \
table row and its column headers.
- glob(pattern): list candidate documents (rarely needed -- the source_document \
is given).
- answer(text): submit the concise final answer and end the task.

Method:
1. grep for the key term(s) in the source_document.
2. read around the best match to align the value with its column/units.
3. The tables are wide; verify you are reading the right year/column before answering.
4. answer() with ONLY the value (a number or short phrase). Match the question's \
units (e.g. millions of dollars) but do not add words -- e.g. answer "507", not \
"507 million dollars".

Be efficient: a handful of grep/read calls should suffice. Do not answer without \
locating the evidence in the document."""


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


SKILL_BLOCK_HEADER = """## Relevant skills (retrieved from past solved/failed tasks)

The following skills were distilled from prior OfficeQA document-navigation
tasks. They are guidance, not commands: if a skill applies to the current task,
follow it; if none applies, rely on your own judgement.

<skills>
{skill_block}
</skills>

""" + PER_STEP_GATE + "\n"


def build_agent_system(skill_block: str | None) -> str:
    """Return the policy system prompt, with retrieved skills injected when
    `skill_block` is non-empty (the `ours` / `orig` conditions). Empty/None
    reproduces the no-bank baseline verbatim."""
    if not skill_block or not skill_block.strip():
        return AGENT_SYSTEM
    return AGENT_SYSTEM + "\n\n" + SKILL_BLOCK_HEADER.format(
        skill_block=skill_block.strip())


# Whole-doc injection: a single consolidated skill library, mandatory-use
# framing, NO retrieval.
SKILL_DOC_HEADER = """## Loaded Skill Library (consolidated from past OfficeQA tasks)

The following is a curated, general skill library distilled from many solved and
failed document-navigation tasks. Use it as your standard operating procedure.

### Mandatory Skill Usage Protocol
- First, analyze the question and the source_document, and plan which of these
  skills apply.
- During execution, if a skill covers the operation you need, you MUST follow it.
- Only act on your own judgment if no skill is relevant, or a skill does not
  cover the specific operation you need.

<skills>
{skill_doc}
</skills>

""" + PER_STEP_GATE + "\n"


def build_agent_system_doc(skill_doc: str | None) -> str:
    """Inject a WHOLE consolidated skill library into the system prompt -- no
    per-task retrieval. Empty/None reproduces the no-bank baseline."""
    if not skill_doc or not skill_doc.strip():
        return AGENT_SYSTEM
    return AGENT_SYSTEM + "\n\n" + SKILL_DOC_HEADER.format(skill_doc=skill_doc.strip())


def build_agent_system_skillopt(skill_doc: str | None) -> str:
    """SkillOpt-FAITHFUL 'direct-chat' injection: append the whole skill as a
    bare `## Skill` section to the system prompt, once -- no usage protocol, no
    per-step gate. Verbatim shape of SkillOpt's codegen_agent._build_system
    (`base + '\\n\\n## Skill\\n' + skill_content`)."""
    if not skill_doc or not skill_doc.strip():
        return AGENT_SYSTEM
    return AGENT_SYSTEM + "\n\n## Skill\n" + skill_doc.strip()


# On-disk injection: only the concise SKILL.md is preloaded into the prompt; the
# full skill folder (incl. references/*.md) sits ON DISK and the agent reads the
# references on demand. Progressive disclosure -- the always-in-context part
# stays small; depth is pulled only when relevant.
SKILL_DIR_HEADER = """## Relevant Skill

The following skill (its SKILL.md) is loaded for your reference. Its FULL skill
folder -- including detailed `references/*.md` files -- is on disk at:
  {skill_dir}

### Mandatory Skill Usage Protocol
- First analyze the question and source_document, and plan how to use the skill.
- SKILL.md below is the concise index. When a skill looks relevant and you need
  its full detail, read the reference files under:
      {skill_dir}/references/
- If the skill contains relevant guidance, you must follow it. Only rely on your
  own judgement when no skill applies or it doesn't cover your specific operation.

<skill_content>
{skill_md}
</skill_content>
"""


def build_agent_system_skilldir(skill_md: str | None, skill_dir: str | None) -> str:
    """On-disk injection: preload SKILL.md, point the agent at the on-disk folder
    to read references/ on demand."""
    if not skill_md or not skill_md.strip() or not skill_dir:
        return AGENT_SYSTEM
    return AGENT_SYSTEM + "\n\n" + SKILL_DIR_HEADER.format(
        skill_dir=skill_dir, skill_md=skill_md.strip())


def task_query(question: str) -> str:
    """Retrieval query for a task = its question (truncated)."""
    return (question or "").strip()[:2000]


# ---------------------------------------------------------------------------
# 2. Trajectory formatting
# ---------------------------------------------------------------------------

# The OfficeQA bridge re-emits the question + docs header on every observation,
# then appends the tool output as the trailing "\n\n"-separated part after this
# instruction sentence. Split on it and keep the tail = the actual tool output.
_OBS_MARKER = "then answer(text) with the concise final answer."


def _step_action(step: dict) -> tuple[str, dict]:
    ra = step.get("raw_action") or {}
    return str(ra.get("name") or ""), dict(ra.get("kwargs") or {})


def _fmt_args(kwargs: dict, max_val: int = 200) -> str:
    parts = []
    for k, v in kwargs.items():
        sv = str(v)
        if len(sv) > max_val:
            sv = sv[:max_val] + "..."
        parts.append(f"{k}={sv!r}")
    return ", ".join(parts)


def _step_output(step: dict, max_chars: int) -> str:
    obs = step.get("filtered_observation") or step.get("raw_observation") or {}
    text = obs.get("text") if isinstance(obs, dict) else ""
    text = text or ""
    if _OBS_MARKER in text:
        text = text.split(_OBS_MARKER, 1)[1].strip()
    return text[:max_chars]


def format_trajectory(steps: list[dict], max_arg_chars: int = 200,
                      max_obs_chars: int = 600,
                      max_think_chars: int = 500) -> str:
    """Render an OfficeQA trajectory as reasoning + tool call + tool output per
    step, for the induction prompt. Tools are glob/read/grep/answer."""
    lines: list[str] = []
    for i, step in enumerate(steps, 1):
        raw = (step.get("policy_raw_response") or "").strip()
        name, kwargs = _step_action(step)
        out = _step_output(step, max_obs_chars)
        block = [f"Step {i}:"]
        if raw:
            block.append(f"  reasoning: {raw[:max_think_chars]}")
        if name == "answer":
            ans = str(kwargs.get("text", ""))
            if len(ans) > max_arg_chars:
                ans = ans[:max_arg_chars] + "..."
            block.append(f"  action: answer({ans!r})")
        elif name:
            block.append(f"  TOOL {name}({_fmt_args(kwargs, max_arg_chars)})")
        if out and name != "answer":
            block.append(f"  output: {out}")
        lines.append("\n".join(block))
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Skill induction (document-navigation domain)
# ---------------------------------------------------------------------------

_OUTPUT_FORMAT = """## Output Format
Your output must strictly follow this Markdown format:

```
# Memory Item i
## Title <imperative, <= 8 words, names the tactic>
## Description <one sentence shaped "When <situation>, <do Y>">
## Content <= 250 chars. MUST name a SPECIFIC primitive: a grep-pattern-choice rule, a read-window tactic, a glob pattern, or a concrete column/header/year verification step. Worked examples of the required specificity: "grep the bare 2- or 4-digit year rather than the full metric label -- parsers split multi-word headers across lines"; "after a numeric grep hit, read the ~8 lines ABOVE it to capture the column-header/units row before trusting the value"; "when the cited document lacks the figure, glob the sibling edition (same series, adjacent month) and grep there".>
```
"""

# Quality gate mirroring SWE-bench's FORBIDDEN enumeration. Without this the
# inducer emits generic/crude skills.
_FORBIDDEN = """## FORBIDDEN -- reject these; better EMPTY than noisy
Emit NOTHING rather than any item that is:
  - Process paranoia restating the task: "verify you read the right column",
    "make sure it's the correct year", "double-check the value matches the question".
  - A platitude: "be careful with units", "read the document carefully",
    "cross-reference the data", "search systematically", "broaden search terms".
  - A non-atomic tool mention: "use grep to find the table", "use read to inspect"
    -- names a tool but no specific pattern-choice or read-window heuristic.
  - Anything naming a specific document / metric / year / value.
Every item must name a concrete grep/read/glob primitive or verification step.
If you cannot find a clean technique-level difference, output NOTHING.
"""

SUCCESS_SI = (
    """You are a document-navigation expert who answers factual questions by """
    """searching parsed financial-table documents (grep / read / glob over U.S. """
    """Treasury Bulletin tables). You are given a question and a trajectory in """
    """which an agent found the correct answer.

Extract at most 3 reusable, transferable navigation PRIMITIVES that would help
answer SIMILAR document-QA questions. Each must name a specific grep-pattern
choice, read-window tactic, glob pattern, or column/header/year verification
step -- not a general strategy. Prefer EMPTY over generic.

"""
    + _FORBIDDEN + "\n"
    + _OUTPUT_FORMAT
)

FAILED_SI = (
    """You are a document-navigation expert who answers factual questions by """
    """searching parsed financial-table documents (grep / read / glob over U.S. """
    """Treasury Bulletin tables). You are given a question and a trajectory in """
    """which an agent FAILED to find the correct answer.

State the root cause in one line, then extract at most 3 preventative
PRIMITIVES -- each naming a specific grep/read/glob technique or verification
step that would have avoided THIS failure (e.g. "read the header row above a
numeric hit before answering"). Prefer EMPTY over generic.

"""
    + _FORBIDDEN + "\n"
    + _OUTPUT_FORMAT
)

PAIRED_DIFF_SI = (
    """You are extracting atomic, transferable document-navigation PRIMITIVES by """
    """COMPARING two trajectories on the SAME question in the SAME environment """
    """(grep / read / glob over parsed financial-table documents).

  - FAILURE: the agent attempted the question and answered WRONG (or not at all).
  - SUCCESS: the agent attempted the same question and answered CORRECTLY.

Identify the SPECIFIC PRIMITIVE the SUCCESS used that the FAILURE did not -- a
better grep term choice, reading the column-header row before the value,
verifying the right year column, globbing a sibling document. If the only
difference is luck or unrelated knowledge, emit NOTHING.
{diagnosis}
## Constraints (every item must satisfy ALL)
  1. ATOMIC: one specific grep/read/glob primitive or verification step -- NOT a
     meta-recommendation, NOT "search systematically".
  2. GENERAL-AT-CORE: the primitive works across document-QA tasks; the situation
     it triggers on may mention domain context, but never a specific document
     name, line number, metric, year, or value.
  3. STRUCTURE: phrasable as "When <situation>, <do specific-Y>" where Y names a
     grep pattern / read window / glob / header-check.
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


def induce_success(question: str, traj_text: str,
                   llm_model: str = "openai/gpt-4.1-mini",
                   max_items: int = 3, gemini_api_key: str | None = None) -> list[dict]:
    user = f"Question: {question[:2000]}\n\nSuccessful trajectory:\n{traj_text[:9000]}"
    return _induce(SUCCESS_SI, user, llm_model, max_items, 0.0, gemini_api_key=gemini_api_key)


def induce_failed(question: str, traj_text: str,
                  llm_model: str = "openai/gpt-4.1-mini",
                  max_items: int = 3, gemini_api_key: str | None = None) -> list[dict]:
    user = f"Question: {question[:2000]}\n\nFailed trajectory:\n{traj_text[:9000]}"
    return _induce(FAILED_SI, user, llm_model, max_items, 0.0, gemini_api_key=gemini_api_key)


def induce_paired_diff(question: str, fail_text: str, succ_text: str,
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
    user = (f"Question: {question[:2000]}\n\n"
            f"FAILURE trajectory:\n{fail_text[:7000]}\n\n"
            f"SUCCESS trajectory:\n{succ_text[:7000]}\n\n"
            "Compare. Extract atomic primitives. If no clean difference, emit nothing.")
    return _induce(system, user, llm_model, max_items, 0.0, gemini_api_key=gemini_api_key)
