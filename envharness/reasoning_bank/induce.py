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

"""Induce memory items from a single trajectory using an LLM.

Prompts copied verbatim from Google Research ReasoningBank's WebArena
`memory_instruction` prompts, with the domain word "web navigation"
replaced by "household text environment" to fit ALFWorld.
"""
from __future__ import annotations

from envharness.infra.model import completion_kwargs
# Verbatim ReasoningBank prompts; kept under envharness/third_party/ with the rest of
# the third-party material, re-exported here so callers are unaffected.
from envharness.third_party.reasoning_bank.prompts import (  # noqa: F401
    FAILED_SI, PARALLEL_SI, SUCCESSFUL_SI,
)
import re
import sys
import time
import litellm



# Exception class NAMES that must never be retried/swallowed: a missing API
# key or exhausted budget fails every call identically, and `return []`
# would let a keyless overnight corpus run "succeed" with zero items.
# Matched by name across the MRO so we don't depend on litellm's module layout.
_FATAL_EXC_NAMES = {
    "AuthenticationError", "PermissionDeniedError",
    "NotFoundError", "BudgetExceededError",
}


def _is_fatal_llm_error(e: Exception) -> bool:
    return any(c.__name__ in _FATAL_EXC_NAMES for c in type(e).__mro__)






# RB's "parallel / scaling" induction prompt -- self-contrast across MULTIPLE
# trajectories of the same task, distilling them into <=5 memory items.
# Verbatim from ReasoningBank's WebArena `memory_instruction` prompts
# (PARALLEL_SI), with the domain word "web navigation" replaced by "text-based
# household environment (ALFWorld)".


_ITEM_RE = re.compile(
    r"#\s*Memory Item\s*\d*\s*\n+"
    r"##\s*Title\s*:?\s*(?P<title>.+?)\n+"
    r"##\s*Description\s*:?\s*(?P<description>.+?)\n+"
    r"##\s*Content\s*:?\s*(?P<content>.+?)(?=\n+#\s*Memory Item|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def _clean(s: str) -> str:
    s = s.strip()
    # Strip leading colon + whitespace (PARALLEL_SI sometimes emits "## Title: X"
    # vs the template "## Title X"; regex captures the colon as part of the value).
    while s.startswith(':') or s.startswith('：'):
        s = s[1:].lstrip()
    # Strip standalone markdown fences. _ITEM_RE's last item's content captures
    # to \Z, so the model's closing ``` fence would otherwise be injected
    # verbatim into eval prompts.
    lines = s.splitlines()
    _fence = lambda ln: re.fullmatch(r"`{3,}[\w-]*", ln.strip()) is not None
    while lines and _fence(lines[0]):
        lines.pop(0)
    while lines and _fence(lines[-1]):
        lines.pop()
    return "\n".join(lines).strip()


def parse_memory_items(md: str) -> list[dict]:
    items: list[dict] = []
    for m in _ITEM_RE.finditer(md or ""):
        items.append({
            "title":       _clean(m.group("title")),
            "description": _clean(m.group("description")),
            "content":     _clean(m.group("content")),
        })
    if items:
        return items
    return _parse_memory_items_loose(md or "")


def _parse_memory_items_loose(md: str) -> list[dict]:
    """Fallback parser for the format variants gemini-flash actually emits.

    The strict regex requires `## Title <text>` with the value on the SAME
    line. Models frequently emit instead:

        # Memory Item 1
        ## <the title text itself>          <- no literal 'Title' keyword
        ## Description
        <value on the NEXT line(s)>         <- keyword-only header line
        ## Content
        <value>

    Without this fallback every such response parses to 0 items, silently
    dropping the task from BOTH banks (paired-diff induction was losing all
    mixed-outcome tasks this way). Applies identically to every condition,
    so the two arms stay symmetric."""
    blocks = re.split(r"^#\s*Memory Item[^\n]*$", md, flags=re.MULTILINE | re.IGNORECASE)
    items: list[dict] = []
    for block in blocks[1:]:
        # Segment the block by ## headers; each segment = (header_line, body).
        segs = re.split(r"^##\s*", block, flags=re.MULTILINE)
        title = description = content = ""
        for seg in segs[1:]:
            lines = seg.splitlines()
            head = (lines[0] if lines else "").strip()
            body = "\n".join(lines[1:]).strip()
            low = head.lower()
            if low.startswith("title"):
                title = _clean(head[len("title"):]) or _clean(body)
            elif low.startswith("description"):
                description = _clean(head[len("description"):]) or _clean(body)
            elif low.startswith("content"):
                content = _clean(head[len("content"):]) or _clean(body)
            elif not title:
                # First non-keyword ## header = the title text itself.
                title = _clean(head)
                if body and not description and not content:
                    # Rare: model writes description prose right under it.
                    description = _clean(body)
        if title and (description or content):
            items.append({"title": title,
                          "description": description or content[:200],
                          "content": content or description})
    return items


def format_trajectory(steps: list[dict], max_obs_chars: int = 350) -> str:
    """Convert per-step records (envharness trace 'steps' list) into a compact
    string acceptable by the induction prompt.

    If `step.policy_raw_response` is present (set when the policy uses
    text_complete / react / think_action format and runner is capturing it),
    the trajectory is formatted as the ReasoningBank recipe expects:
        <think>...</think>
        <action>cmd</action>
        Observation: ...

    Otherwise (legacy traces), falls back to action+observation only.
    """
    import re as _re
    THINK_RE = _re.compile(r"<think>.*?</think>", _re.DOTALL)
    ACTION_RE = _re.compile(r"<action>.*?</action>", _re.DOTALL)
    lines: list[str] = []
    for i, step in enumerate(steps):
        fa = step.get("filtered_action") or {}
        kwargs = fa.get("kwargs") or {}
        action_text = kwargs.get("text") or fa.get("name") or "<?>"
        obs = step.get("filtered_observation") or {}
        obs_text = (obs.get("text") or "")[:max_obs_chars]
        raw = step.get("policy_raw_response") or ""
        if raw:
            think_m = THINK_RE.search(raw)
            act_m = ACTION_RE.search(raw)
            if think_m or act_m:
                pieces = [f"Step {i+1}:"]
                if think_m: pieces.append(think_m.group(0))
                pieces.append(act_m.group(0) if act_m else f"<action>{action_text}</action>")
                pieces.append(f"Observation: {obs_text}")
                lines.append("\n".join(pieces))
                continue
            # raw response present but no think/action tags -- still include it
            lines.append(f"Step {i+1}:\nReasoning: {raw.strip()[:400]}\nAction: {action_text}\nObservation: {obs_text}")
            continue
        # Legacy fallback
        lines.append(f"Step {i+1}:\nAction: {action_text}\nObservation: {obs_text}")
    return "\n\n".join(lines)


def induce_memory_items(
    task_query: str, trajectory_text: str, success: bool,
    llm_model: str = "openai/gpt-4.1-mini",
    max_items: int = 3, retries: int = 4,
) -> list[dict]:
    """Returns at most `max_items` parsed memory items {title, description, content}."""
    system = SUCCESSFUL_SI if success else FAILED_SI
    user = (f"User query: {task_query}\n\n"
            f"Trajectory:\n{trajectory_text}")
    for attempt in range(retries):
        try:
            r = litellm.completion(
                messages=[{"role": "system", "content": system},
                           {"role": "user",   "content": user}],
                **completion_kwargs(llm_model, temperature=0.0),
            )
            txt = r.choices[0].message.content or ""
            items = parse_memory_items(txt)
            return items[:max_items]
        except Exception as e:
            if _is_fatal_llm_error(e):
                raise    # auth/permission/budget errors fail every call; don't retry
            if attempt == retries - 1:
                print(f"[induce_memory_items] WARNING: giving up after "
                      f"{retries} attempts ({type(e).__name__}: {e}); "
                      f"returning 0 items", file=sys.stderr, flush=True)
                return []
            time.sleep(2 ** attempt)
    return []


def induce_memory_items_parallel(
    task_query: str, trajectories_with_success: list[tuple[str, bool]],
    llm_model: str = "openai/gpt-4.1-mini",
    max_items: int = 5, retries: int = 4,
) -> list[dict]:
    """RB's PARALLEL_SI induction: self-contrast across multiple trajectories
    of the SAME task. Each tuple is (trajectory_text, success_flag) -- the
    success flag is kept for the source dict, NOT shown to the model (RB's
    `induce_scaling.py` deliberately doesn't label each trajectory; the
    PARALLEL_SI prompt instructs the model to identify which succeeded by
    self-contrast reasoning).

    Returns at most `max_items` parsed memory items {title, description, content}.
    """
    if not trajectories_with_success:
        return []
    # Match RB's induce_scaling.py format exactly:
    #   **Query:** <task>
    #   **Trajectory 1 :**
    #   <traj>
    #   **Trajectory 2 :**
    #   ...
    parts = [f"**Query:** {task_query}\n"]
    for i, (traj, _success) in enumerate(trajectories_with_success, 1):
        parts.append(f"**Trajectory {i} :**\n{traj}\n")
    user = "\n".join(parts)
    for attempt in range(retries):
        try:
            r = litellm.completion(
                messages=[{"role": "system", "content": PARALLEL_SI},
                           {"role": "user",   "content": user}],
                **completion_kwargs(llm_model, temperature=0.7),
            )
            txt = r.choices[0].message.content or ""
            items = parse_memory_items(txt)
            return items[:max_items]
        except Exception as e:
            if _is_fatal_llm_error(e):
                raise    # auth/permission/budget errors fail every call; don't retry
            if attempt == retries - 1:
                print(f"[induce_memory_items_parallel] WARNING: giving up after "
                      f"{retries} attempts ({type(e).__name__}: {e}); "
                      f"returning 0 items", file=sys.stderr, flush=True)
                return []
            time.sleep(2 ** attempt)
    return []


# ---------------------------------------------------------------------------
# Paired-diff induction: compare FAIL vs SUCCESS on the SAME environment
# ---------------------------------------------------------------------------

PAIRED_DIFF_SI = """
You are extracting ATOMIC GENERAL skills by COMPARING two trajectories on the same task in the same environment.

  - FAILURE: the agent attempted the task and FAILED.
  - SUCCESS: the agent attempted the same task and SUCCEEDED.

Both ran in the SAME environment. The only difference is the agent's strategy.

COMPARE the two. Identify the SPECIFIC TECHNIQUE the SUCCESS used that FAILURE did not. If the only difference is luck, emit nothing.

Constraints:
1. ATOMIC: one specific action pattern or technique, not a meta-recommendation.
2. GENERAL: works across tasks. Domain context OK if the technique is portable.
3. TITLE: <= 10 words. DESCRIPTION: <= 1 sentence. CONTENT: <= 250 chars.
FORBIDDEN: vague advice, platitudes, task-specific references.

Output format:
```
# Memory Item i
## Title <imperative, <= 10 words>
## Description <when to apply>
## Content <specific technique, <= 250 chars>
```
Max 3 items. Empty is OK.
""".strip()


def induce_paired_diff(
    task_query: str, fail_trajectory_text: str, succ_trajectory_text: str,
    llm_model: str = "openai/gpt-4.1-mini",
    max_items: int = 3, retries: int = 4,
) -> list[dict]:
    """Compare a FAIL and SUCCESS trajectory on the same task. Extract skills."""
    user = (
        f"Task: {task_query[:2000]}\n\n"
        f"FAILURE trajectory:\n{fail_trajectory_text[:7000]}\n\n"
        f"SUCCESS trajectory:\n{succ_trajectory_text[:7000]}\n\n"
        f"Compare. Extract atomic skills. If no clean difference, emit NOTHING."
    )
    for attempt in range(retries):
        try:
            r = litellm.completion(
                messages=[{"role": "system", "content": PAIRED_DIFF_SI},
                          {"role": "user", "content": user}],
                **completion_kwargs(llm_model, temperature=0.0),
            )
            txt = r.choices[0].message.content or ""
            items = parse_memory_items(txt)
            return items[:max_items]
        except Exception as e:
            if _is_fatal_llm_error(e):
                raise    # auth/permission/budget errors fail every call; don't retry
            if attempt == retries - 1:
                print(f"[induce_paired_diff] WARNING: giving up after "
                      f"{retries} attempts ({type(e).__name__}: {e}); "
                      f"returning 0 items", file=sys.stderr, flush=True)
                return []
            time.sleep(2 ** attempt)
    return []
