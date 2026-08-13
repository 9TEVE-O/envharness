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

"""SkillOS-style ALFWorld prompt + retrieval helpers (SkillOS: arXiv:2605.06614).

Used by `experiments/alfworld/reasoning_bank_eval.py`. The prompt and memory-block
builders match the SkillOS setup; keep them byte-for-byte stable so evals remain
comparable across runs.
"""
from __future__ import annotations

import re


SKILLOS_TEMPLATE = """You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
## Past Relevant Skills
{retrieved_skills}
## Current Progress
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history} You are now at step {current_step} and your current observation is: {current_observation} Your admissible actions of the current situation are: {admissible_actions}

Now it's your turn to take an action. You should first reason step-by-step about the current situation with the help of past relevant skills. This reasoning process MUST be enclosed within <think> </think> tags. Once you've finished your reasoning, you should choose an admissible action for current step and MUST present it within <action> </action> tags."""


ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)


def extract_task(obs_text: str) -> str:
    for ln in (obs_text or "").splitlines():
        low = ln.strip().lower()
        if low.startswith("task:") or low.startswith("your task is to:"):
            return ln.split(":", 1)[1].strip()
    return ""


def strip_task_and_admissibles(obs_text: str) -> str:
    out = []
    for ln in (obs_text or "").splitlines():
        low = ln.strip().lower()
        if low.startswith("task:") or low.startswith("admissible commands:"):
            continue
        out.append(ln)
    return "\n".join(out).strip()


def normalize_to_admissible(cmd: str, admissibles: list[str]) -> str:
    if not cmd:
        return ""
    if cmd in admissibles:
        return cmd
    low = cmd.lower()
    for c in admissibles:
        if low == c.lower():
            return c
    for c in admissibles:
        if c.lower() in low:
            return c
    return cmd


def format_admissible(adm: list[str]) -> str:
    return "\n ".join(f"'{s}'" for s in adm if s != "help")


def format_history(records: list[dict], history_length: int) -> tuple[str, int]:
    recent = records[-history_length:]
    start = len(records) - len(recent)
    lines = []
    for j, rec in enumerate(recent):
        n = start + j + 1
        lines.append(f"[Observation {n}: '{rec['obs']}', "
                     f"Action {n}: '{rec['action']}']")
    return "\n".join(lines), len(recent)


def build_memory_block(items, style: str = "soft") -> str:
    """Format retrieved Bank items into the per-turn memory block. The
    per-step gate is part of `soft` rather than `legacy`: it instructs the
    policy to check relevance before applying an insight each step."""
    if not items:
        return ""
    gate = ("In each step, before acting, explicitly state whether any "
            "insight applies to the current observation and why; if none "
            "do, ignore them.")
    if style == "soft":
        header = ("Below are some memory items that I accumulated from past "
                  "interactions in this environment that may be helpful to "
                  "solve the task. You can use them when you feel they are "
                  f"relevant.\n\n{gate}\n")
    elif style == "legacy":
        header = ("The following are reasoning insights distilled from prior "
                  "similar tasks. Use them only if relevant; do not blindly "
                  "follow them.\n")
    else:
        raise ValueError(f"unknown inject_style: {style!r}")
    parts = [header]
    for i, it in enumerate(items, 1):
        parts.append(f"### Insight {i}: {it.title}")
        parts.append(f"_When to use_: {it.description}")
        parts.append(it.content)
        parts.append("")
    return "\n".join(parts) + "\n"
