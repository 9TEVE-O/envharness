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

"""Action-string projection for verl-agent's GRPO loop with EnvHarness ALFWorld.

Identical contract to verl-agent's stock `alfworld_projection`: extract the
`<action>...</action>` substring, mark valid iff `<think>...</think>` is also
present and no Chinese characters appear.

After extraction, the action is **normalized** against the per-env
admissible command list (`action_pools[i]`) using the same logic as
`envharness.policy._normalize_command` (case-insensitive match, then
substring match). This aligns train-time action resolution with eval-time
`PolicyAgent.stateless_template`, eliminating a train/eval gap where
training rewarded "Nothing happens" for cosmetic mismatches that eval
would have auto-corrected.
"""
from __future__ import annotations
import re
from typing import List


def _normalize(text: str, admissible: list[str]) -> str:
    """Pick the best admissible command matching `text`."""
    text = text.strip()
    if not admissible:
        return text
    if text in admissible:
        return text
    low = text.lower()
    for c in admissible:
        if low == c.lower():
            return c
    for c in admissible:
        if c.lower() in low:
            return c
    return text


def envharness_alfworld_projection(actions: List[str],
                                    action_pools: List[List[str]]):
    valids = [0] * len(actions)
    for i in range(len(actions)):
        original_str = actions[i]
        actions[i] = actions[i].lower()
        start_tag, end_tag = "<action>", "</action>"
        start_idx = actions[i].find(start_tag)
        end_idx = actions[i].find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                actions[i] = actions[i][-30:]
                continue
            extracted = actions[i][start_idx + len(start_tag):end_idx].strip()
            actions[i] = _normalize(extracted, action_pools[i] if i < len(action_pools) else [])
            valids[i] = 1
        except Exception:
            actions[i] = actions[i][-30:]
        # require <think>...</think>
        if (original_str.find("<think>") == -1
                or original_str.find("</think>") == -1):
            valids[i] = 0
        # reject Chinese characters
        if re.search(r"[一-鿿]", original_str):
            valids[i] = 0
    return actions, valids
