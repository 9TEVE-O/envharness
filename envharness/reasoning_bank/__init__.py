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

"""ReasoningBank for EnvHarness.

Test-time training mechanism: build a bank of reasoning memory items from
agent trajectories during training, then retrieve top-K relevant items at
inference time to augment the policy prompt.

Public surface:
  Bank construction:
    `induce_memory_items`  -- single-trajectory induction (success or fail)
    `induce_memory_items_parallel` -- PARALLEL_SI multi-trajectory variant
    `format_trajectory`    -- convert a Trace's `steps` to the induction text
    `embed_texts`          -- Gemini-embedding-001 via litellm
  Retrieval:
    `Bank`         load a JSONL bank, retrieve top-K by cosine
    `MemoryItem`   one bank entry (title + description + content + embedding)

Adapted from Google Research ReasoningBank
(https://github.com/google-research/reasoning-bank).
"""
from .bank import Bank, MemoryItem
from .embed import embed_texts
from .induce import (
    FAILED_SI, PARALLEL_SI, SUCCESSFUL_SI, PAIRED_DIFF_SI,
    format_trajectory, induce_memory_items, induce_memory_items_parallel,
    induce_paired_diff, parse_memory_items,
)

__all__ = [
    "Bank", "MemoryItem", "embed_texts",
    "induce_memory_items", "induce_memory_items_parallel", "induce_paired_diff",
    "format_trajectory", "parse_memory_items",
    "SUCCESSFUL_SI", "FAILED_SI", "PARALLEL_SI", "PAIRED_DIFF_SI",
]
