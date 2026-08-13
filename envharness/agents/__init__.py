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

"""Agents that DRIVE the harness-search loop.

`HarnessAgent` is the ABC the orchestrator talks to: it
`propose / decide / refine`s candidate harnesses (Setup + Rules stacks)
to apply to the env, and observes the K-rollout outcomes.

Renamed from the old `Rules` ABC because the EnvHarness implementation
that actually wraps the env is now called `Rules` (the `_Rules(Rules)`
the LLM emits); the agent that DRIVES the search is the `HarnessAgent`.
"""
from envharness.agents.harness_agent import (
    ExploringHarnessAgent,
    HarnessAgent,
    HarnessAgentContext,
    LLMHarnessAgent,
    NoopHarnessAgent,
    ScriptedHarnessAgent,
)
from envharness.agents.policy import ActionFormat, PolicyAgent

__all__ = [
    "HarnessAgent", "HarnessAgentContext",
    "LLMHarnessAgent", "NoopHarnessAgent", "ScriptedHarnessAgent",
    "ExploringHarnessAgent",
    "PolicyAgent", "ActionFormat",
]
