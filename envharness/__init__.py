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

"""EnvHarness: autogenous environment scaling through harness mutation.

Top-level layout
----------------
- envharness.core           data contracts + ActionableEnv / EnvHarness
                            ABCs + Tool + code loader + registries
- envharness.harnesses      concrete EnvHarness implementations
                            (Setup, Rules)
- envharness.persistence    save / load for (env + harness stack)
- envharness.bridges        per-benchmark ActionableEnv implementations
                            (toy24, alfworld, swebench, webarena)
- envharness.orchestration  outer loop: Orchestrator, Runner, ...
- envharness.agents         PolicyAgent and search-driving LLM agents
- envharness.infra          cross-cutting: LLM clients, endpoint pool, utils
- envharness.prompts        optional per-bench prompt builders

LLM-emitted Rules code only needs `Rules` plus the data contracts
(`Action`, `Blocked`, `Observation`, `EnvResponse`) -- they are
re-exported here so the LLM can `from envharness import Rules, Action,
Blocked, Observation, EnvResponse` without knowing the subpackage layout.
"""

from envharness.core import (
    Action, ActionableEnv, Blocked, Candidate, DecideResult, Decision,
    EnvHarness, EnvResetResponse, EnvResponse, EvaluationResult,
    FailureAnalysis, RulesCodeError, Observation, ObjectiveSignal, Step,
    StepInfo, Tool, Trace,
    load_rules_instance, load_rules_subclass,
    register_env, register_harness,
    get_env_class, get_harness_class,
    registered_envs, registered_harnesses,
)
from envharness.harnesses import Setup, Link, Rules
from envharness.persistence import (
    Checkpoint, build_stack, dump_stack, load_checkpoint, save_checkpoint,
)

# Auto-register the dependency-free base ActionableEnv (toy24).
# Heavier bridges (alfworld, swebench, webarena) carry runtime deps
# (alfworld[full], docker, playwright) and must be imported explicitly by
# the user before `load_checkpoint` can resolve their env_type tag::
#
#     import envharness.bridges.alfworld   # registers "alfworld"
#     env = load_checkpoint("path/to/cp.json")
#
# This is a deliberate trade-off: keeping `import envharness` cheap and
# dependency-light, at the cost of one extra import line for non-toy24
# benches. The error from load_checkpoint when a tag is unknown lists
# every registered env so the user can see what's missing.
from envharness.bridges import toy24 as _toy24  # noqa: F401

__all__ = [
    # ABCs
    "ActionableEnv", "EnvHarness",
    # Concrete harnesses
    "Setup", "Link", "Rules",
    # Data contracts (also what LLM-emitted Rules code imports)
    "Action", "Blocked", "Observation",
    "EnvResponse", "EnvResetResponse", "EvaluationResult",
    "Step", "StepInfo", "Candidate", "Trace",
    "Decision", "FailureAnalysis", "DecideResult", "ObjectiveSignal",
    # Tool ABC
    "Tool",
    # Rules-source loading
    "RulesCodeError", "load_rules_subclass", "load_rules_instance",
    # Registry
    "register_env", "register_harness",
    "get_env_class", "get_harness_class",
    "registered_envs", "registered_harnesses",
    # Save / load
    "Checkpoint", "dump_stack", "build_stack",
    "save_checkpoint", "load_checkpoint",
]
