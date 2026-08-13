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

"""Core: data contracts + ActionableEnv / EnvHarness ABCs + tooling.

Public surface:

  Data types (re-exported from envharness.core.types):
      Action, Blocked, Observation, EnvResponse, EnvResetResponse,
      EvaluationResult, Step, StepInfo, ...

  ABCs:
      ActionableEnv     universal env interface (reset/step/.../save_state)
      EnvHarness        composable middleware ABC (is-a ActionableEnv)

  Tool:
      Tool              one action available to the agent; auto-schemas

  Code loader:
      load_rules_subclass(code)   compile LLM-emitted Rules source
      load_rules_instance(code, inner=...)
      RulesCodeError

  Registries:
      register_env(tag), register_harness(tag)   class decorators
      get_env_class(tag), get_harness_class(tag)
      registered_envs(), registered_harnesses()
"""
from envharness.core.actionable_env import ActionableEnv
from envharness.core.code_loader import (
    RulesCodeError, load_rules_instance, load_rules_subclass,
)
from envharness.core.envharness import EnvHarness
from envharness.core.registry import (
    get_env_class, get_harness_class,
    register_env, register_harness,
    registered_envs, registered_harnesses,
)
from envharness.core.tool import Tool
from envharness.core.types import (
    Action, Blocked, Candidate, DecideResult, Decision, EnvResetResponse,
    EnvResponse, EvaluationResult, FailureAnalysis, Observation,
    ObjectiveSignal, Step, StepInfo, Trace,
)

__all__ = [
    "ActionableEnv", "EnvHarness", "Tool",
    "RulesCodeError", "load_rules_subclass", "load_rules_instance",
    "register_env", "register_harness",
    "get_env_class", "get_harness_class",
    "registered_envs", "registered_harnesses",
    "Action", "Blocked", "Observation", "EnvResponse", "EnvResetResponse",
    "EvaluationResult", "Step", "StepInfo", "Candidate", "Trace",
    "Decision", "FailureAnalysis", "DecideResult", "ObjectiveSignal",
]
