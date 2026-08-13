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

"""Compile + load Rules-emitted Python source.

Contract: the source MUST define a top-level class named `_Rules` that
subclasses `Rules` (`envharness.harnesses.rules:Rules`). The class
is loaded in a controlled namespace that exposes the public framework
symbols (`Rules`, `Action`, `Blocked`, `Observation`, `EnvResponse`).
Failure modes raise `RulesCodeError` with a message shaped to be useful
when fed back to the LLM for self-repair.

Used by:
- `Rules.from_state(state)` when loading a checkpoint.
- The orchestrator's runner when applying a Candidate that carries
  `rules_code` (a Rules-emitted harness).
"""
from __future__ import annotations

from envharness.core.types import Action, Blocked, EnvResponse, Observation


class RulesCodeError(Exception):
    """Raised when Rules-emitted code cannot be turned into a usable
    Rules subclass. The message is shaped to be useful as feedback to
    the LLM for self-repair."""


def _default_namespace() -> dict:
    # Late import: Rules (and therefore EnvHarness, ActionableEnv) must
    # be importable before this namespace is built. We import lazily to
    # avoid a top-level cycle.
    from envharness.harnesses.rules import Rules
    return {
        "Rules":     Rules,
        "Action":      Action,
        "Blocked":     Blocked,
        "Observation": Observation,
        "EnvResponse": EnvResponse,
    }


def load_rules_subclass(code: str) -> type:
    """Compile `code` and return the `_Rules` class object.

    Returns the CLASS (not an instance) so the caller can decide how to
    instantiate it -- typically `subclass(inner=<env>)`. This separation
    matters because the constructor needs an `inner` ActionableEnv, which
    `from_state` provides.

    Empty code returns the base `Rules` class (all defaults).
    """
    from envharness.harnesses.rules import Rules

    if not code.strip():
        return Rules

    try:
        compiled = compile(code, "<mutator>", "exec")
    except SyntaxError as e:
        raise RulesCodeError(
            f"SyntaxError: {e.msg} at line {e.lineno}, column {e.offset}"
        ) from e

    namespace = _default_namespace()
    try:
        exec(compiled, namespace)  # noqa: S102 -- intentional execution of LLM code
    except Exception as e:
        raise RulesCodeError(
            f"Code raised at module load time: {type(e).__name__}: {e}"
        ) from e

    cls = namespace.get("_Rules")
    if cls is None:
        raise RulesCodeError(
            "Code must define a top-level class named `_Rules`."
        )
    if not isinstance(cls, type) or not issubclass(cls, Rules):
        raise RulesCodeError(
            "`_Rules` must subclass `Rules`."
        )
    return cls


def load_rules_instance(code: str, inner=None):
    """Convenience: compile `code`, return an instance bound to `inner`.

    Equivalent to `load_rules_subclass(code)(inner=inner)` but also
    sets `instance.rules_code = code` so the instance can round-trip
    through `save_state`."""
    cls = load_rules_subclass(code)
    inst = cls(inner=inner)
    inst.rules_code = code if code.strip() else ""
    return inst
