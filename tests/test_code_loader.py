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

"""Tests for code_loader -- how LLM-emitted Python becomes a live Rules.

This is the trust boundary: the LLM writes arbitrary Python and we exec it.
Failure modes (syntax errors, wrong class name, wrong base class, runtime
errors during instantiation) should turn into typed `RulesCodeError`s
with helpful messages, NOT crash the subprocess worker.
"""
import pytest

from envharness import Action, Blocked, Rules
from envharness.core.code_loader import (
    RulesCodeError, load_rules_instance, load_rules_subclass,
)


def test_empty_code_returns_base_mutator_class():
    cls = load_rules_subclass("")
    assert cls is Rules


def test_empty_code_instance_is_passthrough():
    inst = load_rules_instance("", inner=None)
    assert type(inst) is Rules
    assert inst.rules_code == ""
    # Default filter_action passes through.
    a = Action(name="x")
    assert inst.filter_action(a, env_state=None) is a


def test_well_formed_mutation_loads_and_overrides_hook():
    code = """
class _Rules(Rules):
    def filter_action(self, action, env_state):
        return Blocked(reason="nope")
"""
    cls = load_rules_subclass(code)
    assert issubclass(cls, Rules) and cls is not Rules
    inst = cls()    # no inner needed for this hook-only test
    out = inst.filter_action(Action(name="x"), env_state=None)
    assert isinstance(out, Blocked)
    assert out.reason == "nope"


def test_load_rules_instance_records_source_for_roundtrip():
    code = "class _Rules(Rules): pass"
    inst = load_rules_instance(code, inner=None)
    assert inst.rules_code.strip() == code.strip()


def test_syntax_error_raises_typed_error():
    with pytest.raises(RulesCodeError) as exc:
        load_rules_subclass("class _Rules(Rules):\n    def f(")
    assert "SyntaxError" in str(exc.value)


def test_missing_class_raises():
    with pytest.raises(RulesCodeError, match="_Rules"):
        load_rules_subclass("x = 1")


def test_wrong_base_class_raises():
    with pytest.raises(RulesCodeError, match="Rules"):
        load_rules_subclass("class _Rules: pass")


def test_runtime_error_during_load_is_typed():
    """A top-level expression that explodes during exec must surface as
    a typed RulesCodeError, not a bare Exception."""
    with pytest.raises(RulesCodeError, match="ZeroDivisionError"):
        load_rules_subclass("class _Rules(Rules): pass\n1/0")
