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

"""Tests for the Tool ABC and the auto-generated OpenAI function-call schema.

When you add a tool to a new Bridge, this test pattern catches:
  - mis-declared `name` / `description`
  - missing type hints (Tool.get_info would treat them as `str`)
  - required-vs-optional drift (defaults should NOT land in `required`)
"""
from envharness.core.tool import Tool


class _Add(Tool):
    """Add a number to a running sum."""
    name = "add"
    description = "Add a number."

    @classmethod
    def invoke(cls, env_state, a: int, b: int = 0) -> int:
        return a + b


def test_get_info_schema_shape():
    info = _Add.get_info()
    assert info["type"] == "function"
    fn = info["function"]
    assert fn["name"] == "add"
    assert fn["description"] == "Add a number."
    assert fn["parameters"]["type"] == "object"


def test_get_info_excludes_cls_and_env_state():
    """`cls` and `env_state` are framework plumbing, not LLM-visible args."""
    info = _Add.get_info()
    props = info["function"]["parameters"]["properties"]
    assert set(props) == {"a", "b"}


def test_get_info_required_only_no_defaults():
    info = _Add.get_info()
    assert info["function"]["parameters"]["required"] == ["a"]


def test_get_info_types_map_to_json_schema_types():
    info = _Add.get_info()
    props = info["function"]["parameters"]["properties"]
    assert props["a"]["type"] == "integer"
    assert props["b"]["type"] == "integer"


def test_default_name_falls_back_to_class_name():
    class Unnamed(Tool):
        """no name attribute"""
        @classmethod
        def invoke(cls, env_state):
            return None

    info = Unnamed.get_info()
    assert info["function"]["name"] == "Unnamed"
