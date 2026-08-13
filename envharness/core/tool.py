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

"""Tool ABC -- abstracted from tau-bench. Each tool self-describes via get_info()."""
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, get_type_hints


class Tool(ABC):
    """An action available to the agent.

    Subclasses define `name`, `description`, and an `invoke` classmethod.
    `get_info()` introspects `invoke` to produce an OpenAI function-call schema.
    """
    name: str = ""
    description: str = ""

    @classmethod
    @abstractmethod
    def invoke(cls, env_state: Any, **kwargs) -> Any:
        """Execute the tool against the env state. Return value is opaque to the
        framework -- the Bridge interprets it."""
        ...

    @classmethod
    def get_info(cls) -> dict:
        """Return an OpenAI function-call schema for this tool.

        Introspects invoke()'s signature; uses parameter names + types + the
        docstring (first line after the first blank) for the description.
        """
        sig = inspect.signature(cls.invoke)
        hints = get_type_hints(cls.invoke)
        properties: dict[str, dict] = {}
        required: list[str] = []
        for pname, param in sig.parameters.items():
            if pname in ("cls", "env_state"):
                continue
            t = hints.get(pname, str)
            properties[pname] = _python_type_to_schema(t)
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        return {
            "type": "function",
            "function": {
                "name": cls.name or cls.__name__,
                "description": cls.description or (cls.__doc__ or "").strip(),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


def _python_type_to_schema(t: type) -> dict:
    if t is int:    return {"type": "integer"}
    if t is float:  return {"type": "number"}
    if t is bool:   return {"type": "boolean"}
    if t is str:    return {"type": "string"}
    return {"type": "string"}   # fallback
