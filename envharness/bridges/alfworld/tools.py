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

"""ALFWorld tool registry.

Single tool: `do(text)`. The underlying TextWorld engine is held by the
Bridge (not by env_state), so AlfworldBridge.step bypasses Tool.invoke and
dispatches the text command directly. This stub exists only for the schema
that Bridge.tool_schemas() emits for the Policy and Rules prompts.

Bridges may freely choose between (a) routing step() through tool_registry
(toy-style; see Toy24Bridge) and (b) bypassing tool_registry to talk to a
runtime handle (this file's pattern; same approach SWE-bench / WebArena
Bridges will use).
"""
from __future__ import annotations

from typing import Any

from envharness.core.tool import Tool


class Do(Tool):
    name = "do"
    description = (
        "Issue one natural-language ALFWorld command and observe the result. "
        "Examples: 'go to drawer 1', 'open drawer 1', 'take apple 2 from "
        "countertop 1', 'put apple 2 in/on fridge 1', 'examine bowl 1', "
        "'look', 'inventory'. Prefer a command listed in the observation's "
        "admissible commands."
    )

    @classmethod
    def invoke(cls, env_state: Any, text: str) -> Any:
        # Not called: AlfworldBridge.step dispatches `text` straight to the
        # underlying TextWorld engine. Tool.invoke runs only when a Bridge
        # opts into tool_registry dispatch (toy24 does; alfworld does not).
        raise NotImplementedError(
            "Do.invoke is unused; AlfworldBridge.step dispatches the text "
            "directly to the underlying TextWorld engine."
        )
