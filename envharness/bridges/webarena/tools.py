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

"""WebArena tool registry.

Single tool: `do(action_str)`. The Policy emits one browsergym high-level
action per turn (e.g. `click('123')`, `fill('5', 'hello')`, `goto('http://x')`,
`scroll(0, 200)`, `noop(500)`). The underlying Playwright + browsergym env
handle lives on the Bridge, so `WebArenaBridge.step` dispatches the string
directly to `env.step(action_str)` and bypasses `Tool.invoke`.

This stub exists only to surface a schema for `Bridge.tool_schemas()` so the
Policy / Rules prompts have a known signature.

Bridges may choose between (a) routing step() through `tool_registry` and
(b) bypassing it to talk to a runtime handle (this file's pattern; same
choice as the ALFWorld and SWE-bench bridges).
"""
from __future__ import annotations

from typing import Any

from envharness.core.tool import Tool


_HIGHLEVEL_ACTION_DESCRIPTION = (
    "Emit one browsergym high-level action string. Supported action types "
    "(browsergym 0.14.x `HighLevelActionSet(subsets=['bid'])`): "
    "noop(wait_ms), scroll(dx, dy), click(bid, button), dblclick(bid), "
    "hover(bid), fill(bid, value), select_option(bid, options), "
    "press(bid, key_comb), focus(bid), keyboard_press(key), "
    "keyboard_type(text), goto(url), go_back(), go_forward(), tab_close(), "
    "new_tab(), tab_focus(index). `bid` is the bracketed integer ID shown "
    "before each interactive element in the AXTree. Always pick a `bid` "
    "from the AXTree -- inventing ones leads to ElementNotFound. Wrap "
    "string arguments in single quotes."
)


class Do(Tool):
    name = "do"
    description = _HIGHLEVEL_ACTION_DESCRIPTION

    @classmethod
    def invoke(cls, env_state: Any, action_str: str) -> Any:
        # Not called: WebArenaBridge.step dispatches `action_str` straight
        # to the underlying browsergym env. Tool.invoke runs only when a
        # Bridge opts into tool_registry dispatch.
        raise NotImplementedError(
            "Do.invoke is unused; WebArenaBridge.step dispatches the action "
            "string directly to the underlying browsergym env."
        )
