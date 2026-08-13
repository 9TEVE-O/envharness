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

"""EnvHook -- lifecycle observers. Pattern adapted from SWE-agent.

Hooks are for logging, trace collection, external observers. They MUST NOT
mutate environment behavior -- that is the Middleware's job.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from envharness.core.types import Candidate, Trace


class EnvHook:
    def on_init(self, *, bridge: Any) -> None: ...
    def on_episode_start(self, *, bridge: Any, episode_id: str) -> None: ...
    def on_episode_end(self, *, bridge: Any, trace: "Trace") -> None: ...
    def on_mutation_applied(self, *, bridge: Any, mutation: "Candidate") -> None: ...
    def on_close(self, *, bridge: Any) -> None: ...


class CombinedEnvHooks(EnvHook):
    def __init__(self) -> None:
        self._hooks: list[EnvHook] = []

    def add(self, hook: EnvHook) -> None:
        self._hooks.append(hook)

    def on_init(self, *, bridge):
        for h in self._hooks: h.on_init(bridge=bridge)

    def on_episode_start(self, *, bridge, episode_id):
        for h in self._hooks: h.on_episode_start(bridge=bridge, episode_id=episode_id)

    def on_episode_end(self, *, bridge, trace):
        for h in self._hooks: h.on_episode_end(bridge=bridge, trace=trace)

    def on_mutation_applied(self, *, bridge, mutation):
        for h in self._hooks: h.on_mutation_applied(bridge=bridge, mutation=mutation)

    def on_close(self, *, bridge):
        for h in self._hooks: h.on_close(bridge=bridge)
