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

"""OfficeQA tool registry.

Four tools, all dispatched directly by `OfficeQAEnv.step` (the docs root +
file access live on the Bridge, not in env_state, so we bypass `Tool.invoke`
exactly like the alfworld / swebench / sbench bridges). These stubs exist only
so `tool_schemas()` can emit the function-call schema the Policy and the
Rules-driving HarnessAgent see. Tool set + descriptions mirror SkillOpt's
officeqa offline (local-tools) mode.
"""
from __future__ import annotations

from typing import Any

from envharness.core.tool import Tool


class Glob(Tool):
    name = "glob"
    description = (
        "Find candidate local document files by filename or relative-path glob "
        "pattern (e.g. 'treasury_bulletin_1941_*.txt' or '*1941*'). Returns "
        "matching document paths under the docs root."
    )

    @classmethod
    def invoke(cls, env_state: Any, pattern: str) -> Any:
        raise NotImplementedError("Glob.invoke is unused; OfficeQAEnv.step handles it.")


class Read(Tool):
    name = "read"
    description = (
        "Read a line window from a local text document. `path` is a document "
        "path (from the observation or glob); `start` is the 1-based first line "
        "(default 1); `limit` is how many lines to return (default 200). Use it "
        "to inspect the region of the document around your evidence."
    )

    @classmethod
    def invoke(cls, env_state: Any, path: str, start: int = 1, limit: int = 200) -> Any:
        raise NotImplementedError("Read.invoke is unused; OfficeQAEnv.step handles it.")


class Grep(Tool):
    name = "grep"
    description = (
        "Search a local text document for a literal substring (case-insensitive) "
        "and return matching lines with their line numbers. Use it to locate the "
        "figure, year, or heading the question asks about before reading around it."
    )

    @classmethod
    def invoke(cls, env_state: Any, pattern: str, path: str) -> Any:
        raise NotImplementedError("Grep.invoke is unused; OfficeQAEnv.step handles it.")


class Answer(Tool):
    name = "answer"
    description = (
        "Submit your final answer to the question. Call this once, with the "
        "concise answer text (a number, name, or short phrase -- no explanation). "
        "It ends the episode and grades your answer against the ground truth."
    )

    @classmethod
    def invoke(cls, env_state: Any, text: str) -> Any:
        raise NotImplementedError("Answer.invoke is unused; OfficeQAEnv.step handles it.")
