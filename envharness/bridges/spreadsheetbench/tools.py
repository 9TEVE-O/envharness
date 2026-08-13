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

"""SpreadsheetBench tool registry.

Two tools, both dispatched directly by `SpreadsheetBenchEnv.step` (the working
directory + subprocess execution live on the Bridge, not in env_state, so we
bypass `Tool.invoke` exactly like the alfworld / swebench bridges). These stubs
exist only so `tool_schemas()` can emit the function-call schema the Policy and
the Rules-driving HarnessAgent see in their prompts.
"""
from __future__ import annotations

from typing import Any

from envharness.core.tool import Tool


class RunPython(Tool):
    name = "run_python"
    description = (
        "Execute a Python 3 snippet inside the task's working directory "
        "(openpyxl and pandas are available). Use it to inspect the input "
        "spreadsheet and to produce the answer. You MUST save your final "
        "result to the absolute output_path given in the observation. The "
        "snippet's combined stdout+stderr is returned as the next observation. "
        "Commands are stateless between calls (each runs in a fresh process), "
        "so write a self-contained script each time."
    )

    @classmethod
    def invoke(cls, env_state: Any, code: str) -> Any:
        raise NotImplementedError(
            "RunPython.invoke is unused; SpreadsheetBenchEnv.step executes the "
            "code in the episode working directory directly."
        )


class Submit(Tool):
    name = "submit"
    description = (
        "Call this with no arguments once the spreadsheet at output_path is "
        "final and correct. It ends the episode and triggers Online-Judge "
        "grading of output_path against the ground truth at answer_position."
    )

    @classmethod
    def invoke(cls, env_state: Any) -> Any:
        raise NotImplementedError(
            "Submit.invoke is unused; SpreadsheetBenchEnv.step handles "
            "submission termination directly."
        )
