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

"""SWEBenchEnvBashOnly -- think_action-compatible SWE-bench env subclass.

`SWEBenchEnv.tool_registry == [Bash, StrReplaceEditor]` -- two tools, which
the `text_complete` / `think_action` PolicyAgent dispatch path rejects
(it requires exactly one tool with exactly one argument). This three-line
subclass strips the registry to `[Bash]` so the think_action eval / corpus
runs can use the SWE-bench bridge without touching `PolicyAgent`.

Used by:
- `experiments/swebench/corpus.yaml` (Stage 1 corpus generation)
- `experiments/swebench/reasoning_bank_eval.py` (ReasoningBank eval)
"""
from __future__ import annotations
from typing import ClassVar

from envharness.bridges.swebench import SWEBenchEnv
from envharness.bridges.swebench.tools import Bash
from envharness.core.tool import Tool


class SWEBenchEnvBashOnly(SWEBenchEnv):
    tool_registry: ClassVar[list[type[Tool]]] = [Bash]
