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

"""SWE-bench tool registry.

Two tools:
  - `bash(command)` -- arbitrary shell in /testbed (exploration, tests, git).
  - `str_replace_editor(operation, path, ...)` -- structured file edit ops
    matching OpenHands' tool surface, so 7-8B models don't have to wrestle
    `sed -i` escaping just to make a one-line change.

Pattern note: SWEBenchBridge keeps the docker container handle on itself
(not on env_state), so it bypasses `Tool.invoke` and dispatches directly
via `docker exec`. These Tool stubs exist so:
  (a) the Policy / Rules prompts can be auto-generated from their
      schemas via `Tool.get_info()`,
  (b) the action surface is explicit (one place to look for what the
      model can do).

If a future Bridge wants to actually route through `tool_registry`, the
`invoke` methods below contain working implementations.
"""
from __future__ import annotations

from typing import Any

from envharness.core.tool import Tool


# ---------------------------------------------------------------------------
# Bash
# ---------------------------------------------------------------------------


class Bash(Tool):
    name = "bash"
    description = (
        "Execute a bash command in the task's docker container at /testbed "
        "and return its stdout+stderr combined. Each invocation is a fresh "
        "`docker exec` (no persistent shell state across calls; use absolute "
        "paths or chain commands with && / ;). To submit your solution, "
        "echo the sentinel line `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` "
        "followed by your git diff -- the Bridge will detect the sentinel "
        "and capture everything after it as the submitted patch."
    )

    @classmethod
    def invoke(cls, env_state: Any, command: str) -> Any:
        # Not called: SWEBenchBridge.step dispatches `command` straight to
        # docker exec. Tool.invoke runs only when a Bridge opts into
        # tool_registry dispatch.
        raise NotImplementedError(
            "Bash.invoke is unused; SWEBenchBridge.step dispatches the "
            "command directly to docker exec inside the task container."
        )


# ---------------------------------------------------------------------------
# StrReplaceEditor
# ---------------------------------------------------------------------------
#
# Operations (OpenHands-compatible names):
#   view        -- print a file or list a directory (no edits).
#   create      -- write a brand new file with the given content (fails if exists).
#   str_replace -- replace the FIRST exact occurrence of `old_str` with
#                  `new_str` in the file. Fails if old_str is missing or appears
#                  multiple times.
#   insert      -- insert `new_str` AFTER line number `insert_line` (1-indexed).
#
# All paths must be absolute and live under /testbed. Anything else is an
# explicit error returned to the model so it can recover.
#
# The Bridge runs these as a `python -c "..."` inside the container, so the
# editor logic is identical regardless of whether we're on docker, podman, or
# a future runtime.


class StrReplaceEditor(Tool):
    name = "str_replace_editor"
    description = (
        "Structured file editor for /testbed. Use this INSTEAD of `sed -i` "
        "or `printf > file` for any edit -- it's much more reliable for "
        "non-trivial patches. Operations:\n"
        "  - view:        show a file (or list a directory).\n"
        "  - create:      write a NEW file (fails if path exists).\n"
        "  - str_replace: replace the FIRST occurrence of `old_str` with "
        "`new_str` in `path`. Fails if old_str appears 0 or >=2 times.\n"
        "  - insert:      insert `new_str` AFTER line `insert_line` "
        "(1-indexed) in `path`.\n"
        "All paths must be absolute and under /testbed."
    )

    # NOTE: Tool.get_info() introspects this signature for the function-call
    # JSON schema. Scalar types only per envharness/tool.py:
    #   operation:    str   (view | create | str_replace | insert)
    #   path:         str   (absolute path under /testbed)
    #   file_text:    str   (used by `create`; full new-file contents)
    #   old_str:      str   (used by `str_replace`; exact substring to find)
    #   new_str:      str   (used by `str_replace` and `insert`)
    #   insert_line:  int   (used by `insert`; 1-indexed, 0 = top of file)
    #   view_range:   str   (used by `view`; "start-end" e.g. "10-50"; "" = whole file)
    @classmethod
    def invoke(cls, env_state: Any,
               operation: str = "view",
               path: str = "",
               file_text: str = "",
               old_str: str = "",
               new_str: str = "",
               insert_line: int = 0,
               view_range: str = "") -> Any:
        # The Bridge handles this; this stub exists for symmetry with Bash.
        # See SWEBenchBridge._exec_str_replace_editor for the real impl.
        raise NotImplementedError(
            "StrReplaceEditor.invoke is unused; SWEBenchBridge.step "
            "dispatches via _exec_str_replace_editor inside the container."
        )
