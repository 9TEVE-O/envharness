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

"""Save / load for a stack of (ActionableEnv + zero or more EnvHarness layers).

The on-disk format is a single JSON document::

    {
      "schema_version": 1,
      "env":      {"type": "<env_type>", "state": {...}},
      "harnesses": [
        {"type": "<harness_type>", "state": {...}},   # innermost
        ...
        {"type": "<harness_type>", "state": {...}}    # outermost (what agent sees)
      ],
      "metadata": {...}                                # free-form annotation
    }

See `envharness.persistence.checkpoint` for the implementation.
"""
from envharness.persistence.checkpoint import (
    Checkpoint, load_checkpoint, save_checkpoint, dump_stack, build_stack,
)

__all__ = [
    "Checkpoint",
    "load_checkpoint", "save_checkpoint",
    "dump_stack", "build_stack",
]
