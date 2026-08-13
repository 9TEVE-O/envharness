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

"""Concrete EnvHarness implementations.

- `Setup`   replays a list of actions on top of inner.reset to land on a
             mutated starting state (the S₀ axis). Save = list of actions.
- `Rules` wraps inner with 3 per-step transformation hooks defined in
             subclass code (A / T / O; S₀ belongs to Setup, R is
             intentionally absent). Save = Python source string.
- `Link`  composes TWO ActionableEnvs serially within one episode. Verifier
             = a.success AND b.success. Used for long-horizon experiments
             where a single sub-env's horizon is too short to surface
             budget-management / context-management skills.

All three are registered with stable type tags ("setup" / "rules" / "link")
so save files are self-describing without import paths. Link checkpoint is
NOT supported in schema v1 (tree shape, see envharness/harnesses/link.py).
"""
from envharness.harnesses.setup import Setup
from envharness.harnesses.link import Link
from envharness.harnesses.rules import Rules

__all__ = ["Setup", "Link", "Rules"]
