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

"""Subprocess entrypoint: read EpisodeSpec JSON from stdin, run one episode,
write the Trace JSON on stdout's last line. Used by SubprocessRunner.
"""
from __future__ import annotations

import json
import sys

from envharness.core.types import Candidate
from envharness.orchestration.runner import (
    EnvSpec, EpisodeSpec, PolicySpec, run_episode,
)


# obs.data keys that may carry numpy / large runtime objects we DON'T want
# in the persisted Trace (webarena's `browsergym_raw` carries the full
# unflattened dict incl. screenshot + axtree_object). They're useful at
# runtime for Policy/Rules but explode trace size and break Pydantic JSON.
_RUNTIME_ONLY_OBS_DATA_KEYS = ("browsergym_raw",)


def _strip_runtime_only(trace) -> None:
    for step in trace.steps:
        for obs in (step.raw_observation, step.filtered_observation):
            if obs is None:
                continue
            data = getattr(obs, "data", None)
            if not isinstance(data, dict):
                continue
            for k in _RUNTIME_ONLY_OBS_DATA_KEYS:
                data.pop(k, None)


def _spec_from_json(d: dict) -> EpisodeSpec:
    return EpisodeSpec(
        env=EnvSpec(**d["env"]),
        candidate=Candidate.model_validate(d["candidate"]),
        policy=PolicySpec(**d["policy"]),
        iteration_id=d["iteration_id"],
        task_id=d.get("task_id", ""),
        max_steps=d.get("max_steps", 50),
    )


def main() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        print("error: empty stdin", file=sys.stderr)
        return 2
    try:
        spec = _spec_from_json(json.loads(raw))
    except Exception as e:
        print(f"error: spec decode failed: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 2
    trace = run_episode(spec)
    _strip_runtime_only(trace)
    sys.stdout.write(trace.model_dump_json())
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
