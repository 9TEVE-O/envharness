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

"""TraceStore -- JSONL-backed persistent trace storage.

Format is newline-delimited JSON, one Trace per line, training-framework
friendly. Both 'accepted' and 'exploration' traces share the same schema and
the same file; downstream consumers filter by `kind`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from envharness.core.types import Trace


class TraceStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: list[Trace] = []
        if self.path.exists():
            with self.path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self._cache.append(Trace.model_validate_json(line))

    def add(self, trace: Trace) -> None:
        self._cache.append(trace)
        with self.path.open("a") as f:
            f.write(trace.model_dump_json() + "\n")

    def all(self) -> list[Trace]:
        return list(self._cache)

    def recent(self, n: int) -> list[Trace]:
        return self._cache[-n:]

    def filter_kind(self, kind: str) -> Iterator[Trace]:
        return (t for t in self._cache if t.kind == kind)

    def __len__(self) -> int:
        return len(self._cache)
