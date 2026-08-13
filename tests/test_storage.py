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

"""Tests for TraceStore — JSONL-backed Trace persistence.

Round-trip requirements: a TraceStore opened on an existing JSONL file
must yield the same Traces it would have if they'd been added live.
"""
from envharness.orchestration.storage import TraceStore
from envharness.core.types import Candidate, Trace


def _t(eid: str, success: bool, kind: str = "accepted") -> Trace:
    return Trace(
        episode_id=eid, iteration_id="it", task_id="t",
        candidate=Candidate(), success=success, kind=kind,
    )


def test_add_writes_jsonl(tmp_path):
    store = TraceStore(tmp_path / "traces.jsonl")
    store.add(_t("e1", True))
    store.add(_t("e2", False))
    lines = (tmp_path / "traces.jsonl").read_text().splitlines()
    assert len(lines) == 2
    # each line is a valid Trace JSON
    Trace.model_validate_json(lines[0])


def test_reopen_loads_existing_traces(tmp_path):
    p = tmp_path / "traces.jsonl"
    s1 = TraceStore(p)
    s1.add(_t("e1", True))
    s1.add(_t("e2", False))
    s2 = TraceStore(p)                  # reopen from disk
    assert len(s2) == 2
    assert {t.episode_id for t in s2.all()} == {"e1", "e2"}


def test_recent_returns_tail(tmp_path):
    store = TraceStore(tmp_path / "x.jsonl")
    for i in range(5):
        store.add(_t(f"e{i}", success=bool(i % 2)))
    last_two = store.recent(2)
    assert [t.episode_id for t in last_two] == ["e3", "e4"]


def test_filter_kind_separates_groups(tmp_path):
    store = TraceStore(tmp_path / "x.jsonl")
    store.add(_t("a", True, kind="accepted"))
    store.add(_t("b", False, kind="exploration"))
    store.add(_t("c", True, kind="accepted"))
    accepted_ids = [t.episode_id for t in store.filter_kind("accepted")]
    assert accepted_ids == ["a", "c"]
