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

"""OfficeQA dataset loader.

Self-contained: reads `officeqa_full.csv` (the gated databricks/officeqa payload:
uid, question, answer, source_docs, source_files, difficulty) and the id-split
manifest (train/val/test uid lists) so absolute-index selection is stable and
the train / held-out split matches SkillOpt's.

Each task resolves to ONE parsed text document (`source_files`, e.g.
`treasury_bulletin_1941_01.txt`) living under the docs root
(`OFFICEQA_DOCS_DIR` or reset_options['docs_root']), which the agent inspects
with glob/read/grep. Grading is normalized exact-match on the short answer.
"""
from __future__ import annotations

import csv
import glob as _glob
import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache

# Env fallbacks when reset_options omits the paths.
DATA_PATH_ENV = "OFFICEQA_DATA"        # dir holding officeqa_full.csv + officeqa_id_split/
DOCS_DIR_ENV = "OFFICEQA_DOCS_DIR"     # dir holding the parsed treasury-bulletin txt


@dataclass(frozen=True)
class OQATask:
    """One OfficeQA instance. Pure paths + metadata; no file handles.

    OfficeQA tasks may reference MULTIPLE source documents (cross-document QA;
    ~half the set, up to 12 docs) -- `source_files`/`doc_paths` are tuples."""
    uid: str
    question: str
    answer: str
    source_files: tuple[str, ...]    # basenames, e.g. ("treasury_bulletin_1941_01.txt",)
    doc_paths: tuple[str, ...] = ()  # resolved abs paths (only the ones found on disk)
    source_doc: str = ""             # source URL(s)
    difficulty: str = ""             # "easy" | "hard"
    split: str = ""                  # "train" | "val" | "test" | ""
    missing: bool = False            # True if NONE of the source docs resolved on disk


@lru_cache(maxsize=8)
def _load_csv(csv_path: str) -> dict[str, dict]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"OfficeQA officeqa_full.csv not found at {csv_path!r}. Set "
            f"reset_options['data_path'] or ${DATA_PATH_ENV} to the dir that "
            "holds officeqa_full.csv + officeqa_id_split/."
        )
    rows: dict[str, dict] = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows[str(r["uid"]).strip()] = r
    return rows


@lru_cache(maxsize=8)
def _resolve_docs_index(docs_root: str) -> dict[str, str]:
    """basename -> abs path for every *.txt under docs_root (recursive)."""
    idx: dict[str, str] = {}
    for p in _glob.glob(os.path.join(docs_root, "**", "*.txt"), recursive=True):
        idx.setdefault(os.path.basename(p), p)
    return idx


@lru_cache(maxsize=16)
def _split_uids(data_path: str, split: str) -> tuple[str, ...]:
    """Ordered uid list for a split, from officeqa_id_split/<split>/items.json."""
    items_path = os.path.join(data_path, "officeqa_id_split", split, "items.json")
    if not os.path.isfile(items_path):
        raise FileNotFoundError(f"OfficeQA split items not found at {items_path!r}.")
    with open(items_path) as f:
        items = json.load(f)
    return tuple(str(it["uid"]).strip() for it in items)


@lru_cache(maxsize=16)
def load_split(data_path: str, docs_root: str, split: str) -> tuple[OQATask, ...]:
    """Materialize every task in `split` as an ordered tuple of OQATask.

    Ordered + immutable so absolute indexing (seed % len) is stable and the
    result is safe to cache/share across episodes. Tasks whose uid is absent
    from the CSV are skipped; tasks whose doc is missing on disk are KEPT and
    flagged `missing` (select_task raises a clear error if one is selected)."""
    csv_rows = _load_csv(os.path.join(data_path, "officeqa_full.csv"))
    docs_idx = _resolve_docs_index(docs_root)
    tasks: list[OQATask] = []
    for uid in _split_uids(data_path, split):
        row = csv_rows.get(uid)
        if row is None:
            continue
        # source_files may hold several basenames joined by newlines/CRLF.
        names = tuple(p.strip() for p in re.split(r"[\r\n]+",
                      str(row.get("source_files", "")).strip()) if p.strip())
        doc_paths = tuple(docs_idx[n] for n in names if n in docs_idx)
        tasks.append(OQATask(
            uid=uid,
            question=str(row.get("question", "")).strip(),
            answer=str(row.get("answer", "")).strip(),
            source_files=names,
            doc_paths=doc_paths,
            source_doc=str(row.get("source_docs", "")).strip(),
            difficulty=str(row.get("difficulty", "")).strip(),
            split=split,
            missing=not bool(doc_paths),
        ))
    return tuple(tasks)


def select_task(data_path: str, docs_root: str, seed: int | None,
                instance_id: str | None = None,
                split: str = "test") -> tuple[OQATask, int]:
    """Resolve (task, index) for an episode.

    Priority:
      1. `instance_id` (an OfficeQA uid) -> exact match within `split`.
      2. `seed % len(split)` -> deterministic, alfworld/sbench-style.

    The orchestrator injects options['task_id'] = the project label (a string
    like 'officeqa-corpus'), NOT a uid; that must NOT be used for selection --
    the per-task instance comes from `seed` (= ctx.task_id int). `instance_id`
    is the explicit hook the eval driver uses.
    """
    tasks = load_split(data_path, docs_root, split)
    if not tasks:
        raise RuntimeError(f"No OfficeQA tasks loaded for split={split!r} from {data_path!r}.")
    if instance_id:
        for i, t in enumerate(tasks):
            if t.uid == str(instance_id):
                _check_not_missing(t, i)
                return t, i
        raise ValueError(
            f"instance_id={instance_id!r} not found among {len(tasks)} "
            f"tasks in split={split!r}.")
    idx = (int(seed) if seed is not None else 0) % len(tasks)
    _check_not_missing(tasks[idx], idx)
    return tasks[idx], idx


def _check_not_missing(task: OQATask, idx: int) -> None:
    if task.missing:
        raise RuntimeError(
            f"OfficeQA task uid={task.uid!r} (index {idx}): none of the parsed "
            f"documents {task.source_files!r} were found under the docs root. Set "
            f"reset_options['docs_root'] or ${DOCS_DIR_ENV} to the dir that "
            "holds the treasury_bulletins_parsed txt files.")
