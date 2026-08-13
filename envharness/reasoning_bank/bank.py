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

"""MemoryItem + Bank — append-only store with cosine-similarity retrieval.

Persists as JSONL on disk: one line per item with title/description/content,
embedding vector, and provenance (source_task_id, source_condition, success,
trace_id).
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

from .embed import cosine, embed_texts


@dataclass
class MemoryItem:
    title: str
    description: str
    content: str
    embedding: list[float]
    source: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Text representation used both for retrieval embedding and for
        prepending to the policy prompt."""
        return (f"## {self.title}\n"
                f"_When to use_: {self.description}\n"
                f"{self.content}")

    def to_dict(self) -> dict:
        return asdict(self)


class Bank:
    def __init__(self, items: list[MemoryItem] | None = None):
        self.items: list[MemoryItem] = list(items or [])

    # ----- mutation -----
    def add(self, items: Iterable[MemoryItem]) -> None:
        self.items.extend(items)

    # ----- persistence -----
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for it in self.items:
                f.write(json.dumps(it.to_dict()) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "Bank":
        items: list[MemoryItem] = []
        path = Path(path)
        if not path.exists():
            # Returning an empty bank here would silently turn a typo'd
            # --bank path into a full no-bank eval run. Fail loudly instead.
            raise FileNotFoundError(
                f"Bank file not found: {path} — check the --bank / config "
                f"path (a missing bank would otherwise silently run as a "
                f"no-bank condition)"
            )
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                items.append(MemoryItem(**d))
        bank = cls(items)
        bank._source_path = str(path)
        return bank

    # ----- retrieval -----
    def retrieve(self, query_text: str, k: int = 5,
                 query_embedding: list[float] | None = None,
                 mode: str = "cosine",
                 mmr_lambda: float = 0.5,
                 cosine_threshold: float = 0.0,
                 ) -> list[MemoryItem]:
        """Top-k retrieval.

        mode='cosine' (default): naive cosine top-k.
        mode='mmr': Maximal Marginal Relevance -- forces diversity so the
            top-k isn't dominated by near-duplicate items from the same
            source task. Each pick maximizes
            λ·sim(item, query) − (1−λ)·max_sim(item, already-picked).
            λ=1 == cosine top-k; λ=0 == pure diversity. On full banks
            (multiple items per task) this avoids the redundant
            retrievals that naive cosine top-k is prone to.
        cosine_threshold > 0: drop items below this query-similarity before
            ranking (prevents low-confidence retrievals at large bank size).
        """
        if not self.items:
            return []
        if query_embedding is None:
            query_embedding = embed_texts([query_text])[0]
        scored = [(cosine(it.embedding, query_embedding), it)
                  for it in self.items]
        if cosine_threshold > 0:
            scored = [(s, it) for (s, it) in scored if s >= cosine_threshold]
        if not scored:
            return []
        if mode == "cosine":
            scored.sort(key=lambda t: t[0], reverse=True)
            return [it for _, it in scored[:k]]
        # MMR
        selected: list[tuple[float, MemoryItem]] = []
        remaining = list(scored)
        while len(selected) < k and remaining:
            best_idx, best_score = -1, -float("inf")
            for i, (qsim, it) in enumerate(remaining):
                div_pen = (0.0 if not selected else
                           max(cosine(it.embedding, s_it.embedding)
                               for _, s_it in selected))
                mmr = mmr_lambda * qsim - (1 - mmr_lambda) * div_pen
                if mmr > best_score:
                    best_score, best_idx = mmr, i
            if best_idx == -1:
                break
            selected.append(remaining.pop(best_idx))
        return [it for _, it in selected]

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        return f"Bank(n_items={len(self.items)})"
