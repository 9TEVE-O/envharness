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

"""Convert a legacy MutationLayer corpus into a envharness Rules corpus.

Input records (one JSON object per line) carry the legacy shape produced by
the old envharness orchestrator / corpus-indexer:

    {
      "game_file": "/.../game.tw-pddl",
      "mutation_code": "class _Mutation(MutationLayer): ...",   # may be ""
      "in_env_actions": [{"name": "do", "kwargs": {"text": "..."}}, ...],
      ...                                                       # ignored keys
    }

Output records carry the envharness shape consumed by
`envharness_rl.alfworld.envs.EnvharnessAlfworldWorker`:

    {
      "game_file": "/.../game.tw-pddl",
      "rules_code": "class _Rules(Rules): ...",                 # may be ""
      "in_env_actions": [{"name": "do", "kwargs": {"text": "..."}}, ...]
    }

The only code transform is the class declaration rename
`class _Mutation(MutationLayer)` -> `class _Rules(Rules)`. The A/O/T hook
signatures are identical between the legacy `MutationLayer` and the new
`Rules` harness, so hook bodies port verbatim. Legacy-only methods
(`setup_initial_state`, `execute_in_env_mutation`, `step_reward`) survive as
dead methods on the subclass -- `Rules` never calls them -- and S0 intent is
carried instead by `in_env_actions` (replayed by the worker).

Each converted record is verified to load via `load_rules_subclass` before it
is written; records whose code fails to compile are dropped and reported.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

_LEGACY_DECL = "class _Mutation(MutationLayer)"
_NEW_DECL = "class _Rules(Rules)"


def mutation_code_to_rules_code(mutation_code: str | None) -> str:
    """Rename the legacy class declaration. Empty/None -> "" (pass-through)."""
    code = (mutation_code or "").strip()
    if not code:
        return ""
    return code.replace(_LEGACY_DECL, _NEW_DECL)


def _normalize_actions(raw: object) -> list[dict]:
    actions: list[dict] = []
    for a in raw or []:  # type: ignore[union-attr]
        if isinstance(a, dict) and "name" in a:
            actions.append({"name": str(a["name"]),
                            "kwargs": dict(a.get("kwargs") or {})})
    return actions


def convert_records(records: Iterable[dict], *, validate: bool = True
                    ) -> tuple[list[dict], list[tuple[str, str]]]:
    """Convert legacy records to envharness corpus records.

    Returns (converted, errors). `errors` is a list of (game_file, reason)
    for records dropped because their code failed to compile as a `_Rules`
    subclass (only when `validate=True`).
    """
    load_rules_subclass = None
    if validate:
        from envharness.core.code_loader import load_rules_subclass  # noqa

    converted: list[dict] = []
    errors: list[tuple[str, str]] = []
    for rec in records:
        gf = rec.get("game_file")
        if not gf:
            continue
        rules_code = mutation_code_to_rules_code(rec.get("mutation_code"))
        if validate and rules_code:
            try:
                load_rules_subclass(rules_code)  # type: ignore[misc]
            except Exception as e:  # noqa: BLE001
                errors.append((gf, f"{type(e).__name__}: {e}"))
                continue
        converted.append({
            "game_file": gf,
            "rules_code": rules_code,
            "in_env_actions": _normalize_actions(rec.get("in_env_actions")),
        })
    return converted, errors


def convert_file(src: str | Path, dst: str | Path, *, limit: int | None = None,
                 require_mutation: bool = False, validate: bool = True
                 ) -> dict:
    """Read legacy JSONL `src`, write envharness JSONL `dst`.

    `require_mutation`: keep only records that carry non-empty code AND/OR
    in_env_actions (skip pure pass-through records). Useful when building a
    small example corpus where every line should actually exercise a harness.
    `limit`: stop after writing this many records.
    """
    src, dst = Path(src), Path(dst)
    raw = [json.loads(l) for l in src.open() if l.strip()]
    converted, errors = convert_records(raw, validate=validate)

    if require_mutation:
        converted = [r for r in converted
                     if r["rules_code"] or r["in_env_actions"]]
    if limit is not None:
        converted = converted[:limit]

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        for r in converted:
            f.write(json.dumps(r) + "\n")
    return {"written": len(converted), "errors": errors,
            "src": str(src), "dst": str(dst)}
