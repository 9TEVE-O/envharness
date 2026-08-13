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

"""OfficeQAEnv -- wrap OfficeQA (databricks/officeqa) as an ActionableEnv.

Contract follows the EnvHarness ActionableEnv ABC: zero mutation logic here.
The only runtime this Bridge owns is READ-ONLY access to a corpus of parsed
treasury-bulletin text documents (the docs root). Every agent action inspects
those documents -- the file-based analogue of the sbench working dir, but
read-only (the agent produces an ANSWER, not an edited file). Rules-written
code never sees the docs root, only the `OfficeQAEnvState` dataclass (pure data).

Task: a factual question whose answer lives in ONE parsed document
(`source_file`). The agent locates + reads the evidence, then submits a short
answer. Grading is normalized exact-match (evaluator.py) -- deterministic and
teacher-free (no LibreOffice, no LLM judge in the loop).

Action contract (multi-turn, four tools):
    glob(pattern)              -- list candidate document paths under docs root
    read(path, start, limit)   -- read a line window of a document
    grep(pattern, path)        -- case-insensitive substring search in a document
    answer(text)               -- submit the final answer; ends the episode
    If the agent never answers, the episode runs to max_steps and we grade the
    empty answer (a fail).

Reset options (passed through `options` in reset()):
    data_path: str   -- dir with officeqa_full.csv + officeqa_id_split/. Falls
                        back to $OFFICEQA_DATA.
    docs_root: str   -- dir holding the parsed treasury-bulletin *.txt. Falls
                        back to $OFFICEQA_DOCS_DIR.
    split: str       -- 'train' | 'val' | 'test' (default 'test').
    instance_id: str -- explicit OfficeQA uid; wins over seed-based selection.
    obs_truncate: int      -- max chars per tool output (default 6000).
    read_limit_cap: int    -- max lines a single read may return (default 400).
    grep_max_matches: int  -- max grep hits returned (default 60).
    show_doc_path: bool    -- include the source document path in the initial
                              observation (default True). The raw env exposes it;
                              a Rules O-hook is free to withhold it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, ClassVar

from envharness.core.actionable_env import ActionableEnv
from envharness.core.registry import register_env
from envharness.core.tool import Tool
from envharness.core.types import (
    Action, EnvResetResponse, EnvResponse, EvaluationResult, Observation,
    TaskSummary,
)
# The grader is SkillOpt's, vendored under envharness/third_party/.
from envharness.third_party.skillopt import officeqa_evaluator as evaluator
from .dataset import (
    DATA_PATH_ENV, DOCS_DIR_ENV, OQATask, load_split, select_task,
)
from .tools import Answer, Glob, Grep, Read

DEFAULT_OBS_TRUNCATE = 6000
DEFAULT_READ_LIMIT = 200
DEFAULT_READ_LIMIT_CAP = 400
DEFAULT_GREP_MAX_MATCHES = 60


@dataclass
class OfficeQAEnvState:
    """Bridge-exposed view for Rules hooks. Pure data -- no file handles. Rules
    may read every field; the only writable surface is `extras`. NOTE: the gold
    answer is deliberately NOT here (leakage guard) -- it lives on the Bridge."""
    task_id: str = ""                    # OfficeQA uid
    question: str = ""                   # the factual question
    difficulty: str = ""                 # "easy" | "hard"
    source_files: str = ""               # comma-joined basenames of the source document(s)
    doc_paths: list = field(default_factory=list)   # abs paths of the parsed txt document(s)
    docs_root: str = ""                  # root the tools may access
    doc_line_count: int = 0              # total #lines across the source document(s)
    last_tool: str = ""                  # most recent tool name
    last_args: dict = field(default_factory=dict)   # most recent tool kwargs
    last_output: str = ""                # most recent tool output (truncated)
    answered: bool = False               # True once the agent called answer()
    final_answer: str = ""               # the submitted answer text
    step_count: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


@register_env("officeqa")
class OfficeQAEnv(ActionableEnv):
    """OfficeQA ActionableEnv. Read-only parsed-document corpus; multi-turn
    glob/read/grep navigation; normalized exact-match grading."""

    tool_registry: ClassVar[list[type[Tool]]] = [Glob, Read, Grep, Answer]

    def __init__(self) -> None:
        super().__init__()
        self._data_path: str = ""
        self._docs_root: str = ""
        self._split: str = "test"
        self._task: OQATask | None = None
        self._obs_truncate: int = DEFAULT_OBS_TRUNCATE
        self._read_limit_cap: int = DEFAULT_READ_LIMIT_CAP
        self._grep_max_matches: int = DEFAULT_GREP_MAX_MATCHES
        self._show_doc_path: bool = True
        self._terminated: bool = False
        self._eval_cache: EvaluationResult | None = None
        self.state: OfficeQAEnvState = OfficeQAEnvState()
        self._last_reset_seed: int | None = None
        self._last_reset_options: dict[str, Any] = {}

    # -- core env interface -------------------------------------------------

    def reset(self, seed: int | None = None,
              options: dict[str, Any] | None = None) -> EnvResetResponse:
        opts = options or {}
        self._last_reset_seed = seed
        self._last_reset_options = dict(opts)

        self._data_path = (opts.get("data_path")
                           or os.environ.get(DATA_PATH_ENV) or "")
        self._docs_root = (opts.get("docs_root")
                           or os.environ.get(DOCS_DIR_ENV) or "")
        if not self._data_path:
            raise RuntimeError("OfficeQAEnv.reset: no data_path. Set "
                               f"reset_options['data_path'] or ${DATA_PATH_ENV}.")
        if not self._docs_root or not os.path.isdir(self._docs_root):
            raise RuntimeError("OfficeQAEnv.reset: no valid docs_root. Set "
                               f"reset_options['docs_root'] or ${DOCS_DIR_ENV}.")
        self._docs_root = os.path.realpath(self._docs_root)
        self._split = str(opts.get("split") or "test")
        self._obs_truncate = int(opts.get("obs_truncate") or DEFAULT_OBS_TRUNCATE)
        self._read_limit_cap = int(opts.get("read_limit_cap") or DEFAULT_READ_LIMIT_CAP)
        self._grep_max_matches = int(opts.get("grep_max_matches") or DEFAULT_GREP_MAX_MATCHES)
        self._show_doc_path = bool(opts.get("show_doc_path", True))

        instance_id = opts.get("instance_id") or None
        task, _idx = select_task(self._data_path, self._docs_root, seed,
                                 instance_id, split=self._split)
        self._task = task

        line_count = 0
        for dp in task.doc_paths:
            try:
                with open(dp, encoding="utf-8", errors="replace") as f:
                    line_count += sum(1 for _ in f)
            except OSError:
                pass

        self._terminated = False
        self._eval_cache = None
        self.state = OfficeQAEnvState(
            task_id=task.uid,
            question=task.question,
            difficulty=task.difficulty,
            source_files=", ".join(task.source_files),
            doc_paths=list(task.doc_paths),
            docs_root=self._docs_root,
            doc_line_count=line_count,
            step_count=0,
        )
        return EnvResetResponse(
            observation=self._observe(),
            info={"task_id": task.uid, "difficulty": task.difficulty, "won": False},
        )

    def step(self, action: Action) -> EnvResponse:
        if self._terminated:
            return EnvResponse(observation=self._observe(), reward=0.0,
                               terminated=True, truncated=False, info={"won": False})

        name = action.name
        kw = action.kwargs or {}

        if name == Answer.name:
            text = str(kw.get("text") if kw.get("text") is not None else "")
            self.state.answered = True
            self.state.final_answer = text
            self.state.last_tool = name
            self.state.last_args = {"text": text[:200]}
            self._terminated = True
            return EnvResponse(
                observation=Observation(
                    text=f"[answer submitted] {text[:200]}",
                    data={"answered": True, "final_answer": text},
                ),
                reward=0.0, terminated=True, truncated=False,
                info={"answered": True, "won": None},
            )

        if name not in (Glob.name, Read.name, Grep.name):
            return EnvResponse(
                observation=Observation(
                    text=f"[unknown tool: {name}] valid tools: glob(pattern), "
                         "read(path, start, limit), grep(pattern, path), answer(text)",
                    data={"error": "unknown_tool"}),
                reward=0.0, terminated=False, truncated=False,
                info={"error": "unknown_tool"})

        self.state.step_count += 1
        if name == Glob.name:
            out = self._do_glob(str(kw.get("pattern", "")))
        elif name == Grep.name:
            out = self._do_grep(str(kw.get("pattern", "")), str(kw.get("path", "")))
        else:  # read
            out = self._do_read(str(kw.get("path", "")),
                                kw.get("start", 1), kw.get("limit", DEFAULT_READ_LIMIT))
        out = self._truncate(out)
        self.state.last_tool = name
        self.state.last_args = {k: str(v)[:120] for k, v in kw.items()}
        self.state.last_output = out
        return EnvResponse(
            observation=self._observe(tool_output=out),
            reward=0.0, terminated=False, truncated=False,
            info={"result": {"tool": name, "output": out[:1000]}, "won": None},
        )

    def evaluate(self) -> EvaluationResult:
        if self._eval_cache is not None:
            return self._eval_cache
        task = self._task
        if task is None:
            return EvaluationResult(success=False, score=0.0,
                                    metrics={"error": "no active episode"})
        res = evaluator.evaluate(self.state.final_answer, task.answer)
        result = EvaluationResult(
            success=bool(res["em"] >= 1.0),
            score=float(res["em"]),
            metrics={"won": bool(res["em"] >= 1.0), "em": res["em"], "f1": res["f1"],
                     "predicted": res["predicted_answer"][:200], "gold": task.answer,
                     "answered": self.state.answered, "difficulty": task.difficulty,
                     "steps": self.state.step_count},
        )
        self._eval_cache = result
        return result

    def observe(self) -> Observation:
        return self._observe()

    def get_env_state(self) -> OfficeQAEnvState:
        return self.state

    @classmethod
    def env_state_schema(cls) -> str:
        return (
            "env_state is an OfficeQAEnvState dataclass with these fields:\n"
            "  task_id: str          -- OfficeQA uid\n"
            "  question: str         -- the factual question to answer\n"
            "  difficulty: str       -- 'easy' | 'hard'\n"
            "  source_files: str     -- comma-joined basenames of the source document(s)\n"
            "  doc_paths: list[str]  -- abs paths of the parsed txt document(s)\n"
            "  docs_root: str        -- root dir the tools (glob/read/grep) may access\n"
            "  doc_line_count: int   -- total #lines across the source document(s)\n"
            "  last_tool: str        -- most recent tool name (glob|read|grep|answer)\n"
            "  last_args: dict       -- most recent tool kwargs\n"
            "  last_output: str      -- most recent tool output (truncated)\n"
            "  answered: bool        -- True once the agent called answer()\n"
            "  final_answer: str     -- the submitted answer text\n"
            "  step_count: int       -- tool actions taken so far\n"
            "  extras: dict          -- free scratch dict for Rules hook state\n"
            "The gold answer is NOT in env_state (leakage guard). S0 changes go through\n"
            "in_env_actions (Setup), not Rules. Rules hooks may read all fields and write\n"
            "to `extras`. To change what the agent sees, return a new Observation from\n"
            "filter_observation -- do NOT mutate env_state fields."
        )

    # -- save / load --------------------------------------------------------

    def save_state(self) -> dict:
        return {"reset_seed": self._last_reset_seed,
                "reset_options": dict(self._last_reset_options)}

    @classmethod
    def from_state(cls, state: dict) -> "OfficeQAEnv":
        env = cls()
        env._last_reset_seed = state.get("reset_seed")
        env._last_reset_options = dict(state.get("reset_options") or {})
        return env

    def default_reset_args(self) -> tuple[int | None, dict]:
        return self._last_reset_seed, dict(self._last_reset_options)

    # -- agent-driven task selection ---------------------------------------

    @classmethod
    def list_tasks(cls, reset_options: dict | None = None,
                   limit: int | None = None) -> list[TaskSummary]:
        opts = dict(reset_options or {})
        data_path = opts.get("data_path") or os.environ.get(DATA_PATH_ENV) or ""
        docs_root = opts.get("docs_root") or os.environ.get(DOCS_DIR_ENV) or ""
        split = str(opts.get("split") or "test")
        if not data_path or not docs_root:
            raise RuntimeError("OfficeQAEnv.list_tasks: need data_path + docs_root "
                               "(reset_options or $OFFICEQA_DATA/$OFFICEQA_DOCS_DIR).")
        tasks = load_split(data_path, os.path.realpath(docs_root), split)
        n = limit if limit is not None else len(tasks)
        out: list[TaskSummary] = []
        for i, t in enumerate(tasks[:n]):
            out.append(TaskSummary(
                task_idx=i, instance_id=t.uid,
                brief=f"[{t.difficulty}] {t.question[:180]}",
                metadata={"difficulty": t.difficulty,
                          "source_files": list(t.source_files), "missing": t.missing}))
        return out

    def close(self) -> None:
        super().close()

    # -- helpers ------------------------------------------------------------

    def _truncate(self, text: str) -> str:
        if len(text) > self._obs_truncate:
            return text[: self._obs_truncate] + "\n...[output truncated]"
        return text

    def _safe_path(self, path: str) -> str | None:
        """Resolve `path` to an abs path guaranteed under docs_root, or None.
        Accepts an abs path, a docs-root-relative path, or a bare basename."""
        path = (path or "").strip()
        if not path:
            return None
        cands = [path, os.path.join(self._docs_root, path)]
        # bare basename -> match against this task's source doc(s)
        if os.sep not in path and self._task:
            for dp in self._task.doc_paths:
                if os.path.basename(dp) == path:
                    cands.append(dp)
        for c in cands:
            rp = os.path.realpath(c)
            if (rp == self._docs_root or rp.startswith(self._docs_root + os.sep)) \
                    and os.path.isfile(rp):
                return rp
        return None

    def _do_glob(self, pattern: str) -> str:
        import glob as _glob
        pattern = (pattern or "").strip()
        if not pattern:
            return "[glob] empty pattern."
        hits = _glob.glob(os.path.join(self._docs_root, "**", pattern), recursive=True)
        if not hits:
            hits = _glob.glob(os.path.join(self._docs_root, "**", f"*{pattern}*"),
                              recursive=True)
        hits = sorted(h for h in hits if os.path.isfile(h))[:50]
        if not hits:
            return f"[glob] no documents match {pattern!r}."
        return "\n".join(os.path.relpath(h, self._docs_root) for h in hits)

    def _do_read(self, path: str, start: Any, limit: Any) -> str:
        rp = self._safe_path(path)
        if rp is None:
            return f"[read] path not found under docs root: {path!r}"
        try:
            start = max(1, int(start))
        except (TypeError, ValueError):
            start = 1
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = DEFAULT_READ_LIMIT
        limit = max(1, min(limit, self._read_limit_cap))
        lines: list[str] = []
        with open(rp, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, start=1):
                if i < start:
                    continue
                if i >= start + limit:
                    break
                lines.append(f"{i}\t{line.rstrip()}")
        if not lines:
            return f"[read] {os.path.relpath(rp, self._docs_root)}: no lines in range " \
                   f"(start={start})."
        header = f"[read {os.path.relpath(rp, self._docs_root)} lines {start}-{start+len(lines)-1}]"
        return header + "\n" + "\n".join(lines)

    def _do_grep(self, pattern: str, path: str) -> str:
        pattern = (pattern or "").strip()
        if not pattern:
            return "[grep] empty pattern."
        rp = self._safe_path(path)
        if rp is None:
            return f"[grep] path not found under docs root: {path!r}"
        needle = pattern.lower()
        matches: list[str] = []
        with open(rp, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, start=1):
                if needle in line.lower():
                    matches.append(f"{i}\t{line.rstrip()}")
                    if len(matches) >= self._grep_max_matches:
                        matches.append(f"...[more than {self._grep_max_matches} matches; refine pattern]")
                        break
        if not matches:
            return f"[grep] no lines match {pattern!r} in " \
                   f"{os.path.relpath(rp, self._docs_root)}."
        return f"[grep {pattern!r} in {os.path.relpath(rp, self._docs_root)}]\n" + \
               "\n".join(matches)

    def _observe(self, tool_output: str | None = None) -> Observation:
        s = self.state
        parts = [
            f"question: {s.question}",
            f"docs_root: {s.docs_root}",
        ]
        rel_docs = [os.path.relpath(dp, self._docs_root) for dp in s.doc_paths]
        if self._show_doc_path:
            if len(rel_docs) == 1:
                parts.append(f"source_document: {rel_docs[0]} ({s.doc_line_count} lines total)")
            else:
                parts.append("source_documents (the answer may require combining them):\n"
                             + "\n".join(f"  - {d}" for d in rel_docs))
        parts.append(
            "Use grep(pattern, path) to locate the relevant figures/rows, "
            "read(path, start, limit) to read around them, glob(pattern) to find "
            "documents, then answer(text) with the concise final answer.")
        if tool_output is not None:
            parts.append(tool_output)
        return Observation(
            text="\n\n".join(parts),
            data={
                "task_id": s.task_id, "question": s.question,
                "difficulty": s.difficulty, "source_files": s.source_files,
                "doc_paths": list(s.doc_paths) if self._show_doc_path else [],
                "docs_root": s.docs_root, "step_count": s.step_count,
                "answered": s.answered,
            },
        )
