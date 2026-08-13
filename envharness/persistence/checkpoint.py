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

"""Checkpoint: portable save/load of an (env + harness stack).

Design goals:

1. **Round-trip must not silently corrupt state.** Every step of save+load
   passes through JSON-string serialization, surfacing any non-JSON-safe
   field at write time rather than silently dropping it on read.

2. **Self-describing.** No import paths in the file. Env and harnesses
   are identified by string tags via `core.registry`; adding a new
   ActionableEnv or EnvHarness only requires `@register_env(tag)` /
   `@register_harness(tag)`.

3. **Schema versioned.** `schema_version=1`. Future format changes raise
   a clear error rather than silently misinterpreting.

4. **Composable.** Save = `env.save_state()` + `[h.save_state() for h in
   stack]`. Load walks `harnesses` in order, building from inner to outer
   so the agent ends up with the outermost ActionableEnv reference.

Save/load is a correctness-critical surface. Every error path here aims
to be loud and informative rather than silently dropping fields.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from envharness.core.actionable_env import ActionableEnv
from envharness.core.envharness import EnvHarness
from envharness.core.registry import get_env_class, get_harness_class


_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class Checkpoint:
    """In-memory representation of a saved (env + harness stack).

    Layout fields:
        env_type, env_state  -- the base ActionableEnv
        harnesses            -- list of {"type": <tag>, "state": <dict>}; index 0
                                 is the innermost wrap, index -1 is outermost
                                 (what the agent sees)
        metadata             -- free-form annotation; ignored by load
    """

    env_type: str
    env_state: dict[str, Any]
    harnesses: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ----- serialization ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "env": {
                "type": self.env_type,
                "state": dict(self.env_state),
            },
            "harnesses": [
                {"type": str(h["type"]), "state": dict(h.get("state") or {})}
                for h in self.harnesses
            ],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "Checkpoint":
        if not isinstance(obj, dict):
            raise ValueError(
                f"Checkpoint.from_dict: expected dict at top level, got "
                f"{type(obj).__name__}"
            )
        ver = obj.get("schema_version")
        if ver != _SCHEMA_VERSION:
            raise ValueError(
                f"Checkpoint schema_version={ver!r} not supported; "
                f"this build expects {_SCHEMA_VERSION}."
            )
        env_block = obj.get("env")
        if not isinstance(env_block, dict):
            raise ValueError(
                "Checkpoint: 'env' must be a dict with 'type' and 'state'."
            )
        env_type = env_block.get("type")
        env_state = env_block.get("state")
        if not isinstance(env_type, str) or not env_type:
            raise ValueError("Checkpoint.env.type must be a non-empty string.")
        if not isinstance(env_state, dict):
            raise ValueError("Checkpoint.env.state must be a dict.")
        harnesses_raw = obj.get("harnesses") or []
        if not isinstance(harnesses_raw, list):
            raise ValueError("Checkpoint.harnesses must be a list.")
        harnesses: list[dict[str, Any]] = []
        for i, h in enumerate(harnesses_raw):
            if not isinstance(h, dict):
                raise ValueError(
                    f"Checkpoint.harnesses[{i}] must be a dict; got "
                    f"{type(h).__name__}"
                )
            htype = h.get("type")
            hstate = h.get("state")
            if not isinstance(htype, str) or not htype:
                raise ValueError(
                    f"Checkpoint.harnesses[{i}].type must be a non-empty string."
                )
            if hstate is None:
                hstate = {}
            if not isinstance(hstate, dict):
                raise ValueError(
                    f"Checkpoint.harnesses[{i}].state must be a dict (or omitted)."
                )
            harnesses.append({"type": htype, "state": dict(hstate)})
        metadata = obj.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("Checkpoint.metadata must be a dict.")
        return cls(
            env_type=env_type,
            env_state=dict(env_state),
            harnesses=harnesses,
            metadata=dict(metadata),
        )

    # ----- disk I/O --------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write to disk as JSON. Goes through json.dumps so any non-JSON
        value in the state dict raises HERE rather than silently
        corrupting the file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        # Strict round-trip: loading what we just produced must succeed,
        # otherwise our save dict has shape issues and we should refuse
        # to write it.
        try:
            self.from_dict(json.loads(text))
        except Exception as e:
            raise ValueError(
                f"Checkpoint.save: refusing to write a file that wouldn't "
                f"load back. Reason: {e}"
            ) from e
        p.write_text(text, encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Checkpoint":
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text))


# ---------------------------------------------------------------------------
# Stack <-> Checkpoint conversions
# ---------------------------------------------------------------------------

def dump_stack(env: ActionableEnv,
                metadata: dict[str, Any] | None = None) -> Checkpoint:
    """Walk an env / harness stack and capture each layer's save_state.

    Given `Rules(Setup(Toy24Env()))`:

      - we walk from outermost (Rules) inward;
      - the innermost non-harness object is the env;
      - the returned Checkpoint has `harnesses=[Setup.state, Rules.state]`
        (innermost first, outermost last).
    """
    layers: list[EnvHarness] = []
    current: ActionableEnv = env
    while isinstance(current, EnvHarness):
        layers.append(current)
        current = current.inner
    # `current` is now the base ActionableEnv.
    if isinstance(current, EnvHarness):  # pragma: no cover - defensive
        raise RuntimeError("Stack walk failed to reach a non-EnvHarness base.")
    env_obj: ActionableEnv = current

    # Reverse so harnesses[0] is innermost, harnesses[-1] is outermost.
    layers_inner_to_outer = list(reversed(layers))

    # Sanity: every layer must be in the registry. This catches the case
    # of a user writing an EnvHarness subclass without @register_harness
    # and trying to checkpoint it -- we'd otherwise produce a save file
    # that nothing can read.
    harness_entries: list[dict[str, Any]] = []
    for h in layers_inner_to_outer:
        tag = h.harness_type()
        # Round-trip through get_harness_class to confirm the tag exists.
        get_harness_class(tag)
        state = h.save_state()
        if not isinstance(state, dict):
            raise TypeError(
                f"{type(h).__name__}.save_state() must return a dict; "
                f"got {type(state).__name__}"
            )
        harness_entries.append({"type": tag, "state": state})

    env_tag = env_obj.env_type()
    get_env_class(env_tag)
    env_state = env_obj.save_state()
    if not isinstance(env_state, dict):
        raise TypeError(
            f"{type(env_obj).__name__}.save_state() must return a dict; "
            f"got {type(env_state).__name__}"
        )

    return Checkpoint(
        env_type=env_tag,
        env_state=env_state,
        harnesses=harness_entries,
        metadata=dict(metadata or {}),
    )


def build_stack(cp: Checkpoint) -> ActionableEnv:
    """Reconstruct an (env + harness stack) from a Checkpoint.

    Builds from inner to outer. Returns the outermost ActionableEnv
    (which is what the agent should program against).
    """
    env_cls = get_env_class(cp.env_type)
    env = env_cls.from_state(cp.env_state)
    if not isinstance(env, ActionableEnv):
        raise TypeError(
            f"{env_cls.__name__}.from_state returned a "
            f"{type(env).__name__}, not an ActionableEnv."
        )
    current: ActionableEnv = env
    for h_spec in cp.harnesses:
        h_cls = get_harness_class(h_spec["type"])
        # from_state must accept `inner=` per the EnvHarness contract.
        wrapped = h_cls.from_state(h_spec.get("state") or {}, inner=current)
        if not isinstance(wrapped, EnvHarness):
            raise TypeError(
                f"{h_cls.__name__}.from_state returned a "
                f"{type(wrapped).__name__}, not an EnvHarness."
            )
        current = wrapped
    return current


# ---------------------------------------------------------------------------
# Convenience top-level helpers
# ---------------------------------------------------------------------------

def save_checkpoint(env: ActionableEnv, path: str | Path,
                     metadata: dict[str, Any] | None = None) -> Path:
    """`dump_stack` + `Checkpoint.save` in one call."""
    return dump_stack(env, metadata=metadata).save(path)


def load_checkpoint(path: str | Path,
                     auto_reset: bool = True) -> ActionableEnv:
    """`Checkpoint.load` + `build_stack` in one call.

    `auto_reset=True` (default) calls `env.reset(seed, options)` using the
    base env's `default_reset_args()`, so the returned env is ready for
    the agent's first step. Set `auto_reset=False` to inspect the env
    before booting any heavyweight runtime (Docker container, browser).
    """
    cp = Checkpoint.load(path)
    env = build_stack(cp)
    if auto_reset and env.reset_after_load():
        seed, options = env.default_reset_args()
        env.reset(seed=seed, options=options)
    return env
