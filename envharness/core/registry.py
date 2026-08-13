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

"""Type-tag registries for save/load routing.

Save formats name an ActionableEnv via its `env_type` and an EnvHarness via
its `harness_type`. The registry lets `persistence.checkpoint.load(...)`
look up the right class without import paths in the save file (import paths
break on rename / repackage).

Adding a new ActionableEnv or EnvHarness in user code looks like::

    @register_env("my_bench")
    class MyEnv(ActionableEnv):
        ...

    @register_harness("my_harness")
    class MyHarness(EnvHarness):
        ...

The string tag is what appears in save dicts. Once written it is part of
your on-disk format -- treat it as a stable identifier.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, TypeVar

if TYPE_CHECKING:
    from envharness.core.actionable_env import ActionableEnv
    from envharness.core.envharness import EnvHarness


_ENV_REGISTRY: dict[str, type] = {}
_HARNESS_REGISTRY: dict[str, type] = {}


T = TypeVar("T")


def register_env(tag: str) -> Callable[[type[T]], type[T]]:
    """Class decorator: register an ActionableEnv subclass under `tag`."""
    def _wrap(cls: type[T]) -> type[T]:
        if tag in _ENV_REGISTRY and _ENV_REGISTRY[tag] is not cls:
            raise ValueError(
                f"env_type {tag!r} already registered to "
                f"{_ENV_REGISTRY[tag].__name__}; cannot reassign to {cls.__name__}"
            )
        _ENV_REGISTRY[tag] = cls
        # Bind the class-level identifier so cls.env_type() returns `tag`.
        cls.env_type = classmethod(lambda c, _t=tag: _t)  # type: ignore[attr-defined]
        return cls
    return _wrap


def register_harness(tag: str) -> Callable[[type[T]], type[T]]:
    """Class decorator: register an EnvHarness subclass under `tag`."""
    def _wrap(cls: type[T]) -> type[T]:
        if tag in _HARNESS_REGISTRY and _HARNESS_REGISTRY[tag] is not cls:
            raise ValueError(
                f"harness_type {tag!r} already registered to "
                f"{_HARNESS_REGISTRY[tag].__name__}; cannot reassign to {cls.__name__}"
            )
        _HARNESS_REGISTRY[tag] = cls
        cls.harness_type = classmethod(lambda c, _t=tag: _t)  # type: ignore[attr-defined]
        return cls
    return _wrap


def get_env_class(tag: str) -> type:
    if tag not in _ENV_REGISTRY:
        raise KeyError(
            f"Unknown env_type {tag!r}. Registered envs: "
            f"{sorted(_ENV_REGISTRY)}"
        )
    return _ENV_REGISTRY[tag]


def get_harness_class(tag: str) -> type:
    if tag not in _HARNESS_REGISTRY:
        raise KeyError(
            f"Unknown harness_type {tag!r}. Registered harnesses: "
            f"{sorted(_HARNESS_REGISTRY)}"
        )
    return _HARNESS_REGISTRY[tag]


def registered_envs() -> dict[str, type]:
    return dict(_ENV_REGISTRY)


def registered_harnesses() -> dict[str, type]:
    return dict(_HARNESS_REGISTRY)
