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

"""EndpointPool -- thread-safe work-stealing dispatcher for a fixed set of
inference endpoints (e.g. N vLLM instances behind ports).

The driver leases an endpoint via a context manager; the lease blocks if
all endpoints are in use, releases automatically on context exit (success
or exception). This is the cleanest implementation of the
"100 tasks / 8 GPUs / dispatch the 9th when any GPU frees" pattern.

Designed to be the single source of truth for endpoint dispatch. Callers
(Orchestrator K-rollout, sweep eval driver, future batch utilities) lease
an endpoint and inject it into spec.policy.client_kwargs["api_base"]
before handing the spec to the runner. CompletionAPIClient then consumes
that api_base unchanged -- no in-client load-balancing logic.

Typical usage with a ThreadPoolExecutor (sweep driver):

    pool = EndpointPool(["http://localhost:890%d/v1" % i for i in range(1, 9)])

    def run_one(spec):
        with pool.lease() as endpoint:
            new_spec = pin_endpoint(spec, endpoint)
            return runner.run(new_spec)

    with ThreadPoolExecutor(max_workers=len(pool)) as ex:
        results = list(ex.map(run_one, specs))

For mutation K-rollouts, the orchestrator does the same dance per
candidate (K=5 leases against a pool of >=K endpoints; when K < pool size
the unused endpoints sit idle for that outer iter -- the deployment fix
is multiple orchestrators in parallel via N shards or outer-loop
parallelism).
"""
from __future__ import annotations

import contextlib
import queue
from dataclasses import replace
from typing import Any


class EndpointPool:
    """Bounded pool of endpoints with lease/release context-manager semantics.

    Thread-safe (backed by queue.Queue). Subprocess-safe iff each subprocess
    has its own pool instance -- this is a driver-side abstraction; do not
    pickle and share across processes.
    """

    def __init__(self, endpoints: list[str]):
        if not endpoints:
            raise ValueError("EndpointPool requires at least one endpoint")
        self._endpoints: list[str] = list(endpoints)
        self._q: queue.Queue[str] = queue.Queue()
        for ep in self._endpoints:
            self._q.put(ep)

    def __len__(self) -> int:
        return len(self._endpoints)

    @property
    def endpoints(self) -> list[str]:
        return list(self._endpoints)

    @contextlib.contextmanager
    def lease(self, timeout: float | None = None):
        """Block until an endpoint is free, yield it, return it on exit.

        Releases via finally even if the caller raises.
        """
        ep = self._q.get(timeout=timeout) if timeout is not None else self._q.get()
        try:
            yield ep
        finally:
            self._q.put(ep)


# ---------------------------------------------------------------------------
# Helper: build an EndpointPool from a PolicySpec / dict client_kwargs.
# Tolerates either api_base_pool (list) or api_base (str -> singleton pool)
# or both missing (returns None -> caller skips endpoint pinning).
# ---------------------------------------------------------------------------

def pool_from_client_kwargs(client_kwargs: dict[str, Any]) -> "EndpointPool | None":
    """Return an EndpointPool for a config's policy.client_kwargs, or None.

    Resolution order:
        1. `api_base_pool` (list[str]) at top level
        2. `inner_kwargs.api_base_pool` -- when the client is wrapped by
           LoggingLLMClient (run_harness.py does this when log_policy_calls
           is enabled), the real client kwargs are nested under inner_kwargs
        3. `api_base` (str) at top level -- single-endpoint fallback
        4. `inner_kwargs.api_base` (str) -- single-endpoint via wrapper
        5. neither -- returns None; caller forwards spec unchanged
    """
    inner = client_kwargs.get("inner_kwargs") or {}
    eps = client_kwargs.get("api_base_pool") or inner.get("api_base_pool")
    if eps:
        return EndpointPool(list(eps))
    single = client_kwargs.get("api_base") or inner.get("api_base")
    if single:
        return EndpointPool([single])
    return None


# ---------------------------------------------------------------------------
# Helper: build a fresh EpisodeSpec / PolicySpec with api_base pinned to a
# specific endpoint. Keeps the original spec immutable so concurrent workers
# don't race on a shared dict.
# ---------------------------------------------------------------------------

def pin_endpoint(spec, endpoint: str):
    """Return a copy of `spec` with policy.client_kwargs["api_base"] set to
    `endpoint` and `api_base_pool` dropped from the clone.

    Handles the LoggingLLMClient wrap: when the policy spec's client_kwargs
    has `inner_kwargs` (set by run_harness.py when log_policy_calls=True),
    the api_base lives BELOW that wrap and must be pinned in the nested
    dict, not the top-level wrapper kwargs. Without this, the pool fields
    stayed buried inside inner_kwargs untouched and every subprocess
    fell back to inner_kwargs.api_base_pool[0] -- pinning all traffic
    on the first vLLM endpoint.

    Works on any EpisodeSpec whose `policy` is a dataclass with a
    `client_kwargs` dict. Original spec is unchanged.
    """
    new_kwargs = dict(spec.policy.client_kwargs)
    if "inner_kwargs" in new_kwargs:
        # Wrapped client (LoggingLLMClient). Pin inside the nested dict.
        new_inner = dict(new_kwargs["inner_kwargs"])
        new_inner["api_base"] = endpoint
        new_inner.pop("api_base_pool", None)
        new_kwargs["inner_kwargs"] = new_inner
    else:
        new_kwargs["api_base"] = endpoint
        new_kwargs.pop("api_base_pool", None)
    new_policy = replace(spec.policy, client_kwargs=new_kwargs)
    return replace(spec, policy=new_policy)
