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

"""Single place that turns a model id into a ready-to-use client spec.

Every stage -- corpus, induction, evaluation -- asks this module for its
client instead of naming a client class and hand-assembling provider kwargs.
Switching provider is then one string in a config:

    openai/gpt-4.1-mini
    gemini/gemini-3.5-flash
    vertex_ai/claude-sonnet-4-6

A bare name is accepted too (`gpt-4.1`, `claude-sonnet-4-6`, `gemini-3.5-flash`);
the provider is inferred from the family prefix.

Why a resolver at all, when litellm already dispatches on the prefix:

  * Auth differs per provider. OpenAI and Gemini read an API key from the
    environment; Vertex uses ADC plus a project/region pair whose env-var
    names are litellm's (VERTEXAI_*), not the ones the Anthropic SDK uses.
  * Parameters are not portable. `reasoning_effort` is native on Gemini,
    unsupported on `gpt-4.1` (litellm raises UnsupportedParamsError before
    the request leaves the process), and on Claude it is translated into a
    thinking budget that the API rejects unless `max_tokens` exceeds it.
    A config written for one provider must not explode on another.

Both are handled here so callers can stay provider-agnostic.
"""
from __future__ import annotations

import os

CLIENT_FACTORY = "envharness.infra.llm:LiteLLMClient"

# Recognised litellm provider prefixes, in the form callers should use.
PROVIDERS = ("gemini", "openai", "vertex_ai", "anthropic", "azure", "ollama")

# Bare-name -> provider, longest prefix first so "gpt-" cannot shadow a more
# specific family added later.
_FAMILY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("gemini", "gemini"),
    ("gpt-", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("claude", "vertex_ai"),      # Claude is reached through Vertex here
)

# Reasoning knobs a config may carry. On a provider/model that does not take
# them litellm drops them (see `drop_params` below) rather than raising.
_REASONING_PARAMS = ("reasoning_effort", "thinking", "thinking_budget")

# Claude turns `reasoning_effort` into a thinking budget and rejects the
# request unless max_tokens leaves room for it. Requests that ask for
# reasoning without saying how long the answer may be get this floor.
_REASONING_MAX_TOKENS_FLOOR = 4096

# Embedding model per provider, used when a caller does not name one. The
# vector dimension is part of a bank's on-disk format: a bank built with one
# embedding model cannot be queried with another (Bank.retrieve raises on the
# dimension mismatch), so switching this invalidates existing banks.
_DEFAULT_EMBED = {
    "gemini": "gemini/gemini-embedding-001",
    "openai": "openai/text-embedding-3-small",
    # Claude has no embedding model, but Vertex serves its own -- and it
    # authenticates with the same ADC, so a Vertex-only user needs no key.
    "vertex_ai": "vertex_ai/text-embedding-004",
    "anthropic": "openai/text-embedding-3-small",
}

# Where each provider's key lives. Vertex is absent on purpose: it
# authenticates through Application Default Credentials, not a key.
_API_KEY_ENV = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
}

# (pool, single) env vars per provider. The drivers run many workers at once
# and give each its own key where the provider meters per key.
_POOL_ENV = {
    "gemini": ("GEMINI_API_KEYS", "GEMINI_API_KEY"),
    "openai": ("OPENAI_API_KEYS", "OPENAI_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEYS", "ANTHROPIC_API_KEY"),
}


def split_model(model: str) -> tuple[str, str]:
    """`"openai/gpt-4.1"` -> `("openai", "gpt-4.1")`.

    A bare name is resolved through the family prefixes; anything
    unrecognised is returned with an empty provider so litellm can apply its
    own default routing.
    """
    model = (model or "").strip()
    if "/" in model:
        provider, name = model.split("/", 1)
        if provider in PROVIDERS or "/" in name:
            return provider, name
        return provider, name
    low = model.lower()
    for prefix, provider in _FAMILY_PREFIXES:
        if low.startswith(prefix):
            return provider, model
    return "", model


def qualify(model: str) -> str:
    """Model id with its provider prefix, ready for litellm."""
    provider, name = split_model(model)
    return f"{provider}/{name}" if provider else name


def api_key_for(model: str) -> str | None:
    """The key this model's provider needs, or None when it uses ADC."""
    provider, _ = split_model(model)
    for var in _API_KEY_ENV.get(provider, ()):
        key = os.environ.get(var)
        if key:
            return key
    return None


def embedding_model(model: str | None = None) -> str:
    """Embedding model to pair with `model`.

    `EH_EMBED_MODEL` overrides everything, so a run can keep one embedding
    space while the policy provider changes -- which is what you want when
    comparing providers against a bank that already exists.
    """
    override = os.environ.get("EH_EMBED_MODEL")
    if override:
        return override
    provider, _ = split_model(model or "")
    return _DEFAULT_EMBED.get(provider, _DEFAULT_EMBED["openai"])


def effective_model(model: str) -> str:
    """`model`, unless a run-wide override is in force.

    `$EH_MODEL` overrides whatever a config names. Drivers set it from their
    `MODEL` knob so one switch reaches every stage -- including stages behind
    a dispatcher or a subprocess, which inherit the environment but not the
    command line. Unset (the default) means each config keeps the model it
    names, which is how the policy and the harness agent can differ.
    """
    return os.environ.get("EH_MODEL") or model


def max_output_tokens(model: str) -> int | None:
    """The model's own output-token ceiling, or None when litellm has no entry.

    A `max_tokens` tuned for one provider is a hard 400 on a model with a
    smaller ceiling ("max_tokens is too large"), and `drop_params` cannot help
    because the parameter is supported -- only the value is out of range. So
    the value is clamped rather than dropped.
    """
    try:
        import litellm
        info = litellm.get_model_info(qualify(model))
    except Exception:
        return None
    if not isinstance(info, dict):
        return None
    limit = info.get("max_output_tokens") or info.get("max_tokens")
    return int(limit) if limit else None


def clamp_max_tokens(model: str, max_tokens: int | None) -> int | None:
    """`max_tokens`, lowered to what `model` actually accepts."""
    if not max_tokens:
        return max_tokens
    limit = max_output_tokens(model)
    return min(int(max_tokens), limit) if limit else int(max_tokens)


def key_pool(model: str | None = None) -> list[str]:
    """Keys to spread across concurrent workers, for `model`'s provider.

    Gemini bills quota per key, so the drivers hand a different key to each
    worker to avoid throttling one. OpenAI has a single key and Vertex has
    none (ADC), where a pool is meaningless -- so this returns a one-element
    list and an empty one respectively. Round-robin over `key_pool(m) or [None]`
    and every provider works.
    """
    provider, _ = split_model(effective_model(model or ""))
    plural, singular = _POOL_ENV.get(provider, (None, None))
    if plural is None:
        return []
    raw = os.environ.get(plural) or os.environ.get(singular) or ""
    return [k.strip() for k in raw.split(",") if k.strip()]


def key_env(model: str | None, key: str | None) -> dict:
    """Environment a child process needs to use `key` for `model`'s provider.

    The drivers pass one key per worker through the environment. Which
    variable carries it depends on the provider, so callers ask here instead
    of hard-coding `GEMINI_API_KEY`.
    """
    if not key:
        return {}
    provider, _ = split_model(effective_model(model or ""))
    plural, singular = _POOL_ENV.get(provider, (None, None))
    return {singular: key} if singular else {}


def pool_env(model: str | None, keys: list[str] | None = None) -> dict:
    """Environment carrying a whole key pool to a child, for `model`'s provider.

    The counterpart of `key_env` for stages that round-robin the pool
    themselves (bank building, eval) rather than being handed one key.
    """
    model = effective_model(model or "")
    provider, _ = split_model(model)
    plural, singular = _POOL_ENV.get(provider, (None, None))
    keys = [k for k in (keys if keys is not None else key_pool(model)) if k]
    if not plural or not keys:
        return {}
    return {plural: ",".join(keys), singular: keys[0]}


def missing_key_message(model: str | None, minimum: int = 1) -> str | None:
    """None if `model`'s provider has the keys it needs, else what to export.

    Naming the variable the *selected* provider reads is the point: a driver
    that always demands GEMINI_API_KEYS cannot be run on GPT at all.
    """
    model = effective_model(model or "")
    provider, _ = split_model(model)
    if provider in ("vertex_ai",):
        return None                      # ADC, no key to set
    plural, singular = _POOL_ENV.get(provider, (None, None))
    if plural is None:
        return None                      # unknown provider: let litellm decide
    have = len(key_pool(model))
    if have >= max(1, minimum):
        return None
    want = (f"{minimum} comma-separated keys in ${plural}" if minimum > 1
            else f"${singular} (or ${plural} for a comma-separated pool)")
    return f"model {model} needs {want}; got {have}"


def key_override(model: str, key: str | None) -> str | None:
    """A caller-supplied key, kept only if it can belong to `model`'s provider.

    The eval drivers spread load across several `GEMINI_API_KEYS` and hand the
    one they picked to every call they make. That key authenticates a Gemini
    request and nothing else, so on an OpenAI or Vertex model it overwrites the
    key this module resolved and the request fails to authenticate. Returning
    None leaves `api_key_for` in charge.

    The test is the shape of the key: OpenAI issues `sk-...`, Gemini does not,
    and Vertex authenticates through ADC with no key at all.
    """
    if not key:
        return None
    provider, _ = split_model(effective_model(model))
    is_openai_shaped = key.startswith("sk-")
    if provider == "openai":
        return key if is_openai_shaped else None
    if provider == "gemini":
        return None if is_openai_shaped else key
    if provider in ("vertex_ai",):
        return None
    return key


def auth_kwargs(model: str) -> dict:
    """Credentials litellm needs to reach `model`, and nothing else.

    Takes the model verbatim -- no `EH_MODEL` substitution. The embedding path
    needs exactly this: an embedding model must keep its own identity while
    still picking up the provider's credentials, and routing it through
    `client_kwargs` would swap in whatever chat model `EH_MODEL` names.
    """
    provider, _ = split_model(model)
    kwargs: dict = {"model": qualify(model)}

    key = api_key_for(model)
    if key:
        kwargs["api_key"] = key

    if provider == "vertex_ai":
        project = (os.environ.get("VERTEXAI_PROJECT")
                   or os.environ.get("ANTHROPIC_VERTEX_PROJECT")
                   or os.environ.get("GOOGLE_CLOUD_PROJECT"))
        location = (os.environ.get("VERTEXAI_LOCATION")
                    or os.environ.get("ANTHROPIC_VERTEX_REGION")
                    or "global")
        if project:
            kwargs["vertex_project"] = project
        kwargs["vertex_location"] = location
    return kwargs


def client_kwargs(model: str, **overrides) -> dict:
    """Provider-ready `client_kwargs` for `LiteLLMClient`.

    Everything is a plain JSON type, so the result can travel inside a
    PolicySpec through the subprocess runner.
    """
    model = effective_model(model)
    provider, _ = split_model(model)
    kwargs: dict = {"model": qualify(model)}

    # Let a config carry a knob its provider cannot use. litellm removes what
    # the target model does not accept instead of raising, which is what
    # makes one config work across all three providers.
    kwargs["drop_params"] = True

    key = api_key_for(model)
    if key:
        kwargs["api_key"] = key

    if provider == "vertex_ai":
        # litellm reads VERTEXAI_*; the Anthropic SDK reads a different pair.
        # Accept either so a single set of exports drives both.
        project = (os.environ.get("VERTEXAI_PROJECT")
                   or os.environ.get("ANTHROPIC_VERTEX_PROJECT")
                   or os.environ.get("GOOGLE_CLOUD_PROJECT"))
        location = (os.environ.get("VERTEXAI_LOCATION")
                    or os.environ.get("ANTHROPIC_VERTEX_REGION")
                    or "global")
        if project:
            kwargs["vertex_project"] = project
        kwargs["vertex_location"] = location

    kwargs.update({k: v for k, v in overrides.items() if v is not None})

    # A caller that rotates keys passes one on every call, including calls to a
    # provider it cannot authenticate. Drop it there rather than let it shadow
    # the key resolved above.
    if "api_key" in kwargs:
        checked = key_override(model, kwargs["api_key"])
        if checked is None:
            kwargs.pop("api_key")
            if key:
                kwargs["api_key"] = key
        else:
            kwargs["api_key"] = checked

    # A reasoning request needs room for the reasoning. Claude rejects the call
    # outright ("max_tokens must be greater than thinking.budget_tokens"), so
    # it gets a floor even when the config named no cap. Elsewhere only an
    # explicit-but-too-small cap is raised: inventing one where the config had
    # none would truncate a provider that was running unbounded.
    if any(kwargs.get(p) for p in _REASONING_PARAMS):
        current = kwargs.get("max_tokens")
        _prov, _name = split_model(model)
        _is_claude = _prov in ("vertex_ai", "anthropic") and "claude" in _name.lower()
        if (not current and _is_claude) or (
                current and int(current) < _REASONING_MAX_TOKENS_FLOOR):
            kwargs["max_tokens"] = _REASONING_MAX_TOKENS_FLOOR
    if kwargs.get("max_tokens"):
        kwargs["max_tokens"] = clamp_max_tokens(model, kwargs["max_tokens"])

    return kwargs


def reconcile_call_params(model: str, *, temperature: float | None,
                          max_tokens: int | None,
                          params: dict) -> tuple[float | None, int | None]:
    """Make one call's `temperature` / `max_tokens` acceptable to `model`.

    Call-site values win over `params` (the client's construction kwargs);
    this only fills gaps and resolves conflicts the provider would reject.
    Returns the pair to send.

    The rules are the provider constraints observed in practice:

      * A reasoning request needs a `max_tokens` that leaves room for the
        reasoning. Claude rejects the call outright when the thinking budget
        does not fit; Gemini accepts it but can spend the whole budget
        thinking and return nothing.
      * Claude only allows `temperature=1` while thinking is enabled. A
        config carrying both a reasoning knob and the framework's usual
        sampling temperature is otherwise a 400.
    """
    if max_tokens is None:
        max_tokens = params.get("max_tokens")
    wants_reasoning = any(params.get(p) for p in _REASONING_PARAMS)
    if wants_reasoning:
        provider, name = split_model(model)
        if provider in ("vertex_ai", "anthropic") and "claude" in name.lower():
            # Claude rejects the request unless max_tokens exceeds the thinking
            # budget, and only allows temperature=1 while thinking is enabled.
            if not max_tokens or int(max_tokens) < _REASONING_MAX_TOKENS_FLOOR:
                max_tokens = _REASONING_MAX_TOKENS_FLOOR
            temperature = 1.0
        elif max_tokens is not None and int(max_tokens) < _REASONING_MAX_TOKENS_FLOOR:
            # An explicit cap this small would be spent thinking, leaving no
            # answer. An *absent* cap is left absent: imposing one here would
            # silently truncate providers that were running unbounded.
            max_tokens = _REASONING_MAX_TOKENS_FLOOR
    return temperature, clamp_max_tokens(model, max_tokens)


def client_spec(model: str, **overrides) -> tuple[str, dict]:
    """`(client_factory_import_path, client_kwargs)` for `model`.

    The factory is the same for every provider; only the kwargs differ.
    Feed the pair straight into `PolicySpec` or `import_symbol(factory)(**kwargs)`.
    """
    return CLIENT_FACTORY, client_kwargs(model, **overrides)


def completion_kwargs(model: str, *, temperature: float | None = None,
                      max_tokens: int | None = None, **overrides) -> dict:
    """Everything but `messages` for a direct `litellm.completion` call.

    For code that talks to litellm without going through `LLMClient` -- skill
    induction and the bank builders. Applies the same auth and
    parameter reconciliation the client path gets, so those stages switch
    provider with the same model string as the policy does.
    """
    model = effective_model(model)
    kwargs = client_kwargs(model, **overrides)
    temperature, max_tokens = reconcile_call_params(
        model, temperature=temperature, max_tokens=max_tokens, params=kwargs,
    )
    kwargs.pop("max_tokens", None)
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return kwargs


def embedding_kwargs(model: str | None = None, **overrides) -> dict:
    """Everything but `input` for a `litellm.embedding` call.

    Embeddings are a separate model from the policy's, so they need their own
    auth: on Vertex that is a project/region pair, on the others a key. Without
    this the bank builders reach litellm with no credentials and fall back to
    whatever happens to be in the environment.
    """
    embed = overrides.pop("model", None) or embedding_model(model)
    kwargs = auth_kwargs(embed)
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return kwargs


def build_client(model: str, **overrides):
    """Instantiate the client for `model`. For in-process callers; the
    subprocess runner wants `client_spec` instead."""
    from envharness.infra.utils import import_symbol
    factory, kwargs = client_spec(model, **overrides)
    return import_symbol(factory)(**kwargs)
