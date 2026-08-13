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

"""LLMClient abstraction. Default = LiteLLM. Also provides:
- ScriptedClient for offline/deterministic testing without any API keys
- LoggingLLMClient that wraps any other client and JSONL-logs every chat call
  (subprocess-safe; useful for remote/offline debug)
"""
from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Common types
# ---------------------------------------------------------------------------

@dataclass
class Message:
    role: str          # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: list["ToolCall"] | None = None
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None


# ---------------------------------------------------------------------------
# Direct-call helper for eval drivers
# ---------------------------------------------------------------------------

def completion_with_retry(*, max_retries: int = 6,
                          retry_initial_delay: float = 2.0,
                          retry_max_delay: float = 60.0,
                          **completion_kwargs):
    """litellm.completion with the framework's transient-vs-fatal policy,
    for eval drivers that call litellm directly (bypassing LLMClient).

    Transient errors (429 rate-limit / 5xx / connection / timeout) retry IN
    PLACE with exponential backoff -- an API hiccup must not be recorded as
    an episode failure. Fatal errors (auth / bad request) raise on the
    first attempt. Total worst-case wait at the defaults: ~2 minutes."""
    import litellm
    transient: tuple = ()
    for name in ("APIConnectionError", "RateLimitError",
                 "ServiceUnavailableError", "InternalServerError", "Timeout"):
        cls = getattr(litellm, name, None)
        if isinstance(cls, type):
            transient = transient + (cls,)
    for attempt in range(max_retries + 1):
        try:
            return litellm.completion(**completion_kwargs)
        except transient as e:
            if attempt == max_retries:
                raise
            delay = min(retry_max_delay, retry_initial_delay * (2 ** attempt))
            print(f"[completion_with_retry {attempt + 1}/{max_retries}] "
                  f"{type(e).__name__}: sleeping {delay:.0f}s and retrying",
                  file=__import__("sys").stderr, flush=True)
            time.sleep(delay)


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    model_id: str = ""

    @abstractmethod
    def chat(self,
             messages: list[Message],
             tools: list[dict] | None = None,
             tool_choice: str | dict = "auto",
             temperature: float = 0.7,
             max_tokens: int | None = None,
             **kwargs) -> ChatResponse: ...


# ---------------------------------------------------------------------------
# LiteLLM-backed default
# ---------------------------------------------------------------------------

class LiteLLMClient(LLMClient):
    """Default. Supports every provider LiteLLM supports.

    Examples:
      LiteLLMClient("anthropic/claude-opus-4-7")
      LiteLLMClient("openai/gpt-4o")
      LiteLLMClient("openai/qwen2.5-7b-instruct", api_base="http://localhost:8000/v1")
      LiteLLMClient("ollama/llama3.1:70b")

    Retries on transient API errors (5xx / connection / rate-limit / timeout)
    with exponential backoff. Configurable via `max_retries` (default 7) and
    `retry_initial_delay` (default 1.0s). The default budget
    (1+2+4+8+16+30+30 = 91s) outlasts a provider's per-minute token-quota
    reset, so a stage that saturates its TPM waits for the window to refill
    instead of failing every episode in it.
    """

    def __init__(self, model: str, api_base: str | None = None,
                 max_retries: int = 7, retry_initial_delay: float = 1.0,
                 retry_max_delay: float = 30.0, **defaults):
        self.model_id = model
        self.api_base = api_base
        self.max_retries = int(max_retries)
        self.retry_initial_delay = float(retry_initial_delay)
        self.retry_max_delay = float(retry_max_delay)
        self.defaults = defaults

    def chat(self, messages, tools=None, tool_choice="auto",
             temperature=0.7, max_tokens=None, **kwargs):
        import litellm
        # Exception classes that warrant retry. Pull from litellm dynamically
        # because the module structure has shifted across versions. We
        # explicitly DO NOT include `APIError` (the base class): retrying a
        # deterministic 400 (BadRequest / ContextWindowExceeded /
        # AuthenticationError -- all APIError subclasses) burns minutes of
        # backoff before failing identically each time.
        _transient_excs: tuple = ()
        for name in ("APIConnectionError", "RateLimitError",
                      "ServiceUnavailableError", "InternalServerError",
                      "Timeout"):
            cls = getattr(litellm, name, None)
            if isinstance(cls, type):
                _transient_excs = _transient_excs + (cls,)
        params = {**self.defaults, **kwargs}
        # temperature / max_tokens are passed explicitly below; leaving them in
        # `params` too would splat into litellm.completion twice and raise
        # TypeError "got multiple values". They are removed here and folded
        # back in by reconcile_call_params, which resolves them against what
        # this model's provider will accept (see envharness.infra.model).
        from envharness.infra.model import reconcile_call_params
        temperature, max_tokens = reconcile_call_params(
            self.model_id, temperature=temperature, max_tokens=max_tokens,
            params=params,
        )
        params.pop("temperature", None)
        params.pop("max_tokens", None)
        if self.api_base:
            params["api_base"] = self.api_base

        resp = None
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = litellm.completion(
                    model=self.model_id,
                    messages=[_message_to_dict(m) for m in messages],
                    tools=tools,
                    tool_choice=tool_choice if tools else None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **params,
                )
                break
            except _transient_excs as e:
                last_exc = e
                if attempt == self.max_retries:
                    raise
                delay = min(
                    self.retry_max_delay,
                    self.retry_initial_delay * (2 ** attempt),
                )
                # Stderr so the parent orchestrator (which captures stderr per
                # subprocess) sees the retry happened.
                print(
                    f"[LiteLLMClient retry {attempt+1}/{self.max_retries}] "
                    f"{type(e).__name__}: sleeping {delay:.1f}s and retrying",
                    file=__import__("sys").stderr, flush=True,
                )
                time.sleep(delay)
        assert resp is not None  # for type-checkers; loop either returns or raises
        choice = resp.choices[0].message
        tcalls = []
        for tc in (choice.tool_calls or []) if hasattr(choice, "tool_calls") else []:
            args = tc.function.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            tcalls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        return ChatResponse(content=choice.content or "", tool_calls=tcalls, raw=resp)


# ---------------------------------------------------------------------------
# Completion-API + tokenizer-rendered chat template
# ---------------------------------------------------------------------------

_TOKENIZER_CACHE: dict[str, Any] = {}


class CompletionAPIClient(LLMClient):
    """vLLM /v1/completions client with manual chat-template rendering.

    vLLM's /v1/chat/completions injects an empty `<think></think>` prefix when
    enable_thinking=False, which makes Qwen3 emit `</think>` repeatedly instead
    of the action. This client takes the completion-API path instead:
    apply tokenizer.apply_chat_template to the messages
    locally (with chat_template_kwargs), POST the rendered prompt to
    /v1/completions via litellm.text_completion, and return the raw text.

    Use this for trained Qwen3 checkpoints served by vLLM whenever the
    mutation-time / training-time / inference-time SR must match bit-exactly.

    Note: completion API doesn't support tool calls — `tools` kwarg is ignored.
    Pair with PolicyAgent's text_complete / think_action / react formats.

    Example::

        CompletionAPIClient(
            model="openai/qwen3-8b-base",
            api_base="http://localhost:8901/v1",
            api_key="EMPTY",
            tokenizer_path="Qwen/Qwen3-8B",
            chat_template_kwargs={"enable_thinking": False},
        )
    """

    def __init__(self, model: str, tokenizer_path: str,
                 api_base: str | None = None, api_key: str | None = None,
                 api_base_pool: list[str] | None = None,
                 chat_template_kwargs: dict | None = None,
                 default_max_tokens: int = 512,
                 max_retries: int = 5, retry_initial_delay: float = 1.0,
                 retry_max_delay: float = 30.0, **defaults):
        self.model_id = model
        self.tokenizer_path = tokenizer_path
        # Endpoint resolution (highest priority first):
        #   1. ENVHARNESS_API_BASE env var -- per-process override the shard
        #      launcher / driver can set without touching the YAML. Used by
        #      the 8-shard mutation deployment (each shard pins to one
        #      endpoint) and by integration tests that point at a custom
        #      server. Empty string is treated as unset.
        #   2. `api_base` kwarg -- explicit per-instance override. Driver-side
        #      EndpointPool.lease() ultimately routes through this path via
        #      envharness.infra.endpoint_pool.pin_endpoint.
        #   3. `api_base_pool` kwarg -- list of endpoints. CompletionAPIClient
        #      does NOT do its own load-balancing across these. The driver
        #      (Orchestrator / sweep) is expected to construct an
        #      EndpointPool from this list and call pin_endpoint per-spec.
        #      If a client is built with a pool but no driver-side dispatch
        #      (e.g. a direct-Python smoke test), it falls back to the FIRST
        #      endpoint -- consistent and predictable, no PID clustering.
        env_override = os.environ.get("ENVHARNESS_API_BASE") or None
        if env_override:
            api_base = env_override
        elif not api_base and api_base_pool:
            api_base = list(api_base_pool)[0]
        self.api_base = api_base
        self.api_base_pool = list(api_base_pool) if api_base_pool else None
        self.api_key = api_key
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        # /v1/completions defaults to max_tokens=16 — silently truncates Qwen3
        # mid-<think> before it can emit <action>. Default 512 here matches
        # pure_eval / step_60 training.
        self.default_max_tokens = int(default_max_tokens)
        self.max_retries = int(max_retries)
        self.retry_initial_delay = float(retry_initial_delay)
        self.retry_max_delay = float(retry_max_delay)
        self.defaults = defaults

    def _tokenizer(self):
        if self.tokenizer_path not in _TOKENIZER_CACHE:
            from transformers import AutoTokenizer
            _TOKENIZER_CACHE[self.tokenizer_path] = AutoTokenizer.from_pretrained(
                self.tokenizer_path
            )
        return _TOKENIZER_CACHE[self.tokenizer_path]

    def chat(self, messages, tools=None, tool_choice="auto",
             temperature=0.7, max_tokens=None, **kwargs):
        import litellm
        # Match LiteLLMClient's transient-error retry policy
        _transient_excs: tuple = ()
        for name in ("APIConnectionError", "RateLimitError",
                      "ServiceUnavailableError", "InternalServerError",
                      "Timeout", "APIError"):
            cls = getattr(litellm, name, None)
            if isinstance(cls, type):
                _transient_excs = _transient_excs + (cls,)

        tok = self._tokenizer()
        # NOTE: template kwargs must be SPREAD into apply_chat_template.
        # Passing `chat_template_kwargs={...}` as one named kwarg only drops
        # an unused variable into the Jinja context, and the intended flags
        # (e.g. enable_thinking=False) silently never reach the template --
        # verified against transformers 4.57 + Qwen3.
        rendered = tok.apply_chat_template(
            [_message_to_dict(m) for m in messages],
            add_generation_prompt=True, tokenize=False,
            **self.chat_template_kwargs,
        )
        # "openai/foo" -> "text-completion-openai/foo"; bare names pass through
        raw_model = self.model_id.split("/", 1)[1] if "/" in self.model_id else self.model_id
        params = {**self.defaults, **kwargs}
        # temperature is passed explicitly to litellm.text_completion below;
        # pop it so a config that also sets it in client_kwargs doesn't raise
        # TypeError "got multiple values". PolicySpec.temperature is the
        # canonical knob — a client_kwargs temperature is ignored.
        # (max_tokens is assigned into params below, so it can't collide.)
        params.pop("temperature", None)
        if self.api_base:
            params["api_base"] = self.api_base
        if self.api_key:
            params["api_key"] = self.api_key
        params["max_tokens"] = int(max_tokens) if max_tokens is not None else self.default_max_tokens

        resp = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = litellm.text_completion(
                    model="text-completion-openai/" + raw_model,
                    prompt=rendered,
                    temperature=temperature,
                    **params,
                )
                break
            except _transient_excs as e:
                if attempt == self.max_retries:
                    raise
                delay = min(
                    self.retry_max_delay,
                    self.retry_initial_delay * (2 ** attempt),
                )
                print(
                    f"[CompletionAPIClient retry {attempt+1}/{self.max_retries}] "
                    f"{type(e).__name__}: sleeping {delay:.1f}s and retrying",
                    file=__import__("sys").stderr, flush=True,
                )
                time.sleep(delay)
        assert resp is not None
        text = (resp.choices[0].text or "")
        return ChatResponse(content=text, tool_calls=[], raw=resp)


def _message_to_dict(m: Message) -> dict:
    d: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        d["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
            for tc in m.tool_calls
        ]
    if m.tool_call_id:
        d["tool_call_id"] = m.tool_call_id
    if m.name:
        d["name"] = m.name
    return d


# ---------------------------------------------------------------------------
# Scripted client -- deterministic responses for offline tests
# ---------------------------------------------------------------------------

class ScriptedClient(LLMClient):
    """A canned LLM client. Two construction modes:

    1. JSON-serializable (works through subprocess):
         ScriptedClient(model_id="scripted/seq",
                        script=[{"name": "combine", "kwargs": {"i":0,"j":1,"op":"add"}},
                                {"name": "stop", "kwargs": {}}])
       Each chat() call advances the script index and returns the next entry
       as a tool_call. Past the end, returns an empty response.

    2. In-process callable (NOT subprocess-safe):
         ScriptedClient(responder=fn, model_id="scripted/fn")
       The callable receives (messages, tools) and returns a ChatResponse.

    For demonstrating Rules BEHAVIOR, do NOT use this; use the
    LLMHarnessAgent with a real LLM. ScriptedClient exists only to test framework plumbing."""

    def __init__(self, model_id: str = "scripted/test",
                 script: list[dict] | None = None,
                 responder: Callable[[list[Message], list[dict] | None], ChatResponse] | None = None):
        if script is not None and responder is not None:
            raise ValueError("ScriptedClient: pass either script or responder, not both")
        self.model_id = model_id
        self.script = script or []
        self.responder = responder
        self._idx = 0

    def chat(self, messages, tools=None, tool_choice="auto",
             temperature=0.7, max_tokens=None, **kwargs):
        if self.responder is not None:
            return self.responder(messages, tools)
        if self._idx >= len(self.script):
            return ChatResponse(content="(end of script)", tool_calls=[])
        item = self.script[self._idx]
        self._idx += 1
        if "name" in item:
            return ChatResponse(
                content=item.get("content", ""),
                tool_calls=[ToolCall(id=f"t{self._idx}", name=item["name"],
                                      arguments=item.get("kwargs", {}))],
            )
        return ChatResponse(content=item.get("content", ""), tool_calls=[])


# ---------------------------------------------------------------------------
# In-process vLLM stub (interface only; users wire their own training framework)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Logging wrapper -- subprocess-safe, JSONL output
# ---------------------------------------------------------------------------

class LoggingLLMClient(LLMClient):
    """Wraps any LLMClient and appends every chat call to a JSONL file.

    Subprocess-safe: uses os.O_APPEND for atomic writes of small records.
    Construction is JSON-serializable (uses inner_factory + inner_kwargs by
    import path), so this can sit inside a PolicySpec for SubprocessRunner.

    Each line of the JSONL is one record:
      {ts, role, model, duration_ms, messages, tools, response, error?, pid}
    """

    def __init__(self, inner_factory: str, inner_kwargs: dict,
                 log_path: str, role: str = "",
                 truncate_messages_to_chars: int | None = 8000):
        from envharness.infra.utils import import_symbol
        cls = import_symbol(inner_factory)
        self.inner: LLMClient = cls(**inner_kwargs)
        self.log_path = Path(log_path)
        # When the orchestrator runs K rollouts concurrently it sets
        # ENVHARNESS_LOG_PER_PID=1 so each subprocess writes to its own file
        # (avoids POSIX O_APPEND races on lines > PIPE_BUF). Merge with
        # `cat <run>/policy_calls.*.jsonl` or `jq -s . <run>/policy_calls.*.jsonl`.
        if os.environ.get("ENVHARNESS_LOG_PER_PID"):
            base = self.log_path
            self.log_path = base.with_name(f"{base.stem}.{os.getpid()}{base.suffix}")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.role = role
        self.truncate_messages_to_chars = truncate_messages_to_chars
        self.model_id = self.inner.model_id

    def chat(self, messages, tools=None, tool_choice="auto",
             temperature=0.7, max_tokens=None, **kwargs):
        t0 = time.time()
        error: str | None = None
        exc: Exception | None = None   # original exception, re-raised after logging
        resp: ChatResponse
        try:
            resp = self.inner.chat(messages, tools=tools, tool_choice=tool_choice,
                                    temperature=temperature, max_tokens=max_tokens,
                                    **kwargs)
        except Exception as e:
            exc = e
            error = f"{type(e).__name__}: {e}"
            resp = ChatResponse(content="", tool_calls=[])
        elapsed_ms = int((time.time() - t0) * 1000)

        record = {
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pid": os.getpid(),
            "role": self.role,
            "model": self.model_id,
            "duration_ms": elapsed_ms,
            "messages": [self._serialize_message(m) for m in messages],
            "tools": tools,
            "tool_choice": tool_choice,
            "response": {
                "content": resp.content,
                "tool_calls": [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in resp.tool_calls
                ],
            },
            "error": error,
        }
        try:
            line = json.dumps(record, default=str, ensure_ascii=False) + "\n"
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass   # never break the run because of logging

        if exc is not None:
            # Re-raise the ORIGINAL exception (not a RuntimeError wrapper) so
            # outer code can still classify litellm exception types.
            raise exc
        return resp

    def _serialize_message(self, m: Message) -> dict:
        d = _message_to_dict(m)
        cap = self.truncate_messages_to_chars
        if cap and isinstance(d.get("content"), str) and len(d["content"]) > cap:
            d["content"] = d["content"][:cap] + f"...[truncated {len(d['content']) - cap} chars]"
        return d


# ---------------------------------------------------------------------------
# In-process vLLM stub (interface only; users wire their own training framework)
# ---------------------------------------------------------------------------

class VLLMInProcessClient(LLMClient):
    """Direct vLLM Python API client.

    Optional. Use when running inside a training framework (Verl/OpenRLHF) that
    holds vLLM weights in-process and wants to avoid the HTTP hop. Stub here
    -- wire to your framework's vLLM engine when needed.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "VLLMInProcessClient is a stub. Wire to your training framework's "
            "vLLM engine. For most use cases, LiteLLMClient pointing at a vLLM "
            "OpenAI-compatible endpoint is sufficient."
        )

    def chat(self, *args, **kwargs):
        raise NotImplementedError
