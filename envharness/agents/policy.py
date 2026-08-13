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

"""PolicyAgent -- consumes Tool schemas + observation, emits Actions.

FUNCTION_CALLING (the default) requires an LLM backend with tool use; the
other formats drive a single-tool bridge from raw text.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from envharness.infra.llm import LLMClient, Message
from envharness.core.types import Action, Observation


class ActionFormat(str, Enum):
    FUNCTION_CALLING = "function_calling"
    TEXT_COMPLETE = "text_complete"   # raw line of text -> single-tool, single-arg
    # Model emits `<think>reasoning</think><action>command</action>`; the
    # command is normalized against the bridge's admissible_commands (same as
    # TEXT_COMPLETE) and dispatched to the single tool. This is the verl-agent
    # / GiGPO prompt format.
    THINK_ACTION = "think_action"
    # WebArena: delegate per-turn act() to ReasoningBank's GenericAgent so
    # corpus generation, mutation rollouts, and test eval all use the
    # SAME prompt the no-bank test baseline used. PolicyAgent reads the
    # full raw browsergym observation via `obs.data["browsergym_raw"]`.
    # Rules.filter_observation can mutate `obs.data["browsergym_raw"]`
    # in-place; the RB agent sees the mutated dict.
    WEBARENA_RB = "webarena_rb"


# ---------------------------------------------------------------------------
# Internal: response parsing
# ---------------------------------------------------------------------------

# `<action>command</action>`. We take the LAST
# match so any preamble or thinking that mentions `<action>` doesn't
# confuse the parser.
THINK_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)


# Strip <think>...</think> reasoning blocks (hybrid-thinking models like Qwen3).
# Also handles unclosed <think> when the model gets truncated mid-thought.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_OPEN_THINK_RE = re.compile(r"<think>.*", re.DOTALL)


def _normalize_command(text: str, admissible: list[str]) -> str:
    """Mirror of pure_eval._normalize_command. Pick a single command line from
    the model's output and match it to an admissible command. Returns the raw
    first line if no admissible match (the env will then echo 'Nothing happens'
    and the model gets another chance)."""
    text = _OPEN_THINK_RE.sub("", _THINK_BLOCK_RE.sub("", text or ""))
    text = text.strip().strip("`").strip('"').strip("'")
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    if not admissible:
        return line
    if line in admissible:
        return line
    low = line.lower()
    for c in admissible:
        if low == c.lower():
            return c
    # Substring pass: longest admissible command first, so "go to drawer 12"
    # in the model output can't be shadowed by "go to drawer 1" appearing
    # earlier in the admissible list (prefix shadowing).
    for c in sorted(admissible, key=len, reverse=True):
        if c in line:
            return c
    return line



# ---------------------------------------------------------------------------
# PolicyAgent
# ---------------------------------------------------------------------------

@dataclass
class PolicyAgent:
    client: LLMClient
    tools: list[dict]                         # tool schemas (from Bridge)
    task_prompt: str = ""                     # task description
    action_format: ActionFormat = ActionFormat.FUNCTION_CALLING
    max_history: int = 50
    temperature: float = 0.4                  # see PolicySpec.temperature
    # Extra per-format config (WEBARENA_RB reads rb_model_name and
    # max_response_tokens from here).
    prompt_builder_kwargs: dict | None = None

    def __post_init__(self):
        self._history: list[Message] = []
        self._last_admissible: list[str] = []
        self._text_complete_target: tuple[str, str] | None = None
        # (id, name) of the tool call awaiting its result message. OpenAI and
        # Anthropic reject the next request unless an assistant message
        # carrying tool_calls is followed immediately by a tool message for
        # each id; Gemini tolerates its absence. The next observation IS that
        # result, so act() emits it with role="tool" while this is set.
        self._pending_tool_call: tuple[str, str] | None = None
        # Last raw LLM response (includes any <think>...</think> blocks).
        # Runner reads this after each act() and writes it into Step.policy_raw_response
        # so RB induction can use it. Set to "" on reset.
        self.last_raw_response: str = ""
        if self.action_format in (ActionFormat.TEXT_COMPLETE,
                                    ActionFormat.THINK_ACTION):
            if len(self.tools) != 1:
                raise ValueError(
                    f"{self.action_format.value} requires exactly one tool; "
                    f"got {len(self.tools)}"
                )
            fn = self.tools[0]["function"]
            props = fn.get("parameters", {}).get("properties", {})
            if len(props) != 1:
                raise ValueError(
                    f"{self.action_format.value} requires the single tool to "
                    f"have exactly one argument; got {list(props)}"
                )
            self._text_complete_target = (fn["name"], next(iter(props)))
        # WEBARENA_RB: lazy-init RB GenericAgent inside _act_webarena_rb (the
        # browsergym/playwright import graph is heavy + monkey-patches litellm;
        # we want it to run only after the bridge is up so the subprocess
        # image stays clean). Pull RB-specific config from
        # prompt_builder_kwargs (rb_model_name, max_response_tokens).
        self._reasoning_bank_agent = None
        if self.action_format == ActionFormat.WEBARENA_RB:
            if len(self.tools) != 1:
                raise ValueError(
                    "webarena_rb requires exactly one tool (do); "
                    f"got {len(self.tools)}"
                )
            fn = self.tools[0]["function"]
            self._text_complete_target = (fn["name"], "action_str")
            pk = self.prompt_builder_kwargs or {}
            client_model = (getattr(self.client, "model_id", None)
                            or getattr(self.client, "model", None)
                            or "openai/gpt-4.1-mini")
            default_rb_model = (client_model if str(client_model).startswith("litellm/")
                                else f"litellm/{client_model}")
            self._rb_model_name = pk.get("rb_model_name", default_rb_model)
            self._rb_max_response_tokens = int(pk.get("max_response_tokens", 512))
        if self.task_prompt:
            self._history.append(Message(role="system", content=self._system_prompt()))

    # ----- public API -----

    def reset(self) -> None:
        self._history = []
        self._pending_tool_call = None
        if self.task_prompt:
            self._history.append(Message(role="system", content=self._system_prompt()))

    def act(self, observation: Observation) -> Action:
        """Given an observation, return the next Action."""
        if self.action_format in (ActionFormat.TEXT_COMPLETE,
                                    ActionFormat.THINK_ACTION):
            # Track admissible_commands from the latest obs for normalization.
            self._last_admissible = list(
                (observation.data or {}).get("admissible_commands") or []
            )
        if self.action_format == ActionFormat.WEBARENA_RB:
            return self._act_webarena_rb(observation)
        obs_text = self._format_obs(observation)
        if self._pending_tool_call is not None:
            call_id, call_name = self._pending_tool_call
            self._history.append(Message(role="tool", content=obs_text,
                                          tool_call_id=call_id, name=call_name))
            self._pending_tool_call = None
        else:
            self._history.append(Message(role="user", content=obs_text))
        if self.action_format == ActionFormat.FUNCTION_CALLING:
            return self._act_fc()
        if self.action_format == ActionFormat.TEXT_COMPLETE:
            return self._act_text_complete()
        if self.action_format == ActionFormat.THINK_ACTION:
            return self._act_think_action()
        raise NotImplementedError(self.action_format)

    # ----- history slicing (preserves leading system message) -----

    def _recent_history(self) -> list[Message]:
        """Return up to max_history messages, always including the leading
        system message if one exists. Pure-eval-style baselines keep full
        history; default max_history=50 silently drops the system prompt on
        long episodes (>= ~25 steps)."""
        if not self._history:
            return []
        if (self._history[0].role == "system"
                and len(self._history) > self.max_history):
            head, tail = [self._history[0]], self._history[-(self.max_history - 1):]
        else:
            head, tail = [], self._history[-self.max_history:]
        # A tool message whose assistant turn fell outside the window is an
        # orphan the API rejects; drop those leading results.
        while tail and tail[0].role == "tool":
            tail = tail[1:]
        return head + tail

    # ----- FC path -----

    def _act_fc(self) -> Action:
        resp = self.client.chat(messages=self._recent_history(),
                                 tools=self.tools,
                                 temperature=self.temperature)
        self.last_raw_response = resp.content or ""
        if resp.tool_calls:
            tc = resp.tool_calls[0]
            self._history.append(Message(role="assistant", content=resp.content,
                                          tool_calls=[tc]))
            self._pending_tool_call = (tc.id, tc.name)
            return Action(name=tc.name, kwargs=tc.arguments)
        # Model returned plain text instead of a tool call -- treat as a
        # malformed action; downstream will mark it failed.
        self._history.append(Message(role="assistant", content=resp.content))
        return Action(name="__noop__", kwargs={"raw": resp.content})

    # ----- WebArena RB-agent path (delegates to ReasoningBank GenericAgent
    # so corpus generation, mutation rollouts, and eval all share the exact
    # same prompt as the no-bank test baseline) -----

    def _act_webarena_rb(self, observation: Observation) -> Action:
        if self._reasoning_bank_agent is None:
            from envharness.prompts.webarena_reasoning_bank_agent import build_reasoning_bank_agent
            self._reasoning_bank_agent = build_reasoning_bank_agent(
                model_name=self._rb_model_name,
                temperature=self.temperature,
                max_tokens=self._rb_max_response_tokens,
            )
        data = observation.data or {}
        raw_obs = data.get("browsergym_raw")
        if raw_obs is None:
            # No raw browsergym dict on this observation -- happens on
            # bridge error paths ([unknown tool] / [no action emitted]) and
            # when a Rules O-hook rebuilt obs.data without the key. Recover
            # with a cheap noop instead of crashing the episode: the next
            # bridge.step returns a full observation. Skipping the RB
            # agent's get_action here also keeps its internal obs/action
            # history aligned.
            self.last_raw_response = (
                "<error: obs.data['browsergym_raw'] missing, "
                f"got keys={list(data.keys())}>"
            )
            tool_name, arg_name = self._text_complete_target
            return Action(name=tool_name, kwargs={arg_name: "noop(1000)"})
        processed = self._reasoning_bank_agent.obs_preprocessor(raw_obs)
        try:
            action_str, _ans = self._reasoning_bank_agent.get_action(processed)
        except Exception as e:  # noqa: BLE001
            self.last_raw_response = f"<reasoning_bank_agent_error: {type(e).__name__}: {e}>"
            return Action(name="__noop__", kwargs={"raw": self.last_raw_response})
        action_str = action_str or ""
        self.last_raw_response = action_str
        tool_name, arg_name = self._text_complete_target
        return Action(name=tool_name, kwargs={arg_name: action_str})

    # ----- Text-complete path (raw line of text; mirrors pure-eval loop) -----

    def _act_text_complete(self) -> Action:
        resp = self.client.chat(messages=self._recent_history(), tools=None,
                                 temperature=self.temperature)
        raw = resp.content or ""
        self.last_raw_response = raw
        normalized = _normalize_command(raw, self._last_admissible)
        # Match pure_eval: store the normalized command in history (falls back
        # to raw if normalization produced nothing).
        self._history.append(Message(role="assistant", content=normalized or raw))
        tool_name, arg_name = self._text_complete_target
        return Action(name=tool_name, kwargs={arg_name: normalized})

    # ----- Think-Action path -----
    # Model emits: <think>...reasoning...</think><action>command</action>
    # We extract the LAST <action>...</action>, normalize against the
    # bridge's admissible_commands, and dispatch through the single tool.

    def _act_think_action(self) -> Action:
        resp = self.client.chat(messages=self._recent_history(), tools=None,
                                 temperature=self.temperature)
        raw = resp.content or ""
        self.last_raw_response = raw
        # Store the full response (think + action) in history so subsequent
        # turns see the model's reasoning context.
        self._history.append(Message(role="assistant", content=raw))
        # Parse FIRST <action>...</action>: enforces one-action-per-turn
        # at the parser level when the model dumps a multi-action plan
        # (common on long-horizon benches like SWE-bench). For ALFWorld
        # where models emit one action per turn anyway, first == last so
        # behavior is unchanged.
        matches = list(THINK_ACTION_RE.finditer(raw))
        if matches:
            cmd_text = matches[0].group(1).strip()
        else:
            # Fallback: strip <think> blocks and take the last non-empty line
            stripped = _OPEN_THINK_RE.sub("", _THINK_BLOCK_RE.sub("", raw)).strip()
            cmd_text = next((ln.strip() for ln in stripped.splitlines() if ln.strip()), "")
        # Multi-line preservation for open-vocab tools (admissible empty):
        # SWE-bench bash commands need multi-line support (heredocs, multi-
        # line python -c). When admissible is empty, the Bridge accepts
        # whatever text we pass; preserve the full action content. For
        # ALFWorld (admissible populated), keep the canonical first-line
        # normalization path.
        if self._last_admissible:
            normalized = _normalize_command(cmd_text, self._last_admissible)
        else:
            normalized = cmd_text.strip()
        tool_name, arg_name = self._text_complete_target
        return Action(name=tool_name, kwargs={arg_name: normalized})

    # ----- prompts -----

    def _system_prompt(self) -> str:
        if self.action_format == ActionFormat.FUNCTION_CALLING:
            return self.task_prompt
        if self.action_format == ActionFormat.TEXT_COMPLETE:
            # text_complete: task_prompt is the whole protocol. The Bridge's
            # observation already carries the admissible commands; no tool
            # listing or output format scaffolding is added by the framework.
            return self.task_prompt
        if self.action_format == ActionFormat.THINK_ACTION:
            # think_action: task_prompt is the whole protocol. We rely on
            # the instruction inside the user prompt (built
            # in the eval script's prompt template) so the system slot just
            # carries the task description.
            return self.task_prompt
        # ReAct: include tool list inline
        tool_lines = []
        for ts in self.tools:
            f = ts["function"]
            params = f["parameters"]["properties"]
            arglist = ", ".join(f"{k}: {v.get('type','any')}" for k, v in params.items())
            tool_lines.append(f"- {f['name']}({arglist}): {f['description']}")
        return (
            f"{self.task_prompt}\n\n"
            "You have access to these tools:\n"
            + "\n".join(tool_lines)
            + "\n\nFormat each turn as:\n"
              "Thought: <your reasoning>\n"
              "Action: tool_name(arg1=value1, arg2=value2)\n"
        )

    def _format_obs(self, obs: Observation) -> str:
        # text_complete + think_action: pure-eval-style obs (admissible
        # commands already in obs.text via the Bridge's _observe()). For
        # think_action we append a one-line reminder of the response format
        # so the model emits <think>...</think><action>cmd</action>.
        if self.action_format == ActionFormat.TEXT_COMPLETE:
            return obs.text
        if self.action_format == ActionFormat.THINK_ACTION:
            # Admissibles may live in obs.text (wrapped obs_style) or only in
            # obs.data["admissible_commands"] (raw obs_style). Surface them
            # in the reminder when the Bridge exposes them via obs.data so
            # the model isn't told "choose from the list above" when no list
            # appears above. Falls back to the generic phrasing when neither
            # source is available (the Bridge isn't reporting admissibles).
            admissibles = (obs.data or {}).get("admissible_commands") or []
            if admissibles:
                admissible_block = ("Admissible actions: ["
                                     + ", ".join(admissibles) + "].")
            else:
                admissible_block = "Choose an admissible action."
            return (
                obs.text
                + f"\n\n{admissible_block}\n\n"
                  "Reason about the current situation inside `<think> </think>` tags, "
                  "then present ONE admissible action inside `<action> </action>` tags. "
                  "Example:\n"
                  "<think>...</think><action>go to drawer 1</action>"
            )
        if obs.data:
            return f"{obs.text}\n\nState: {json.dumps(obs.data, ensure_ascii=False)}"
        return obs.text

    def observe_tool_result(self, tool_name: str, result: str, tool_call_id: str | None = None) -> None:
        """Inject a tool result back into the FC conversation. For ReAct, the
        next call to act() will format it as the user message."""
        if self.action_format == ActionFormat.FUNCTION_CALLING and tool_call_id:
            self._history.append(Message(role="tool", content=result,
                                          tool_call_id=tool_call_id, name=tool_name))
