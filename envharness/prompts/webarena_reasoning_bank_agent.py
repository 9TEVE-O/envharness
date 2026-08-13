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

"""Build a GenericAgent for WebArena (ReasoningBank-compatible).

Used by PolicyAgent when action_format=webarena_rb. Lazy-imported so the
heavy browsergym/playwright import graph only loads inside the subprocess
where the bridge is already up.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

# reasoning_bank_agent is vendored under envharness/third_party/. Add that dir so
# `import reasoning_bank_agent` resolves.
_RB_PARENT = str(Path(__file__).resolve().parents[1] / "third_party")
if _RB_PARENT not in sys.path:
    sys.path.insert(0, _RB_PARENT)


_LITELLM_PATCHED = False


def _install_litellm_dispatcher(rb_chat_api) -> None:
    global _LITELLM_PATCHED
    if _LITELLM_PATCHED:
        return
    import litellm as _litellm
    from typing import Any, List
    from langchain_core.language_models.chat_models import SimpleChatModel
    from langchain_core.messages import BaseMessage
    from pydantic import PrivateAttr, Field

    def _to_litellm(messages):
        out = []
        for m in messages:
            cls = m.__class__.__name__
            content = m.content if isinstance(m.content, str) else str(m.content)
            role = {"SystemMessage": "system", "HumanMessage": "user",
                    "ChatMessage": "user", "AIMessage": "assistant"}.get(cls, "user")
            out.append({"role": role, "content": content})
        return out

    class ChatLitellm(SimpleChatModel):
        _model: str = PrivateAttr()
        _temperature: float = PrivateAttr()
        _max_tokens: Optional[int] = PrivateAttr()
        n_retry_server: int = Field(default=4)

        def __init__(self, model_name, temperature=0.0, max_tokens=None, **kw):
            super().__init__()
            self._model = model_name.removeprefix("litellm/")
            self._temperature = temperature
            self._max_tokens = max_tokens

        @property
        def _llm_type(self):
            return "litellm"

        def _call(self, messages, stop=None, run_manager=None, **kw):
            for attempt in range(self.n_retry_server):
                try:
                    r = _litellm.completion(
                        model=self._model, messages=_to_litellm(messages),
                        temperature=self._temperature, max_tokens=self._max_tokens,
                        drop_params=True)
                    return r.choices[0].message.content or ""
                except Exception as e:
                    if attempt == self.n_retry_server - 1:
                        raise
                    time.sleep(min(2 ** attempt, 16))

    original = rb_chat_api.ChatModelArgs.make_chat_model

    def patched(self):
        if isinstance(self.model_name, str) and self.model_name.startswith("litellm/"):
            return ChatLitellm(model_name=self.model_name,
                               temperature=self.temperature,
                               max_tokens=self.max_new_tokens)
        return original(self)

    rb_chat_api.ChatModelArgs.make_chat_model = patched
    _LITELLM_PATCHED = True


_TOKENIZER_PATCHED = False


def _install_tokenizer_fallback() -> None:
    global _TOKENIZER_PATCHED
    if _TOKENIZER_PATCHED:
        return
    from reasoning_bank_agent.utils import llm_utils
    from reasoning_bank_agent import dynamic_prompting
    import tiktoken
    _orig_get = llm_utils.get_tokenizer
    _orig_cnt = llm_utils.count_tokens

    def _tok(model_name="openai/gpt-4"):
        if isinstance(model_name, str) and model_name.startswith("litellm/"):
            return tiktoken.encoding_for_model("gpt-4")
        return _orig_get(model_name)

    def _cnt(text, model="openai/gpt-4"):
        if isinstance(model, str) and model.startswith("litellm/"):
            try:
                return len(tiktoken.encoding_for_model("gpt-4").encode(text))
            except Exception:
                return max(1, len(text) // 4)
        return _orig_cnt(text, model)

    llm_utils.get_tokenizer = _tok
    llm_utils.count_tokens = _cnt
    if hasattr(dynamic_prompting, "count_tokens"):
        dynamic_prompting.count_tokens = _cnt
    _TOKENIZER_PATCHED = True


def build_reasoning_bank_agent(model_name: str = "litellm/openai/gpt-4.1-mini",
                   temperature: float = 0.7,
                   max_tokens: int = 65536,
                   memory_path: str | None = None):
    """Build a GenericAgent matching the eval worker's setup."""
    from reasoning_bank_agent.agent import GenericAgentArgs
    from reasoning_bank_agent.dynamic_prompting import Flags
    from reasoning_bank_agent.utils import chat_api as rb_chat_api

    _install_litellm_dispatcher(rb_chat_api)
    _install_tokenizer_fallback()

    flags = Flags(
        use_html=False, use_ax_tree=True, use_thinking=True,
        use_error_logs=True, use_memory=memory_path is not None,
        use_history=True, use_diff=False, use_past_error_logs=True,
        use_action_history=True, multi_actions=False,
        use_abstract_example=True, use_concrete_example=True,
        use_screenshot=False, enable_chat=False, demo_mode="off",
        memory_path=memory_path, action_space="bid",
        html_type="pruned_html",
    )
    chat_args = rb_chat_api.ChatModelArgs(
        model_name=model_name, temperature=temperature,
        max_total_tokens=128_000, max_input_tokens=126_000,
        max_new_tokens=max_tokens,
    )
    return GenericAgentArgs(
        chat_model_args=chat_args, flags=flags, max_retry=4,
    ).make_agent()
