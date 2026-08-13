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

"""Single-task WebArena eval worker using GenericAgent (browsergym RB agent).

Usage:
  PY=<your-webarena-env>/bin/python
  $PY experiments/webarena/worker.py \
      --task-id 123 --container forum_1 --container-url http://127.0.0.1:19998 \
      --bank path/to/bank.jsonl --top-k 5 --out results.jsonl
"""
from __future__ import annotations
import argparse, fcntl, json, os, signal, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
# the vendored ReasoningBank agent
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party"))

from envharness.bridges.webarena.bridge import WebArenaEnv
from envharness.core.types import Action
from envharness.reasoning_bank import Bank

# ---------------------------------------------------------------------------
# Container management
# ---------------------------------------------------------------------------

_WA_DEFAULTS = {
    "WA_REDDIT": "http://127.0.0.1:19999",
    "WA_SHOPPING": "http://127.0.0.1:17770",
    "WA_SHOPPING_ADMIN": "http://127.0.0.1:17780/admin",
    "WA_GITLAB": "http://127.0.0.1:18023",
    "WA_WIKIPEDIA": "http://stub", "WA_MAP": "http://stub",
    "WA_HOMEPAGE": "http://stub",
}
_SITE_KEY = {
    "forum": ("WA_REDDIT", "REDDIT"),
    "shopping_admin": ("WA_SHOPPING_ADMIN", "SHOPPING_ADMIN"),
    "shopping": ("WA_SHOPPING", "SHOPPING"),
    "gitlab": ("WA_GITLAB", "GITLAB"),
}


def restart_container(name: str, url: str) -> None:
    subprocess.run(["docker", "restart", name], capture_output=True, timeout=60)
    is_gitlab = "gitlab" in name
    max_wait = 240 if is_gitlab else 120
    import urllib.error
    for _ in range(max_wait):
        try:
            try:
                resp = urllib.request.urlopen(url, timeout=5)
            except urllib.error.HTTPError:
                time.sleep(2 if is_gitlab else 1)
                continue
            body = resp.read().decode("utf-8", errors="ignore")
            if is_gitlab:
                if "username" not in body.lower() and "password" not in body.lower():
                    time.sleep(2)
                    continue
                # Double-check: try loading the sign-in page directly
                try:
                    resp2 = urllib.request.urlopen(url + "/users/sign_in", timeout=5)
                    body2 = resp2.read().decode("utf-8", errors="ignore")
                    if "username" not in body2.lower():
                        time.sleep(5)
                        continue
                except Exception:
                    time.sleep(5)
                    continue
                time.sleep(10)
                return
            settle = 5 if "shopping_admin" in name else 1
            time.sleep(settle)
            return
        except Exception:
            time.sleep(2 if is_gitlab else 1)
    time.sleep(8)


def setup_env(container_name: str, container_url: str) -> None:
    base = container_name
    if container_name.count("_") >= 1 and container_name.split("_")[-1].isdigit():
        base = "_".join(container_name.split("_")[:-1]) if container_name.count("_") >= 2 else container_name.split("_")[0]
    for prefix, (wa_key, short) in _SITE_KEY.items():
        if base == prefix:
            url = container_url + "/admin" if prefix == "shopping_admin" else container_url
            os.environ[wa_key] = url
            os.environ[short] = url
            break
    for k, v in _WA_DEFAULTS.items():
        os.environ.setdefault(k, v)
        os.environ.setdefault(k[3:], v)


# ---------------------------------------------------------------------------
# Litellm dispatcher: patches ChatModelArgs to route "litellm/" models
# ---------------------------------------------------------------------------

_LITELLM_PATCHED = False

def _install_litellm_dispatcher(rb_chat_api) -> None:
    global _LITELLM_PATCHED
    if _LITELLM_PATCHED:
        return
    import litellm as _litellm
    from typing import Any, List, Optional
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
        def _llm_type(self): return "litellm"

        def _call(self, messages, stop=None, run_manager=None, **kw):
            # Route through the resolver rather than calling litellm directly:
            # it applies $EH_MODEL (so `MODEL=...` reaches this agent, which
            # the dispatcher has no flag to pass), the provider's auth, and the
            # per-provider parameter fixes.
            from envharness.infra.model import completion_kwargs
            for attempt in range(self.n_retry_server):
                try:
                    r = _litellm.completion(
                        messages=_to_litellm(messages),
                        **completion_kwargs(self._model,
                                            temperature=self._temperature,
                                            max_tokens=self._max_tokens))
                    return r.choices[0].message.content or ""
                except Exception as e:
                    if attempt == self.n_retry_server - 1: raise
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
            try: return len(tiktoken.encoding_for_model("gpt-4").encode(text))
            except: return max(1, len(text) // 4)
        return _orig_cnt(text, model)

    llm_utils.get_tokenizer = _tok
    llm_utils.count_tokens = _cnt
    if hasattr(dynamic_prompting, "count_tokens"):
        dynamic_prompting.count_tokens = _cnt
    _TOKENIZER_PATCHED = True


# ---------------------------------------------------------------------------
# Per-episode eval (uses GenericAgent from browsergym ReasoningBank)
# ---------------------------------------------------------------------------

def run_single(task_id: int, container_name: str, container_url: str,
               bank: Bank | None, top_k: int, model: str, temperature: float,
               max_tokens: int, max_steps: int,
               skip_restart: bool = False) -> dict:
    if not skip_restart:
        restart_container(container_name, container_url)
    setup_env(container_name, container_url)

    from reasoning_bank_agent.agent import GenericAgentArgs
    from reasoning_bank_agent.dynamic_prompting import Flags
    from reasoning_bank_agent.utils import chat_api as rb_chat_api
    ChatModelArgs = rb_chat_api.ChatModelArgs
    _install_litellm_dispatcher(rb_chat_api)
    _install_tokenizer_fallback()

    try:
        signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError()))
        signal.alarm(300)
    except Exception:
        pass

    t0 = time.time()
    rec = {"task_id": task_id, "success": False, "duration_steps": 0,
           "duration_ms": 0, "final_reward": 0.0, "error": "",
           "retrieved_titles": []}
    env = WebArenaEnv()
    tmp_path = None
    try:
        reset = env.reset(seed=task_id, options={
            "task_id": task_id, "headless": True, "obs_style": "wrapped"})

        memory_path = None
        if bank and top_k > 0:
            intent = (reset.info or {}).get("goal", "")
            if not intent:
                intent = (reset.observation.text or "")[:500]
            _mode = os.environ.get("WEBARENA_RETRIEVAL_MODE", "cosine")
            retrieved = bank.retrieve(intent, k=top_k, mode=_mode)
            rec["retrieved_titles"] = [it.title for it in retrieved]
            if retrieved:
                tmp_path = tempfile.mktemp(suffix=".txt")
                with open(tmp_path, "w") as f:
                    f.write("\n\n".join(it.text for it in retrieved) + "\n")
                memory_path = tmp_path

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
        chat_args = ChatModelArgs(
            model_name=model, temperature=temperature,
            max_total_tokens=128_000, max_input_tokens=126_000,
            max_new_tokens=max_tokens,
        )
        agent = GenericAgentArgs(
            chat_model_args=chat_args, flags=flags, max_retry=4,
        ).make_agent()
        raw_obs = env._latest_obs

        for step in range(max_steps):
            processed = agent.obs_preprocessor(raw_obs)
            action_str = None
            for _retry in range(3):
                hist_len = len(getattr(agent, "obs_history", []) or [])
                try:
                    action_str, _ = agent.get_action(processed)
                    break
                except Exception as e:
                    # GenericAgent.get_action appends the obs to
                    # agent.obs_history BEFORE the LLM call, so a failed
                    # call leaves a dangling obs; retrying would re-append
                    # the same obs and History.__init__'s
                    # len(history_obs) == len(actions) + 1 assert fails
                    # deterministically. Pop the just-appended obs first.
                    hist = getattr(agent, "obs_history", None)
                    if hist is not None and len(hist) > hist_len:
                        del hist[hist_len:]
                    if _retry == 2:
                        rec["error"] = f"get_action: {type(e).__name__}: {e}"
                    else:
                        time.sleep(2 ** _retry)
            if rec["error"]:
                break
            if not action_str:
                rec["error"] = "empty action"
                break

            env_resp = env.step(Action(name="do", kwargs={"action_str": action_str}))
            rec["final_reward"] = float(env_resp.reward or 0.0)
            rec["success"] = bool((env_resp.info or {}).get("won", False))
            raw_obs = env._latest_obs
            rec["duration_steps"] = step + 1
            if env_resp.terminated or env_resp.truncated:
                break
    except TimeoutError:
        rec["error"] = "timeout"
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    finally:
        try: signal.alarm(0)
        except: pass
        try: env.close()
        except: pass
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
    rec["duration_ms"] = int((time.time() - t0) * 1000)
    return rec


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--container", required=True)
    p.add_argument("--container-url", required=True)
    p.add_argument("--bank", default=None)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--model", default="litellm/openai/gpt-4.1-mini")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-response-tokens", type=int, default=65536)
    p.add_argument("--out", required=True)
    p.add_argument("--skip-restart", action="store_true")
    args = p.parse_args()

    bank = Bank.load(args.bank) if args.bank else None

    rec = run_single(
        task_id=args.task_id, container_name=args.container,
        container_url=args.container_url, bank=bank, top_k=args.top_k,
        model=args.model, temperature=args.temperature,
        max_tokens=args.max_response_tokens, max_steps=args.max_steps,
        skip_restart=args.skip_restart)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(json.dumps(rec) + "\n")
        f.flush()
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    status = "OK" if rec["success"] else "FAIL"
    print(f"[task {args.task_id}] {status} steps={rec['duration_steps']} "
          f"err={rec.get('error','')}", flush=True)


if __name__ == "__main__":
    main()
