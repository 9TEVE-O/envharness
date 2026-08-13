# EnvHarness modifications to verl-agent

These notes document how the tree at `third_party/verl-agent/` differs from
[verl-agent](https://github.com/langfengQ/verl-agent) upstream `master`
@ `796ed310287fa605c9292a0fce07a86d79fde05e`. verl-agent is **not checked
into this repository** — `rl/scripts/fetch_verl_agent.sh` clones upstream at
that commit and applies the patch, reproducing the tree the RL experiments
ran against.

**The default fetch applies a SINGLE change vs upstream — `env_manager.py`
(the env route), via `rl/integration/verl_agent_env_manager.patch`.** For the
default Qwen2.5-1.5B ALFWorld example that is all that is needed. The other
adaptations below (Qwen3-8B / DAPO / webshop / SWE-Gym) are preserved in
`rl/integration/verl_agent_all_changes.patch` — fetch with `PATCH=all` to
apply them instead.

Below is every change we originally made relative to upstream (10 files);
only the first (`env_manager.py`) is applied by the default fetch.

## Environment integration (the core adaptation — LIVE)

| file | change |
|---|---|
| `agent_system/environments/env_manager.py` | **Additive env routes** selected by `env.env_name`, all inside the existing `alfworld` / new branches — they build EnvHarness-wrapped envs and reuse the stock EnvironmentManagers:<br>• `envharness_rl/alfworld` → release_version `AlfworldEnv` + `Rules` (via `release_version_rl`)<br>• `envharness/alfworld` → legacy `AlfworldBridge` + `MutationLayer`<br>• `envharness_webshop` → `envharness.bridges.webshop.WebshopBridge`<br>• `envharness_swegym` / `envharness_swegym_openhands` → SWE-Gym bridges<br>Legacy/stock routes are left untouched. |
| `agent_system/environments/prompts/alfworld.py` | Added non-task-specific anti-loop **tips** to the ALFWorld prompt templates (nudges against repeated/ineffective commands; +~10pp SR for Qwen3-8B non-thinking OOD). |
| `agent_system/environments/env_package/webshop/projection.py` | Added a lightweight **`WS_PROJ_DIAG` diagnostic print** (running valid_action_ratio + chinese_rate), since the webshop env manager doesn't report valid_action_ratio. No effect on returned actions/valids. |

## Rollout

| file | change |
|---|---|
| `agent_system/multi_turn_rollout/rollout_loop.py` | (1) `ENVHARNESS_DISABLE_THINKING=1` → passes `chat_template_kwargs={"enable_thinking": False}` for Qwen3 non-thinking rollouts. (2) Chat-path support: when the env manager returns a multi-turn message **list** (not a flat string), use it directly as the chat so training is in-distribution for chat-SFT'd models (used by the SWE-Gym/OpenHands path). Plain-list handling (not np.array) to avoid `apply_chat_template` truthiness errors. |

## Trainer (DAPO + large-model support)

| file | change |
|---|---|
| `verl/trainer/ppo/ray_trainer.py` | **DAPO dynamic sampling** in the agentic `fit()` loop: after reward, keep only prompt-groups with reward std > 0, accumulate across gen-batches up to `max_num_gen_batches`, fall back to the last unfiltered batch if none are informative. |
| `verl/trainer/constants_ppo.py` | DAPO-related constants. |
| `verl/workers/fsdp_workers.py` | transformers-5.x fallback: `AutoModelForVision2Seq` → try/except `AutoModelForImageTextToText` (Qwen3-class models). |
| `verl/utils/checkpoint/fsdp_checkpoint_manager.py` | Same tf5 `AutoModelForImageTextToText` fallback on checkpoint save/load. |
| `verl/utils/vllm_utils.py` | Small util patch for Qwen3 / newer vLLM. |
| `verl/workers/sharding_manager/fsdp_vllm.py` | FSDP→vLLM weight-sync patch for the above. |

## Not required for the basic examples

For the default **Qwen2.5-1.5B ALFWorld example** (`release_version_rl`), only the
`env_manager.py` route is strictly required. The `rollout_loop.py`,
`prompts/alfworld.py`, and `verl/*` changes matter only for Qwen3-8B / DAPO /
webshop / SWE-Gym runs.

The full diff vs upstream is also saved at
`release_version_rl/integration/verl_agent_all_changes.patch` in the EnvHarness repo.
