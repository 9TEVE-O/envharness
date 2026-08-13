# EnvHarness RL — ALFWorld GRPO

RL-training adapter that drives [verl-agent](https://github.com/langfengQ/verl-agent)
GRPO on the base `envharness` package's ALFWorld harness (`AlfworldEnv` + `Rules`). The env is
selected purely by `env.env_name=envharness_rl/alfworld`; verl-agent is NOT
checked in — a fetch script reproduces the exact tree (upstream at a pinned
commit + one additive route).

## Environment

- conda env `verl-agent` (Python 3.12; TextWorld's PDDL grammar is not 3.13-safe),
  with verl / vLLM 0.11 / flash-attn installed.
- ALFWorld game data in `~/.cache/alfworld/`.
- GPUs (2 for the smoke, 8 for the full run).

```bash
# env layer loads (no GPU)
PYTHONPATH=..:. \
  ~/miniconda3/envs/verl-agent/bin/python scripts/smoke_worker.py
```

## Getting verl-agent (clone + patch)

verl-agent is not part of this repository — you reproduce the exact tree the
experiments ran against by cloning upstream at a pinned commit and applying
our patch. The one-command way:

```bash
bash rl/scripts/fetch_verl_agent.sh
```

That clones [verl-agent](https://github.com/langfengQ/verl-agent) at commit
`796ed310287fa605c9292a0fce07a86d79fde05e` into `third_party/verl-agent/`
(gitignored) and applies `rl/integration/verl_agent_env_manager.patch` — the
single additive change (the `envharness_rl/alfworld` env route). It is
idempotent: re-running on an already-patched tree is a no-op.

Equivalent manual steps, if you'd rather drive git yourself or place the
tree elsewhere (then point `$VERL_AGENT` at it):

```bash
git clone https://github.com/langfengQ/verl-agent third_party/verl-agent
cd third_party/verl-agent
git checkout 796ed310287fa605c9292a0fce07a86d79fde05e
git apply ../../rl/integration/verl_agent_env_manager.patch
```

Two patches are available (see `rl/integration/ENVHARNESS_CHANGES.md` for
the per-file breakdown):

| Patch | What it applies | When |
|---|---|---|
| `verl_agent_env_manager.patch` | the env route only — the single live change | default; all you need for the ALFWorld runs |
| `verl_agent_all_changes.patch` | superset: env route + DAPO / Qwen3-8B / webshop / SWE-Gym adaptations | `PATCH=all bash rl/scripts/fetch_verl_agent.sh`, or apply manually INSTEAD of the default |

Apply exactly one of the two — they overlap, so applying both fails.

## Run

```bash
# 2-GPU GRPO smoke: Qwen2.5-1.5B, 2 epochs, on the 6 bundled mutated games.
bash scripts/run_grpo.sh

# unmutated control (no corpus, full TRAIN)
MUTATION_CORPUS= TRAIN_SUBSET_PATH= bash scripts/run_grpo.sh

# full 8-GPU run (Qwen3-8B)
MODE=full bash scripts/run_grpo.sh
```

Output lands under `runs/grpo_envrl_alfworld_<mode>_<ts>/`. `val_before_train`
reports per-task-type success rates; each step logs `actor/pg_loss`,
`episode/reward/mean`, `episode/success_rate`.

## Knobs (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `MODE` | `smoke` | `smoke` (Qwen2.5-1.5B, 2 GPU) or `full` (Qwen3-8B, 8 GPU) |
| `MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | policy model |
| `MUTATION_CORPUS` | bundled `example_corpus.jsonl` | `{game_file, rules_code, in_env_actions}` per line; empty = unmutated |
| `ALFWORLD_DATA` | -- | ALFWorld data root; the bundled JSONLs list `game_file` relative to it |
| `TRAIN_SUBSET_PATH` | bundled `train_subset.jsonl` | restrict TRAIN to these games; empty = full TRAIN |
| `VERL_AGENT` | `third_party/verl-agent` | which verl-agent to run (populate with `scripts/fetch_verl_agent.sh`) |
| `VLLM_ATTENTION_BACKEND` | `FLASH_ATTN` | keep FLASH_ATTN on vLLM 0.11 (XFORMERS V1 needs block_size % 256) |
| `N_GPUS` / `TP` | `2 / 2` (smoke) | GPUs / tensor-parallel |

## Files

| File | Purpose |
|---|---|
| `scripts/run_grpo.sh` | GRPO launcher (smoke / full) |
| `scripts/fetch_verl_agent.sh` | fetch verl-agent @ pinned commit + apply the env route patch |
| `scripts/smoke_worker.py` | no-GPU env sanity check |
| `scripts/build_corpus.py` | build the example corpus + subset from a legacy corpus |
| `envharness_rl/alfworld/envs.py` | Ray-actor parallel `AlfworldEnv` + `Rules` workers |
| `envharness_rl/alfworld/projection.py` | `<action>`/`<think>` extraction + admissible-command normalize |
| `experiments/alfworld/data/` | `example_corpus.jsonl` (6 mutated games) + `train_subset.jsonl` |
| `../third_party/verl-agent/` | fetched verl-agent (gitignored; upstream + the `envharness_rl/alfworld` route) |
| `integration/ENVHARNESS_CHANGES.md` | what differs from upstream |
| `integration/verl_agent_env_manager.patch` | the single live change (the env route) — applied by the fetch script |
| `integration/verl_agent_all_changes.patch` | full patch (re-enables DAPO / Qwen3-8B / webshop / SWE-Gym) |

## Acknowledgements

The RL experiments run on the third-party
[**verl-agent**](https://github.com/langfengQ/verl-agent) repository (GiGPO;
Apache-2.0), at upstream commit `796ed31` — fetched by
`scripts/fetch_verl_agent.sh`, not redistributed here. All RL training /
rollout infrastructure is theirs — we only add the additive
`envharness_rl/alfworld` env route (see `integration/ENVHARNESS_CHANGES.md`).
verl-agent in turn builds on [verl](https://github.com/volcengine/verl).
Please cite/credit them when using this.
