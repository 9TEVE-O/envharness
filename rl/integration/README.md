# verl-agent integration

`release_version_rl` plugs into verl-agent through a single ADDITIVE route in:

    third_party/verl-agent/agent_system/environments/env_manager.py

inside the `elif "alfworld" in config.env.env_name.lower():` branch. The new
route fires first:

```python
if config.env.env_name.lower().startswith("envharness_rl/alfworld"):
    # put release_version/ AND release_version_rl/ on sys.path
    from envharness_rl.alfworld import (
        build_envharness_alfworld_envs, envharness_alfworld_projection)
    ...
    return envs, val_envs
```

It is intentionally separate from the legacy `envharness/alfworld` route that
follows it -- active training runs depend on the legacy path, so it is left
untouched. Selecting between them is purely `env.env_name`:

| `env.env_name`            | env layer |
|---------------------------|-----------|
| `envharness_rl/alfworld`  | release_version `AlfworldEnv` + `Rules` (this package) |
| `envharness/alfworld`     | legacy `AlfworldBridge` + `MutationLayer` |
| `alfworld/AlfredTWEnv`    | stock verl-agent alfworld |

## Env vars the route reads

| var | effect |
|---|---|
| `ALFWORLD_TRAIN_SUBSET_PATH` | JSONL of `{"game_file": ...}`; restricts TRAIN. Train-only. |
| `ENVHARNESS_MUTATION_CORPUS` | JSONL of `{game_file, rules_code, in_env_actions}`; applies the matching `Rules` per train episode. Train-only. |
| `ENVHARNESS_SUBSET_AUTHORITATIVE` | `1/true/yes` -> treat the subset as authoritative (load game COPIES outside alfworld's scanned dir). Train-only. |

`scripts/run_grpo.sh` sets these from the bundled example data by default.

> Attribution: the `.patch` files in this directory are diffs against
> [verl-agent](https://github.com/langfengQ/verl-agent) (Apache-2.0) and
> therefore contain verbatim upstream context lines alongside our additions.

## Fetching verl-agent

verl-agent is not checked into this repository. `scripts/fetch_verl_agent.sh`
clones upstream at the pinned commit (`796ed310`) into
`third_party/verl-agent/` (gitignored) and applies
`verl_agent_env_manager.patch` — the additive block above, and the ONLY live
change vs upstream. `PATCH=all` applies `verl_agent_all_changes.patch`
instead (a superset: DAPO / Qwen3-8B / webshop / SWE-Gym support; see
`ENVHARNESS_CHANGES.md` for the per-file breakdown).
