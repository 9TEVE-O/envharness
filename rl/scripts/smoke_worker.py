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

#!/usr/bin/env python
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

"""No-GPU smoke for the rl/ ALFWorld worker.

Instantiates EnvharnessAlfworldWorker against the bundled example corpus,
resets onto a mutated game, steps a few admissible commands, and checks the
returned (text, score, done, info) shape. Run before the GPU GRPO smoke.

  PYTHONPATH=..:.  ~/miniconda3/envs/verl-agent/bin/python scripts/smoke_worker.py
"""
from __future__ import annotations

import os
from pathlib import Path

RL_ROOT = Path(__file__).resolve().parents[1]
ROOT = RL_ROOT.parent
DATA = RL_ROOT / "experiments" / "alfworld" / "data"
# $VERL_AGENT wins so an external checkout can be used; otherwise the vendored
# copy under third_party/.
VERL_AGENT = Path(os.environ.get("VERL_AGENT")
                  or ROOT / "third_party" / "verl-agent")


def main() -> None:
    # Default to the vendored verl-agent's ALFWorld config.
    os.environ.setdefault(
        "ALFWORLD_CONFIG",
        str(VERL_AGENT / "agent_system" / "environments"
            / "env_package" / "alfworld" / "configs" / "config_tw.yaml"))

    from envharness_rl.alfworld.envs import EnvharnessAlfworldWorker

    w = EnvharnessAlfworldWorker(
        seed=0, split="train", repetition_threshold=0, obs_style="raw",
        train_subset_path=str(DATA / "train_subset.jsonl"),
        mutation_corpus_path=str(DATA / "example_corpus.jsonl"),
        subset_authoritative=False)

    text, info = w.reset()
    gf = info["extra.gamefile"][0]
    mutated = info["mutation_active"][0]
    admissible = info["admissible_commands"][0]
    assert "Your task is to:" in text, f"task line missing:\n{text[:300]}"
    assert isinstance(admissible, list) and admissible, "no admissible commands"
    print(f"[reset] gamefile={gf}")
    print(f"[reset] mutation_active={mutated}  n_admissible={len(admissible)}")
    if info.get("mutation_error"):
        print(f"[reset] WARNING mutation_error={info['mutation_error'][0]}")
    print(f"[reset] obs head: {text[:200]!r}")

    for i in range(3):
        act = admissible[0] if admissible else "look"
        text, score, done, info = w.step(act)
        won = info["won"][0]
        admissible = info["admissible_commands"][0]
        assert isinstance(won, bool), f"won not bool: {won!r}"
        assert isinstance(score, float), f"score not float: {score!r}"
        print(f"[step {i}] act={act!r} score={score} done={done} won={won} "
              f"n_admissible={len(admissible)}")
        if done:
            break

    w.close()
    print("SMOKE OK")


if __name__ == "__main__":
    main()
