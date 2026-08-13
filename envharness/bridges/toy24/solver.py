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

"""Ground-truth solver + puzzle generator for the 24-point game."""
from __future__ import annotations

import random
from itertools import permutations, product
from typing import Iterable

EPS = 1e-6
OPS = ["add", "sub", "mul", "div"]


def _apply(op: str, a: float, b: float) -> float | None:
    if op == "add": return a + b
    if op == "sub": return a - b
    if op == "mul": return a * b
    if op == "div":
        if abs(b) < EPS: return None
        return a / b
    return None


def has_solution(numbers: list[int], target: int = 24,
                  allowed_ops: Iterable[str] = OPS) -> bool:
    """True iff (numbers, allowed_ops) can reach `target`. Brute force; fine
    for 4 numbers.
    """
    allowed = list(allowed_ops)
    if not numbers:
        return False
    if len(numbers) == 1:
        return abs(numbers[0] - target) < EPS

    for i in range(len(numbers)):
        for j in range(len(numbers)):
            if i == j: continue
            rest = [n for k, n in enumerate(numbers) if k not in (i, j)]
            for op in allowed:
                v = _apply(op, numbers[i], numbers[j])
                if v is None: continue
                if has_solution(rest + [v], target, allowed):
                    return True
    return False


# Curated lists of (numbers, target=24) examples by difficulty.
EASY_PUZZLES = [
    [8, 8, 3, 3], [6, 6, 6, 6], [4, 4, 6, 6], [3, 3, 8, 8],
    [2, 4, 4, 8], [1, 5, 5, 5], [2, 2, 6, 8], [3, 4, 4, 6],
]
MEDIUM_PUZZLES = [
    [3, 3, 7, 7], [1, 4, 6, 6], [2, 3, 5, 12], [4, 6, 6, 8],
    [1, 2, 3, 4], [2, 5, 5, 10], [1, 3, 4, 6], [2, 2, 4, 9],
]
HARD_PUZZLES = [
    [3, 3, 8, 8], [1, 3, 4, 6], [3, 7, 8, 9], [4, 7, 8, 8],
    [5, 5, 5, 1], [3, 8, 3, 8], [6, 7, 8, 9], [3, 4, 5, 6],
]
UNSOLVABLE = [
    [1, 1, 1, 1], [2, 2, 2, 2], [1, 1, 1, 2], [1, 1, 2, 3],
]


def generate_puzzle(difficulty: str = "easy", target: int = 24,
                     allow_unsolvable: bool = False,
                     rng: random.Random | None = None) -> list[int]:
    rng = rng or random.Random()
    pool = {
        "easy": EASY_PUZZLES,
        "medium": MEDIUM_PUZZLES,
        "hard": HARD_PUZZLES,
    }.get(difficulty, EASY_PUZZLES)
    if allow_unsolvable and rng.random() < 0.3:
        return list(rng.choice(UNSOLVABLE))
    return list(rng.choice(pool))
