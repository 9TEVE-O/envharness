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

"""Text embedding via litellm, provider-agnostic.

Going through litellm (rather than a provider SDK) means the same model
string, key handling and retry policy used for chat applies to embeddings:
`gemini/gemini-embedding-001`, `openai/text-embedding-3-small`, ...

The model is resolved in this order:
  1. the `model=` argument
  2. `$EH_EMBED_MODEL`
  3. the default paired with `$EH_MODEL`'s provider
  4. `openai/text-embedding-3-small`

A bank stores its vectors, so the embedding model is part of its on-disk
format: querying a bank with a different model raises on the dimension
mismatch (see `cosine`). Rebuild the bank after changing it.
"""
from __future__ import annotations
import os
import time
import litellm

DEFAULT_MODEL = "openai/text-embedding-3-small"
EMBED_BATCH = 10                # gemini-embedding-001 caps per-call inputs
EMBED_RETRY = 5


def default_embed_model() -> str:
    """Embedding model for the currently configured provider."""
    from envharness.infra.model import embedding_model
    return embedding_model(os.environ.get("EH_MODEL", ""))


def embed_texts(texts: list[str], model: str | None = None,
                 batch_size: int = EMBED_BATCH) -> list[list[float]]:
    """Embed a list of texts; returns one vector per input (in order)."""
    model = model or default_embed_model()
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        for attempt in range(EMBED_RETRY):
            try:
                from envharness.infra.model import embedding_kwargs
                r = litellm.embedding(input=chunk,
                                       **embedding_kwargs(model=model))
                # Parse the WHOLE chunk into a local list first; only extend
                # `out` once parsing succeeded. Appending inside the loop meant
                # a mid-iteration failure re-appended the chunk on retry,
                # silently misaligning every later embedding with its text.
                local = [d["embedding"] for d in r.data]
                out.extend(local)
                break
            except Exception as e:
                if attempt == EMBED_RETRY - 1:
                    raise
                time.sleep(2 ** attempt)
    return out


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(
            f"embedding dim mismatch: {len(a)} vs {len(b)} "
            f"(bank built with a different embedding model?)"
        )
    s = na = nb = 0.0
    for x, y in zip(a, b):
        s += x * y; na += x * x; nb += y * y
    return s / max((na * nb) ** 0.5, 1e-9)
