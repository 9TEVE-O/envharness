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

"""Preflight environment check, per benchmark.

Every failure class we've hit was "a dependency silently missing at
runtime" (swebench scorer package absent -> every eval graded fail;
reasoning_bank_agent's google-genai/langchain imports absent -> every webarena episode
dead on policy.act). Run this BEFORE launching a pipeline; it verifies the
CURRENT interpreter can actually run the benchmark.

Usage (run with the SAME python that will run the experiment):
    <alfworld-env>/bin/python scripts/check_env.py alfworld
    <swebench-env>/bin/python scripts/check_env.py swebench
    <webarena-env>/bin/python scripts/check_env.py webarena
    python scripts/check_env.py officeqa
    python scripts/check_env.py spreadsheetbench
    python scripts/check_env.py all      # union of importable checks
"""
from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILS: list[str] = []


def ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def fail(msg: str, hint: str = "") -> None:
    FAILS.append(msg)
    print(f"  [FAIL] {msg}" + (f"\n         fix: {hint}" if hint else ""))


def need_import(mod: str, hint: str) -> bool:
    try:
        importlib.import_module(mod)
        ok(f"import {mod}")
        return True
    except Exception as e:
        fail(f"import {mod}: {type(e).__name__}: {e}", hint)
        return False


def need_keys() -> None:
    """Check the credentials of whatever provider $MODEL names.

    Demanding a Gemini key regardless of provider would stop a GPT-only or
    Vertex-only user at the preflight.
    """
    from envharness.infra.model import (key_pool, missing_key_message,
                                        split_model)
    model = os.environ.get("EH_MODEL") or os.environ.get("MODEL") \
        or "openai/gpt-4.1-mini"
    provider, _ = split_model(model)
    missing = missing_key_message(model)
    if missing:
        fail(missing, f"export the key for {provider or 'your provider'}")
    elif provider == "vertex_ai":
        ok(f"{model}: Vertex uses ADC, no API key needed")
    else:
        ok(f"{model}: {len(key_pool(model))} key(s) available")


def need_docker() -> None:
    if not shutil.which("docker"):
        fail("docker CLI not found", "install docker")
        return
    r = subprocess.run(["docker", "info"], capture_output=True)
    if r.returncode == 0:
        ok("docker daemon reachable")
    else:
        fail("docker daemon unreachable from this process",
             "add user to docker group (usermod -aG docker $USER) or wrap "
             "commands with `sg docker -c ...`")


def check_common() -> None:
    for m in ("pydantic", "yaml", "litellm"):
        need_import(m, "pip install pydantic pyyaml litellm")
    need_keys()


def check_alfworld() -> None:
    check_common()
    if sys.version_info[:2] != (3, 12):
        fail(f"python {sys.version.split()[0]} (alfworld/textworld needs 3.12; "
             "3.13 hits a textworld NameError)",
             "create the env with python=3.12 and run this with its interpreter")
    else:
        ok("python 3.12")
    if need_import("alfworld", 'pip install "alfworld[full]"'):
        need_import("alfworld.agents.environment",
                    'pip install "alfworld[full]" (the [full] extra matters)')
    data = Path(os.environ.get("ALFWORLD_DATA",
                                Path.home() / ".cache/alfworld"))
    if (data / "json_2.1.1").exists():
        ok(f"game data at {data}")
    else:
        fail(f"alfworld game data missing at {data}", "alfworld-download")


def check_swebench() -> None:
    check_common()
    # The scorer runs via `sys.executable -m swebench.harness.run_evaluation`
    # -- swebench MUST be importable by THIS interpreter, or every
    # evaluate() silently grades success=False.
    need_import("swebench", "pip install swebench  (into THIS python env)")
    need_import("datasets", "pip install datasets")
    need_docker()


def check_webarena(stack: bool = True) -> None:
    check_common()
    for m, hint in (
        ("browsergym.core", "pip install browsergym==0.14.1 browsergym-webarena==0.14.1"),
        ("playwright", "pip install playwright && playwright install chromium"),
        # reasoning_bank_agent's import chain (third_party/reasoning_bank_agent/**):
        ("google.genai", "pip install google-genai"),
        ("langchain_core", "pip install langchain-core"),
        ("langchain_community", "pip install langchain_community"),
        ("langchain_anthropic", "pip install langchain_anthropic"),
        ("openai", "pip install openai"),
        ("transformers", "pip install transformers"),
        ("tiktoken", "pip install tiktoken"),
    ):
        need_import(m, hint)
    try:
        import nltk
        nltk.data.find("tokenizers/punkt_tab")
        ok("nltk punkt_tab data")
    except Exception:
        fail("nltk punkt_tab missing/corrupt",
             "python -m nltk.downloader punkt_tab (rm -rf "
             "~/nltk_data/tokenizers/punkt_tab* first if corrupt)")
    # Full reasoning_bank_agent build (catches anything the import list above missed).
    try:
        sys.path.insert(0, str(ROOT))
        from envharness.prompts.webarena_reasoning_bank_agent import build_reasoning_bank_agent
        build_reasoning_bank_agent(model_name="litellm/openai/gpt-4.1-mini",
                       temperature=0.7, max_tokens=64)
        ok("reasoning_bank_agent GenericAgent builds")
    except Exception as e:
        fail(f"reasoning_bank_agent build: {type(e).__name__}: {e}",
             "see experiments/webarena/README.md prereqs")
    if stack:
        import urllib.request
        ports = [19999, 19998, 19997, 17770, 17769, 17768,
                 17780, 17779, 17778, 18023, 18022, 18021]
        bad = []
        for p in ports:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{p}", timeout=5)
            except Exception:
                bad.append(p)
        if bad:
            fail(f"webarena stack ports not serving: {bad}",
                 "IMAGE_DIR=~/webarena_images bash "
                 "experiments/webarena/setup_stack.sh")
        else:
            ok("webarena stack: 12/12 ports serving")


def check_spreadsheetbench() -> None:
    check_common()
    for m in ("openpyxl", "pandas"):
        need_import(m, "pip install openpyxl pandas")
    if shutil.which("soffice") or shutil.which("libreoffice"):
        ok("LibreOffice (soffice) present")
    else:
        fail("soffice not found", "sudo apt install -y libreoffice-calc")
    repo = Path(__file__).resolve().parents[1]
    data = repo / "experiments/spreadsheetbench/data"
    for sub, hint in (
        ("_dl/all_data_912_v0.1",
         "curl -L -o /tmp/sb912.tar.gz https://raw.githubusercontent.com/"
         "RUCKBReasoning/SpreadsheetBench/main/data/spreadsheetbench_912_v0.1.tar.gz"
         " && mkdir -p experiments/spreadsheetbench/data/_dl && tar xzf "
         "/tmp/sb912.tar.gz -C experiments/spreadsheetbench/data/_dl"),
        ("spreadsheetbench_verified_400",
         "curl -L -o /tmp/sb400.tar.gz https://raw.githubusercontent.com/"
         "RUCKBReasoning/SpreadsheetBench/main/data/"
         "spreadsheetbench_verified_400.tar.gz && tar xzf /tmp/sb400.tar.gz "
         "-C experiments/spreadsheetbench/data"),
    ):
        if (data / sub / "dataset.json").is_file():
            ok(f"dataset {sub}")
        else:
            fail(f"dataset missing: {data / sub}", hint)


def check_officeqa() -> None:
    check_common()
    repo = Path(__file__).resolve().parents[1]
    split = repo / "experiments/officeqa/data/officeqa_id_split/test/items.json"
    if split.is_file():
        ok("officeqa id split (ships in the repo)")
    else:
        fail(f"missing {split}", "re-clone the repo; the id split ships with it")
    qa = repo / "experiments/officeqa/data/officeqa_full.csv"
    if qa.is_file():
        ok("officeqa question/answer payload")
    else:
        fail(f"officeqa payload missing at {qa}",
             "the dataset is gated: accept access to databricks/officeqa on HF "
             "and place its officeqa_full.csv there (it is not redistributed here)")
    docs = Path(os.path.expanduser(
        os.environ.get("OFFICEQA_DOCS_DIR")
        or os.environ.get("DOCS_DIR")
        or "~/officeqa/treasury_bulletins_parsed"))
    if docs.is_dir() and any(docs.rglob("*.txt")):
        ok(f"parsed doc corpus at {docs}")
    else:
        fail(f"officeqa parsed docs missing at {docs}",
             "accept the gated databricks/officeqa dataset on HF, materialize "
             "treasury_bulletins_parsed/, then export OFFICEQA_DOCS_DIR")


def check_toy24() -> None:
    check_common()
    ok("toy24 needs nothing else (pure stdlib env)")


CHECKS = {
    "alfworld": check_alfworld,
    "swebench": check_swebench,
    "webarena": check_webarena,
    "spreadsheetbench": check_spreadsheetbench,
    "officeqa": check_officeqa,
    "toy24": check_toy24,
}


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(CHECKS) if target == "all" else [target]
    for n in names:
        if n not in CHECKS:
            print(f"unknown benchmark {n!r}; choose from {list(CHECKS)}")
            return 2
        print(f"[{n}] (interpreter: {sys.executable})")
        CHECKS[n]()
        print()
    if FAILS:
        print(f"{len(FAILS)} problem(s) found -- fix before launching.")
        return 1
    print("environment OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
