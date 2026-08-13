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

"""ReasoningBank `memory_instruction` prompts, SWE-bench wording.

Not ours. From Google Research's ReasoningBank (arXiv:2509.25140), Apache 2.0.
Same prompts as `prompts.py`; the domain word is "software-engineering tasks
(SWE-bench)" instead of the household one.

Imported by experiments/swebench/bank_distillation/induce.py. The paired-diff
prompts in that module are EnvHarness's own and stay there.
"""

SUCCESSFUL_SI = """
You are an expert in software-engineering tasks (SWE-bench). You will be given a problem statement (a GitHub issue) and the corresponding trajectory that represents **how an agent successfully resolved the issue** (the official SWE-bench harness graded its submitted patch as passing).

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's successful trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar bugs -- a future agent should be able to retrieve these lessons and apply them when facing a structurally similar problem.

## Important notes
  - You must first think why the trajectory is successful, and then summarize the insights.
  - You can extract *at most 3* memory items from the trajectory.
  - You must not repeat similar or overlapping items.
  - Prefer concrete, actionable procedures over abstract principles. Do not embed the specific instance_id, repo-specific file paths only useful to this one bug, or literal string contents from the issue.
  - DO mention transferable tools / techniques: `grep -rn`, python heredocs, `pytest <path>::<test>`, where to look for fixture defs, how to read tracebacks, etc.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary describing when or when NOT to use the memory item>
## Content <1-3 sentences describing the insights learned to successfully resolving similar bugs in the future>
```
""".strip()

FAILED_SI = """
You are an expert in software-engineering tasks (SWE-bench). You will be given a problem statement (a GitHub issue) and the corresponding trajectory that represents **how an agent attempted to resolve the issue but failed** (either did not submit, or submitted a patch that the official SWE-bench harness graded as not passing).

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's failed trajectory.
The goal of summarized memory items is to warn a future agent about pitfalls and to suggest recovery strategies -- a future agent facing a structurally similar bug should retrieve these lessons and avoid the same trap.

## Important notes
  - You must first reflect on why the trajectory failed (got stuck searching, submitted a wrong-shape patch, broke unrelated tests, ran out of steps without inspecting the right file, etc.), and then summarize what to do differently.
  - You can extract *at most 3* memory items from the trajectory.
  - You must not repeat similar or overlapping items.
  - Prefer concrete, actionable recovery procedures over abstract principles. Do not embed the specific instance_id, repo-specific file paths only useful to this one bug, or literal string contents from the issue.
  - DO mention transferable failure modes: "tried sed -i on a file with special chars and broke it -- use python heredoc instead", "patched the wrong function because didn't read the failing test's imports first", "submitted before running pytest -- always verify first", etc.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary describing when or when NOT to use the memory item>
## Content <1-3 sentences describing the insights learned to avoid such failures and successfully resolve similar bugs in the future>
```
""".strip()
