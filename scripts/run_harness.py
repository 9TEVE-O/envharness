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

"""Entry point for orchestrator runs.

Two modes:
  --config <path>           load YAML and run with LiteLLM-backed agents
  --scripted                offline plumbing test (no API keys, ScriptedClient)

Common options:
  --run-name NAME           Directory name under ./runs/ (default: derived
                            from config filename + timestamp). The config's
                            {run_name} placeholders are substituted.
  --in-process              With --scripted, use InProcessRunner (skip subprocess).

YAML schema (excerpt)::

    env:
      import_path: envharness.bridges.toy24:Toy24Env
      reset_options: { ... }
    policy:
      client_factory: envharness.infra.llm:LiteLLMClient
      client_kwargs: { model: openai/gpt-4.1-mini }
      action_format: function_calling
    agent:
      type: llm           # or "noop" / "explore"
      client_factory: envharness.infra.llm:LiteLLMClient
      client_kwargs: { ... }
    objective:
      type: difficulty_zone
      target_band: [0.4, 0.6]
    budget:
      type: capped_adaptive
      max_k: 2
    orchestrator: { ... }
    runner: { type: subprocess, timeout_seconds: 600 }
    storage: { trace_path: ./runs/{run_name}/traces.jsonl }
    logging: { log_dir: ./runs/{run_name} }
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envharness.agents.harness_agent import (
    ExploringHarnessAgent,
    LLMHarnessAgent,
    NoopHarnessAgent,
    ScriptedHarnessAgent,
)
from envharness.bridges.toy24 import Toy24Env
from envharness.core.types import (
    Candidate, DecideResult, Decision, FailureAnalysis,
)
from envharness.infra.llm import LiteLLMClient, LoggingLLMClient
from envharness.infra.utils import import_symbol
from envharness.orchestration.budget import CappedAdaptive, FixedBudget, ObjectiveDriven
from envharness.orchestration.objectives import DifficultyZone, RedTeam
from envharness.orchestration.orchestrator import Orchestrator, OrchestratorConfig
from envharness.orchestration.runner import (
    EnvSpec, EpisodeRunner, InProcessRunner, PolicySpec, SubprocessRunner,
)
from envharness.orchestration.storage import TraceStore


# ---------------------------------------------------------------------------
# YAML config -> Orchestrator (with logging wired in)
# ---------------------------------------------------------------------------

def _substitute_run_name(config: dict, run_name: str) -> dict:
    """Replace `{run_name}` placeholders throughout the config tree."""
    def _walk(node):
        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(x) for x in node]
        if isinstance(node, str):
            return node.replace("{run_name}", run_name)
        return node
    return _walk(config)


def _resolve_env_block(cfg: dict) -> dict:
    """Accept either the new `env:` block or the legacy `bridge:` block."""
    if "env" in cfg:
        return cfg["env"]
    if "bridge" in cfg:
        return cfg["bridge"]
    raise KeyError("config must have an `env:` (or legacy `bridge:`) block.")


def _client_from_block(block: dict) -> tuple[str, dict]:
    """(client_factory, client_kwargs) for a `policy:` / `agent:` block.

    Two accepted shapes. The short one names a model and lets
    `envharness.infra.model` supply the provider's auth and parameter
    handling, so the same block works on any provider:

        policy:
          model: openai/gpt-4.1          # or gemini/... or vertex_ai/claude-...
          client_kwargs: {...}           # optional extras, merged in

    The explicit one names the client class itself, for a backend the
    resolver does not cover (a local vLLM endpoint, CompletionAPIClient, ...):

        policy:
          client_factory: envharness.infra.llm:CompletionAPIClient
          client_kwargs: {...}
    """
    from envharness.infra.model import CLIENT_FACTORY, client_spec, effective_model

    extras = dict(block.get("client_kwargs") or {})
    factory = block.get("client_factory")
    named = block.get("model") or extras.get("model")
    # Route through the resolver whenever the block names a model the resolver
    # can serve. Doing this only when a run-wide override *changes* the model
    # would leave the un-overridden case with whatever kwargs the config spells
    # out by hand -- which is how a config lost `drop_params` and died on a
    # `reasoning_effort` the target model does not take.
    if named and (factory in (None, CLIENT_FACTORY)
                  or effective_model(named) != named):
        extras.pop("model", None)
        return client_spec(named, **extras)
    if factory:
        return factory, extras
    raise KeyError(
        "config block needs either `model:` (recommended) or "
        "`client_factory:` + `client_kwargs:`"
    )


def _resolve_agent_block(cfg: dict) -> dict:
    """Accept either the new `agent:` block or the legacy `mutator:` block."""
    if "agent" in cfg:
        return cfg["agent"]
    if "mutator" in cfg:
        return cfg["mutator"]
    raise KeyError("config must have an `agent:` (or legacy `mutator:`) block.")


def build_from_config(cfg_path: Path, run_name: str,
                       overrides: dict | None = None,
                       model: str | None = None) -> Orchestrator:
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg = _substitute_run_name(cfg, run_name)

    # `--model` replaces the model on BOTH roles, so a driver can drive a whole
    # run onto one provider without editing the config. It clears any
    # `client_factory`/`client_kwargs.model` the block carried, otherwise the
    # explicit factory would win and the override would silently do nothing.
    if model:
        for block_name in ("policy", "mutator", "agent"):
            block = cfg.get(block_name)
            if not isinstance(block, dict):
                continue
            block["model"] = model
            block.pop("client_factory", None)
            (block.get("client_kwargs") or {}).pop("model", None)

    log_cfg = cfg.get("logging") or {}
    log_dir = Path(log_cfg["log_dir"]) if log_cfg.get("log_dir") else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Raw pre-substitution / pre-override copy. The FINAL resolved
            # config is dumped as config_used.yaml further down, once CLI
            # overrides have been merged into the orchestrator block.
            shutil.copy(cfg_path, log_dir / "config_original.yaml")
        except Exception:
            pass

    env_block = _resolve_env_block(cfg)
    env_spec = EnvSpec(
        import_path=env_block["import_path"],
        reset_options=env_block.get("reset_options") or {},
        reset_seed=env_block.get("reset_seed"),
    )

    # Policy LLM client config -- if logging requested, wrap with LoggingLLMClient.
    policy_cf, policy_ck = _client_from_block(cfg["policy"])
    if log_dir and log_cfg.get("log_policy_calls", True):
        policy_cf, policy_ck = "envharness.infra.llm:LoggingLLMClient", {
            "inner_factory": policy_cf,
            "inner_kwargs": policy_ck,
            "log_path": str(log_dir / "policy_calls.jsonl"),
            "role": "policy",
        }
    policy_prompt = (cfg["policy"].get("task_description")
                     or cfg["orchestrator"]["task_description"])
    policy_spec_kwargs = dict(
        client_factory=policy_cf,
        client_kwargs=policy_ck,
        action_format=cfg["policy"].get("action_format", "function_calling"),
        task_prompt=policy_prompt,
    )
    if "max_history" in cfg["policy"]:
        policy_spec_kwargs["max_history"] = int(cfg["policy"]["max_history"])
    if "temperature" in cfg["policy"]:
        policy_spec_kwargs["temperature"] = float(cfg["policy"]["temperature"])
    if "prompt_builder_kwargs" in cfg["policy"]:
        policy_spec_kwargs["prompt_builder_kwargs"] = (
            cfg["policy"].get("prompt_builder_kwargs") or {}
        )
    policy_spec = PolicySpec(**policy_spec_kwargs)

    # HarnessAgent: type "llm" (default) wires LLMHarnessAgent; "noop" gives
    # the pass-through baseline. "explore" wires ExploringHarnessAgent.
    agent_block = _resolve_agent_block(cfg)
    agent_type = (agent_block.get("type") or "llm").lower()
    if agent_type == "noop":
        harness_agent = NoopHarnessAgent()
    elif agent_type in ("llm", "explore"):
        agent_cf, agent_ck = _client_from_block(agent_block)
        if log_dir and log_cfg.get("log_agent_calls",
                                     log_cfg.get("log_mutator_calls", True)):
            agent_client = LoggingLLMClient(
                inner_factory=agent_cf,
                inner_kwargs=agent_ck,
                log_path=str(log_dir / "agent_calls.jsonl"),
                role="harness_agent",
            )
        else:
            agent_client = import_symbol(agent_cf)(**agent_ck)
        if agent_type == "llm":
            harness_agent = LLMHarnessAgent(
                client=agent_client,
                system_prompt=agent_block.get("system_prompt") or "",
                extra_instructions=agent_block.get("extra_instructions") or "",
            )
        else:  # "explore"
            harness_agent = ExploringHarnessAgent(
                client=agent_client,
                env_spec=env_spec,
                max_explore_steps=int(agent_block.get("max_explore_steps", 10)),
                explore_temperature=float(agent_block.get("explore_temperature", 0.7)),
                system_prompt=agent_block.get("system_prompt") or None,
            )
    else:
        raise ValueError(f"Unknown agent.type: {agent_type!r}")

    obj_cfg = cfg["objective"]
    if obj_cfg["type"] == "difficulty_zone":
        objective = DifficultyZone(
            target_band=tuple(obj_cfg.get("target_band", [0.3, 0.7])),
            window=obj_cfg.get("window", 10),
        )
    elif obj_cfg["type"] == "red_team":
        objective = RedTeam(window=obj_cfg.get("window", 10))
    else:
        raise ValueError(obj_cfg["type"])

    bcfg = cfg["budget"]
    if bcfg["type"] == "fixed":             budget = FixedBudget(bcfg["k"])
    elif bcfg["type"] == "capped_adaptive": budget = CappedAdaptive(max_k=bcfg["max_k"])
    elif bcfg["type"] == "objective_driven":
        budget = ObjectiveDriven(score_threshold=bcfg["score_threshold"],
                                  max_k=bcfg.get("max_k", 20))
    else:
        raise ValueError(bcfg["type"])

    rcfg = cfg.get("runner") or {"type": "subprocess"}
    if rcfg["type"] == "subprocess":
        runner: EpisodeRunner = SubprocessRunner(
            timeout=rcfg.get("timeout_seconds", 600.0),
            subprocess_log_dir=(log_dir / "subprocess") if log_dir else None,
        )
    else:
        runner = InProcessRunner()

    Path(cfg["storage"]["trace_path"]).parent.mkdir(parents=True, exist_ok=True)
    store = TraceStore(cfg["storage"]["trace_path"])
    orch_dict = dict(cfg["orchestrator"])
    if overrides:
        orch_dict.update(overrides)
    orch_cfg = OrchestratorConfig(**orch_dict)

    if log_dir:
        try:
            final_cfg = dict(cfg)
            final_cfg["orchestrator"] = orch_dict
            (log_dir / "config_used.yaml").write_text(
                yaml.safe_dump(final_cfg, sort_keys=False))
        except Exception:
            pass

    env_cls = import_symbol(env_block["import_path"])
    tool_schemas = env_cls.tool_schemas()
    env_state_schema = env_cls.env_state_schema()

    return Orchestrator(
        env_spec=env_spec, policy_spec=policy_spec,
        harness_agent=harness_agent,
        objective=objective, budget=budget,
        runner=runner, trace_store=store,
        tool_schemas=tool_schemas, env_state_schema=env_state_schema,
        config=orch_cfg, log_dir=log_dir,
    )


# ---------------------------------------------------------------------------
# Scripted (offline) mode -- plumbing only
# ---------------------------------------------------------------------------

_PLACEHOLDER_RULES_CODE = """
class _Rules(Rules):
    pass
""".strip()


def _scripted_agent_fn(call_kind, args):
    if call_kind == "propose":
        return Candidate(rules_code=_PLACEHOLDER_RULES_CODE,
                          rationale="plumbing test: pass-through Rules")
    if call_kind == "decide":
        traces = args["traces"]
        any_success = any(t.success for t in traces)
        return DecideResult(
            decision=Decision.ACCEPT,
            rationale=f"plumbing test: K={len(traces)} done, any_success={any_success}",
            failure_analysis=(None if any_success else FailureAnalysis(
                primary_axis="task_understanding",
                label="placeholder", description="placeholder")),
        )
    if call_kind == "refine":
        return args["candidate"]
    raise ValueError(call_kind)


def build_scripted(use_subprocess: bool, run_name: str) -> Orchestrator:
    tool_schemas = Toy24Env.tool_schemas()
    env_state_schema = Toy24Env.env_state_schema()

    log_dir = Path(f"./runs/{run_name}")
    log_dir.mkdir(parents=True, exist_ok=True)

    env_spec = EnvSpec(
        import_path="envharness.bridges.toy24:Toy24Env",
        reset_options={"numbers": [3, 3, 7, 7], "target": 24},
    )
    policy_spec = PolicySpec(
        client_factory="envharness.infra.llm:ScriptedClient",
        client_kwargs={"model_id": "scripted/seq",
                        "script": [
                            {"name": "combine", "kwargs": {"i": 2, "j": 0, "op": "mul"}},
                            {"name": "combine", "kwargs": {"i": 2, "j": 0, "op": "add"}},
                            {"name": "stop", "kwargs": {}},
                        ]},
        task_prompt="Combine to make 24.", action_format="function_calling",
    )
    runner: EpisodeRunner = (
        SubprocessRunner(timeout=30, subprocess_log_dir=log_dir / "subprocess")
        if use_subprocess else InProcessRunner()
    )

    return Orchestrator(
        env_spec=env_spec, policy_spec=policy_spec,
        harness_agent=ScriptedHarnessAgent(_scripted_agent_fn),
        objective=DifficultyZone(),
        budget=CappedAdaptive(max_k=2),
        runner=runner,
        trace_store=TraceStore(log_dir / "traces.jsonl"),
        tool_schemas=tool_schemas, env_state_schema=env_state_schema,
        config=OrchestratorConfig(
            task_id="toy24-scripted",
            task_description="Combine to make 24.",
            n_iterations=2, max_episode_steps=8,
            k_per_candidate=2, base_seed=0,
            compute_baseline=False,
        ),
        log_dir=log_dir,
    )


# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--scripted", action="store_true")
    p.add_argument("--in-process", action="store_true")
    p.add_argument("--run-name", type=str, default=None,
                    help="Directory under ./runs/ for this run's logs/traces.")
    p.add_argument("--n-tasks", type=int, default=None,
                    help="Override orchestrator.n_iterations (n_tasks).")
    p.add_argument("--task-offset", type=int, default=None,
                    help="Override orchestrator.task_id_base_offset.")
    p.add_argument("--model", type=str, default=None,
                    help="Override the model on the policy and mutator blocks "
                    "(e.g. openai/gpt-4.1-mini, gemini/gemini-3.5-flash, "
                    "vertex_ai/claude-sonnet-4-6).")
    p.add_argument("--task-ids", type=str, default=None,
                    help="Comma-separated explicit task_ids (e.g. "
                    "'7,42,103,...'). OVERRIDES n-tasks + task-offset "
                    "stride formula; orchestrator iterates exactly this "
                    "list.")
    args = p.parse_args(argv)

    if args.scripted:
        run_name = args.run_name or f"scripted-{time.strftime('%Y%m%d-%H%M%S')}"
        orch = build_scripted(use_subprocess=not args.in_process,
                                run_name=run_name)
    elif args.config is not None:
        run_name = (args.run_name
                     or f"{args.config.stem}-{time.strftime('%Y%m%d-%H%M%S')}")
        overrides: dict = {}
        if args.n_tasks is not None:
            # Set BOTH aliases coherently: overriding only the legacy
            # n_iterations against a YAML that uses the preferred n_tasks
            # key trips OrchestratorConfig.__post_init__'s
            # "conflicting n_tasks vs n_iterations" check.
            overrides["n_tasks"] = args.n_tasks
            overrides["n_iterations"] = args.n_tasks
        if args.task_offset is not None:
            overrides["task_id_base_offset"] = args.task_offset
        if args.task_ids is not None:
            ids = [int(x) for x in args.task_ids.split(",") if x.strip()]
            overrides["explicit_task_ids"] = ids
            overrides["n_tasks"] = len(ids)
            overrides["n_iterations"] = len(ids)
        orch = build_from_config(args.config, run_name,
                                  overrides=overrides or None,
                                  model=args.model)
    else:
        p.error("specify --scripted or --config")

    print(f"[run] run_name={run_name}")
    accepted = orch.run()

    print(f"\n=== run finished: {len(accepted)} accepted episodes ===")
    for t in accepted:
        print(f"  cand={t.candidate_id} roll={t.rollout_idx} ep={t.episode_id} "
              f"success={t.success} reward={t.final_reward:.2f} "
              f"steps={t.duration_steps} "
              f"err={t.error or '-'} "
              f"failure={t.failure_analysis.label if t.failure_analysis else '-'}")


if __name__ == "__main__":
    main()
