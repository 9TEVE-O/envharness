# Bank distillation — SWE-bench

Stage 2 of the SWE-bench pipeline: distill the per-task mutation corpus
(Stage 1, `../corpus.yaml`) into the two skill banks that `../reasoning_bank_eval.py`
evaluates against the no-bank baseline.

## Files

| file | role |
|---|---|
| `induce.py` | LLM distillation prompts (`SUCCESSFUL_SI`, `FAILED_SI`, `induce_paired_diff_atomic`, `format_trajectory_swebench`). Sibling of `envharness/reasoning_bank/induce.py` but with the SWE-bench domain phrasing. |
| `build_orig_bank.py` | `orig` bank builder. Run with `--pair-mode intra_baseline`: pairs each task's failing baseline traces with succeeding baseline traces (un-mutated env only). Its balanced-cap logic (`cap = min(cascade_pairs, intra_pairs)` per task) keeps the two banks' structural footprint matched. |
| `build_ours_bank.py` | `ours` (EnvHarness) bank builder. Run with `--pair-mode cascade`: per-task succ-source tiered cascade (accepted mutation succ → last exploration mutation succ → baseline succ), paired against baseline failures. Same `induce_paired_diff_atomic` prompt as the orig build. |

`../reproduce.py` (Stage 2) calls `build_orig_bank.py --pair-mode
intra_baseline` for the `orig` bank and `build_ours_bank.py --pair-mode
cascade` for the `ours` bank.

## Why the pairing is structural

Both builds:
- Use the **same distillation prompt** (`induce_paired_diff_atomic` from
  `induce.py`), the same distillation LLM, and a per-task pair cap.
- The ONLY differential between the two banks is which trace fills the
  success slot of each induction pair. For `orig`, it's always the
  original-task baseline succ. For `ours`, it falls back through
  Tier 1 (accepted mutation succ) → Tier 2 (last exploration mutation succ)
  → Tier 3 (baseline succ).

That structural pairing is what makes any orig-vs-ours delta attributable
to the mutation step alone, not to bank size or prompt differences.

## Dependencies

These scripts import from the root EnvHarness package
(`envharness.reasoning_bank` for `Bank` / `MemoryItem` / `embed_texts`).
They put the EnvHarness repo root on `PYTHONPATH` themselves (via
`sys.path.insert(0, ...)` at the top), so they can be run directly.
