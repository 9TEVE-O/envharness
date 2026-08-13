#!/usr/bin/env bash
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

# GRPO training on ALFWorld via verl-agent, with the env routed through
# envharness's AlfworldEnv + Rules harness
# (env.env_name=envharness_rl/alfworld -- the ADDITIVE route added to
# third_party/verl-agent/.../env_manager.py).
#
# Modes:
#   MODE=smoke (default)  -- tiny batch / 2 epochs, sanity-check the pipeline
#   MODE=full             -- stock recipe scaled to 8 GPUs
#
# Defaults to the bundled 6-game example corpus + train subset under
# experiments/alfworld/data/, so a fresh checkout trains on mutated games out
# of the box. Override with MUTATION_CORPUS= / TRAIN_SUBSET_PATH= (empty to
# disable, a path to point elsewhere).
#
# Examples:
#   bash rl/scripts/run_grpo.sh                 # smoke, example data
#   MUTATION_CORPUS= TRAIN_SUBSET_PATH= bash .../run_grpo.sh    # unmutated control
#   MODE=full bash rl/scripts/run_grpo.sh       # full

set -e
set -u

RL_ROOT=$(cd "$(dirname "$0")/.." && pwd)              # rl/
ROOT=$(cd "$RL_ROOT/.." && pwd)                        # repo root (has envharness/)
RELEASE=${RELEASE:-$ROOT}
VERL_AGENT=${VERL_AGENT:-$ROOT/third_party/verl-agent}
[ -d "$VERL_AGENT" ] || { echo "verl-agent not found at $VERL_AGENT; run \`bash rl/scripts/fetch_verl_agent.sh\` first (or set \$VERL_AGENT)" >&2; exit 1; }
PY=${PY:-$HOME/miniconda3/envs/verl-agent/bin/python}
MODE=${MODE:-smoke}

# --- mode-specific hyperparameters ---
if [ "$MODE" = "smoke" ]; then
    TRAIN_BS=${TRAIN_BS:-8}
    VAL_BS=${VAL_BS:-8}
    GROUP_N=${GROUP_N:-4}
    PPO_MINI_BS=${PPO_MINI_BS:-8}
    PPO_MICRO_BS_PER_GPU=${PPO_MICRO_BS_PER_GPU:-2}
    LOG_PROB_MICRO_BS_PER_GPU=${LOG_PROB_MICRO_BS_PER_GPU:-2}
    EPOCHS=${EPOCHS:-2}
    N_GPUS=${N_GPUS:-2}
    TP=${TP:-2}
    MAX_STEPS=${MAX_STEPS:-15}
    TEST_FREQ=${TEST_FREQ:-1}
    VAL_BEFORE=${VAL_BEFORE:-True}
    EXP_NAME_DEFAULT="grpo_envrl_alfworld_qwen1.5b_smoke"
elif [ "$MODE" = "full" ]; then
    TRAIN_BS=${TRAIN_BS:-16}
    VAL_BS=${VAL_BS:-128}
    GROUP_N=${GROUP_N:-8}
    PPO_MINI_BS=${PPO_MINI_BS:-256}
    PPO_MICRO_BS_PER_GPU=${PPO_MICRO_BS_PER_GPU:-8}
    LOG_PROB_MICRO_BS_PER_GPU=${LOG_PROB_MICRO_BS_PER_GPU:-8}
    EPOCHS=${EPOCHS:-150}
    N_GPUS=${N_GPUS:-8}
    TP=${TP:-4}
    MAX_STEPS=${MAX_STEPS:-50}
    HISTORY_LENGTH=${HISTORY_LENGTH:-50}
    TEST_FREQ=${TEST_FREQ:-5}
    VAL_BEFORE=${VAL_BEFORE:-True}
    EXP_NAME_DEFAULT="grpo_envrl_alfworld_qwen7b_full"
else
    echo "MODE must be smoke or full"; exit 2
fi
HISTORY_LENGTH=${HISTORY_LENGTH:-50}
if [ "$MODE" = "full" ]; then
    ENFORCE_EAGER=${ENFORCE_EAGER:-True}
else
    ENFORCE_EAGER=${ENFORCE_EAGER:-False}
fi

MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
ENGINE=${ENGINE:-vllm}
NUM_CPUS_PER_ENV=${NUM_CPUS_PER_ENV:-0.1}
WANDB_PROJECT=${WANDB_PROJECT:-envharness_rl_alfworld}
EXP_NAME=${EXP_NAME:-$EXP_NAME_DEFAULT}_$(date +%H%M%S)
SAVE_FREQ=${SAVE_FREQ:--1}

TS=$(date +%H%M%S)
RUN_DIR=$ROOT/runs/grpo_envrl_alfworld_${MODE}_${TS}
mkdir -p "$RUN_DIR"
ln -sfn "$RUN_DIR" "$ROOT/runs/grpo_envrl_alfworld_${MODE}_latest"

echo "=== rl ALFWorld GRPO ($MODE) ==="
echo "  model=$MODEL  engine=$ENGINE"
echo "  train_bs=$TRAIN_BS  val_bs=$VAL_BS  group_n=$GROUP_N"
echo "  n_gpus=$N_GPUS  TP=$TP  max_steps=$MAX_STEPS  epochs=$EPOCHS  save_freq=$SAVE_FREQ"
echo "  exp_name=$EXP_NAME  run_dir=$RUN_DIR"
date
echo ""

export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export ENVHARNESS_DISABLE_THINKING=${ENVHARNESS_DISABLE_THINKING:-1}

# Bundled example data defaults (override TRAIN_SUBSET_PATH= / MUTATION_CORPUS=
# to disable, or to a path to point elsewhere).
DATA_DIR=$RL_ROOT/experiments/alfworld/data
TRAIN_SUBSET_PATH=${TRAIN_SUBSET_PATH-$DATA_DIR/train_subset.jsonl}
if [ -n "$TRAIN_SUBSET_PATH" ] && [ -f "$TRAIN_SUBSET_PATH" ]; then
    export ALFWORLD_TRAIN_SUBSET_PATH="$TRAIN_SUBSET_PATH"
    echo "  using TRAIN subset: $ALFWORLD_TRAIN_SUBSET_PATH ($(wc -l < "$ALFWORLD_TRAIN_SUBSET_PATH") games)"
else
    unset ALFWORLD_TRAIN_SUBSET_PATH || true
    echo "  using FULL TRAIN (~3553 games)"
fi

MUTATION_CORPUS=${MUTATION_CORPUS-$DATA_DIR/example_corpus.jsonl}
if [ -n "$MUTATION_CORPUS" ] && [ -f "$MUTATION_CORPUS" ]; then
    export ENVHARNESS_MUTATION_CORPUS="$MUTATION_CORPUS"
    echo "  using MUTATION corpus: $ENVHARNESS_MUTATION_CORPUS ($(wc -l < "$ENVHARNESS_MUTATION_CORPUS") games)"
elif [ -n "$MUTATION_CORPUS" ]; then
    echo "  FATAL: MUTATION_CORPUS=$MUTATION_CORPUS not found"; exit 3
else
    unset ENVHARNESS_MUTATION_CORPUS || true
    echo "  using UNMUTATED env (control)"
fi

SUBSET_AUTHORITATIVE=${SUBSET_AUTHORITATIVE:-}
if [ -n "$SUBSET_AUTHORITATIVE" ]; then
    export ENVHARNESS_SUBSET_AUTHORITATIVE="$SUBSET_AUTHORITATIVE"
    echo "  subset_authoritative=$ENVHARNESS_SUBSET_AUTHORITATIVE"
else
    unset ENVHARNESS_SUBSET_AUTHORITATIVE || true
fi

RESUME_FROM=${RESUME_FROM:-}
if [ -n "$RESUME_FROM" ] && [ ! -e "$RESUME_FROM" ]; then
    echo "  FATAL: RESUME_FROM=$RESUME_FROM does not exist"; exit 4
fi

# Both envharness (base AlfworldEnv + Rules) and rl (the
# adapter) must be importable inside verl-agent's Ray workers.
export PYTHONPATH="$RELEASE:$RL_ROOT:${PYTHONPATH:-}"
export ALFWORLD_CONFIG="$VERL_AGENT/agent_system/environments/env_package/alfworld/configs/config_tw.yaml"

cd "$VERL_AGENT"

"$PY" examples/data_preprocess/prepare.py \
    --mode 'text' \
    --train_data_size $TRAIN_BS \
    --val_data_size $VAL_BS

set -x
"$PY" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$TRAIN_BS \
    data.val_batch_size=$VAL_BS \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BS \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BS_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BS_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL:-0.6} \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BS_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=envharness_rl/alfworld \
    env.seed=0 \
    env.history_length=$HISTORY_LENGTH \
    env.max_steps=$MAX_STEPS \
    env.rollout.n=$GROUP_N \
    env.resources_per_worker.num_cpus=$NUM_CPUS_PER_ENV \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.default_local_dir=$RUN_DIR/ckpts \
    actor_rollout_ref.actor.checkpoint.contents=['model','optimizer','extra','hf_model'] \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$EPOCHS \
    trainer.val_before_train=$VAL_BEFORE \
    ${RESUME_FROM:+trainer.resume_mode=resume_path trainer.resume_from_path=$RESUME_FROM} \
    ${EXTRA_HYDRA:-} \
    2>&1 | tee "$RUN_DIR/train.log"
