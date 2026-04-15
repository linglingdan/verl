#!/usr/bin/env bash
# 特权信息蒸馏 —— Agent（多轮工具调用）版本
#
# 与单轮版 run_qwen_gsm8k_privileged.sh 的核心区别：
#   1. actor_rollout_ref.rollout.multi_turn.enable=True：开启多轮 agent 推理
#   2. 需要配置 tool_config_path / format 等多轮参数
#   3. 数据集每条样本仍需包含 "privilege" 列（神谕提示文本）
#
# 特权蒸馏在 agent 路径下的工作方式与单轮完全相同：
#   rollout 结束后，trainer 将 [prompt | privilege | responses] 拼接，
#   计算 P(response | prompt, privilege, response_{<t}) 作为额外监督信号。
#   多轮 responses 中工具 observation 部分（response_mask=0）自动被屏蔽，不计入损失。
#
# 用法：
#   DATA_PATH=/path/to/data TOOL_CONFIG_PATH=/path/to/tools.yaml bash run_qwen_gsm8k_privileged_agent.sh
set -xeuo pipefail

############################ 环境 ############################

VERL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"

############################ 快速配置 ############################

ROLLOUT_NAME="vllm"  # sglang or vllm

STUDENT_MODEL=/chubao/tj-train-ssd-21/liuchengwei/models/qwen/Qwen3-0.6B
TEACHER_MODEL=/chubao/tj-train-ssd-21/liuchengwei/models/qwen/Qwen3-8B

USE_POLICY_GRADIENT=True
DISTILLATION_LOSS_MODE="k1"
USE_FUSED_KERNELS=False

DISTILLATION_LOSS_MAX_CLAMP=10.0
DISTILLATION_LOG_PROB_MIN_CLAMP=-10.0

# ── 特权蒸馏设置 ───────────────────────────────────────────────────────────────
# 教师 KL 损失权重，0.0 = 关闭教师（同时跳过教师前向推理）
TEACHER_LOSS_WEIGHT=0.0
# 特权 KL 损失权重，0.0 = 关闭特权蒸馏
PRIVILEGE_LOSS_WEIGHT=1.0
# privilege 字段的最大 token 长度（超长截断）
PRIVILEGE_MAX_LENGTH=512
# ──────────────────────────────────────────────────────────────────────────────

# ── Agent / 多轮设置 ───────────────────────────────────────────────────────────
# 工具配置文件路径（yaml），定义 agent 可调用的工具列表
# 示例: /path/to/tools.yaml
: "${TOOL_CONFIG_PATH:?请设置 TOOL_CONFIG_PATH 环境变量，指向工具配置 yaml 文件}"
# 对话格式：hermes / chatml / llama3 等，需与模型 chat template 一致
MULTI_TURN_FORMAT=hermes
# 最大工具调用轮数
MAX_TURNS=5
# agent_loop 名称，需在代码中注册，默认为 tool_agent
AGENT_LOOP=tool_agent
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_NAME='verl_privileged_agent_gsm8k'

MAX_PROMPT=512
MAX_RESPONSE_LENGTH=1024   # agent 多轮，response 更长
# 特权前向需要额外 PRIVILEGE_MAX_LENGTH token 的空间
MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE_LENGTH + PRIVILEGE_MAX_LENGTH + 1 ))
TRAIN_PROMPT_BSZ=64        # agent 轨迹更长，适当减小 batch size
STUDENT_MICRO_BATCH_SIZE_PER_GPU=1
STUDENT_MAX_TOKEN_LEN_PER_GPU=$(( STUDENT_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + PRIVILEGE_MAX_LENGTH + MAX_RESPONSE_LENGTH) ))
USE_DYNAMIC_BSZ=True

STUDENT_WORLD_SIZE=2
TEACHER_RESOURCE_POOL=False
TEACHER_WORLD_SIZE=4
SP=1

EXP_NAME="agent/student-${STUDENT_MODEL}/loss-${DISTILLATION_LOSS_MODE}/priv-${PRIVILEGE_LOSS_WEIGHT}/turn-${MAX_TURNS}"

ENFORCE_EAGER=True

############################ 数据路径 ############################

# 数据集要求：parquet 文件，每条样本包含 "privilege" 列（agent 任务的神谕提示）
: "${DATA_PATH:?请设置 DATA_PATH 环境变量，指向包含 privilege 列的 parquet 数据目录}"
TRAIN_FILES="['${DATA_PATH}/train.parquet']"
TEST_FILES="['${DATA_PATH}/test.parquet']"

############################ 参数组 ############################

DATA=(
    data.train_files="$TRAIN_FILES"
    data.val_files="$TEST_FILES"
    data.max_prompt_length=$MAX_PROMPT
    data.max_response_length=$MAX_RESPONSE_LENGTH
    data.train_batch_size=$TRAIN_PROMPT_BSZ
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
)

MODEL=(
    actor_rollout_ref.model.path="${STUDENT_MODEL}"
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=$USE_FUSED_KERNELS
    +actor_rollout_ref.model.override_config.attn_implementation=eager
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER
)

DISTILLATION=(
    distillation.enabled=True
    distillation.num_workers=8
    distillation.privilege_max_length=$PRIVILEGE_MAX_LENGTH
    distillation.teacher_model.enable_resource_pool=$TEACHER_RESOURCE_POOL
    distillation.teacher_model.n_gpus_per_node=$TEACHER_WORLD_SIZE
    distillation.teacher_model.nnodes=1
    distillation.teacher_model.model_path="${TEACHER_MODEL}"
    distillation.teacher_model.inference.tensor_model_parallel_size=1
    distillation.teacher_model.inference.name=$ROLLOUT_NAME
    distillation.teacher_model.inference.gpu_memory_utilization=0.3
    distillation.teacher_model.inference.enforce_eager=$ENFORCE_EAGER
    distillation.teacher_model.inference.max_model_len=$MAX_NUM_TOKENS
    distillation.teacher_model.inference.max_num_batched_tokens=$MAX_NUM_TOKENS
    distillation.teacher_model.inference.max_num_seqs=$MAX_NUM_TOKENS
    distillation.distillation_loss.loss_mode=$DISTILLATION_LOSS_MODE
    distillation.distillation_loss.topk=64
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=$USE_POLICY_GRADIENT
    distillation.distillation_loss.loss_max_clamp=$DISTILLATION_LOSS_MAX_CLAMP
    distillation.distillation_loss.log_prob_min_clamp=$DISTILLATION_LOG_PROB_MIN_CLAMP
    distillation.distillation_loss.teacher_loss_weight=$TEACHER_LOSS_WEIGHT
    distillation.distillation_loss.privilege_loss_weight=$PRIVILEGE_LOSS_WEIGHT
)

STUDENT=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_PROMPT_BSZ
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3
    actor_rollout_ref.rollout.calculate_log_probs=False
    actor_rollout_ref.rollout.max_model_len=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.n=1
    # ── 多轮 Agent 配置 ──────────────────────────────────────────────────────
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$MAX_TURNS
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$MAX_TURNS
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$TOOL_CONFIG_PATH
    actor_rollout_ref.rollout.multi_turn.format=$MULTI_TURN_FORMAT
    actor_rollout_ref.rollout.agent.default_agent_loop=$AGENT_LOOP
    # ────────────────────────────────────────────────────────────────────────
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

TRAINER=(
    trainer.logger='["console"]'
    trainer.project_name=$PROJECT_NAME
    trainer.experiment_name=$EXP_NAME
    trainer.n_gpus_per_node=$STUDENT_WORLD_SIZE
    trainer.nnodes=1
    trainer.save_freq=200
    trainer.test_freq=5
    trainer.total_epochs=15
    trainer.val_before_train=False
    trainer.use_legacy_worker_impl=disable
    trainer.resume_mode=disable
    trainer.log_val_generations=5
)

############################ 启动 ############################

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_trainer.yaml' \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${DISTILLATION[@]}" \
    "${ROLLOUT[@]}" \
    "${STUDENT[@]}" \
    "${TRAINER[@]}" \
    "$@"
