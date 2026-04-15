# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig

from .rollout import RolloutConfig

__all__ = ["DistillationLossConfig", "DistillationTeacherModelConfig", "DistillationConfig", "OnlinePrivilegeConfig"]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class DistillationLossConfig(BaseConfig):
    """Configuration for distillation loss settings.

    loss_mode (str):
        Distillation loss function to use.
    topk (int, optional):
        Number of top tokens to consider for top-k distillation losses.
    use_task_rewards (bool):
        Whether to include task rewards alongside distillation loss.
    distillation_loss_coef (float):
        Coefficient for distillation loss when combined with task rewards.
    loss_max_clamp (float, optional):
        Maximum value to clamp distillation loss. If None, no clamping is applied.
    log_prob_min_clamp (float, optional):
        Minimum value to clamp log probabilities for stability, e.g., log q - log p where p or q are
        very close to zero. If None, no clamping is applied.
    use_policy_gradient (bool):
        Whether to incorporate distillation loss as a reward, as done
        by https://thinkingmachines.ai/blog/on-policy-distillation/. Recommended to use loss_mode=k1.
        Otherwise, distillation loss is directly backpropagated as a supervised loss,
        as in https://arxiv.org/abs/2306.13649. Recommended to use loss_mode=k3 or forward_kl_topk.
    policy_loss_mode (str):
        Name of the policy loss to use when use_policy_gradient is true.
    clip_ratio (float):
        PPO clipping ratio for policy loss.
    clip_ratio_low (float):
        Lower bound for PPO clipping ratio.
    clip_ratio_high (float):
        Upper bound for PPO clipping ratio.
    loss_settings (DistillationLossSettings, optional):
        Runtime-populated settings based on loss_mode. Not set by user.
    """

    loss_mode: str = "k3"
    topk: Optional[int] = 128
    use_task_rewards: bool = True
    distillation_loss_coef: float = 1.0
    loss_max_clamp: Optional[float] = 10.0
    log_prob_min_clamp: Optional[float] = -10.0

    use_policy_gradient: bool = True
    policy_loss_mode: str = "vanilla"
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2

    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # batch_num_tokens: number of valid tokens in global batch
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)

    # Weight applied to the teacher KL loss term.
    # Set to 0.0 to disable teacher distillation entirely (skips teacher forward pass too).
    teacher_loss_weight: float = 1.0

    # Weight applied to the privileged k1 loss term (student vs privileged-student).
    # Set to 0.0 to disable. Only effective when the dataset has a "privilege" column.
    privilege_loss_weight: float = 1.0

    # Whether to collect normal-context attention (no privilege) during compute_old_log_prob
    # and multiply it element-wise with f_attention_privileged as a joint token-importance weight.
    # Requires attn_implementation=eager. Adds a second attention forward pass cost.
    # Only effective when f_attention_privileged is also active.
    use_normal_attention_weight: bool = True

    # Store distillation loss settings for computing the specified loss_mode
    # Not set by user, populated at runtime
    loss_settings: Optional[dict] = None

    def __post_init__(self):
        self._mutable_fields.add("loss_settings")
        from verl.trainer.distillation.losses import DistillationLossSettings, get_distillation_loss_settings

        self.loss_settings: DistillationLossSettings = get_distillation_loss_settings(self.loss_mode)

        if self.policy_loss_mode != "vanilla":
            raise NotImplementedError(
                f"Only vanilla policy loss is currently supported when use_policy_gradient is True, "
                f"but got {self.policy_loss_mode}."
            )

        if self.use_policy_gradient and self.loss_mode == "forward_kl_topk":
            print(
                "WARNING: forward_kl_topk is most effective as a supervised distillation loss "
                "(use_policy_gradient=False). With policy gradient, the update uses only the sampled"
                " token's logprob ∇logπ(a), so the top-k distributional signal (how non-sampled logits "
                "should move) is largely unused."
            )

        if not self.use_policy_gradient and self.loss_mode == "k1" and self.teacher_loss_weight > 0.0:
            raise ValueError(
                "Directly backpropagating k1 loss is incorrect since gradient of k1 loss"
                " wrt model weights does not depend on teacher log probabilities."
                " Set use_policy_gradient=True, or set teacher_loss_weight=0.0 to use"
                " k1 for privileged-only distillation."
            )


@dataclass
class DistillationTeacherModelConfig(BaseConfig):
    """Configuration for on-policy distillation teacher.

    enable_resource_pool (bool):
        Whether to enable separate resource pool for teacher model(s).
    n_gpus_per_node (int):
        Number of GPUs per node to use for distillation teacher model(s).
    nnodes (int):
        Number of nodes to use for distillation teacher model(s).
    model_path (str, optional):
        Model path for the teacher model. Can be a local path or a Hugging Face model
    inference (RolloutConfig):
        Rollout configuration for the teacher model inference during distillation.
    """

    _mutable_fields = BaseConfig._mutable_fields

    enable_resource_pool: bool = False
    n_gpus_per_node: int = 0
    nnodes: int = 0
    model_path: Optional[str] = None
    inference: RolloutConfig = field(default_factory=RolloutConfig)


@dataclass
class OnlinePrivilegeConfig(BaseConfig):
    """Configuration for online privilege generation via an external LLM API.

    When enabled, after each rollout the trainer calls an OpenAI-compatible
    chat completions endpoint to generate a per-response "privilege" string.
    The input to the LLM includes the current response, its reward, and the
    global set of all responses (and rewards) for the same prompt (uid group).

    Requires ``actor_rollout_ref.rollout.n > 1`` so that multiple responses
    per prompt are available in the same batch.
    """

    # Master switch.  Set to True to activate online privilege generation.
    enabled: bool = False

    # OpenAI-compatible chat completions endpoint of the privilege LLM.
    # e.g. "http://localhost:8001/v1/chat/completions"
    server_url: str = "http://localhost:8001/v1/chat/completions"

    # Model identifier sent to the endpoint.
    model_name: str = ""

    # Maximum tokens to generate for each privilege string.
    max_tokens: int = 200

    # Per-request HTTP timeout in seconds.
    timeout: float = 30.0

    # Maximum simultaneous HTTP requests (asyncio semaphore).
    max_concurrency: int = 32

    # Minimum number of correct responses (reward > 0) required in a uid group
    # before generating privileges.  Groups below this threshold are skipped
    # and their privilege is set to "" (no privileged loss for those samples).
    min_correct: int = 1

    # How many "other" responses to include as context in the prompt sent to the LLM.
    max_context_examples: int = 3

    # Maximum characters allowed in the user message sent to the privilege LLM.
    # Prompts longer than this are hard-truncated with "...[truncated]" appended.
    # This is a safety guard against VLLMValidationError when the combined length
    # of question + responses exceeds the server's context window.
    # Rule of thumb: max_prompt_chars ≈ (server_max_model_len - max_tokens) × 3
    # e.g. 8192-token window, 200 output tokens → (8192-200)×3 ≈ 24000 chars.
    # Default 6000 is conservative and fits most 4096-token servers.
    max_prompt_chars: int = 6000

    # Optional system-prompt override.  Leave empty to use the built-in default.
    system_prompt: str = ""


@dataclass
class DistillationConfig(BaseConfig):
    """Configuration for on-policy distillation.

    enabled (bool):
        Whether on-policy distillation is enabled.
    num_workers (int):
        Number of teacher model replicas.
    teacher_model (TeacherModelConfig):
        Configuration for the teacher model used for distillation.
    distillation_loss (DistillationLossConfig):
        Configuration for distillation loss settings.
    """

    _mutable_fields = BaseConfig._mutable_fields | {"online_privilege"}

    enabled: bool = False
    num_workers: int = 8
    teacher_model: DistillationTeacherModelConfig = field(default_factory=DistillationTeacherModelConfig)
    distillation_loss: DistillationLossConfig = field(default_factory=DistillationLossConfig)
    # Max token length to truncate each sample's "privilege" string when building the
    # privileged forward sequence [prompt | privilege | response].
    # Only read when the dataset parquet contains a "privilege" column.
    privilege_max_length: int = 192

    # When True, the teacher computes log-probs on [prompt | privilege | response]
    # instead of the default [prompt | response].  Requires the dataset to have a
    # "privilege" column.  Only effective when teacher_loss_weight > 0.
    teacher_use_privilege: bool = True

    # Online privilege generation settings.  When online_privilege.enabled=True,
    # the "privilege" column in the dataset is REPLACED each step by fresh LLM output.
    online_privilege: OnlinePrivilegeConfig = field(default_factory=OnlinePrivilegeConfig)

    def __post_init__(self):
        # Prompt + Response from student are fed into teacher as context
        max_model_len = self.teacher_model.inference.max_model_len
        max_num_batched_tokens = self.teacher_model.inference.max_num_batched_tokens
        student_prompt_length = self.teacher_model.inference.prompt_length
        student_response_length = self.teacher_model.inference.response_length
        if self.enabled:
            required_context_len = student_prompt_length + student_response_length + 1
            if max_model_len is not None and required_context_len > max_model_len:
                raise ValueError(
                    "Distillation teacher inference requires room for the student prompt, the full student "
                    f"response, and one generated token, but got {student_prompt_length=}, "
                    f"{student_response_length=}, {required_context_len=}, {max_model_len=}."
                )
            if max_num_batched_tokens is not None and required_context_len > max_num_batched_tokens:
                raise ValueError(
                    "Distillation teacher inference requires room for the student prompt, the full student "
                    f"response, and one generated token within the engine batching budget, but got "
                    f"{student_prompt_length=}, {student_response_length=}, {required_context_len=}, "
                    f"{max_num_batched_tokens=}."
                )

        self.teacher_model.inference.prompt_length = (
            self.teacher_model.inference.prompt_length + self.teacher_model.inference.response_length
        )
        self.teacher_model.inference.response_length = 1

        # Ensure max log probs is aligned with top-k
        engine_name = self.teacher_model.inference.name
        engine_kwargs = self.teacher_model.inference.engine_kwargs
        if not self.distillation_loss.loss_settings.use_topk or self.distillation_loss.topk is None or not self.enabled:
            return
        match engine_name:
            case "vllm":
                vllm_engine_kwargs = dict(engine_kwargs.get("vllm", {}))
                max_logprobs = vllm_engine_kwargs.get("max_logprobs")
                if max_logprobs is None:
                    vllm_engine_kwargs["max_logprobs"] = self.distillation_loss.topk
                    max_logprobs = self.distillation_loss.topk
                if max_logprobs < self.distillation_loss.topk:
                    raise ValueError(
                        f"VLLM max_logprobs ({max_logprobs}) must be >= distillation_loss topk "
                        f"({self.distillation_loss.topk}) to enable distillation loss computation."
                    )
                engine_kwargs["vllm"] = vllm_engine_kwargs
            case _:
                raise NotImplementedError(
                    f"DistillationTeacherModelConfig does not support inference engine {engine_name}"
                )
