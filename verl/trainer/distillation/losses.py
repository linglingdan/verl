# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import logging

import torch
from tensordict import TensorDict

from verl.base_config import BaseConfig
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, DistillationConfig, DistillationLossConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding

logger = logging.getLogger(__name__)

DistillationLossFn = Callable[
    [
        ActorConfig,  # actor_config
        DistillationConfig,  # distillation_config
        dict,  # model_output
        TensorDict,  # micro batch input
    ],
    tuple[torch.Tensor, dict[str, Any]],
]


def is_distillation_enabled(config: Optional[DistillationConfig]) -> bool:
    """Check if distillation is enabled based on the provided configuration."""
    if config is None:
        return False
    return config.enabled


@dataclass
class DistillationLossSettings(BaseConfig):
    """
    Settings for a distillation loss function to be registered.

    Args:
        names (str | list[str]): Name(s) to register the distillation loss function under.
        use_topk (bool): Whether the loss function uses top-k log probabilities.
        use_estimator (bool): Whether the loss function uses single-sample KL estimators.
    """

    names: str | list[str] = field(default_factory=list)
    use_topk: bool = False
    use_estimator: bool = False

    _mutable_fields = {"names"}

    def __post_init__(self):
        self.names = [self.names] if isinstance(self.names, str) else self.names
        if sum([self.use_topk, self.use_estimator]) != 1:
            raise ValueError(
                f"Expected only one of use_estimator, use_topk, but got {self.use_estimator=}, {self.use_topk=}."
            )


DISTILLATION_LOSS_REGISTRY: dict[str, DistillationLossFn] = {}
DISTILLATION_SETTINGS_REGISTRY: dict[str, DistillationLossSettings] = {}


def register_distillation_loss(
    loss_settings: DistillationLossSettings,
) -> Callable[[DistillationLossFn], DistillationLossFn]:
    """Register a distillation loss function with the given name."""

    def decorator(func: DistillationLossFn) -> DistillationLossFn:
        for name in loss_settings.names:
            if name in DISTILLATION_LOSS_REGISTRY:
                raise ValueError(f"Distillation loss function with name '{name}' is already registered.")
            DISTILLATION_LOSS_REGISTRY[name] = func
            DISTILLATION_SETTINGS_REGISTRY[name] = loss_settings
        return func

    return decorator


def get_distillation_loss_fn(loss_name: str) -> DistillationLossFn:
    """Get the distillation loss function with a given name."""
    if loss_name not in DISTILLATION_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_LOSS_REGISTRY.keys())}"
        )
    return DISTILLATION_LOSS_REGISTRY[loss_name]


def get_distillation_loss_settings(loss_name: str) -> DistillationLossSettings:
    """Get the distillation loss settings with a given name."""
    if loss_name not in DISTILLATION_SETTINGS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_SETTINGS_REGISTRY.keys())}"
        )
    return DISTILLATION_SETTINGS_REGISTRY[loss_name]


def compute_distillation_loss_range(
    distillation_losses: torch.Tensor, response_mask: torch.Tensor
) -> dict[str, Metric]:
    """Compute min and max distillation loss over valid response tokens."""
    distillation_losses_response = distillation_losses[response_mask.bool()]
    return {
        "distillation/loss_min": Metric(AggregationType.MIN, distillation_losses_response.min()),
        "distillation/loss_max": Metric(AggregationType.MAX, distillation_losses_response.max()),
    }


def compute_topk_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    data: TensorDict,
    student_logits: torch.Tensor,
    data_format: str,
) -> torch.Tensor:
    """Compute the topk loss in logit processor.

    Returns:
    - distillation_losses: (bsz, seqlen/cp_size)
    - student_mass: (bsz, seqlen/cp_size)
    - teacher_mass: (bsz, seqlen/cp_size)
    """
    match config.strategy:
        case "fsdp":
            import verl.trainer.distillation.fsdp.losses as fsdp_losses

            distillation_loss_fn = fsdp_losses.compute_forward_kl_topk
        case "megatron":
            import verl.trainer.distillation.megatron.losses as megatron_losses

            distillation_loss_fn = megatron_losses.compute_forward_kl_topk
        case _:
            raise NotImplementedError(f"Unsupported strategy: {config.strategy=}")

    outputs = distillation_loss_fn(
        student_logits=student_logits,
        teacher_topk_log_probs=data["teacher_logprobs"],
        teacher_topk_ids=data["teacher_ids"],
        config=distillation_config,
        data_format=data_format,
    )

    expected_shape = student_logits.shape[:2]
    for k, v in outputs.items():
        assert v.shape == expected_shape, f"Expected shape {expected_shape}, but got {v.shape} for {k=}."

    return outputs


def distillation_ppo_loss(
    config: ActorConfig,
    distillation_config: Optional[DistillationConfig],
    model_output: dict = None,
    data: TensorDict = None,
    dp_group=None,
    student_logits: torch.Tensor = None,
    data_format: str = "thd",
):
    """Loss function used both for logit processor and final policy loss.
    - student_logits is not None, compute the topk loss in logit processor.
    - student_logits is None, compute final policy loss.

    [split sequence across sp/cp groups]
                   |
    [model forward and output logits: (bsz, seqlen/cp_size, vocab_size/tp_size)]
                   |
    [logits processor compute topk loss: (bsz, seqlen/cp_size)]
                   |
    [all gather topk loss across sp/cp groups: (bsz, seqlen)]
                   |
    [combine topk loss with policy loss]

    Args:
        config: Actor configuration.
        distillation_config: Distillation configuration.
        model_output: Model output, including log_probs, entropy.
        data: Micro input batch, contains
          - teacher_logprobs: (bsz, seqlen, topk)
          - teacher_ids: (bsz, seqlen, topk)
        student_logits: (bsz, seqlen/cp_size, vocab_size/tp_size).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - student_logits is not None, return the topk loss tensor (bsz, seqlen/cp_size).
    - student_logits is None, return the final policy loss scalar and metrics.
    """

    # Called as logits processor
    if student_logits is not None:
        return compute_topk_loss(config, distillation_config, data, student_logits, data_format)

    # Called as final policy loss
    distillation_loss_config = distillation_config.distillation_loss
    distill_loss, distill_metrics = distillation_loss(config, distillation_config, model_output, data)
    policy_loss, policy_metrics = ppo_loss(config, model_output, data, dp_group)
    if not distillation_loss_config.use_task_rewards:
        policy_loss = 0.0

    # Combine distillation with policy loss
    policy_metrics.update(distill_metrics)
    distillation_loss_coef = (
        distillation_loss_config.distillation_loss_coef if distillation_loss_config.use_task_rewards else 1.0
    )
    policy_loss += distill_loss * distillation_loss_coef
    policy_metrics["distillation/loss"] = Metric(value=distill_loss, aggregation=AggregationType.SUM)

    return policy_loss, policy_metrics


def distillation_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics.

    Returns:
    - distillation_loss: Aggregated distillation loss scalar.
    - distillation_metrics: Dictionary of metrics.
    """
    assert distillation_config is not None
    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_loss_fn = get_distillation_loss_fn(loss_config.loss_mode)
    distillation_losses, distillation_metrics, f_attn_for_adv = distillation_loss_fn(
        config=config,
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )
    response_mask = data["response_mask"]
    loss_agg_mode = config.loss_agg_mode

    distillation_metrics.update(
        compute_distillation_loss_range(distillation_losses=distillation_losses, response_mask=response_mask)
    )
    if loss_config.loss_max_clamp is not None:
        # clamping min is for k1 loss which can be negative
        distillation_losses = distillation_losses.clamp(min=-loss_config.loss_max_clamp, max=loss_config.loss_max_clamp)

    if loss_config.use_policy_gradient:
        # When f_attn is available: multiply it onto the GRPO task advantage so that PPO
        # focuses gradient updates on tokens the model attends to most under privilege context.
        # When f_attn is not available: fall back to using -KL as the per-token advantage,
        # which is the original on-policy distillation behaviour.
        policy_loss_fn = get_policy_loss_fn(loss_config.policy_loss_mode)
        for k, v in config.global_batch_info.items():
            loss_config.global_batch_info[k] = v
        log_prob = no_padding_2_padding(model_output["log_probs"], data)
        old_log_prob = data["old_log_probs"]
        rollout_is_weights = data.get("rollout_is_weights", None)

        if f_attn_for_adv is not None and "advantages" in data.keys():
            # New path: attention-weighted task advantage.
            # f_attn_for_adv is already normalised (mean=1 over response tokens) and masked.
            # Multiplying preserves the overall advantage magnitude while re-distributing
            # gradient emphasis to high-attention token positions.
            task_advantages = data["advantages"].to(f_attn_for_adv.dtype)
            weighted_advantages = task_advantages * f_attn_for_adv
            distillation_metrics["distillation/adv_attn_weighted"] = Metric(
                AggregationType.MEAN,
                weighted_advantages[response_mask.bool()].mean(),
            )
        else:
            # Original path: treat -KL per token as the per-token advantage.
            weighted_advantages = -distillation_losses.detach()

        distillation_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=weighted_advantages,
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=loss_config,
            rollout_is_weights=rollout_is_weights,
        )
        pg_metrics = {f"distillation/{k[len('actor/') :]}": v for k, v in pg_metrics.items()}
        distillation_metrics.update(pg_metrics)
    else:
        # Directly backpropagate distillation loss as a supervised loss, as in https://arxiv.org/abs/2306.13649.
        distillation_loss = agg_loss(
            loss_mat=distillation_losses,
            loss_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )

    return distillation_loss, distillation_metrics


@register_distillation_loss(DistillationLossSettings(names=["forward_kl_topk"], use_topk=True))  # type: ignore[arg-type]
def compute_forward_kl_topk(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute forward KL distillation loss and related metrics using top-k log probabilities.

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    # topk loss has been computed in logits processor
    distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    student_mass = no_padding_2_padding(model_output["student_mass"], data)
    teacher_mass = no_padding_2_padding(model_output["teacher_mass"], data)
    response_mask_bool = data["response_mask"].bool()
    assert distillation_losses.shape == student_mass.shape == teacher_mass.shape == response_mask_bool.shape

    # Log amount of mass in the top-k log probabilities for both student and teacher.
    student_mass = student_mass[response_mask_bool]
    teacher_mass = teacher_mass[response_mask_bool]
    distillation_metrics = {
        "distillation/student_mass": student_mass.mean().item(),
        "distillation/student_mass_min": Metric(AggregationType.MIN, student_mass.min()),
        "distillation/student_mass_max": Metric(AggregationType.MAX, student_mass.max()),
        "distillation/teacher_mass": teacher_mass.mean().item(),
        "distillation/teacher_mass_min": Metric(AggregationType.MIN, teacher_mass.min()),
        "distillation/teacher_mass_max": Metric(AggregationType.MAX, teacher_mass.max()),
    }

    # Due to use of top-k, student and teacher distributions don't sum to 1 -> divergences can be negative.
    distillation_losses = distillation_losses.clamp_min(0.0)

    # topk path has no f_attention
    return distillation_losses, distillation_metrics, None


@register_distillation_loss(
    DistillationLossSettings(names=["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_distillation_loss_reverse_kl_estimator(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics using single-sample KL estimators.

    Uses the kl_penalty function from core_algos which supports various KL divergence
    estimators: "kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3".

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    response_mask_bool = data["response_mask"].bool()

    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    if loss_config.teacher_loss_weight > 0.0:
        teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
        assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape
        distillation_losses = kl_penalty(
            logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty=loss_config.loss_mode
        ) * loss_config.teacher_loss_weight
    else:
        distillation_losses = torch.zeros_like(student_log_probs)
    # Since k1 can be negative, log the mean absolute loss.
    metrics = {
        "distillation/abs_loss": Metric(AggregationType.MEAN, distillation_losses[response_mask_bool].abs().mean()),
    }

    # Compute f_attn here so it is available both for privileged_losses weighting
    # and (when use_policy_gradient=True) for multiplying onto the task advantage.
    # f_attn_for_adv is None when no attention keys are present.
    f_attn_for_adv: torch.Tensor | None = None

    # Privileged distillation: push student toward its own privileged-context self.
    if "privileged_log_probs" in data.keys():
        privileged_log_probs = data["privileged_log_probs"]  # (bsz, resp_len), already padded
        assert privileged_log_probs.shape == student_log_probs.shape, (
            f"privileged_log_probs shape {privileged_log_probs.shape} "
            f"!= student_log_probs shape {student_log_probs.shape}"
        )
        privileged_losses = kl_penalty(
            logprob=student_log_probs,
            ref_logprob=privileged_log_probs,
            kl_penalty=loss_config.loss_mode,
        )
        if "f_attention_privileged" in data.keys():
            f_attn = data["f_attention_privileged"].to(privileged_losses.dtype)
            # Joint weighting: multiply privileged attention with normal-context attention.
            # f_attention_normal comes from a no-privilege forward pass over [prompt | response],
            # giving the model's baseline per-token importance without oracle context.
            # The product highlights tokens that are BOTH important normally AND shift under privilege.
            if "f_attention_normal" in data.keys():
                f_attn_normal = data["f_attention_normal"].to(f_attn.dtype)
                f_attn = f_attn * f_attn_normal
                f_attn_normal_valid = f_attn_normal[response_mask_bool]
                metrics["distillation/f_attn_normal_mean"] = Metric(AggregationType.MEAN, f_attn_normal_valid.mean())
                metrics["distillation/f_attn_normal_max"] = Metric(AggregationType.MEAN, f_attn_normal_valid.max())
            # Mask out non-response positions and normalise so that the mean weight = 1
            # (keeps loss magnitude stable regardless of absolute attention values).
            f_attn = f_attn * response_mask_bool
            f_attn_sum = f_attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            f_attn_normalised = f_attn / f_attn_sum * response_mask_bool.sum(dim=-1, keepdim=True)
            privileged_losses = privileged_losses * f_attn_normalised
            # Save for re-use in the policy_gradient advantage branch below.
            f_attn_for_adv = f_attn_normalised
            f_attn_valid = f_attn_normalised[response_mask_bool]
            metrics["distillation/f_attn_mean"] = Metric(AggregationType.MEAN, f_attn_valid.mean())
            metrics["distillation/f_attn_max"] = Metric(AggregationType.MEAN, f_attn_valid.max())
            logger.info(
                "[f_attention] active | f_attn_mean=%.4f | f_attn_max=%.4f",
                f_attn_valid.mean().item(),
                f_attn_valid.max().item(),
            )
        distillation_losses = distillation_losses + loss_config.privilege_loss_weight * privileged_losses
        priv_abs = privileged_losses[response_mask_bool].abs().mean()
        metrics["distillation/privileged_abs_loss"] = Metric(AggregationType.MEAN, priv_abs)
        logger.info(
            "[privileged distillation] active | bsz=%d | privilege_loss_weight=%.3f | "
            "privileged_abs_loss=%.4f | teacher_abs_loss=%.4f",
            student_log_probs.shape[0],
            loss_config.privilege_loss_weight,
            priv_abs.item(),
            distillation_losses[response_mask_bool].abs().mean().item(),
        )

    return distillation_losses, metrics, f_attn_for_adv
