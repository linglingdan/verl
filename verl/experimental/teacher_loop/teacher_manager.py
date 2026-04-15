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
import asyncio
import logging
import os
from typing import Any, Optional
from uuid import uuid4

import ray
import torch
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARNING"))
from tensordict import TensorDict
from torch.nn import functional as F

from verl.experimental.agent_loop import AsyncLLMServerManager
from verl.protocol import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.config import DistillationConfig, DistillationLossConfig


def _get_teacher_sampling_params(
    distillation_config: DistillationConfig,
    distillation_loss_config: DistillationLossConfig,
) -> dict[str, Any]:
    """Get sampling parameters for teacher model when computing log probabilities for distillation."""
    if distillation_config.teacher_model.inference.temperature != 1.0:
        raise NotImplementedError("vLLM does not support temperature for prompt_logprobs.")

    num_logprobs = distillation_loss_config.topk if distillation_loss_config.loss_settings.use_topk else 0
    return {
        "max_tokens": 1,
        "temperature": distillation_config.teacher_model.inference.temperature,
        "prompt_logprobs": num_logprobs,
    }


def _pad_teacher_outputs(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    prompt_width: int,
    response_width: int,
    prompt_length: int,
    response_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO(wuxibin): remove padding and use tensordict.
    left_pad_size = prompt_width - prompt_length
    right_pad_size = response_width - response_length
    padding = (0, 0, left_pad_size, right_pad_size)
    return (
        F.pad(teacher_ids, padding, value=pad_token_id).unsqueeze(0),
        F.pad(teacher_logprobs, padding, value=0.0).unsqueeze(0),
    )


def _unpad_teacher_inputs(
    data: DataProto,
    privilege_text: Optional[str] = None,
    tokenizer=None,
    privilege_max_length: int = 0,
) -> tuple[list[int], int, int, int]:
    """Unpad valid sequence ids and prompt/response lengths from a single sample.
    The sample is a left-padded prompt concatenated with a right-padded response.

    When privilege_text is provided, the privilege tokens are inserted between
    prompt and response, yielding [prompt | privilege | response] for the teacher.
    Returns (sequence_ids, valid_prompt_length, valid_response_length, priv_length).
    TODO(wuxibin): remove padding and use tensordict.
    """
    assert len(data) == 1, "Teacher logprob computation expects a single sample"

    input_ids = data.batch["input_ids"][0]
    attention_mask = data.batch["attention_mask"][0]
    prompt_width = data.batch["prompts"][0].shape[0]
    response_width = data.batch["responses"][0].shape[0]
    assert attention_mask.shape[0] == prompt_width + response_width, (
        "attention_mask sequence length must match prompt and response widths"
    )
    valid_prompt_length = int(attention_mask[:prompt_width].sum().item())
    valid_response_length = int(attention_mask[-response_width:].sum().item())
    prompt_num_padding = prompt_width - valid_prompt_length
    prompt_ids = input_ids[prompt_num_padding : prompt_width]
    response_ids = input_ids[prompt_width : prompt_width + valid_response_length]

    if privilege_text is not None and tokenizer is not None:
        max_len = privilege_max_length if privilege_max_length > 0 else None
        priv_enc = tokenizer(
            privilege_text,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=max_len is not None,
            max_length=max_len,
        )
        priv_ids = priv_enc["input_ids"][0].to(input_ids.device)
        priv_length = len(priv_ids)
        sequence_ids = torch.cat([prompt_ids, priv_ids, response_ids], dim=0)
        logger.warning(
            "[teacher_privilege] seq_len=%d (prompt=%d + priv=%d + response=%d)",
            len(sequence_ids),
            len(prompt_ids),
            priv_length,
            len(response_ids),
        )
    else:
        priv_length = 0
        sequence_ids = torch.cat([prompt_ids, response_ids], dim=0)
        logger.warning(
            "[teacher_privilege] seq_len=%d (prompt=%d + response=%d, no privilege)",
            len(sequence_ids),
            len(prompt_ids),
            len(response_ids),
        )

    sequence_ids = normalize_token_ids(sequence_ids)
    return sequence_ids, valid_prompt_length, valid_response_length, priv_length


class AsyncTeacherLLMServerManager(AsyncLLMServerManager):
    """Teacher-specific async client used for distillation logprob computation."""

    def __init__(
        self,
        config: DictConfig,
        servers: list[tuple[str, ray.actor.ActorHandle]],
        load_balancer_handle: ray.actor.ActorHandle,
        distillation_config: DictConfig | DistillationConfig,
        pad_token_id: int,
        tokenizer=None,
    ):
        super().__init__(config=config, servers=servers, load_balancer_handle=load_balancer_handle)
        if isinstance(distillation_config, DistillationConfig):
            self.distillation_config = distillation_config
        else:
            self.distillation_config: DistillationConfig = omega_conf_to_dataclass(distillation_config)
        self.distillation_loss_config: DistillationLossConfig = self.distillation_config.distillation_loss
        self.pad_token_id = pad_token_id
        # Tokenizer used to encode privilege strings when teacher_use_privilege=True
        self.tokenizer = tokenizer

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        multi_modal_data: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute teacher log probabilities for a single unpadded sequence."""
        multi_modal_data = multi_modal_data or {}
        teacher_output = await self.generate(
            request_id=uuid4().hex,
            prompt_ids=sequence_ids,
            sampling_params=_get_teacher_sampling_params(self.distillation_config, self.distillation_loss_config),
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
        )
        # Shapes: # S, (1 or K), where S is the response length, K is either 1 or topk depending on
        # the distillation loss settings.
        teacher_ids = torch.tensor(teacher_output.extra_fields["prompt_ids"], dtype=torch.int32)
        teacher_logprobs = torch.tensor(teacher_output.extra_fields["prompt_logprobs"])
        assert teacher_ids.shape[0] == teacher_logprobs.shape[0] == len(sequence_ids)
        return teacher_ids, teacher_logprobs

    async def compute_teacher_logprobs_batch(self, data: DataProto) -> DataProto:
        """Compute teacher log probabilities for a batch of prompt-response pairs."""
        multi_modal_data_batch = data.non_tensor_batch.get("teacher_multi_modal_data")
        # Privilege support: if teacher_use_privilege is set, pass privilege texts so the
        # teacher sees [prompt | privilege | response] instead of [prompt | response].
        use_privilege = getattr(self.distillation_config, "teacher_use_privilege", False)
        privilege_texts = data.non_tensor_batch.get("privilege") if use_privilege else None
        privilege_max_length = getattr(self.distillation_config, "privilege_max_length", 0)
        tokenizer = getattr(self, "tokenizer", None)
        logger.warning(
            "[teacher_privilege] use_privilege=%s, privilege_texts_available=%s, batch_size=%d",
            use_privilege,
            privilege_texts is not None,
            len(data),
        )
        tasks = []
        lengths = []
        prompt_width = data.batch["prompts"].shape[1]
        response_width = data.batch["responses"].shape[1]

        priv_lengths = []

        # Compute logprobs for each sample in the batch
        for i in range(len(data)):
            item = data[i : i + 1]
            privilege_text = privilege_texts[i] if privilege_texts is not None else None
            sequence_ids, prompt_length, response_length, priv_length = _unpad_teacher_inputs(
                item,
                privilege_text=privilege_text,
                tokenizer=tokenizer,
                privilege_max_length=privilege_max_length,
            )
            multi_modal_data = None if multi_modal_data_batch is None else multi_modal_data_batch[i]
            lengths.append((prompt_length, response_length))
            priv_lengths.append(priv_length)
            tasks.append(
                asyncio.create_task(
                    self.compute_teacher_logprobs_single(
                        sequence_ids=sequence_ids,
                        multi_modal_data=multi_modal_data,
                    )
                )
            )
        outputs = await asyncio.gather(*tasks)

        # Pad the teacher logprobs and ids
        padded_teacher_ids = []
        padded_teacher_logprobs = []
        for (teacher_ids, teacher_logprobs), (prompt_length, response_length), priv_length in zip(
            outputs, lengths, priv_lengths, strict=True
        ):
            # Strip the privilege tokens (middle segment) so the tensor is back to
            # [prompt | response] shape before padding to [prompt_width | response_width].
            if priv_length > 0:
                teacher_ids = torch.cat(
                    [teacher_ids[:prompt_length], teacher_ids[prompt_length + priv_length :]], dim=0
                )
                teacher_logprobs = torch.cat(
                    [teacher_logprobs[:prompt_length], teacher_logprobs[prompt_length + priv_length :]], dim=0
                )
            padded_ids, padded_logprobs = _pad_teacher_outputs(
                teacher_ids,
                teacher_logprobs,
                prompt_width=prompt_width,
                response_width=response_width,
                prompt_length=prompt_length,
                response_length=response_length,
                pad_token_id=self.pad_token_id,
            )
            padded_teacher_ids.append(padded_ids)
            padded_teacher_logprobs.append(padded_logprobs)

        batch = TensorDict(
            {
                "teacher_ids": torch.cat(padded_teacher_ids),
                "teacher_logprobs": torch.cat(padded_teacher_logprobs),
            },
            batch_size=len(data),
        )
        return DataProto(batch=batch)
