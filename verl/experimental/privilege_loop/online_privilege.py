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
"""Online privilege generation.

For each response in a rollout batch, generates a personalized "privilege" string
by calling an external LLM with:
  - The current response and its reward (correct / incorrect)
  - A global context of all responses to the same prompt (grouped by uid)

The generated privilege is written back into ``batch.non_tensor_batch["privilege"]``
so that the existing privileged-distillation forward pass (``_compute_privileged_log_prob``)
and teacher forward pass (``teacher_use_privilege=True``) can consume it without change.
"""
import asyncio
import json
import logging
import os
from collections import defaultdict
from typing import Optional

import aiohttp
import numpy as np

from verl.protocol import DataProto

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARNING"))

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that analyzes reasoning quality. "
    "Respond in the same language as the question. Be concise (≤150 words)."
)

_CORRECT_TEMPLATE = """\
Question: {prompt}

The following answer is CORRECT (reward = 1):
{current_response}

Other attempts by the same model for reference:
{other_responses}

In ≤150 words, identify the key steps that make this answer correct and what to avoid."""

_INCORRECT_TEMPLATE = """\
Question: {prompt}

The following answer is INCORRECT (reward = 0):
{current_response}

Correct answers from other attempts of the same model:
{correct_examples}

Incorrect attempts (for comparison):
{wrong_examples}

In ≤150 words, pinpoint the error in the incorrect answer and explain the correct reasoning."""


def _build_privilege_prompt(
    prompt: str,
    all_responses: list[str],
    all_rewards: list[float],
    current_response: str,
    current_reward: float,
    max_context_examples: int = 3,
) -> str:
    """Build the user message sent to the privilege LLM."""
    correct = [r for r, s in zip(all_responses, all_rewards) if s > 0 and r != current_response]
    wrong = [r for r, s in zip(all_responses, all_rewards) if s <= 0 and r != current_response]

    def fmt_list(items: list[str], max_n: int) -> str:
        items = items[:max_n]
        if not items:
            return "(none)"
        return "\n---\n".join(f"[{i+1}] {t}" for i, t in enumerate(items))

    if current_reward > 0:
        return _CORRECT_TEMPLATE.format(
            prompt=prompt,
            current_response=current_response,
            other_responses=fmt_list(correct + wrong, max_context_examples),
        )
    else:
        return _INCORRECT_TEMPLATE.format(
            prompt=prompt,
            current_response=current_response,
            correct_examples=fmt_list(correct, max_context_examples),
            wrong_examples=fmt_list(wrong, max_context_examples),
        )


# ---------------------------------------------------------------------------
# Single async HTTP call
# ---------------------------------------------------------------------------


async def _call_privilege_api(
    session: aiohttp.ClientSession,
    server_url: str,
    model_name: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int,
    timeout: float,
    semaphore: asyncio.Semaphore,
    max_prompt_chars: int = 0,
) -> Optional[str]:
    """Call the OpenAI-compatible chat completions endpoint and return the text.

    Args:
        max_prompt_chars: If > 0, truncate ``user_message`` to this many characters
            before sending.  Use this to stay within the server's context-length
            limit when prompts are long (e.g. n=8 rollouts with long responses).
            A value of 6000 chars ≈ 2000 tokens, leaving ample room for 200 output
            tokens inside an 8192-token window.
    """
    if max_prompt_chars > 0 and len(user_message) > max_prompt_chars:
        user_message = user_message[:max_prompt_chars] + "\n...[truncated]"
        logger.warning(
            "[online_privilege] user_message truncated to %d chars", max_prompt_chars
        )
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    async with semaphore:
        try:
            async with session.post(
                server_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.warning("[online_privilege] API call failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Batch generation entry point
# ---------------------------------------------------------------------------


async def _generate_privileges_async(
    prompts: list[str],
    responses: list[str],
    rewards: list[float],
    uids: list[str],
    server_url: str,
    model_name: str,
    system_prompt: str,
    max_tokens: int,
    timeout: float,
    max_concurrency: int,
    min_correct: int,
    max_context_examples: int,
    max_prompt_chars: int = 6000,
) -> list[Optional[str]]:
    """Async core: generate one privilege per (uid, response) pair concurrently."""

    # Group indices by uid so each sample sees the full set of responses for its prompt
    uid_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, uid in enumerate(uids):
        uid_to_indices[uid].append(i)

    semaphore = asyncio.Semaphore(max_concurrency)
    privileges: list[Optional[str]] = [None] * len(prompts)

    async with aiohttp.ClientSession() as session:
        tasks = []
        task_indices = []

        for uid, indices in uid_to_indices.items():
            group_responses = [responses[i] for i in indices]
            group_rewards = [rewards[i] for i in indices]
            # Skip the whole group if there are not enough correct answers
            num_correct = sum(1 for r in group_rewards if r > 0)
            if num_correct < min_correct:
                logger.warning(
                    "[online_privilege] uid=%s skipped: %d correct / %d total (min_correct=%d)",
                    uid,
                    num_correct,
                    len(indices),
                    min_correct,
                )
                continue

            for i in indices:
                user_msg = _build_privilege_prompt(
                    prompt=prompts[i],
                    all_responses=group_responses,
                    all_rewards=group_rewards,
                    current_response=responses[i],
                    current_reward=rewards[i],
                    max_context_examples=max_context_examples,
                )
                tasks.append(
                    _call_privilege_api(
                        session=session,
                        server_url=server_url,
                        model_name=model_name,
                        system_prompt=system_prompt,
                        user_message=user_msg,
                        max_tokens=max_tokens,
                        timeout=timeout,
                        semaphore=semaphore,
                        max_prompt_chars=max_prompt_chars,
                    )
                )
                task_indices.append(i)

        results = await asyncio.gather(*tasks)
        for idx, result in zip(task_indices, results):
            privileges[idx] = result

    return privileges


def generate_online_privileges(
    batch: DataProto,
    tokenizer,
    server_url: str,
    model_name: str,
    system_prompt: str = "",
    max_tokens: int = 200,
    timeout: float = 30.0,
    max_concurrency: int = 32,
    min_correct: int = 1,
    max_context_examples: int = 3,
    max_prompt_chars: int = 6000,
) -> DataProto:
    """Synchronous wrapper — blocks until all privileges are generated.

    See :func:`submit_online_privileges` for a non-blocking version that can
    overlap API calls with GPU computation.
    """
    future = submit_online_privileges(
        batch=batch,
        tokenizer=tokenizer,
        server_url=server_url,
        model_name=model_name,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        timeout=timeout,
        max_concurrency=max_concurrency,
        min_correct=min_correct,
        max_context_examples=max_context_examples,
        max_prompt_chars=max_prompt_chars,
    )
    return collect_online_privileges(batch, future)


def submit_online_privileges(
    batch: DataProto,
    tokenizer,
    server_url: str,
    model_name: str,
    system_prompt: str = "",
    max_tokens: int = 200,
    timeout: float = 30.0,
    max_concurrency: int = 32,
    min_correct: int = 1,
    max_context_examples: int = 3,
    max_prompt_chars: int = 6000,
) -> "concurrent.futures.Future[list[Optional[str]]]":

    """Start privilege generation in a background thread and return a Future.

    The caller can do other work (e.g. GPU forward passes) while the HTTP
    requests are in flight, then call :func:`collect_online_privileges` when
    the privilege strings are actually needed.

    Example::

        future = submit_online_privileges(batch, tokenizer, ...)
        old_log_prob = compute_old_log_prob(batch)   # GPU runs here
        batch = collect_online_privileges(batch, future)
        priv_result = compute_privileged_log_prob(batch)
    """
    import concurrent.futures

    system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT

    # Extract all needed data eagerly so the background thread does not race
    # with any in-place mutations to `batch` on the main thread.
    prompt_texts: list[str] = tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
    response_texts: list[str] = tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
    rewards: list[float] = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
    uids: list[str] = batch.non_tensor_batch["uid"].tolist()

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _run():
        return asyncio.run(
            _generate_privileges_async(
                prompts=prompt_texts,
                responses=response_texts,
                rewards=rewards,
                uids=uids,
                server_url=server_url,
                model_name=model_name,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                timeout=timeout,
                max_concurrency=max_concurrency,
                min_correct=min_correct,
                max_context_examples=max_context_examples,
                max_prompt_chars=max_prompt_chars,
            )
        )

    future = executor.submit(_run)
    # Attach executor so the caller can shut it down if desired.
    future._executor = executor  # type: ignore[attr-defined]
    return future


def collect_online_privileges(
    batch: DataProto,
    future: "concurrent.futures.Future[list[Optional[str]]]",
) -> DataProto:
    """Wait for a :func:`submit_online_privileges` future and write results into *batch*.

    Blocks until the background API calls complete, then populates
    ``batch.non_tensor_batch["privilege"]``.
    """
    privileges: list[Optional[str]] = future.result()
    # Shut down the thread-pool that was created in submit_online_privileges.
    executor = getattr(future, "_executor", None)
    if executor is not None:
        executor.shutdown(wait=False)

    privilege_arr = np.array([p if p is not None else "" for p in privileges], dtype=object)
    batch.non_tensor_batch["privilege"] = privilege_arr

    n_generated = int((privilege_arr != "").sum())
    logger.warning(
        "[online_privilege] generated %d / %d privileges",
        n_generated,
        len(privileges),
    )
    return batch
