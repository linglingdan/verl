# VERL：如何从 vLLM 返回额外向量并在 generate_sequences 中获取

## 问题描述

在 `generate_sequences` 的返回结果中，除了生成的 token 序列和 log_probs 之外，
如何携带 vLLM 生成时产生的额外数据（如某个向量、隐状态等）？

---

## 完整数据流路径

数据从 vLLM 推理引擎流向最终调用方，**经过 5 层传递**，全程通过 `extra_fields` 字典携带：

```
[1] vllm_async_server.py: generate()
         │  构造 extra_fields dict，放入自定义向量
         │  return TokenOutput(extra_fields=extra_fields)
         ↓
[2] single_turn_agent_loop.py: run()
         │  TokenOutput → AgentLoopOutput
         │  AgentLoopOutput(extra_fields=token_output.extra_fields)  ← 直接赋值
         ↓
[3] agent_loop.py: _agent_loop_postprocess()
         │  AgentLoopOutput → _InternalAgentLoopOutput
         │  _InternalAgentLoopOutput(extra_fields=output.extra_fields)  ← 直接赋值
         ↓
[4] agent_loop.py: _postprocess()
         │  自动遍历所有样本的 extra_fields，收集每个 key
         │  non_tensor_batch["my_vector"] = np.array([样本0的向量, 样本1的向量, ...])
         ↓
[5] generate_sequences() 调用方
         gen_batch_output.non_tensor_batch["my_vector"]  ← 在这里取到
```

---

## 关键代码说明（逐层注释）

### 层 1：vLLM 服务端写入 extra_fields

**文件**：[verl/workers/rollout/vllm_rollout/vllm_async_server.py](../../verl/workers/rollout/vllm_rollout/vllm_async_server.py)  约第 516-552 行

```python
async def generate(self, prompt_ids, sampling_params, request_id, ...) -> TokenOutput:

    # ...运行 vLLM 推理...
    final_res: Optional[RequestOutput] = None
    async for output in generator:
        final_res = output

    # ① extra_fields 是一个普通 dict，默认只存 global_steps
    extra_fields = {"global_steps": self.global_steps}

    # ② 如果你想携带额外向量，在这里加入
    # 例如：把 vLLM 返回的某个字段存进去
    # extra_fields["my_vector"] = final_res.outputs[0].YOUR_FIELD

    token_ids = final_res.outputs[0].token_ids

    # ③ extra_fields 通过 TokenOutput 向上传递
    return TokenOutput(
        token_ids=token_ids,
        log_probs=log_probs,
        routed_experts=routed_experts,
        stop_reason=stop_reason,
        num_preempted=num_preempted,
        extra_fields=extra_fields,   # ← 携带自定义数据
    )
```

**`TokenOutput` 的定义**（[verl/workers/rollout/replica.py](../../verl/workers/rollout/replica.py) 第 39 行）：

```python
class TokenOutput(BaseModel):
    token_ids: list[int]
    log_probs: Optional[list[float]] = None
    routed_experts: Optional[Any] = None
    stop_reason: Optional[str] = None
    num_preempted: Optional[int] = None
    extra_fields: dict[str, Any] = {}   # ← 专门用于扩展的字段
```

---

### 层 2：SingleTurnAgentLoop 直接透传

**文件**：[verl/experimental/agent_loop/single_turn_agent_loop.py](../../verl/experimental/agent_loop/single_turn_agent_loop.py) 第 42 行

```python
async def run(self, sampling_params, **kwargs) -> AgentLoopOutput:
    # 调用服务端生成
    output: TokenOutput = await self.server_manager.generate(
        request_id=uuid4().hex,
        prompt_ids=prompt_ids,
        sampling_params=sampling_params,
        ...
    )

    # TokenOutput → AgentLoopOutput：extra_fields 直接赋值，无损传递
    output: AgentLoopOutput = AgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=output.token_ids[:self.response_length],
        response_logprobs=output.log_probs[:self.response_length] if output.log_probs else None,
        ...
        extra_fields=output.extra_fields,   # ← 直接赋值，my_vector 在这里
    )
    return output
```

---

### 层 3：_agent_loop_postprocess 继续透传

**文件**：[verl/experimental/agent_loop/agent_loop.py](../../verl/experimental/agent_loop/agent_loop.py) 第 615 行

```python
async def _agent_loop_postprocess(self, output, validate, **kwargs) -> _InternalAgentLoopOutput:
    # 注意：这里会往 extra_fields 里额外加 raw_prompt
    output.extra_fields["raw_prompt"] = kwargs["raw_prompt"]

    # teacher_ids / teacher_logprobs 会被 pop 出来单独处理
    teacher_ids, teacher_logprobs = (
        output.extra_fields.pop("teacher_ids", None),
        output.extra_fields.pop("teacher_logprobs", None),
    )

    # AgentLoopOutput → _InternalAgentLoopOutput：extra_fields 再次直接赋值
    return _InternalAgentLoopOutput(
        ...
        teacher_logprobs=teacher_logprobs,
        teacher_ids=teacher_ids,
        extra_fields=output.extra_fields,   # ← my_vector 仍在这里
    )
```

---

### 层 4：_postprocess 自动收集所有 extra_fields（核心！）

**文件**：[verl/experimental/agent_loop/agent_loop.py](../../verl/experimental/agent_loop/agent_loop.py) 第 870 行

```python
def _postprocess(self, inputs: list[_InternalAgentLoopOutput], ...) -> DataProto:

    # ... 处理 tensor 字段 ...

    # ★★★ 关键：自动遍历所有样本的 extra_fields，不需要显式声明 key
    extra_fields = {}
    default_extra_keys = {"turn_scores", "tool_rewards", "min_global_steps", "max_global_steps", "extras"}

    # 收集所有样本中出现过的 key（取并集）
    all_keys = set(key for input_item in inputs for key in input_item.extra_fields) | default_extra_keys

    for key in all_keys:
        temp_arr = np.empty(len(inputs), dtype=object)
        # 每个样本的该 key 值放到对应位置
        temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
        extra_fields[key] = temp_arr   # shape: (batch_size,)，dtype=object

    # 合并到 non_tensor_batch
    non_tensor_batch.update(extra_fields)

    # 最终 DataProto 中 non_tensor_batch["my_vector"] 就有了
    return DataProto(
        batch=batch,
        non_tensor_batch=non_tensor_batch,
        meta_info=meta_info,
    )
```

---

## 实际操作：只需改一处

### 修改位置

**文件**：[verl/workers/rollout/vllm_rollout/vllm_async_server.py](../../verl/workers/rollout/vllm_rollout/vllm_async_server.py)

找到 `generate()` 方法中构造 `extra_fields` 的位置，加入你的向量：

```python
# 原始代码（约第 516 行）
extra_fields = {"global_steps": self.global_steps}
extract_prompt_logprobs(
    output=final_res,
    num_prompt_logprobs=sampling_params.prompt_logprobs,
    result_dict=extra_fields,
)

# ✅ 在这里加入你的向量（以 hidden_states 为例）
extra_fields["my_vector"] = final_res.outputs[0].YOUR_CUSTOM_FIELD
```

### 在调用方取出数据

```python
# ray_trainer.py 或你的自定义 trainer 中
gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)

# 直接从 non_tensor_batch 读取
my_vectors = gen_batch_output.non_tensor_batch["my_vector"]
# 类型：np.ndarray，shape=(batch_size,)，dtype=object
# my_vectors[i] 是第 i 个样本对应的向量

# 如果是 numpy 数组，可以 stack 成张量：
import numpy as np
import torch
my_tensor = torch.from_numpy(np.stack(my_vectors.tolist()))
```

---

## 为什么这样设计

`extra_fields` 的设计哲学是**约定优于配置**：

- `TokenOutput.extra_fields` 是一个开放字典，专门为扩展保留
- `_postprocess` 中的自动收集逻辑（`all_keys` 取并集）确保任何新 key 都无需注册即可透传
- 已有的 `teacher_logprobs` 和 `teacher_ids` 也走这套机制（只是收集后被 pop 出来单独走 TensorDict）

不需要修改以下文件：
- `single_turn_agent_loop.py` —— 直接 `extra_fields=output.extra_fields`，自动透传
- `agent_loop.py: _agent_loop_postprocess` —— 直接 `extra_fields=output.extra_fields`，自动透传
- `agent_loop.py: _postprocess` —— 自动收集所有 key，无需修改
- `AgentLoopOutput` / `_InternalAgentLoopOutput` 数据类 —— `extra_fields: dict` 已经支持任意内容

---

## 注意事项

### 1. extra_fields 存储在 non_tensor_batch，不在 batch（TensorDict）

`non_tensor_batch` 使用 `dtype=object` 的 numpy 数组，每个元素可以是任意 Python 对象（列表、numpy 数组、张量等）。如果需要放入 TensorDict 参与训练前向计算，需要手动 stack：

```python
my_vectors = gen_batch_output.non_tensor_batch["my_vector"]  # np.ndarray of objects
my_tensor = torch.stack([torch.tensor(v) for v in my_vectors])  # 转为真正的 Tensor
```

### 2. extra_fields 的 teacher_ids / teacher_logprobs 被特殊处理

在 `_agent_loop_postprocess` 中，`teacher_ids` 和 `teacher_logprobs` 会被 `pop` 出来，经过 padding 后放入 `_InternalAgentLoopOutput` 的专用字段，最终写入 `batch`（TensorDict）而非 `non_tensor_batch`。如果你的向量也需要参与 FSDP 训练计算，可以参考这个模式。

### 3. key 命名避免与已有字段冲突

已被占用的 `extra_fields` key：

| Key | 用途 |
|-----|------|
| `global_steps` | 当前训练步数 |
| `teacher_ids` | 蒸馏教师 top-k token IDs（会被 pop） |
| `teacher_logprobs` | 蒸馏教师 log probs（会被 pop） |
| `raw_prompt` | 原始 prompt 消息列表 |
| `reward_extra_info` | 奖励额外信息 |
| `turn_scores` | 多轮对话分数 |
| `tool_rewards` | 工具调用奖励 |

---

## 完整示例

假设 vLLM 新增了返回每个 token 的 hidden state 向量，步骤如下：

**Step 1**：在 `vllm_async_server.py` 的 `generate()` 中写入：

```python
# 第 534 行附近
extra_fields = {"global_steps": self.global_steps}
extract_prompt_logprobs(output=final_res, ...)

# 新增：把 hidden_state 存入 extra_fields
if hasattr(final_res.outputs[0], "hidden_states") and final_res.outputs[0].hidden_states is not None:
    extra_fields["hidden_states"] = final_res.outputs[0].hidden_states  # 例如 np.ndarray (seq_len, hidden_dim)

return TokenOutput(..., extra_fields=extra_fields)
```

**Step 2**：在 trainer 的 `fit()` 中读取：

```python
gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)

# 取出向量，每个元素是一个样本的 hidden_states
hidden_states_list = gen_batch_output.non_tensor_batch["hidden_states"]  # (batch_size,) dtype=object

# 转为 Tensor（假设每个样本的 hidden_states 形状相同）
hidden_states_tensor = torch.from_numpy(np.stack(hidden_states_list.tolist()))
# shape: (batch_size, seq_len, hidden_dim)
```

**无需改动中间任何文件**，数据自动流过 5 层传递链。
