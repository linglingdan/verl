# async_rollout_manager 接口分析

## 📋 当前暴露的主要方法

### 在 `AgentLoopManager` 中

**文件**：[verl/experimental/agent_loop/agent_loop.py](../../verl/experimental/agent_loop/agent_loop.py) 第992-1250行

#### 1. `generate_sequences(prompts: DataProto)` - 主方法 ✅

```python
@auto_await
async def generate_sequences(self, prompts: DataProto) -> DataProto:
    """分割输入批次并分发给agent loop workers
    
    Args:
        prompts (DataProto): 输入批次
    
    Returns:
        DataProto: 完整的响应序列（包含生成的tokens）
    """
```

**功能**：
- ✅ 接收 DataProto (包含 prompts/input_ids)
- ✅ 调用 vLLM/sglang 完整生成流程（prefill + decode）
- ✅ 输出完整响应序列和日志概率

**不支持**：
- ❌ 纯 prefill 阶段
- ❌ 直接获取 logits

---

#### 2. 其他辅助方法

```python
async def clear_kv_cache(self)
    """清除所有rollout的KV缓存"""

async def start_profile(self, **kwargs)
    """启动性能分析"""

async def stop_profile(self)
    """停止性能分析"""
```

---

## 🔍 底层推理引擎现状

### vLLM 调用链

```
AgentLoopManager.generate_sequences()
    ↓
AgentLoopWorker.generate_sequences()
    ↓
AsyncLLMServerManager.generate()
    ↓
server.generate.remote()
    ↓ (Ray Remote Actor)
ServerAdapter(vllm_rollout.py)
    ↓
vLLM LLMEngine
    └─ 完整推理流程：prefill + decode
    └ 输出：完整生成序列
```

### vLLM 底层接口 (但未暴露给 VERL)

vLLM 底层实际支持：
- `engine.generate()` - 完整生成
- `engine.encode()` - 仅 prefill（存储在 KV 缓存）
- `engine.generate_greedy()` / `generate_beam_search()`

But VERL 的 OpenAI 兼容 API 层只暴露：
- Chat completions API
- 不支持低级的 prefill/decode 分离调用

---

## ❓ 你的需求分析

### 需求：输入batch，仅prefill，输出logits

```python
# 你想要的：
gen_batch_output = self.async_rollout_manager.forward(  # 不存在
    batch=batch,
    mode='prefill_only'  # 仅对prompt做prefill
)
# 输出：logits, hidden_states 等
```

### 现状：

**不直接支持**。当前架构设计为：
1. 所有推理完全由远程推理引擎（vLLM/sglang）管理
2. VERL trainer 只与推理引擎通过 OpenAI API 交互
3. 低级操作（prefill/decode 分离）不暴露

---

## 💡 替代方案

### 方案1️⃣：直接使用 vLLM 引擎（推荐）

如果你只是需要 prefill 的 logits，不需要完整生成序列：

```python
from vllm import LLM, SamplingParams

# 直接用vLLM而不是VERL rollout
llm = LLM(model="Qwen/Qwen2.5-3B-Instruct", tensor_parallel_size=1)

# Prefill only - 不生成新tokens
outputs = llm.generate(
    prompts=batch["input_ids"],
    sampling_params=SamplingParams(
        max_tokens=1,  # 最小值，实际只做prefill
        logprobs=64,   # 返回logits
    )
)

logits = outputs[0].logits  # 获取logits
```

### 方案2️⃣：修改 VERL 模型引擎

如果要在 VERL 框架中集成，需要：

1. **添加低级 API 到推理引擎**：
   - 在 vllm_rollout.py 中添加 `prefill_only` 方法
   - 或在 ServerAdapter 中暴露低级方法

2. **代码示例**（伪代码）：

```python
# 在 verl/workers/rollout/vllm_rollout.py 中添加

class ServerAdapter:
    async def prefill_only(self, prompt_ids, return_logits=True):
        """仅对prompt做prefill，不生成新tokens"""
        output = await self.llm_engine.generate(
            prompt_ids=prompt_ids,
            sampling_params=SamplingParams(
                max_tokens=1,      # 不生成
                logprobs=None,     # 可选
            ),
            request_id=uuid4().hex,
        )
        if return_logits:
            return output.logits
        else:
            return output.hidden_states
```

3. **在 AsyncLLMServerManager 中暴露**：

```python
# 在 verl/experimental/agent_loop/agent_loop.py 中

class AsyncLLMServerManager:
    async def prefill_only(self, request_id, prompt_ids, **kwargs):
        """Prefill only，返回logits"""
        server_id, server = await self._acquire_server(request_id)
        try:
            output = await server.prefill_only.remote(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                **kwargs,
            )
            return output
        finally:
            self._release_server(server_id)
```

4. **最后在 AgentLoopManager 中暴露**：

```python
class AgentLoopManager:
    async def prefill_only(self, prompts: DataProto) -> DataProto:
        """仅Prefill，返回logits"""
        chunks = prompts.chunk(len(self.agent_loop_workers))
        outputs = await asyncio.gather(
            *[
                worker.prefill_only.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunks)
            ]
        )
        return DataProto.concat(outputs)
```

### 方案3️⃣：使用教师模型推理

如果你的实际需求是获取某个模型在prompt上的概率分布，VERL 蒸馏中已经这样做了：

```python
# 在 VERL distillation 中，教师模型通过以下方式获取logits：
teacher_output = await teacher_server_manager.compute_teacher_logprobs_batch(
    prompt_ids=batch['input_ids'],
    topk=64,  # 返回top-64 logprobs
)

# 这实际是：
# 1. Prefill：处理整个序列到model最后一层
# 2. 提取：返回vocab的高概率部分
```

---

## 📊 功能对比表

| 功能 | 当前支持 | 难度 | 实现位置 |
|------|---------|------|---------|
| 完整生成序列 | ✅ | - | `generate_sequences()` |
| 仅 prefill + logits | ❌ | 中 | 需要修改3个文件 |
| 仅 prefill + hidden states | ❌ | 高 | 需要vLLM底层改动 |
| 获取attention map | ❌ | 高 | 需要vLLM返回中间结果 |

---

## 🎯 总结

**直接回答**：

你要找的 `forward` 或 `prefill` 方法**不存在**。

原因是：
- VERL 的设计哲学是"生成为中心"，所有推理都是为了生成完整序列
- 底层推理引擎（vLLM）通过 OpenAI API 与 VERL 通信
- 低级别的 prefill/decode 分离不暴露在接口层

**最佳选择**：
1. 如果只需要 logits - 直接用 vLLM（方案1️⃣）
2. 如果要在 VERL 框架中使用 - 自己扩展接口（方案2️⃣）
3. 如果要获得教师指导 - 用现有的蒸馏机制（方案3️⃣）

