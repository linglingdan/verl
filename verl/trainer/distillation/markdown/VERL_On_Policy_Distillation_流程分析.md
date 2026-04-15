# VERL On-Policy Distillation 流程详解

## 📌 概述

本文档详细分析VERL (Versatile Reinforcement Learning) 中的On-Policy Knowledge Distillation (蒸馏)训练流程，从数据加载、教师模型推理、学生模型生成、到损失计算的完整过程。

**配置参考**：
- 入口脚本：`/Users/liudan/code/verl/examples/on_policy_distillation_trainer/run_qwen_gsm8k.sh`
- 学生模型：`Qwen2.5-0.5B`
- 教师模型：`Qwen2.5-3B-Instruct`
- 数据集：`GSM8K (General School Math 8K)`

---

## 🚀 第一部分：流程总览

### 1.1 从入口函数开始

**入口点**：`verl.trainer.main_ppo`

```bash
# run_qwen_gsm8k.sh 最后的执行命令
python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_trainer.yaml' \
    ${DATA[@]} ${MODEL[@]} ${DISTILLATION[@]} ...
```

**对应代码文件**：[verl/trainer/main_ppo.py](../../verl/trainer/main_ppo.py)

### 1.2 主函数执行流程

```python
# main_ppo.py 第 34-40 行
@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management."""
    # 自动设置设备（GPU/NPU）
    auto_set_device(config)
    # 迁移旧版reward实现
    config = migrate_legacy_reward_impl(config)
    # 执行PPO训练
    run_ppo(config)
```

**关键步骤**：
1. Hydra加载配置文件
2. 初始化Ray集群
3. 创建TaskRunner远程对象
4. 调用TaskRunner.run()执行任务

---

## 🔄 第二部分：数据流转

### 2.1 阶段1️⃣ - 数据加载（Raw Data → Training Batch）

#### A. 数据文件加载

**源文件**：[verl/trainer/main_ppo.py](../../verl/trainer/main_ppo.py) 第 377-425 行

```python
def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True, max_samples: int = -1):
    """创建强化学习数据集"""
    
    # 第1步：获取数据集类
    from verl.utils.dataset.rl_dataset import get_dataset_class
    dataset_cls = get_dataset_class(data_config)  # 返回RLHFDataset类
    
    # 第2步：实例化数据集
    dataset = dataset_cls(
        data_files=data_paths,              # ['path/gsm8k/train.parquet']
        tokenizer=tokenizer,                # 模型tokenizer
        processor=processor,                # 多模态处理器（可选）
        config=data_config,                 # {'max_prompt_length': 256, ...}
        max_samples=max_samples,            # -1 表示全部加载
    )
    
    return dataset
```

**配置参数**（来自run_qwen_gsm8k.sh）：
```yaml
data:
  train_files: ['path/to/gsm8k/train.parquet']
  max_prompt_length: 256
  max_response_length: 512
  train_batch_size: 128
  filter_overlong_prompts: True
  truncation: 'error'  # 长度超过max_prompt_length时抛错
```

#### B. RLHFDataset 的 __getitem__

**源文件**：[verl/utils/dataset/rl_dataset.py](../../verl/utils/dataset/rl_dataset.py)

> ⚠️ **重要**：新版 VERL 的 `RLHFDataset.__getitem__` **不做 tokenization**，tokenization 已移至 AgentLoop 内部。

```python
def __getitem__(self, item):
    """For rollout, apply_chat_template has been moved to AgentLoop, so we only return raw_prompt here."""
    row_dict: dict = self.dataframe[item]
    # raw_prompt: Parquet 中的 prompt 字段构建出的 chat messages list
    # 例如 [{"role": "user", "content": "What is 2+2?"}]
    row_dict["raw_prompt"] = self._build_messages(row_dict)

    # 一个占位 tensor，确保 DataProto.batch 不为空
    row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

    row_dict["index"] = row_dict.get("extra_info", {}).get("index", 0)
    row_dict["tools_kwargs"] = row_dict.get("extra_info", {}).get("tools_kwargs", {})
    row_dict["interaction_kwargs"] = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
    return row_dict
```

**每个样本返回的字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `dummy_tensor` | `torch.Tensor (1,)` | 占位符 |
| `raw_prompt` | `list[dict]` | chat messages（**未 tokenize 的原始对话**） |
| `data_source` | `str` | 数据集来源标识 |
| `extra_info` | `dict` | Parquet 额外字段 |
| `index` | `int` | 样本序号 |
| `tools_kwargs` | `dict` | 工具调用参数 |
| `interaction_kwargs` | `dict` | 交互参数 |

#### C. DataLoader 输出的 Batch

DataLoader 经 `collate_fn` 合并后，**每个 iteration 给出的原始 batch** 内容如下：

**`batch.batch`（tensor，由 `collate_fn` 的 torch.stack 生成）：**
| 字段 | 形状 | 说明 |
|------|------|------|
| `dummy_tensor` | (128, 1) | 占位 tensor |

**`batch.non_tensor_batch`（np.ndarray of object）：**
| 字段 | 说明 |
|------|------|
| `raw_prompt` | ⭐ 每个样本的 chat messages list（**未 tokenize**） |
| `data_source` | 数据集来源 |
| `extra_info` | Parquet 元数据 |
| `index` | 样本序号 |
| `tools_kwargs` | 工具调用参数 |
| `interaction_kwargs` | 交互参数 |

**`batch.meta_info`（在 fit() 中追加）：**
| 字段 | 说明 |
|------|------|
| `temperature` | 来自 `config.actor_rollout_ref.rollout.temperature` |
| `uid` | 每样本随机 UUID（追加到 non_tensor_batch） |

> ⚠️ **DataLoader 出来的 batch 里没有 `input_ids` / `attention_mask` / `prompts`**，这些都在后续 `generate_sequences` 的 AgentLoop 内部生成。

---

## 🤖 第三部分：学生和教师推理

### 3.1 阶段2️⃣ - 学生模型生成响应

**流程文件**：[verl/trainer/ppo/ray_trainer.py](../../verl/trainer/ppo/ray_trainer.py) 第 1370+ 行

#### A. 真实调用方式

```python
# fit() 内部 — 实际代码（已简化注释）
gen_batch = self._get_gen_batch(batch)  # 从 batch 中提取生成所需字段
gen_batch.meta_info["global_steps"] = self.global_steps
# 如果 rollout.n > 1，每个 prompt 重复采样 n 次
gen_batch_output = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

# ⭐ 核心调用：调用异步生成管理器生成响应
gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

# 将 prompt batch 也 repeat n 次，然后与生成结果合并
batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
batch = batch.union(gen_batch_output)   # 将生成结果 merge 进 batch，不是手动赋值
```

> ⚠️ **注意**：不存在 `batch['responses'] = gen_batch_output['response_ids']` 这样的手动赋值，实际使用 `batch.union()` 合并。  
> `temperature` 等采样参数通过 `meta_info["temperature"]` 传递，不是 `sampling_params` 参数。

#### B. `generate_sequences` 内部流程

`generate_sequences` 内部由 **AgentLoop** 执行：

```
gen_batch.non_tensor_batch["raw_prompt"]   ← chat messages list
         ↓
AgentLoop._run_agent_loop()
  tokenizer.apply_chat_template(raw_prompt)  ← 在这里才做 tokenization
         ↓
  vLLM 推理引擎生成 response tokens
         ↓
AgentLoop._agent_loop_postprocess()
  left-pad prompt_ids  →  prompts   (prompt_length)
  right-pad response_ids → responses (response_length)
  concat → input_ids   (prompt + response)
```

#### C. `generate_sequences` 输出字段

**生成后 `gen_batch_output`（batch.union 合并进 batch）**：

| 字段 | 形状 | 说明 |
|------|------|------|
| `prompts` | (128, 256) | **左填充**的 prompt token IDs |
| `responses` | (128, 512) | **右填充**的生成 response token IDs |
| `input_ids` | (128, 768) | prompts + responses 拼接 |
| `attention_mask` | (128, 768) | 0=padding, 1=real token |
| `response_mask` | (128, 512) | 1=LLM生成token, 0=padding/tool响应 |
| `position_ids` | (128, 768) | 递增位置编号 |

> **注意**：本配置 `calculate_log_probs=False`，生成阶段**不会产生** `rollout_log_probs`。

### 3.2 阶段3️⃣ - 教师模型推理（关键！）

**关键概念**：教师模型通过返回**top-k token的日志概率**来指导学生模型学习。

#### A. 教师推理的两种部署模式

```
配置项：distillation.teacher_model.enable_resource_pool
  = False（本配置）→ 教师与学生 colocated，共享 GPU 资源
  = True           → 教师使用独立 Ray resource pool
```

本配置 `TEACHER_RESOURCE_POOL=False`，触发 colocated 路径：

```python
# ray_trainer.py fit() 中
if self._should_compute_teacher_colocate(batch):   # enable_resource_pool=False 时为 True
    batch_teacher = self._compute_teacher_colocate(batch)
    batch = batch.union(batch_teacher)
```

```python
def _compute_teacher_colocate(self, batch: DataProto) -> DataProto:
    """Compute teacher logprobs after rollout when teacher and student are colocated."""
    teacher_batch = self.teacher_model_manager.compute_logprobs(batch)
    return teacher_batch
```

#### B. 教师推理的输入与输出

**输入**：`batch`（此时已包含生成完毕的 `input_ids`，即完整的 prompt + response 序列）

**输出**（通过 `batch.union()` 合并进 batch）：

| 字段 | 形状 | 说明 |
|------|------|------|
| `teacher_logprobs` | (128, 768, 64) | 每个 token 位置 top-64 的对数概率 |
| `teacher_ids` | (128, 768, 64) | 对应的 top-64 token IDs |

教师模型使用 vLLM 推理引擎，通过设置 `prompt_logprobs=topk` 实现对已有序列每个位置返回 top-k 概率分布，不生成新 token（即整个序列作为 prompt 传入）。

#### C. 教师模型推理的工程细节

```python
# run_qwen_gsm8k.sh中的教师配置
distillation:
  teacher_model:
    model_path: "Qwen/Qwen2.5-3B-Instruct"
    enable_resource_pool: False           # colocated 模式
    n_gpus_per_node: 4                    # 使用4个GPU运行教师
    inference:
      name: "vllm"                        # 使用vLLM推理引擎
      tensor_model_parallel_size: 1       # 不用TP并行
      gpu_memory_utilization: 0.3         # 每个GPU使用30%显存
      enforce_eager: True                 # 不使用CUDA图加速
      max_model_len: 769                  # 最大sequence长度
  distillation_loss:
    topk: 64                              # 每个位置返回 top-64
    loss_mode: "k1"
    use_task_rewards: False               # 不使用外部奖励，纯蒸馏
    use_policy_gradient: True             # 使用PPO更新
```

---

## 📊 第四部分：损失计算

### 4.1 阶段4️⃣ - 蒸馏损失计算（最重要！）

**源文件**：[verl/trainer/distillation/losses.py](../../verl/trainer/distillation/losses.py)

#### A. 损失模式选择

在run_qwen_gsm8k.sh中配置：
```bash
DISTILLATION_LOSS_MODE="k1"  # 使用k1损失模式
USE_POLICY_GRADIENT=True      # 使用PPO梯度进行优化
```

#### B. k1损失的计算原理

对于每个token位置 $t$ 和采样的token $a_t$：

$$\text{k1\_loss}_t = \begin{cases}
\log q_t(a_t) - \log p_t(a_t) & \text{if } a_t \in \text{top-64} \\
0 & \text{otherwise}
\end{cases}$$

其中：
- $q_t(a_t)$：学生模型在位置$t$对token $a_t$的概率
- $p_t(a_t)$：教师模型在位置$t$对token $a_t$的概率
- top-64：教师返回的概率最高的64个token

**关键特点**：
- **稀疏奖励**：只对教师高概率的token进行优化
- **负奖励**：如果学生概率 < 教师概率，loss为负
- **正奖励**：如果学生概率 > 教师概率，loss为正

#### C. 代码实现

**主蒸馏损失函数**：[verl/trainer/distillation/losses.py](../../verl/trainer/distillation/losses.py) 第 238-284 行

```python
def distillation_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    计算蒸馏损失和相关指标
    
    Args:
        config: Actor配置
        distillation_config: 蒸馏配置
        model_output: 模型输出，包含学生日志概率
        data: 包含教师日志概率的数据
    
    Returns:
        distillation_loss: 标量损失值
        distillation_metrics: 指标字典
    """
    
    # 获取蒸馏损失配置
    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    
    # 获取损失计算函数（根据loss_mode选择，如"k1", "k3"等）
    distillation_loss_fn = get_distillation_loss_fn(loss_config.loss_mode)
    
    # 调用具体的损失函数
    distillation_losses, distillation_metrics = distillation_loss_fn(
        config=config,
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )
    
    # distillation_losses: (batch_size, response_length) 每个token的损失
    
    # 获取response mask（用于忽略padding token）
    response_mask = data["response_mask"]
    
    # 对损失进行clamping（防止过大的梯度）
    if loss_config.loss_max_clamp is not None:
        distillation_losses = distillation_losses.clamp(
            min=-loss_config.loss_max_clamp,
            max=loss_config.loss_max_clamp
        )
        # loss_max_clamp=10.0（来自run_qwen_gsm8k.sh）
    
    # 🔑 关键选择：是否使用Policy Gradient
    if loss_config.use_policy_gradient:
        # ✅ 使用PPO + 蒸馏（这是GSM8K配置使用的方式）
        
        # 获取PPO损失函数
        policy_loss_fn = get_policy_loss_fn(loss_config.policy_loss_mode)
        
        # 关键insight：负蒸馏损失作为奖励！
        # 蒸馏损失低 → 优势高 → 增加该action的概率
        advantages = -distillation_losses.detach()  # 🔴 负损失 = 奖励
        
        # 调用PPO损失计算（使用蒸馏损失作为奖励信号）
        distillation_loss, pg_metrics = policy_loss_fn(
            old_log_prob=data["old_log_probs"],      # 旧策略日志概率
            log_prob=model_output["log_probs"],      # 新策略日志概率
            advantages=advantages,                    # 蒸馏-based优势
            response_mask=response_mask,
            loss_agg_mode=config.loss_agg_mode,      # "token-mean"
            config=loss_config,
        )
        
        # 添加指标
        distillation_metrics.update(pg_metrics)
    else:
        # 直接反向传播蒸馏损失（SFT方式）
        distillation_loss = agg_loss(
            loss_mat=distillation_losses,
            loss_mask=response_mask,
            loss_agg_mode=config.loss_agg_mode,
        )
    
    return distillation_loss, distillation_metrics
```

#### D. k1损失具体计算

**源文件**：[verl/trainer/distillation/losses.py](../../verl/trainer/distillation/losses.py) 第 317-365 行

```python
@register_distillation_loss(
    DistillationLossSettings(names=["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"], use_estimator=True)
)
def compute_distillation_loss_reverse_kl_estimator(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    使用单样本KL估计器计算蒸馏损失
    
    支持多种KL散度估计方式：
    - "kl": 标准KL散度
    - "k1": 学生log概率 - 教师log概率（单步估计）
    - "k3": 更复杂的估计器
    - 等等...
    """
    
    # 获取学生模型的日志概率
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    # 形状: (batch_size, response_length)
    
    # 获取教师模型的日志概率（squeeze掉topk维度）
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    # data["teacher_logprobs"] 原始形状: (batch, seq_len, topk)
    # squeeze(-1) 后: (batch, seq_len)
    # 为什么可以squeeze？因为我们只取采样token的概率！
    
    # 获取response mask（用于过滤padding token）
    response_mask_bool = data["response_mask"].bool()
    
    # 确认形状一致
    assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape
    
    # 获取损失配置
    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    
    # 🔑 计算KL惩罚（支持多种模式）
    distillation_losses = kl_penalty(
        logprob=student_log_probs,        # 学生概率 shape: (batch, resp_len)
        ref_logprob=teacher_log_probs,    # 教师概率 shape: (batch, resp_len)
        kl_penalty=loss_config.loss_mode  # "k1" 或其他模式
    )
    # 输出 distillation_losses: (batch, resp_len)
    
    # 计算指标（如果需要）
    distillation_metrics = {
        "distillation/loss_mean": distillation_losses[response_mask_bool].mean().item(),
        # ... 其他指标
    }
    
    return distillation_losses, distillation_metrics
```

#### E. kl_penalty 核心计算

**源文件**：[verl/trainer/ppo/core_algos.py](../../verl/trainer/ppo/core_algos.py) （需要在实际代码中查看）

```python
def kl_penalty(logprob, ref_logprob, kl_penalty="k1"):
    """
    计算KL惩罚的多种变体
    
    Args:
        logprob: 学生模型日志概率 shape: (batch, seq_len)
        ref_logprob: 教师模型日志概率 shape: (batch, seq_len)
        kl_penalty: 惩罚模式，如 "k1", "k2", "k3", "abs", "mse"
    
    Returns:
        loss: 每个token的损失 shape: (batch, seq_len)
    """
    
    if kl_penalty == "k1":
        # k1: 学生 - 教师
        # 1️⃣ 直观解释：
        #    - 如果学生概率 > 教师 → loss > 0（学生过自信）
        #    - 如果学生概率 < 教师 → loss < 0（学生不够自信）
        loss = logprob - ref_logprob
        return loss
    
    elif kl_penalty == "k2":
        # k2: 2 * (学生 - 教师)
        loss = 2.0 * (logprob - ref_logprob)
        return loss
    
    elif kl_penalty == "k3":
        # k3: (logprob - ref_logprob) ^ 2
        loss = (logprob - ref_logprob) ** 2
        return loss
    
    elif kl_penalty == "abs":
        # 绝对值差
        loss = torch.abs(logprob - ref_logprob)
        return loss
    
    elif kl_penalty == "mse":
        # 均方误差
        loss = (logprob - ref_logprob) ** 2
        return loss
    
    # ... 其他模式
```

**k1 vs k3 对比**：

| 损失模式 | 公式 | 含义 | 优点 | 缺点 |
|--------|------|------|------|------|
| k1 | $\log q - \log p$ | 直接KL估计 | 简单，有负值 | 可能不稳定 |
| k3 | $(\log q - \log p)^2$ | 二次惩罚 | 对大差异惩罚更重 | 总是非负 |
| k2 | $2(\log q - \log p)$ | 缩放k1 | 增加梯度 | 可能发散 |

---

## 🎯 第五部分：PPO 训练循环

### 5.1 综合流程：从数据到权重更新

```
每个训练iteration的完整流程：
┌─────────────────────────────────────────────────┐
│  0. DataLoader 给出 batch                       │
│     仅含: raw_prompt (未tokenize), dummy_tensor │
└──────────────────┬──────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────┐
│  1. generate_sequences(gen_batch)               │
│     AgentLoop: tokenize → vLLM 生成             │
│     输出: prompts, responses, input_ids,        │
│           attention_mask, response_mask         │
└──────────────────┬──────────────────────────────┘
                   │ batch.union(gen_batch_output)
                   ↓
┌─────────────────────────────────────────────────┐
│  2. _compute_teacher_colocate(batch)            │
│     vLLM 对 input_ids 算 top-64 logprobs        │
│     输出: teacher_logprobs (128, 768, 64)       │
└──────────────────┬──────────────────────────────┘
                   │ batch.union(batch_teacher)
                   ↓
┌─────────────────────────────────────────────────┐
│  3. _compute_old_log_prob(batch)                │
│     Actor model forward（no_grad）              │
│     输出: old_log_probs (128, 512)              │
│     作为后续 PPO clip 的锚点                     │
└──────────────────┬──────────────────────────────┘
                   │ batch.union(old_log_prob)
                   ↓
┌─────────────────────────────────────────────────┐
│  4. compute_advantage(batch, adv=grpo)          │
│     reward=0（use_task_rewards=False）          │
│     → advantages=0（外部奖励不参与）             │
└──────────────────┬──────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────┐
│  5. _update_actor(batch)                        │
│     engine 注入 loss_fn=distillation_ppo_loss   │
│     → model_forward → k1 loss → PPO update     │
│     → backward() + optimizer.step()            │
└─────────────────────────────────────────────────┘
```

### 5.2 fit()方法的真实执行逻辑

**源文件**：[verl/trainer/ppo/ray_trainer.py](../../verl/trainer/ppo/ray_trainer.py)

> ⚠️ 以下是去除无关逻辑后的**精简真实代码**，蒸馏 distillation_loss **不在 fit() 中显式调用**，而是封装在 `_update_actor()` → engine → `distillation_ppo_loss()` 内部。

```python
def fit(self):
    for epoch in range(self.config.trainer.total_epochs):
        for batch_dict in self.train_dataloader:
            # ── 步骤0：DataLoader 给出原始 batch ──────────────────────────
            batch: DataProto = DataProto.from_single_dict(batch_dict)
            # batch 此时只有 dummy_tensor + raw_prompt + data_source 等非tensor字段
            batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
            batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch))])

            # ── 步骤1：学生模型生成响应 ──────────────────────────────────
            gen_batch = self._get_gen_batch(batch)        # 提取生成所需字段（raw_prompt等）
            gen_batch.meta_info["global_steps"] = self.global_steps
            gen_batch_output = gen_batch.repeat(repeat_times=rollout_n, interleave=True)

            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
            # gen_batch_output 新增：prompts, responses, input_ids,
            #                       attention_mask, response_mask, position_ids

            # align prompt batch 并合并生成结果
            batch = batch.repeat(repeat_times=rollout_n, interleave=True)
            batch = batch.union(gen_batch_output)

            # ── 步骤2：教师模型推理（colocated 模式）──────────────────────
            if self._should_compute_teacher_colocate(batch):
                batch_teacher = self._compute_teacher_colocate(batch)
                batch = batch.union(batch_teacher)
            # batch 新增：teacher_logprobs (128, 768, 64), teacher_ids (128, 768, 64)

            # ── 步骤3：计算 old_log_probs（额外一次 actor forward，不更新权重）
            old_log_prob, _ = self._compute_old_log_prob(batch)
            # actor_rollout_wg.compute_log_prob(batch) — 在当前权重下做一次 forward
            # 输出：old_log_probs (128, 512), entropys (128, 512)
            batch = batch.union(old_log_prob)
            # batch 新增：old_log_probs (128, 512)

            # ── 步骤4：计算 reward 和 advantages ─────────────────────────
            # use_task_rewards=False：外部 reward=0，纯蒸馏驱动
            batch.batch["token_level_scores"] = reward_tensor  # 全零（无外部奖励）
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            # GRPO advantage（adv_estimator=grpo），基于 token_level_rewards（此处为0）
            batch = compute_advantage(batch, adv_estimator="grpo", ...)
            # batch 新增：advantages (128, 512)，此时全为0（因reward=0）

            # ── 步骤5：更新 Actor（蒸馏损失在这里计算）──────────────────
            actor_output = self._update_actor(batch)
            # 内部调用：engine.train_batch(batch, loss_function=distillation_ppo_loss)
            #   → forward_step →
            #       model_output = model(**inputs)          # 新 log_probs
            #       distillation_ppo_loss(model_output, data=micro_batch)
            #         → distillation_loss() → k1 loss per token
            #         → -k1 loss 作为 per-token advantages → PPO update
```

### 5.3 蒸馏损失的真实调用路径

```
_update_actor(batch)
  └→ actor_rollout_wg.update_actor(batch)        [engine_workers.py]
       └→ engine.train_batch(data, loss_function=distillation_ppo_loss)
            └→ forward_step(micro_batch, loss_function)
                 ├─ model(**inputs) → raw_output
                 ├─ prepare_model_outputs() → model_output
                 │    model_output["log_probs"]  ← 本次 forward 新算的学生 log_probs
                 └─ distillation_ppo_loss(model_output=model_output, data=micro_batch)
                      ├─ distillation_loss(config, distillation_config, model_output, data)
                      │    └─ compute_distillation_loss_reverse_kl_estimator()
                      │         ├─ student_log_probs = model_output["log_probs"]
                      │         ├─ teacher_log_probs = data["teacher_logprobs"]  ← 教师推理结果
                      │         └─ k1_loss = student_log_probs - teacher_log_probs
                      │    → advantages = -k1_loss  (负损失作为 per-token 奖励)
                      │    → policy_loss_fn(old_log_prob=data["old_log_probs"],
                      │                     log_prob=model_output["log_probs"],
                      │                     advantages=advantages)
                      └─ ppo_loss = 0 (use_task_rewards=False，外部奖励不参与)
                      total_loss = distillation_ppo_loss only
```

**关键区分**：`data["old_log_probs"]`（步骤3中独立 forward 算好，作为 PPO clip 的锚点）vs `model_output["log_probs"]`（本次 forward 实时计算，有梯度，用于更新权重）。

### 5.3 PPO 损失的计算

**关键：如何从蒸馏损失到PPO损失**

```python
# 簡化的PPO损失計算

def ppo_loss(old_log_probs, log_probs, advantages, response_mask, clip_ratio=0.2):
    """
    计算PPO损失
    
    Args:
        old_log_probs: 旧策略的日志概率 (batch, seq_len)
        log_probs: 新策略的日志概率 (batch, seq_len)
        advantages: 优势函数值，这里来自 -distillation_losses (batch, seq_len)
        response_mask: 用于mask padding token
        clip_ratio: PPO clip参数，通常0.2
    
    Returns:
        ppo_loss: 标量损失值
    """
    
    # 计算概率比率
    ratio = torch.exp(log_probs - old_log_probs)
    # 含义：新/旧 策略的概率比
    
    # 计算未clip的PPO目标
    pg_loss1 = -advantages * ratio
    
    # 计算clipped PPO目标（防止过大的策略更新）
    ratio_clipped = torch.clamp(ratio, min=1-clip_ratio, max=1+clip_ratio)
    pg_loss2 = -advantages * ratio_clipped
    
    # 取两者最大值（当ratio过大/过小时使用clipped版本）
    pg_loss = torch.max(pg_loss1, pg_loss2)
    
    # 应用response mask
    pg_loss = pg_loss * response_mask
    
    # 平均损失
    ppo_loss = pg_loss.sum() / response_mask.sum()
    
    return ppo_loss
```

**PPO的工作原理**（在蒸馏背景下）：

```
优势 > 0（蒸馏损失负）:
  该token的学生输出好 → 增加该action的概率
  ratio > 1 → pg_loss = -advantages*ratio < 0 → 减少损失 ✓

优势 < 0（蒸馏损失正）:
  该token的学生输出差 → 减少该action的概率
  ratio < 1 → pg_loss = -advantages*ratio > 0 → 增加损失 ✓

clip_ratio限制每次更新的幅度：
  防止ratio过大导致训练不稳定
```

---

## 📋 第六部分：代码追踪指南

### 6.1 快速导航表

| 功能模块 | 源文件 | 关键函数 | 用途 |
|---------|--------|---------|------|
| **入口** | `verl/trainer/main_ppo.py` | `main()`, `run_ppo()` | 程序入口 |
| **训练器** | `verl/trainer/ppo/ray_trainer.py` | `RayPPOTrainer.fit()` | 主训练循环 |
| **数据加载** | `verl/utils/dataset/rl_dataset.py` | `RLHFDataset.__getitem__()` | 返回 raw_prompt（未 tokenize） |
| **生成（AgentLoop）** | `verl/experimental/agent_loop/agent_loop.py` | `AgentLoopManager.generate_sequences()` | tokenize + vLLM 生成 |
| **教师推理** | `verl/trainer/ppo/ray_trainer.py` | `_compute_teacher_colocate()` | 获取 top-k teacher logprobs |
| **old_log_prob** | `verl/trainer/ppo/ray_trainer.py` | `_compute_old_log_prob()` | actor forward（no_grad），PPO anchor |
| **Advantage** | `verl/trainer/ppo/ray_trainer.py` | `compute_advantage()` | GRPO advantage（此配置为 0） |
| **蒸馏损失** | `verl/trainer/distillation/losses.py` | `distillation_ppo_loss()`, `distillation_loss()` | 在 engine forward_step 内调用 |
| **k1 estimator** | `verl/trainer/distillation/losses.py` | `compute_distillation_loss_reverse_kl_estimator()` | k1 = student_lp - teacher_lp |
| **PPO loss** | `verl/trainer/ppo/core_algos.py` | `get_policy_loss_fn()` | clip PPO，以 -k1 为 advantage |
| **FSDP engine** | `verl/workers/engine/fsdp/transformer_impl.py` | `forward_step()` | 调用 loss_fn，执行 backward |

### 6.2 数据结构追踪（完整真实流程）

```
① DataLoader 给出 batch：
   batch.batch:
     └── dummy_tensor: (128, 1)
   batch.non_tensor_batch:
     ├── raw_prompt: (128,) object  ← chat messages list，未 tokenize
     ├── data_source, extra_info, index, tools_kwargs, interaction_kwargs
     └── uid: (128,) object  ← fit() 中追加的 UUID
   batch.meta_info:
     └── temperature: float

         ↓ generate_sequences（AgentLoop 内部 tokenize + vLLM 生成）

② 生成后 batch.union(gen_batch_output)，batch 新增：
   batch.batch:
     ├── prompts:        (128, 256)  ← 左对齐 prompt token IDs
     ├── responses:      (128, 512)  ← 右对齐生成 response token IDs
     ├── input_ids:      (128, 768)  ← prompts + responses 拼接
     ├── attention_mask: (128, 768)  ← 0=padding, 1=real
     ├── response_mask:  (128, 512)  ← 1=LLM生成, 0=padding/tool
     └── position_ids:   (128, 768)

         ↓ _compute_teacher_colocate（vLLM 对完整 input_ids 算 top-k logprobs）

③ batch.union(batch_teacher)，batch 新增：
   batch.batch:
     ├── teacher_logprobs: (128, 768, 64)  ← 每个位置 top-64 对数概率
     └── teacher_ids:      (128, 768, 64)  ← 对应 top-64 token IDs

         ↓ _compute_old_log_prob（actor 在当前权重下再 forward 一次，no_grad）

④ batch.union(old_log_prob)，batch 新增：
   batch.batch:
     └── old_log_probs: (128, 512)  ← 更新前的学生策略 log prob（PPO anchor）

         ↓ compute_advantage（adv_estimator=grpo，reward=0 因 use_task_rewards=False）

⑤ batch 新增：
   batch.batch:
     ├── token_level_scores:  (128, 512)  ← reward，此时为 0
     ├── token_level_rewards: (128, 512)  ← 同上（无 KL penalty）
     └── advantages:          (128, 512)  ← GRPO advantage，此时为 0

         ↓ _update_actor → engine.train_batch → forward_step（有梯度）

⑥ 在 distillation_ppo_loss 内部（micro_batch 级别，非全局 batch）：
   model_output["log_probs"]  (bsz, resp_len)  ← 本次 forward 实时计算（有梯度）
   data["teacher_logprobs"]   (bsz, seq_len, 64)  ← 第③步写入
   data["old_log_probs"]      (bsz, resp_len)  ← 第④步写入
   
   k1_loss = student_log_probs - teacher_log_probs  (bsz, resp_len)
   advantages_distill = -k1_loss                    (bsz, resp_len)
   
   distillation_loss = PPO(old_log_probs, new_log_probs, advantages_distill)
   total_loss = distillation_loss  (use_task_rewards=False, ppo_loss=0)
   
         ↓ backward() + optimizer.step()
   
   学生模型权重更新
```

---

## 🔑 第七部分：关键概念和参数解释

### 7.1 为什么使用top-k而不是full vocab？

```
问题：full vocab有50000+个token，计算KL散度太慢

解决方案：只看教师概率最高的64个token

好处：
✓ 计算加速 (50000 → 64)
✓ 聚焦学习 (学习教师偏好的token)
✓ 减少噪声 (忽略low-prob token)

代价：
✗ 可能忽略少数情况
✗ 分布归一化需要处理（add_tail参数）
```

### 7.2 "负损失=奖励" 的含义

```python
# 这是蒸馏中的关键insight

蒸馏损失 = log(学生) - log(教师)

# k1_loss < 0 时（学生置信度高于教师）
# advantages = -k1_loss > 0  → 正奖励 → 增加该action
# PPO会增加产生这个action的概率

# k1_loss > 0 时（学生不足教师）
# advantages = -k1_loss < 0  → 负奖励 → 减少该action
# PPO会减少产生这个action的概率

# 这自动实现了"向教师学习"的目标！
```

### 7.3 为什么需要old_log_probs?

```
old_log_probs = 生成时的日志概率

用途1：PPO的重要性采样
  ratio = exp(log_prob_new - log_prob_old)
  防止策略变动过大

用途2：KL奖励（可选）
  approximate_kl = log_prob_new - log_prob_old
  在reward中加入KL惩罚

用途3：稳定性
  保留生成时刻的策略信息
  用于多步训练时的参考
```

### 7.4 配置参数详解（from run_qwen_gsm8k.sh）

```yaml
# 数据配置
data:
  max_prompt_length: 256          # prompt最大长度限制
  max_response_length: 512        # response最大长度限制
  train_batch_size: 128           # 全局batch size
  filter_overlong_prompts: True   # 过滤超长prompt
  
# 蒸馏配置
distillation:
  enabled: True                              # 启用蒸馏
  num_workers: 8                             # 并行处理的worker数
  teacher_model:
    model_path: "Qwen/Qwen2.5-3B-Instruct"
    n_gpus_per_node: 4                       # 教师使用4个GPU
    inference:
      name: "vllm"                           # 推理引擎
      gpu_memory_utilization: 0.3            # 显存使用率
  distillation_loss:
    loss_mode: "k1"                          # k1模式（直接差分）
    topk: 64                                 # 只看top-64
    use_policy_gradient: True                # 使用PPO而不是SFT
    loss_max_clamp: 10.0                     # 损失裁切值
    log_prob_min_clamp: -10.0                # 日志概率下界

# PPO学生配置
actor_rollout_ref:
  actor:
    optim:
      lr: 1e-6                               # 学习率（很小）
    ppo_mini_batch_size: 128                 # mini batch大小
    ppo_micro_batch_size_per_gpu: 2          # 每个GPU的批量
    ppo_max_token_len_per_gpu: 1024          # 每个GPU的最大token数
    
# 训练配置
trainer:
  total_epochs: 15                           # 训练epoch数
  save_freq: 200                             # 每200步保存checkpoint
  test_freq: 5                               # 每5步测试一次
  n_gpus_per_node: 2                         # 学生使用2个GPU（DP）
```

---

## 📊 第八部分：数据维度完整表

在整个流程中的各个阶段，数据的形状变换：

| 阶段 | 变量名 | 形状 | 说明 |
|------|-------|------|------|
| **① DataLoader** | `dummy_tensor` | (128, 1) | 占位 tensor |
| | `raw_prompt` | (128,) object | 未 tokenize 的 chat messages list |
| | `uid` | (128,) object | 随机 UUID |
| **② generate_sequences 后** | `prompts` | (128, 256) | 左对齐 prompt token IDs |
| | `responses` | (128, 512) | 右对齐生成 response token IDs |
| | `input_ids` | (128, 768) | prompt + response 完整序列 |
| | `attention_mask` | (128, 768) | 0=padding, 1=real token |
| | `response_mask` | (128, 512) | 1=LLM生成, 0=padding |
| | `position_ids` | (128, 768) | 位置编号 |
| **③ 教师推理后** | `teacher_logprobs` | (128, 768, 64) | 每个位置 top-64 对数概率 |
| | `teacher_ids` | (128, 768, 64) | 对应 top-64 token IDs |
| **④ old_log_prob 计算后** | `old_log_probs` | (128, 512) | 更新前学生策略 log prob |
| **⑤ advantage 计算后** | `token_level_scores` | (128, 512) | reward（此配置为 0） |
| | `advantages` | (128, 512) | GRPO advantage（此配置为 0） |
| **⑥ forward_step 内** | `model_output["log_probs"]` | (bsz, resp_len) | 当前 forward 新算的 log_probs（有梯度） |
| | `k1_loss` | (bsz, resp_len) | `student_log_probs - teacher_log_probs` |
| | `advantages_distill` | (bsz, resp_len) | `-k1_loss`（per-token 蒸馏奖励） |

**总结**：
- Batch size = 128 (全局)
- Max sequence length = 768 (prompt + response + 1)
- Prompt length = 256
- Response length = 512
- Teacher top-k = 64
- 真正有梯度更新的：`model_output["log_probs"]`（本次 forward）
- 作为 PPO anchor 的：`data["old_log_probs"]`（独立 forward，no_grad）

---

## 🎓 第九部分：从入口到损失的完整代码路径

```
entry point: python3 -m verl.trainer.main_ppo
  ↓
verl/trainer/main_ppo.py::main() → run_ppo()
  ↓
  • ray.init()
  • RayPPOTrainer(config, tokenizer, ...)
  • RayPPOTrainer.init_workers()
  • RayPPOTrainer.fit()
    ↓
    verl/trainer/ppo/ray_trainer.py::RayPPOTrainer.fit()
      for each batch_dict in train_dataloader:
        
        0. batch = DataProto.from_single_dict(batch_dict)
           batch 只含 dummy_tensor + raw_prompt + data_source + uid 等
        
        1. gen_batch = _get_gen_batch(batch)
           gen_batch_output = async_rollout_manager.generate_sequences(gen_batch_output)
           └→ verl/experimental/agent_loop/agent_loop.py::AgentLoopManager.generate_sequences()
              └→ AgentLoop.run(raw_prompt)
                 └→ tokenizer.apply_chat_template(raw_prompt)   ← tokenization 在这里
                 └→ vLLM 生成 response
                 └→ 输出：prompts, responses, input_ids, attention_mask, response_mask, position_ids
           batch = batch.union(gen_batch_output)
        
        2. batch_teacher = _compute_teacher_colocate(batch)
           └→ teacher_model_manager.compute_logprobs(batch)
              └→ vLLM 对 input_ids 计算 prompt_logprobs（top-64）
              └→ 输出：teacher_logprobs (bsz, seq_len, 64), teacher_ids (bsz, seq_len, 64)
           batch = batch.union(batch_teacher)
        
        3. old_log_prob, _ = _compute_old_log_prob(batch)
           └→ actor_rollout_wg.compute_log_prob(batch)  ← student model forward（no_grad）
              └→ 输出：old_log_probs (bsz, resp_len)
           batch = batch.union(old_log_prob)
        
        4. compute_advantage(batch, adv_estimator=grpo)
           └→ batch["advantages"] = 0（因 use_task_rewards=False，reward=0）
        
        5. _update_actor(batch)
           └→ actor_rollout_wg.update_actor(batch)
              └→ engine.train_batch(batch, loss_function=distillation_ppo_loss)
                 └→ forward_step(micro_batch, loss_function)
                    ├─ model(**inputs) → model_output["log_probs"]  ← 有梯度
                    └─ distillation_ppo_loss(model_output, data=micro_batch)
                       ├─ distillation_loss()
                       │   └─ compute_distillation_loss_reverse_kl_estimator()
                       │      ├─ student_lp = model_output["log_probs"]
                       │      ├─ teacher_lp = data["teacher_logprobs"]  (squeeze topk→1)
                       │      └─ k1 = student_lp - teacher_lp  (bsz, resp_len)
                       │   → advantages = -k1.detach()
                       │   → policy_loss_fn(old_log_probs=data["old_log_probs"],
                       │                    log_probs=model_output["log_probs"],
                       │                    advantages=advantages)  ← PPO clip
                       ├─ ppo_loss = 0  (use_task_rewards=False)
                       └─ total_loss = distillation_ppo_loss
                    └─ backward() + optimizer.step()  ← 更新学生模型权重
```

---

## 💡 总结：核心要点

### 核心流程六步走

1. **📥 数据加载**：Parquet → `raw_prompt`（chat messages，**未 tokenize**）→ DataLoader Batch

2. **🤖 学生模型生成**：AgentLoop 内部 tokenize → vLLM 生成 responses（输出: `prompts/responses/input_ids/attention_mask/response_mask`）

3. **👨‍🏫 教师模型推理**：对完整 `input_ids` 算 top-64 logprobs（输出: `teacher_logprobs (128, 768, 64)`）

4. **🔒 计算 old_log_probs**：Actor 在**当前权重**下额外 forward 一次（no_grad），作为 PPO clip 的锚点

5. **📉 蒸馏损失（在 engine 内部）**：`k1 = student_log_probs - teacher_log_probs` → `-k1` 作为 per-token advantage → PPO update

6. **⬆️ 权重更新**：`backward()` + `optimizer.step()` 更新学生模型

> ⚠️ **关键纠正**：蒸馏损失**不在 fit() 中显式调用**，而是通过 `loss_fn=distillation_ppo_loss` 注入 engine，在 `forward_step` 中与 model forward 结合执行。  
> GRPO advantage（基于外部 reward）虽然也被计算，但因 `use_task_rewards=False` 被置零，实际训练完全由蒸馏信号驱动。

### 关键参数一览

| 参数 | 值 | 含义 |
|------|-----|------|
| Batch size | 128 | 每个训练step的样本数 |
| Learning rate | 1e-6 | 学生模型学习率（很小） |
| Top-k | 64 | 教师返回的token数 |
| Loss mode | k1 | 蒸馏损失使用k1模式 |
| Policy gradient | True | 使用PPO而不是直接监督 |
| Epochs | 15 | 总训练周期数 |
| Teacher GPUs | 4 | 教师模型并行GPU数 |
| Student GPUs | 2 | 学生模型并行GPU数 |

### 为什么这样设计好

✅ **分布式推理**：学生和教师分离，充分利用GPU
✅ **稳定训练**：PPO clip机制提供稳定的策略更新
✅ **计算高效**：top-k而不是full vocab，训练速度快
✅ **知识迁移**：蒸馏loss自动实现向教师学习的目标
✅ **可扩展性**：支持多种损失模式（k1, k3等）和推理引擎（vLLM, sglang）

---

## 📚 附录：常见问题解答

### Q1: 为什么教师模型用max_tokens=1？

A: `max_tokens=1`意味着推理引擎不生成新token，只计算输入序列（prompt+response）中每个位置的概率分布。这是因为我们要获得教师对已有序列的评价，而不是让它生成新内容。

### Q2: 蒸馏损失可以是负数吗？

A: 可以的。在k1模式中，如果学生概率 > 教师概率，loss为负。这代表学生已经"超越"了教师。负loss作为优势时会变成正奖励，鼓励这个行为。

### Q3: 为什么要分离old_log_probs和新的log_probs？

A: old_log_probs来自生成阶段（多步采样），新log_probs来自最后一层softmax。这两个值用于计算策略变化的幅度(ratio)，是PPO prevent-update-too-large机制的核心。

### Q4: distillation_loss和token_level_scores有什么区别？

A: 它们在概念上是同一个东西，只是在不同地方有不同的名称。distillation_loss是计算出的值，token_level_scores是存储在batch中用于记录和分析的版本。

### Q5: 为什么需要response_mask？

A: response_mask用于标记哪些是真实的response token，哪些是padding。在计算损失时乘以response_mask，确保padding token不参与梯度计算，避免对模型的错误更新。

---

## 参考资源

- VERL官方文档：https://github.com/volcengine/verl
- On-Policy Distillation论文参考：https://thinkingmachines.ai/blog/on-policy-distillation/
- vLLM推理引擎：https://github.com/vllm-project/vllm
- PPO算法原文：Proximal Policy Optimization Algorithms (Schulman et al., 2017)

---

**文档完成日期**：2025年4月
**基于代码版本**：VERL main branch
**配置参考**：run_qwen_gsm8k.sh
