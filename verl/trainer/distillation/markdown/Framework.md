1. vllm获取attention
2. 讲attention的结果在 verl在**文件**：[verl/workers/rollout/vllm_rollout/vllm_async_server.py](../../verl/workers/rollout/vllm_rollout/vllm_async_server.py)  约第 516-552 行进行透传
3. 在处理好的parquet数据集中增加privilege information字段。
4. ① DataLoader → batch 含 raw_prompt + privilege(非tensor_batch)

② generate_sequences → batch.union(gen_batch_output)
   batch 新增: prompts (128,256), responses (128,512), input_ids (128,768), ...

③ [新增] 构造特权 forward batch:
   privilege_ids = tokenizer(batch.non_tensor_batch["privilege"])
   new_input_ids  = [prompts ; privilege_ids ; responses]   # (128, 256+priv_len+512)
   new_attn_mask  = [1...1   ; 1...1        ; 1...1   ]
   new_position_ids 重新递增

   privileged_batch.batch["input_ids"]      = new_input_ids
   privileged_batch.batch["responses"]      = responses  （不变！）
   privileged_batch.batch["attention_mask"] = new_attn_mask
   privileged_batch.batch["position_ids"]   = new_position_ids

   privileged_log_probs = actor_rollout_wg.compute_log_prob(privileged_batch)
   # shape: (128, 512) ← 和 old_log_probs 一样
   batch = batch.union(DataProto(batch={"privileged_log_probs": privileged_log_probs}))

④ _compute_teacher_colocate → teacher_logprobs (保持不变)

⑤ _compute_old_log_prob → old_log_probs (保持不变)

⑥ _update_actor → engine.forward_step → distillation_ppo_loss
   [修改 losses.py] 新增计算:
   privileged_lp = no_padding_2_padding(data["privileged_log_probs"], data)
   student_lp    = no_padding_2_padding(model_output["log_probs"], data)
   k1_privilege  = student_lp - privileged_lp
   # 然后用 k1_privilege 作为额外 loss 项