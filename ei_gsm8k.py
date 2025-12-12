# ei_gsm8k.py
import math
import random
import argparse
import time
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

# ---- 复用在 sft_gsm8k.py 里已经写好的工具 ----
from sft_gsm8k import (
    JsonlSFTDataset,
    tokenize_prompt_and_output,
    sft_loss,
    extract_final_answer,
    evaluate_with_vllm,
    init_vllm,
    load_policy_into_vllm_instance,
)


# ============== 1. 内存版 SFT Dataset ==============
class InMemorySFTDataset(Dataset):
    """EI 每一步生成的 (prompt, response) 对，直接放在内存里。"""

    def __init__(self, items: List[Dict[str, str]]):
        # items: [{"prompt": str, "response": str}, ...]
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ============== 2. 用当前 policy 生成 EI 训练数据 ==============
@torch.no_grad()
def build_ei_sft_dataset(
    llm,
    base_items: List[Dict[str, str]],
    ei_db_size: int,
    num_rollouts: int,
    max_new_tokens: int,
    temperature: float = 0.7,
    seed: int = 42,
):
    """
    从 base_items 里采样 ei_db_size 个问题，用当前 policy 生成 num_rollouts 个解，
    用 final answer 是否正确作为 reward=1/0，过滤出正样本。
    返回:
        ei_sft_items: list[{"prompt":..., "response":...}]
        frac_correct: 正确的 rollouts 占总 rollouts 的比例（监控用）
    """
    from vllm import SamplingParams  # 用的是 sft_gsm8k 里导入的同一个 vLLM

    rng = random.Random(seed)
    if ei_db_size > len(base_items):
        ei_db_size = len(base_items)

    # 采样 Db
    Db = rng.sample(base_items, ei_db_size)
    prompts = [x["prompt"] for x in Db]
    refs = [x["response"] for x in Db]   # 里面有标准答案

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        min_tokens=4,      # 作业里推荐，避免空串
        n=num_rollouts,    # 每个问题 G 个 rollouts
    )

    # vLLM 批量生成
    outs = llm.generate(prompts, sampling_params=sampling_params)

    ei_sft_items: List[Dict[str, str]] = []
    total_rollouts = 0
    num_correct = 0

    for q, ref, out in zip(prompts, refs, outs):
        gold = extract_final_answer(ref)

        for cand in out.outputs:
            total_rollouts += 1
            text = cand.text
            pred = extract_final_answer(text)

            if gold is not None and pred is not None and pred == gold:
                num_correct += 1
                # 保留这条 (prompt, response) 用来做 SFT
                ei_sft_items.append({"prompt": q, "response": text})

    frac_correct = num_correct / max(total_rollouts, 1)

    return ei_sft_items, frac_correct


# ============== 3. 在给定数据集上做若干 epoch SFT ==============
def sft_train_on_dataset(
    args,
    model,
    tokenizer,
    train_ds,
    test_ds,
    llm,
    ei_step: int,
    num_epochs: int,
    start_train_step: int = 0,
    start_eval_step: int = 0,
):
    """
    在给定的 train_ds 上做 num_epochs 个 epoch 的 SFT。
    复用你 SFT 代码里的训练逻辑 + vLLM eval。
    """

    device = next(model.parameters()).device
    collate = lambda batch: tokenize_prompt_and_output(batch, tokenizer, max_len=args.max_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=2,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    num_update_steps_per_epoch = math.ceil(len(train_loader))
    max_train_steps = num_epochs * num_update_steps_per_epoch
    warmup = int(args.warmup_ratio * max_train_steps)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=max_train_steps)

    global_train_step = start_train_step
    eval_step = start_eval_step

    for epoch in range(num_epochs):
        for batch in train_loader:
            global_train_step += 1

            torch.cuda.synchronize()
            t0 = time.time()

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits  # (B, S, V)

            loss = sft_loss(logits, labels, response_mask)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            # ==== 修改后的低显存 Entropy 计算 ====
            # 既然作业要求记录 Entropy，我们不能删，但要“省着算”
            # 原理：不要一次性算整个 Batch，而是用循环一个样本一个样本算，算完扔掉中间结果
            with torch.no_grad():
                # logits: (B, L, V)
                # response_mask: (B, L)
                bsz = logits.size(0)
                total_entropy = 0.0
                valid_tokens = 0.0
                
                for i in range(bsz):
                    # 取出第 i 个样本，维度变为 (L, V)
                    # 显存占用瞬间降低为原来的 1/Batch_Size
                    logit_i = logits[i] 
                    mask_i = response_mask[i].float()
                    
                    if mask_i.sum() == 0: continue # 如果没有 response 部分则跳过

                    # 只计算 mask 为 1 的部分的 entropy，进一步节省计算
                    # 甚至可以只对 response 部分做 softmax，减少计算量
                    # 这里为了代码简单，还是对整句算，但只累加 mask 部分
                    
                    probs_i = torch.softmax(logit_i, dim=-1)
                    log_probs_i = torch.log_softmax(logit_i, dim=-1)
                    
                    # Entropy = - sum(p * log(p))
                    ent_i = -(probs_i * log_probs_i).sum(dim=-1) # (L,)
                    
                    # 只取 response 部分
                    total_entropy += (ent_i * mask_i).sum().item()
                    valid_tokens += mask_i.sum().item()
                
                # 计算平均 Entropy
                entropy = total_entropy / max(valid_tokens, 1.0)
            
            # 删除 logits 释放显存 (这是一个好习惯)
            del logits
            # ===================================

            torch.cuda.synchronize()
            step_time = time.time() - t0
            tokens = input_ids.numel()
            toks_per_sec = tokens / step_time

            if global_train_step % args.log_every == 0:
                current_lr = sched.get_last_lr()[0]
                mem_alloc = torch.cuda.max_memory_allocated(device) / 1024**3
                mem_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
                wandb.log(
                    {
                        f"train/loss": loss.item(),
                        f"train/entropy": entropy,
                        f"train/lr": current_lr,
                        f"train/tok_s": toks_per_sec,
                        f"train/mem_alloc_GB": mem_alloc,
                        f"train/mem_reserved_GB": mem_reserved,
                        "train_step": global_train_step,
                        "ei_step": ei_step,
                    }
                )

            # 定期在验证集上评测（用 vLLM）
            if llm is not None and (global_train_step % args.eval_every == 0):
                eval_step += 1
                load_policy_into_vllm_instance(model, llm)
                k = min(args.eval_n, len(test_ds))
                sample = list(test_ds.items[:k])
                eval_prompts = [x["prompt"] for x in sample]
                eval_refs = [x["response"] for x in sample]
                metrics = evaluate_with_vllm(llm, eval_prompts, eval_refs, max_new_tokens=args.gen_max_new_tokens)
                wandb.log(
                    {
                        f"eval/accuracy": metrics["accuracy"],
                        f"eval/denom": metrics["denom"],
                        f"eval/format_accuracy": metrics["format_accuracy"],
                        f"eval/format_denom[": metrics["format_denom"],
                        f"eval/avg_gen_len": metrics["avg_gen_len"],
                        f"eval/trunc_rate": metrics["trunc_rate"],
                        "eval_step": eval_step,
                        "train_step": global_train_step,
                        "ei_step": ei_step,
                    }
                )

    return global_train_step, eval_step


# ============== 4. Expert Iteration 主流程 ==============
def run_expert_iteration(args):
    rank0 = True

    if rank0:
        wandb.define_metric("train_step")
        wandb.define_metric("eval_step")
        wandb.define_metric("ei_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("eval/*", step_metric="eval_step")
        wandb.define_metric("ei/*",   step_metric="ei_step") 

    # ---- 1) 加载原始 GSM8K 数据 ----
    base_train_ds = JsonlSFTDataset(args.train_path)  # 里面有 .items
    test_ds = JsonlSFTDataset(args.test_path)

    # ---- 2) tokenizer & 初始 policy 模型 ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=None,
    ).to(device).train()

    # ---- 3) vLLM on GPU1 做生成 ----
    if torch.cuda.device_count() >= 2:
        llm = init_vllm(args.model_id, device="cuda:1", seed=42, gpu_memory_utilization=args.vllm_mem_util)
        load_policy_into_vllm_instance(model, llm)
    else:
        llm = None
        raise RuntimeError("Expert iteration 需要第二块 GPU 跑 vLLM，请确认有至少 2 张卡。")

    global_train_step = 0
    eval_step = 0

    # ---- 4) EI 外环 ----
    for ei_step in range(1, args.n_ei_steps + 1):
        # 4.1 用当前 policy + vLLM 生成 EI 训练数据
        ei_sft_items, frac_correct = build_ei_sft_dataset(
            llm=llm,
            base_items=base_train_ds.items,
            ei_db_size=args.ei_db_size,
            num_rollouts=args.ei_num_rollouts,
            max_new_tokens=args.gen_max_new_tokens,
            temperature=args.ei_temperature,
            seed=args.seed + ei_step,
        )
        wandb.log(
            {
                f"ei/frac_correct_rollouts": frac_correct,
                f"ei/num_sft_pairs": len(ei_sft_items),
                "ei_step": ei_step,
            }
        )

        if len(ei_sft_items) == 0:
            print(f"[EI step {ei_step}] 没有生成任何正确样本，跳过这个 EI step。")
            continue

        ei_sft_ds = InMemorySFTDataset(ei_sft_items)

        # 4.2 在 EI 数据集上做若干 epoch 的 SFT
        global_train_step, eval_step = sft_train_on_dataset(
            args=args,
            model=model,
            tokenizer=tokenizer,
            train_ds=ei_sft_ds,
            test_ds=test_ds,
            llm=llm,
            ei_step=ei_step,
            num_epochs=args.ei_sft_epochs,
            start_train_step=global_train_step,
            start_eval_step=eval_step,
        )

        # 4.3 每个 EI step 结束时再 eval 一次（保证有一个对齐 ei_step 的点）
        eval_step += 1
        load_policy_into_vllm_instance(model, llm)
        k = min(args.eval_n, len(test_ds))
        sample = list(test_ds.items[:k])
        eval_prompts = [x["prompt"] for x in sample]
        eval_refs = [x["response"] for x in sample]
        metrics = evaluate_with_vllm(llm, eval_prompts, eval_refs, max_new_tokens=args.gen_max_new_tokens)
        wandb.log(
            {
                f"eval/accuracy_final": metrics["accuracy"],
                f"eval/denom_final": metrics["denom"],
                "eval_step": eval_step,
                "train_step": global_train_step,
                "ei_step": ei_step,
            }
        )

    print("Expert iteration 完成。")


# ============== 5. argparse & main ==============
def main():
    # TODO: 改变wandb输出, 现在的样式是散点图，很奇怪
    # 以及适当调参
    p = argparse.ArgumentParser()
    # 和 SFT 基本一致的超参
    p.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--train_path", type=str, default="data/sft_train.jsonl")
    p.add_argument("--test_path", type=str, default="data/sft_test.jsonl")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--eval_every", type=int, default=200)  # EI 里可以稍微稀一点
    p.add_argument("--eval_n", type=int, default=256)
    p.add_argument("--gen_max_new_tokens", type=int, default=1024)
    p.add_argument("--vllm_mem_util", type=float, default=0.85)

    # EI 特有的超参
    p.add_argument("--n_ei_steps", type=int, default=5)
    p.add_argument("--ei_db_size", type=int, default=2048)      # |Db|，作业要在 {512,1024,2048} 中试几个
    p.add_argument("--ei_num_rollouts", type=int, default=4)    # G
    p.add_argument("--ei_sft_epochs", type=int, default=1)      # 每个 EI step 内部 SFT 的 epoch 数
    p.add_argument("--ei_temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--project", type=str, default="ei-gsm8k")
    p.add_argument("--run_name", type=str, default="qwen1.5b-ei")
    args = p.parse_args()

    wandb.init(project=args.project, name=args.run_name, config=vars(args))

    run_expert_iteration(args)


if __name__ == "__main__":
    main()
