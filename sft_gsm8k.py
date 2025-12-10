# sft_gsm8k.py
import os, json, math, random, re, argparse, itertools
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup, PreTrainedModel

# -------- vLLM 初始化（按题面提供的写法） --------
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    state_dict = {k: v.detach().to("cpu") for k, v in policy.state_dict().items()}
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

# -------- 数据集与打包 --------
class JsonlSFTDataset(Dataset):
    def __init__(self, path: str):
        self.items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                # 需要字段 {"prompt": str, "response": str}
                self.items.append({"prompt": obj["prompt"], "response": obj["response"]})

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]

def tokenize_prompt_and_output(batch, tokenizer, max_len: int = 1024):
    prompts = [x["prompt"] for x in batch]
    outputs = [x["response"] for x in batch]

    # 分别编码，再拼接（<BOS> 由 tokenizer 决定）
    enc_p = tokenizer(prompts, add_special_tokens=False)
    enc_o = tokenizer(outputs, add_special_tokens=False)

    input_ids, labels, response_mask = [], [], []
    for p_ids, o_ids in zip(enc_p["input_ids"], enc_o["input_ids"]):
        ids = p_ids + o_ids
        # 截断到 max_len
        ids = ids[:max_len]
        # 训练标签为下一个 token，最后一个 token 无标签
        x = ids[:-1]
        y = ids[1:]
        # 构建 response_mask：prompt 为 0，response 为 1
        p_len = min(len(p_ids), len(ids))
        r_len = len(ids) - p_len
        # 与 y 对齐（因为我们切掉了最后一个 token）
        p_len_y = max(0, min(p_len - 1, len(y)))
        r_len_y = len(y) - p_len_y
        mask = [0]*p_len_y + [1]*r_len_y

        input_ids.append(torch.tensor(x, dtype=torch.long))
        labels.append(torch.tensor(y, dtype=torch.long))
        response_mask.append(torch.tensor(mask, dtype=torch.bool))

    # pad
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    def pad_to_max(tensors, pad_val):
        maxlen = max(t.size(0) for t in tensors)
        out = []
        for t in tensors:
            if t.size(0) < maxlen:
                pad = torch.full((maxlen - t.size(0),), pad_val, dtype=t.dtype)
                out.append(torch.cat([t, pad], dim=0))
            else:
                out.append(t)
        return torch.stack(out, dim=0)

    input_ids = pad_to_max(input_ids, pad_id)
    labels = pad_to_max(labels, -100)            # -100 忽略
    response_mask = pad_to_max(response_mask, 0) # 0/1 mask

    return {
        "input_ids": input_ids,           # (B, S)
        "labels": labels,                 # (B, S)
        "response_mask": response_mask,   # (B, S) bool
        "attention_mask": (input_ids != pad_id),
    }

# -------- 训练一步（只在 response token 上计算 loss） --------
def sft_loss(logits, labels, response_mask):
    # CE over vocab
    vocab = logits.size(-1)
    loss_all = F.cross_entropy(logits.view(-1, vocab), labels.view(-1), reduction="none")
    loss_all = loss_all.view(labels.size())
    # 仅在 response 部位取平均
    mask = response_mask.float()
    denom = mask.sum().clamp_min(1.0)
    return (loss_all * mask).sum() / denom

# -------- 评测：用 vLLM 生成 + 抽取“#### 最终答案”计算 accuracy --------
ANS_RE = re.compile(r"####\s*([\-]?\d+\.?\d*)")

def extract_final_answer(text: str):
    m = ANS_RE.search(text)
    if m: return m.group(1).strip()
    # 兜底：取最后一个数字
    nums = re.findall(r"[\-]?\d+\.?\d*", text)
    return nums[-1].strip() if nums else None

@torch.no_grad()
def evaluate_with_vllm(llm: LLM, eval_prompts: List[str], references: List[str], max_new_tokens=512) -> Dict[str, Any]:
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outs = llm.generate(eval_prompts, sampling_params=sampling)
    preds = []
    for out in outs:
        # vLLM 返回一个 RequestOutput，取第一条候选文本
        text = out.outputs[0].text
        preds.append(text)

    ref_ans = [extract_final_answer(r) for r in references]
    pred_ans = [extract_final_answer(p) for p in preds]

    # ------ 数值正确率（result accuracy）------
    correct = 0
    total = 0
    for pa, ra in zip(pred_ans, ref_ans):
        if ra is None:
            continue
        total += 1
        if pa is not None and pa == ra:
            correct += 1
    acc = correct / total if total > 0 else 0.0

    # ------ 格式正确率（format accuracy）------
    # 严格定义：是否出现 "#### 数字"
    format_correct = 0
    for p in preds:
        if ANS_RE.search(p) is not None:
            format_correct += 1
    format_denom = len(preds)
    format_acc = format_correct / format_denom if format_denom > 0 else 0.0

    return {
        "accuracy": acc,
        "denom": total,
        "format_accuracy": format_acc,
        "format_denom": format_denom,
        "pred_texts": preds,
        "pred_ans": pred_ans,
        "ref_ans": ref_ans,
    }


# -------- 主流程 --------
def run_one_size(args, train_path, test_path, train_size, run_suffix):
    rank0 = True  # 单机脚本，简化

    # wandb 初始化（把不同 train_size 作为不同 run 或者同一 run 的分组均可，这里同一 run 下用名字区分）
    # 约定：你先在命令行里 `wandb login`
    if rank0:
        wandb.define_metric("train_step")
        wandb.define_metric("eval_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("eval/*", step_metric="eval_step")

    # 加载数据
    train_ds_full = JsonlSFTDataset(train_path)
    test_ds = JsonlSFTDataset(test_path)

    if train_size > 0:
        rng = random.Random(42)
        idx = list(range(len(train_ds_full)))
        rng.shuffle(idx)
        idx = idx[:train_size]
        train_items = [train_ds_full[i] for i in idx]
        train_ds = Subset(train_ds_full, idx)
        # train_ds = type(train_ds_full)("")  # hack: 临时 dataset
        train_ds.items = train_items
    else:
        train_ds = train_ds_full

    # tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=None
    ).to(device).train()

    # dataloader
    collate = lambda batch: tokenize_prompt_and_output(batch, tokenizer, max_len=args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=2)

    # 优化器与调度器
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    num_update_steps_per_epoch = math.ceil(len(train_loader))
    max_train_steps = args.epochs * num_update_steps_per_epoch
    warmup = int(args.warmup_ratio * max_train_steps)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=max_train_steps)

    # vLLM（第二块 GPU）
    if torch.cuda.device_count() >= 2:
        llm = init_vllm(args.model_id, device="cuda:1", seed=42, gpu_memory_utilization=args.vllm_mem_util)
        load_policy_into_vllm_instance(model, llm)
    else:
        llm = None

    global_train_step = 0
    eval_step = 0

    for epoch in range(args.epochs):
        for batch in train_loader:
            global_train_step += 1
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

            if rank0 and (global_train_step % args.log_every == 0):
                wandb.log({f"train/loss[{run_suffix}]": loss.item(), "train_step": global_train_step})

            # 定期评测（在 cuda:1 上）
            if llm is not None and (global_train_step % args.eval_every == 0):
                eval_step += 1
                # 每次评测把最新权重 load 进 vLLM
                load_policy_into_vllm_instance(model, llm)
                # 取一个固定子集评测以稳定时间
                k = min(args.eval_n, len(test_ds))
                sample = list(itertools.islice(test_ds, 0, k))
                eval_prompts = [x["prompt"] for x in sample]
                eval_refs = [x["response"] for x in sample]
                metrics = evaluate_with_vllm(
                    llm,
                    eval_prompts,
                    eval_refs,
                    max_new_tokens=args.gen_max_new_tokens
                )
                wandb.log({
                    f"eval/accuracy[{run_suffix}]": metrics["accuracy"],
                    f"eval/denom[{run_suffix}]": metrics["denom"],
                    f"eval/format_accuracy[{run_suffix}]": metrics["format_accuracy"],
                    f"eval/format_denom[{run_suffix}]": metrics["format_denom"],
                    "eval_step": eval_step,
                    "train_step": global_train_step,
                })

        # 轮末再评测一次
        if llm is not None:
            eval_step += 1
            load_policy_into_vllm_instance(model, llm)
            k = min(args.eval_n, len(test_ds))
            sample = list(itertools.islice(test_ds, 0, k))
            eval_prompts = [x["prompt"] for x in sample]
            eval_refs = [x["response"] for x in sample]
            metrics = evaluate_with_vllm(
                llm,
                eval_prompts,
                eval_refs,
                max_new_tokens=args.gen_max_new_tokens
            )
            wandb.log({
                f"eval/accuracy[{run_suffix}]": metrics["accuracy"],
                f"eval/denom[{run_suffix}]": metrics["denom"],
                f"eval/format_accuracy[{run_suffix}]": metrics["format_accuracy"],
                f"eval/format_denom[{run_suffix}]": metrics["format_denom"],
                "eval_step": eval_step,
                "train_step": global_train_step,
            })


def main():
    p = argparse.ArgumentParser()
    # TODO: tune lr and batch_size to get better performance, adjust eval freq and epochs to reduce time
    p.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--train_path", type=str, default="sft_train.jsonl")
    p.add_argument("--test_path", type=str, default="sft_test.jsonl")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--eval_every", type=int, default=20)
    p.add_argument("--eval_n", type=int, default=256)
    p.add_argument("--gen_max_new_tokens", type=int, default=512)
    p.add_argument("--vllm_mem_util", type=float, default=0.85)
    p.add_argument("--project", type=str, default="sft-gsm8k")
    p.add_argument("--run_name", type=str, default="qwen1.5b-sft")
    p.add_argument("--train_sizes", type=int, nargs="+", default=[128, 256, 512, 1024, -1])  # -1=full
    args = p.parse_args()

    wandb.init(project=args.project, name=args.run_name)

    for sz in args.train_sizes:
        suffix = f"sz={sz if sz>0 else 'full'}"
        run_one_size(args, args.train_path, args.test_path, sz, suffix)

if __name__ == "__main__":
    main()
