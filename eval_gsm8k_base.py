# eval_gsm8k_base.py
import json
import re
import argparse
import random
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset

from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch


# ---------------- vLLM 初始化（加载 base 模型，不做微调） ----------------
def init_vllm(model_id: str,
              device: str = "cuda",
              seed: int = 42,
              gpu_memory_utilization: float = 0.85) -> LLM:
    vllm_set_random_seed(seed)

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )

    with world_size_patch, profiling_patch:
        llm = LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    return llm


# ---------------- 数据集读取：{"prompt": ..., "response": ...} ----------------
class JsonlSFTDataset(Dataset):
    def __init__(self, path: str):
        self.items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.items.append({"prompt": obj["prompt"],
                                   "response": obj["response"]})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ---------------- 抽取答案 & 格式检查 ----------------
# 和训练脚本保持一致：优先匹配 #### 18 这样的模式
ANS_RE = re.compile(r"####\s*([\-]?\d+\.?\d*)")


def extract_final_answer(text: str):
    """
    先找 '#### <number>'，
    找不到就取文本中的最后一个数字。
    """
    m = ANS_RE.search(text)
    if m:
        return m.group(1).strip()

    nums = re.findall(r"[\-]?\d+\.?\d*", text)
    return nums[-1].strip() if nums else None


@torch.no_grad()
def evaluate_with_vllm(
    llm: LLM,
    eval_prompts: List[str],
    references: List[str],
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """
    - 用 vLLM 生成完整回答
    - 用 extract_final_answer 抽数值答案
    - result accuracy：数值是否等于参考答案
    - format accuracy：输出中是否出现 '#### <number>'
    """
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outs = llm.generate(eval_prompts, sampling_params=sampling)

    preds = []
    for out in outs:
        text = out.outputs[0].text
        preds.append(text)

    ref_ans = [extract_final_answer(r) for r in references]
    pred_ans = [extract_final_answer(p) for p in preds]

    # ------ 结果准确率（数值对不对） ------
    correct = 0
    total = 0
    for pa, ra in zip(pred_ans, ref_ans):
        if ra is None:
            continue  # 标准答案都抽不出来就跳过
        total += 1
        if pa is not None and pa == ra:
            correct += 1
    result_acc = correct / total if total > 0 else 0.0

    # ------ 格式准确率（有没有 '#### <number>'） ------
    format_correct = 0
    for p in preds:
        if ANS_RE.search(p) is not None:
            format_correct += 1
    format_total = len(preds)
    format_acc = format_correct / format_total if format_total > 0 else 0.0

    return {
        "accuracy": result_acc,
        "denom": total,
        "format_accuracy": format_acc,
        "format_denom": format_total,
        "pred_texts": preds,
        "pred_ans": pred_ans,
        "ref_ans": ref_ans,
    }


# ---------------- 主函数：评估 3 次，取平均 ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-1.5B",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="sft_test.jsonl",
    )
    parser.add_argument(
        "--eval_n",
        type=int,
        default=256,  # 每次评估抽多少条
    )
    parser.add_argument(
        "--gen_max_new_tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--vllm_mem_util",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    print(">>> 加载测试集：", args.test_path)
    test_ds = JsonlSFTDataset(args.test_path)
    n_test = len(test_ds)
    print(f"测试集总样本数 = {n_test}")

    print(">>> 初始化 vLLM（加载 base 模型）……")
    llm = init_vllm(
        model_id=args.model_id,
        device=args.device,
        seed=args.seed,
        gpu_memory_utilization=args.vllm_mem_util,
    )

        # -------- 评估 3 次，随机抽样 eval_n 条，取平均 --------
    num_runs = 3
    result_accs = []
    format_accs = []

    for run_idx in range(num_runs):
        print(f"\n========== Run {run_idx + 1}/{num_runs} ==========")
        rnd = random.Random(args.seed + run_idx)

        indices = list(range(n_test))
        rnd.shuffle(indices)
        k = min(args.eval_n, n_test)
        chosen = indices[:k]

        sample = [test_ds[i] for i in chosen]
        eval_prompts = [x["prompt"] for x in sample]
        eval_refs = [x["response"] for x in sample]

        print(f"本次评估使用样本数 = {k}")
        metrics = evaluate_with_vllm(
            llm,
            eval_prompts,
            eval_refs,
            max_new_tokens=args.gen_max_new_tokens,
        )

        result_acc = metrics["accuracy"]
        format_acc = metrics["format_accuracy"]
        result_accs.append(result_acc)
        format_accs.append(format_acc)

        print(f"[Run {run_idx + 1}] result accuracy = {result_acc * 100:.2f}% "
              f"(denom = {metrics['denom']})")
        print(f"[Run {run_idx + 1}] format accuracy = {format_acc * 100:.2f}% "
              f"(format_denom = {metrics['format_denom']})")

        # ========= 加样例输出，看具体长什么样 =========
        num_examples_to_show = 3
        print(f"\n[Run {run_idx + 1}] 样例输出（最多展示 {num_examples_to_show} 条）:")
        for i in range(min(num_examples_to_show, k)):
            print(f"\n----- Example {i} -----")
            print("Prompt:")
            print(eval_prompts[i])
            print("\nGround-truth response:")
            print(eval_refs[i])
            print("\nModel output:")
            print(metrics["pred_texts"][i])
            print("\n抽取出的答案：")
            print(f"  ref_ans  = {metrics['ref_ans'][i]}")
            print(f"  pred_ans = {metrics['pred_ans'][i]}")
            print("------------------------------")

     # -------- 计算 3 次的平均值 --------
    mean_result_acc = sum(result_accs) / num_runs if num_runs > 0 else 0.0
    mean_format_acc = sum(format_accs) / num_runs if num_runs > 0 else 0.0

    print("\n========== 最终平均结果（3 次评估） ==========")
    print(f"模型: {args.model_id}")
    print(f"平均 result accuracy: {mean_result_acc * 100:.2f}%")
    print(f"平均 format accuracy: {mean_format_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
