#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
import argparse
from typing import Any, Dict, List, Tuple

from tqdm import tqdm
from vllm import LLM, SamplingParams

# 直接用你给的 drgrpo_grader.py
from drgrpo_grader import r1_zero_reward_fn, extract_boxed_answer, r1_zero_reward_fn_loose


# r1_zero 的关键：必须出现 "</think> <answer>"（中间一个空格）
R1_ZERO_PROMPT = (
    "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
    "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
    "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n"
    "User: {problem}\n"
    "Assistant: <think>"
)


def build_prompt(problem: str) -> str:
    return R1_ZERO_PROMPT.format(problem=problem)


def load_math_dir(root_dir: str, split: str = "test") -> List[Dict[str, Any]]:
    """
    读取你这种目录结构：
      root_dir/{train|test}/{subject}/problem_*.json
    每个 json 通常包含: problem, solution, level, type
    """
    pattern = os.path.join(root_dir, split, "*", "problem_*.json")
    files = sorted(glob.glob(pattern))
    examples = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            ex = json.load(f)
        ex["_file"] = fp
        ex["_subject"] = os.path.basename(os.path.dirname(fp))
        examples.append(ex)
    return examples


def get_ground_truth_from_solution(ex: Dict[str, Any]) -> str:
    """
    官方 MATH 的 GT 主要藏在 solution 的 \\boxed{...}
    这里直接复用 drgrpo_grader.extract_boxed_answer
    """
    sol = ex.get("solution", "")
    gt = extract_boxed_answer(sol)
    if gt is None:
        # 极少数异常情况：没 boxed 就退回整个 solution（但 grader 期望字符串）
        gt = sol
    return gt


def evaluate_vllm(
    llm: LLM,
    prompts: List[str],
    gts: List[str],
    metas: List[Dict[str, Any]],
    sampling: SamplingParams,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    outputs = llm.generate(prompts, sampling)

    rows: List[Dict[str, Any]] = []
    format_ok = 0
    answer_ok = 0
    reward_sum = 0.0
    gen_len_sum = 0

    for i, (out, prompt, gt, meta) in enumerate(zip(outputs, prompts, gts, metas)):
        text = out.outputs[0].text
        token_len = len(out.outputs[0].token_ids)

        # 【核心修改点】
        # 因为 Prompt 以 "<think>" 结尾，vLLM 生成的内容是从 think 之后开始的。
        # 为了让 reward_fn 能正则匹配到完整的 <think>...</think>，必须手动拼回去。
        full_text = "<think>" + text

        # 使用补全后的 full_text 进行打分
        r = r1_zero_reward_fn_loose(full_text, gt, fast=True)

        format_ok += 1 if r.get("format_reward", 0.0) >= 1.0 else 0
        answer_ok += 1 if r.get("answer_reward", 0.0) >= 1.0 else 0
        reward_sum += float(r.get("reward", 0.0))
        gen_len_sum += int(token_len)

        rows.append(
            {
                "id": i,
                "meta": meta,
                "prompt": prompt,
                "generation": full_text, # 存完整的文本以便人类查看
                "ground_truth": gt,
                "rewards": r,
                "gen_len_tokens": int(token_len),
            }
        )

    n = max(len(rows), 1)
    metrics = {
        "num_examples": len(rows),
        "format_acc": format_ok / n,
        "answer_acc": answer_ok / n,
        "avg_reward": reward_sum / n,
        "avg_gen_len_tokens": gen_len_sum / n,
    }
    return rows, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    ap.add_argument("--math_root", type=str, default="data/MATH", help="包含 train/test 的根目录")
    ap.add_argument("--split", type=str, default="test", choices=["train", "test"])
    ap.add_argument("--limit", type=int, default=0, help=">0 则只评测前 limit 条")
    ap.add_argument("--out_dir", type=str, default="./math_baseline_out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_jsonl = os.path.join(args.out_dir, "results.jsonl")
    out_metrics = os.path.join(args.out_dir, "metrics.json")

    raw = load_math_dir(args.math_root, args.split)
    if args.limit and args.limit > 0:
        raw = raw[: args.limit]

    prompts: List[str] = []
    gts: List[str] = []
    metas: List[Dict[str, Any]] = []

    for ex in raw:
        problem = ex["problem"]
        gt = get_ground_truth_from_solution(ex)

        prompts.append(build_prompt(problem))
        gts.append(gt)
        metas.append(
            {
                "subject": ex.get("_subject"),
                "file": ex.get("_file"),
                "level": ex.get("level"),
                "type": ex.get("type"),
            }
        )

    llm = LLM(model=args.model, seed=args.seed)

    sampling = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    rows, metrics = evaluate_vllm(llm, prompts, gts, metas, sampling)

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"results -> {out_jsonl}")
    print(f"metrics -> {out_metrics}")


if __name__ == "__main__":
    main()
