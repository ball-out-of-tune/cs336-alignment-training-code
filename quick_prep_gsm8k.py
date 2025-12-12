# cs336_alignment/quick_prep_gsm8k.py
import json
import io
import os
import sys

def to_prompt(question: str) -> str:
    return (
        "You are a helpful math tutor. Please reason step by step and give the final answer after '####'.\n\n"
        f"Question: {question}"
    )

def write_item(fout, q, a):
    obj = {"prompt": to_prompt(q), "response": a}
    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

def convert_any(in_path: str, out_path: str):
    # 用 io.open 兼容 BOM
    with io.open(in_path, "r", encoding="utf-8-sig") as fin:
        data = fin.read()

    # 去掉两端空白
    data_stripped = data.lstrip()

    items = []
    if not data_stripped:
        raise ValueError(f"Input file is empty: {in_path}")

    if data_stripped[0] == "[":
        # 情况2：整个文件是JSON数组
        try:
            arr = json.loads(data_stripped)
        except json.JSONDecodeError as e:
            raise ValueError(f"File looks like a JSON array but failed to parse: {in_path}\n{e}") from e
        for i, obj in enumerate(arr, 1):
            if "question" not in obj or "answer" not in obj:
                raise KeyError(f"Missing 'question' or 'answer' at array index {i} in {in_path}")
            items.append((obj["question"], obj["answer"]))
    else:
        # 情况1/3：按行读取（JSONL），跳过空行/逗号
        for ln, line in enumerate(data.splitlines(), 1):
            s = line.strip()
            if not s:
                continue
            # 兼容有些JSONL来自数组导出，行末多了逗号
            if s.endswith(","):
                s = s[:-1].rstrip()
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error at {in_path}:{ln}\nLine content: {line[:200]}") from e
            if "question" not in obj or "answer" not in obj:
                raise KeyError(f"Missing 'question' or 'answer' at line {ln} in {in_path}")
            items.append((obj["question"], obj["answer"]))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout:
        for q, a in items:
            write_item(fout, q, a)

    print(f"Converted {len(items)} examples -> {out_path}")

if __name__ == "__main__":
    convert_any("data/gsm8k_original_train.jsonl", "data/sft_train.jsonl")
    convert_any("data/gsm8k_original_test.jsonl", "data/sft_test.jsonl")
