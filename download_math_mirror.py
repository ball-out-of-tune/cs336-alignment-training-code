import os
import json
from modelscope.msdatasets import MsDataset

def download_and_reconstruct():
    print("🚀 正在从 ModelScope (魔搭) 下载 MATH 数据集...")
    
    # 1. 从 ModelScope 加载数据集
    # 【修正点】增加了 trust_remote_code=True 参数
    try:
        print("正在加载 Train 集...")
        ds_train = MsDataset.load('competition_math', split='train', trust_remote_code=True)
        
        print("正在加载 Test 集...")
        ds_test = MsDataset.load('competition_math', split='test', trust_remote_code=True)
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return

    # 2. 定义输出路径
    base_output_dir = "./data/MATH"
    print(f"📦 下载完成，正在重构目录结构到: {base_output_dir} ...")

    # 定义要处理的数据集切片
    splits = [('train', ds_train), ('test', ds_test)]

    for split_name, dataset in splits:
        print(f"正在处理 {split_name} 集...")
        split_dir = os.path.join(base_output_dir, split_name)
        
        type_counters = {}

        for item in dataset:
            # ModelScope 的字段通常是 'problem', 'solution', 'type', 'level'
            problem_type = item.get('type', 'Uncategorized')
            
            # 创建类型文件夹 (如 Algebra)
            type_dir = os.path.join(split_dir, problem_type)
            os.makedirs(type_dir, exist_ok=True)
            
            # 计数器生成文件名
            if problem_type not in type_counters:
                type_counters[problem_type] = 0
            type_counters[problem_type] += 1
            
            filename = f"problem_{type_counters[problem_type]}.json"
            file_path = os.path.join(type_dir, filename)
            
            # 构造 JSON 内容
            json_content = {
                "problem": item.get('problem', ''),
                "level": item.get('level', ''),
                "type": problem_type,
                "solution": item.get('solution', '')
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(json_content, f, ensure_ascii=False, indent=4)

    print(f"\n✅ 成功！数据已保存在: {os.path.abspath(base_output_dir)}")
    print("目录结构示例: data/MATH/train/Algebra/problem_1.json")

if __name__ == "__main__":
    download_and_reconstruct()