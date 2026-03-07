#!/usr/bin/env python3
"""
上传中间数据集到 HuggingFace Hub

使用方法:
1. 首先安装 huggingface_hub: pip install huggingface_hub
2. 登录 HuggingFace: huggingface-cli login
3. 运行此脚本: python upload_to_huggingface.py
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, Repository, create_repo
import json

# ==================== 配置区域 ====================

# 你的 HuggingFace 用户名或组织名
USERNAME = "wingchiuloong"  # 请修改为你的用户名

# 数据集名称
DATASET_NAME = "CopyPasteSeed365"

# 数据集文件路径
DATASET_FILE = "data/CopyPasteSeed365.jsonl"

# 是否设置为公开（True）或私有（False）
PUBLIC = True

# ==================== 配置区域结束 ====================


def create_dataset_card():
    """创建数据集卡片 README.md"""
    return """---
license: mit
task_categories:
- text-generation
language:
- en
size_categories:
- 1K<n<10K
tags:
- RAG
- hallucination
- knowledge-conflict
- DPO
- preference-optimization
---

# CopyPasteSeed365

This dataset was used to train [CopyPasteLLM-L3-8B](https://huggingface.co/wingchiuloong/CopyPasteLLM-L3-8B), presented in the paper [Copy-Paste to Mitigate Large Language Model Hallucinations](https://huggingface.co/papers/2510.00508).

## Dataset Description

CopyPasteSeed365 is a high-quality seed dataset derived from three major RAG (Retrieval-Augmented Generation) benchmarks: **PubMedQA**, **FaithEval**, and **RAGTruth**. This dataset contains intermediate data from the DPO (Direct Preference Optimization) preparation pipeline, featuring complete responses and comprehensive evaluation metrics from 6 different generation pipelines.

The dataset employs a novel "copy-paste" strategy to mitigate hallucinations in large language models by promoting answers that extract and directly copy content from the given context.

## Source Datasets

This dataset is constructed from the following RAG hallucination benchmarks:

- **[qiaojin/PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA)**: Biomedical literature QA dataset
- **[Salesforce/FaithEval-counterfactual-v1.0](https://huggingface.co/datasets/Salesforce/FaithEval-counterfactual-v1.0)**: Counterfactual reasoning evaluation for faithfulness
- **[wandb/RAGTruth-processed](https://huggingface.co/datasets/wandb/RAGTruth-processed)**: RAG hallucination detection and truthfulness benchmark

## Dataset Structure

Each record in this dataset contains:

### Top-level Fields
- `sample_id`: Unique sample identifier
- `dataset`: Source dataset name (ragtruth, faith, pubmed)
- `prompt`: Formatted prompt used for generation
- `context`: Original context passage
- `query`: Original question/query
- `original_answer`: Ground truth answer (when available)
- `responses`: Array of 6 pipeline responses with complete metrics
- `selection_info`: Information about the two-stage selection process
- `metadata`: Configuration and base model information

### Response Object Fields
Each response in the `responses` array contains:
- `pipeline`: Pipeline name (Refine, Strict, Bridge, Base, Attributed, Citations)
- `response`: Generated response text
- `category`: Either "chosen" or "reject"
- `is_final_selection`: Boolean indicating if this was the finally selected pipeline
- `stage1_passed`: Whether the pipeline passed Stage 1 inclusion criteria
- `elo_score`: Elo rating from pairwise comparisons
- `metrics`: Object containing 14 evaluation metrics
- `additional_info`: Pipeline-specific additional data (e.g., response_history for Refine)

### Metrics Included
- `alignscore`: Alignment score with the question
- `minicheck`: Factual accuracy score
- `extractiveness_coverage`: Coverage of extracted information from context
- `extractiveness_density`: Density of extracted information
- `qwen3_embedding`: Semantic similarity using Qwen3 embedding
- `ppl`: Perplexity score
- `ratio`: Copy ratio from original text (key metric for copy-paste strategy)
- `relevancy`: Relevance score
- `hallucination_eval`: Hallucination evaluation score
- `bleu`: BLEU score
- `rouge_l`: ROUGE-L score
- `bge_m3`: BGE-M3 embedding score
- `speed`: Processing speed
- `processing_time_seconds`: Total processing time

## Selection Process

### Stage 1: Inclusion Criteria Filtering
Pipelines are filtered based on the following thresholds designed to select high-quality, grounded responses:
- alignscore > 0.93
- minicheck > 0.94
- extractiveness_coverage > 0.8
- extractiveness_density > 5.0
- qwen3_embedding > 0.65
- ppl < 33.0
- ratio > 1.2 (promotes copy-paste behavior)

### Stage 2: Elo Rating Selection
Among pipelines that pass Stage 1, the best one is selected using Elo ratings computed from pairwise comparisons, ensuring the highest quality response is chosen.

## Pipelines

1. **Refine**: Iterative refinement approach
2. **Strict**: Strict adherence to context
3. **Bridge**: Bridging context and question
4. **Base**: Base generation without special handling
5. **Attributed**: Response with source attribution
6. **Citations**: Response with inline citations

## Base Model

All responses were generated using: `deepseek-ai/DeepSeek-V3`

## Dataset Statistics

- Total samples: [To be updated after upload]
- Responses per sample: 6
- Total responses: [To be updated after upload]
- Pipelines: 6

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{username}/{dataset_name}")

# Access a sample
sample = dataset["train"][0]

# Get all responses for a sample
responses = sample["responses"]

# Find the final selected response
final_response = next(r for r in responses if r["is_final_selection"])

# Compare two pipelines
refine_response = next(r for r in responses if r["pipeline"] == "Refine")
base_response = next(r for r in responses if r["pipeline"] == "Base")

print(f"Refine alignscore: {refine_response['metrics']['alignscore']}")
print(f"Base alignscore: {base_response['metrics']['alignscore']}")
```

## Use Cases

- **RAG System Training**: Train models to prefer grounded, copy-paste style responses
- **Pipeline Comparison**: Compare different generation strategies side-by-side
- **Metric Analysis**: Analyze which metrics correlate with quality
- **Selection Method Research**: Study different pipeline selection strategies
- **Hallucination Mitigation**: Research methods to reduce model hallucinations

## Citation

If you use this dataset, please cite our paper:

```bibtex
@inproceedings{{long2026copypaste,
  title={{Copy-Paste to Mitigate Large Language Model Hallucinations}},
  author={{Yongchao Long and Yingying Zhang and Xianbin Wen and Xian Wu and Yuxi Zhou and Shenda Hong}},
  booktitle={{The Fourteenth International Conference on Learning Representations}},
  year={{2026}},
  url={{https://openreview.net/forum?id=crKJJ4Ej60}}
}}
```

## Code

The code used to create this dataset is available at: [https://github.com/longyongchao/CopyPasteLLM](https://github.com/longyongchao/CopyPasteLLM)

## License

MIT License

## Contact

For questions and support, please open an issue on the [GitHub repository](https://github.com/longyongchao/CopyPasteLLM).

---

**Note**: This dataset is designed for research and educational purposes focused on mitigating RAG hallucinations through copy-paste strategies. Please ensure responsible use and compliance with applicable laws and regulations.
""".replace("{username}", USERNAME).replace("{dataset_name}", DATASET_NAME)


def upload_to_huggingface():
    """上传数据集到 HuggingFace Hub"""

    print("🚀 开始上传到 HuggingFace Hub...")

    # 检查文件是否存在
    if not os.path.exists(DATASET_FILE):
        print(f"❌ 文件不存在: {DATASET_FILE}")
        print("请先运行 prepare_data.py 生成中间数据集")
        return False

    # 初始化 API
    api = HfApi()

    # 创建仓库 ID
    repo_id = f"{USERNAME}/{DATASET_NAME}"

    print(f"\n📁 数据集仓库: {repo_id}")

    # 1. 创建仓库（如果不存在）
    print(f"\n📝 创建/检查仓库...")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=not PUBLIC,
            exist_ok=True
        )
        print(f"✅ 仓库准备完成")
    except Exception as e:
        print(f"⚠️  创建仓库时出现警告: {e}")

    # 2. 创建临时目录用于组织文件
    temp_dir = Path("temp_upload")
    temp_dir.mkdir(exist_ok=True)

    # 3. 复制数据文件
    print(f"\n📄 准备数据文件...")
    import shutil
    data_file = temp_dir / "data.jsonl"
    shutil.copy(DATASET_FILE, data_file)

    # 获取文件大小
    file_size_mb = os.path.getsize(DATASET_FILE) / (1024 * 1024)
    print(f"  文件大小: {file_size_mb:.2f} MB")

    # 统计样本数
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        num_samples = sum(1 for _ in f)
    print(f"  样本数量: {num_samples}")

    # 4. 创建数据集卡片
    print(f"\n📝 创建数据集卡片...")
    readme_content = create_dataset_card()
    readme_path = temp_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"✅ README.md 已创建")

    # 5. 上传文件
    print(f"\n📤 上传文件到 HuggingFace Hub...")

    files_to_upload = [
        ("data.jsonl", data_file),
        ("README.md", readme_path)
    ]

    for file_path_obj, file_path in files_to_upload:
        print(f"  上传 {file_path_obj}...")
        try:
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=file_path_obj,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Upload {file_path_obj}"
            )
            print(f"  ✅ {file_path_obj} 上传成功")
        except Exception as e:
            print(f"  ❌ {file_path_obj} 上传失败: {e}")
            return False

    # 7. 清理临时文件
    print(f"\n🧹 清理临时文件...")
    shutil.rmtree(temp_dir)
    print(f"✅ 临时文件已清理")

    # 8. 完成信息
    print(f"\n✅ 上传完成！")
    print(f"\n📊 数据集信息:")
    print(f"  仓库: {repo_id}")
    print(f"  文件: data.jsonl")
    print(f"  样本数: {num_samples}")
    print(f"  大小: {file_size_mb:.2f} MB")
    print(f"  可见性: {'公开' if PUBLIC else '私有'}")
    print(f"\n🔗 访问链接: https://huggingface.co/datasets/{repo_id}")

    print(f"\n💡 使用示例:")
    print(f"  from datasets import load_dataset")
    print(f"  dataset = load_dataset('{repo_id}')")

    return True


def update_dataset_readme():
    """仅更新 README（不重新上传数据）"""
    api = HfApi()
    repo_id = f"{USERNAME}/{DATASET_NAME}"

    readme_content = create_dataset_card()

    api.upload_file(
        path_or_fileobj=readme_content.encode('utf-8'),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Update dataset card"
    )

    print(f"✅ README 已更新: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--readme-only":
        update_dataset_readme()
    else:
        # 检查配置
        if USERNAME == "your-username":
            print("❌ 错误: 请先修改脚本中的 USERNAME 配置")
            print("   在脚本顶部将 USERNAME = 'your-username' 改为你的 HuggingFace 用户名")
            sys.exit(1)

        upload_to_huggingface()
