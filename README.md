# Copy-Paste to Mitigate Large Language Model Hallucinations

[![Arxiv](https://img.shields.io/badge/arXiv-2510.00508-B21A1B)](https://arxiv.org/abs/2510.00508)
[![OpenReview](https://img.shields.io/badge/OpenReview-ICLR%202026-8C1B13)](https://openreview.net/forum?id=crKJJ4Ej60)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-blue)](https://huggingface.co/wingchiuloong/CopyPasteLLM-L3-8B)
[![WeChat Blog](https://img.shields.io/badge/WeChat-公众号文章-07C160?logo=wechat&logoColor=white)](https://mp.weixin.qq.com/s/KMv6dQfwSk5rRd6c3I49LQ)

**Authors**: **Yongchao Long**$^{1,2}$  **Yingying Zhang**$^{3}$ **Xianbin Wen**$^{1}$ **Xian Wu**$^{3,\dagger}$ **Yuxi Zhou**$^{1,\dagger}$ **Shenda Hong**$^{2,\dagger}$

$^{1}$ Department of Computer Science, Tianjin University of Technology, Tianjin, China
$^{2}$ National Institute of Health Data Science, Peking University, Beijing, China
$^{3}$ Tencent Jarvis Lab, Shenzhen, China
$^\dagger$ Corresponding author


## News 📰

- **[2026-03-07]** ✨ **Training Data Released!** We've released [CopyPasteSeed365](https://huggingface.co/datasets/wingchiuloong/CopyPasteSeed365), a high-copying seed dataset derived from PubMedQA, FaithEval, and RAGTruth.
- **[2026-01-26]** 🎉 This paper has been accepted by **ICLR 2026**! We will release the full code soon. 🚀
- **[2025-12-17]** 🏆 This paper received the **Best Poster Award** at BIBM 2025 Advancing Data for Better Health Workshop.
- **[2025-09-28]** 🤗 Released the model weights of [CopyPasteLLM](https://huggingface.co/wingchiuloong/CopyPasteLLM-L3-8B) (based on Llama3-8B-Instract)!


## Paper Overview

![Introduction](img/intro.png)

![Method](img/method.png)

## Model Weights 📥

Fine-tuned model: [🤗CopyPasteLLM-L3-8B](https://huggingface.co/wingchiuloong/CopyPasteLLM-L3-8B) (Llama3-8B-Instruct)

---

## Quick Start 🚀

### Installation

```bash
# Clone repository
git clone https://github.com/wingchiuloong/CopyPasteLLM.git
cd CopyPasteLLM

# Install dependencies
pip install -r requirements.txt

# Install package (editable mode)
pip install -e .
```

### Configuration

1. Copy environment template:
```bash
cp .env.example .env
```

2. Edit `.env` with your settings:
```bash
# Required
OPENAI_API_KEY="sk-your-api-key-here"
OPENAI_BASE_URL="https://api.openai.com/v1"  # Or your compatible endpoint

# Optional (with defaults)
DEFAULT_MODEL="gpt-4o-mini"
DEFAULT_PIPELINE="cp-refine"
DEFAULT_TEMPERATURE=0.1
DEFAULT_TIMEOUT=180
VERBOSE=false
```

### Run Examples

```bash
# Basic usage
python examples/basic_usage.py

# Compare all three pipelines
python examples/pipeline_comparison.py

# Batch processing
python examples/batch_processing.py
```

---

## Usage 📚

### Basic Example

```python
from CopyPasteLLM import CopyPasteClient

# Initialize client (reads from .env)
client = CopyPasteClient(verbose=True)

# Generate response
response = client.responses.create(
    context="Your context document here...",
    query="Your question here?",
    pipeline="cp-refine"  # Options: cp-order, cp-link, cp-refine
)

# Access results
print(response.content)              # Generated text
print(response.extractiveness_score) # 0.0 to 1.0 (higher = more extractive)

# Visualize extractiveness
print(response.render_heatmap(context))

# Get extracted fragments
fragments = response.get_fragments(context, min_length=2)
```

### Three Pipelines

| Pipeline | Description | Best For |
|----------|-------------|----------|
| **CP-Order** | Extracts and reorders sentences |
| **CP-Link** | Extracts core sentences + generates transitions |
| **CP-Refine** | Iterative refinement with two-agent system |

### Extractiveness Metrics

- **Score** (0-1): Overall extractiveness (higher = more grounded)
- **Coverage** (0-1): Percentage from context
- **Density**: Average fragment length

---

## API Reference 🔧

### CopyPasteClient

```python
client = CopyPasteClient(
    api_key="sk-...",           # Optional (auto-detects from env)
    base_url="...",             # Optional (default: OpenAI)
    model="gpt-4o-mini",        # Optional (default: from .env)
    default_pipeline="cp-refine", # Optional (default: from .env)
    temperature=0.1,            # Optional (default: from .env)
    timeout=180,                # Optional (default: from .env)
    verbose=True,               # Optional (default: from .env)
    enable_thinking=False,      # Optional (for supported models)
    thinking_budget=4096        # Optional (when enable_thinking=True)
)
```

### CopyPasteResponse

```python
@dataclass
class CopyPasteResponse:
    content: str                      # Generated response
    extractiveness_score: float        # 0.0 to 1.0
    extractiveness_coverage: float     # 0.0 to 1.0
    extractiveness_density: float      # Average fragment length
    pipeline: str                     # Pipeline name
    processing_time: float            # Seconds
    extra: Dict[str, Any]             # Pipeline-specific data

    def render_heatmap(self, context: str, show_legend: bool = True) -> str
    def get_fragments(self, context: str, min_length: int = 2) -> List[str]
```

---

## Troubleshooting 🔍

**"API key not found"**
- Set `OPENAI_API_KEY` in `.env` file or environment variable

**"Model does not support parameter enable_thinking"**
- Remove `enable_thinking=True` (not all models support thinking mode)

**Empty responses**
- Verify `OPENAI_BASE_URL` is correct
- Check API key has valid credits
- Enable `verbose=True` for debug output

---

## Citation

If you find this project helpful for your research, please consider citing our paper.

```bibtex
@inproceedings{
long2026copypaste,
title={Copy-Paste to Mitigate Large Language Model Hallucinations},
author={Yongchao Long and Yingying Zhang and Xianbin Wen and Xian Wu and Yuxi Zhou and Shenda Hong},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=crKJJ4Ej60}
}
```

## Acknowledgement

Welcome to follow our research group: **[PKU Digital Health](https://github.com/PKUDigitalHealth)** (Shenda Hong's Lab, National Institute of Health Data Science, Peking University).

Mirror repository: [PKUDigitalHealth/CopyPasteLLM](https://github.com/PKUDigitalHealth/CopyPasteLLM)

Feel free to check out our other works as well!
