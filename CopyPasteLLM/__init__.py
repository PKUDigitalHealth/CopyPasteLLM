"""
CopyPasteLLM - Copy-paste pipelines for extractive RAG with hallucination mitigation

This package provides three copy-paste RAG pipelines that prioritize extracting
content from context rather than generating new information, significantly reducing
hallucinations in RAG systems.

## Pipelines

- **CP-Order**: Ultra-strict extraction with intelligent sentence ordering
- **CP-Link**: Cohesive responses with transition generation
- **CP-Refine**: Multi-agent iterative refinement for highest quality

## Quick Start

```python
from CopyPasteLLM import CopyPasteClient

# Initialize client (auto-detects OPENAI_API_KEY from environment)
client = CopyPasteClient(
    model="gpt-4o-mini",
    default_pipeline="cp-order"
)

# Generate extractive response
response = client.responses.create(
    context="Your context document...",
    query="Your question here?"
)

print(response.content)
print(f"Extractiveness: {response.extractiveness_score:.3f}")
```

## Citation

If you use this package, please cite our paper:

```bibtex
@inproceedings{long2026copypaste,
  title={Copy-Paste to Mitigate Large Language Model Hallucinations},
  author={Yongchao Long and Yingying Zhang and Xianbin Wen and Xian Wu and Yuxi Zhou and Shenda Hong},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=crKJJ4Ej60}
}
```
"""

__version__ = "0.1.0"

from CopyPasteLLM.client import CopyPasteClient, CopyPasteResponse
from CopyPasteLLM.pipelines import (
    PipelineBase,
    CPOrderPipeline,
    CPLinkPipeline,
    CPRefinePipeline
)
from CopyPasteLLM.metrics import ExtractivenessMetrics
from CopyPasteLLM.utils import LLMBackend, TextProcessor, create_standard_response_structure

__all__ = [
    # Version
    '__version__',

    # Client
    'CopyPasteClient',
    'CopyPasteResponse',

    # Pipelines
    'PipelineBase',
    'CPOrderPipeline',
    'CPLinkPipeline',
    'CPRefinePipeline',

    # Metrics
    'ExtractivenessMetrics',

    # Utilities
    'LLMBackend',
    'TextProcessor',
    'create_standard_response_structure',
]
