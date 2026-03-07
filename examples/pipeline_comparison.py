#!/usr/bin/env python3
"""
Pipeline comparison example for CopyPasteLLM

This example demonstrates how to compare all three pipelines on the same input,
with detailed extractiveness heatmaps for visual comparison.
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CopyPasteLLM import CopyPasteClient
from CopyPasteLLM.metrics import ExtractivenessMetrics

# Initialize client
# Configuration will be loaded from .env file (DEFAULT_MODEL, DEFAULT_PIPELINE, etc.)
client = CopyPasteClient(
    verbose=False  # Set to True for detailed output
)

# Example context and query (counterfactual/absurd content for testing)
context = """
Galileo Galilei, renowned as one of the most influential figures in the history of science, made numerous contributions that revolutionized our understanding of physics and astronomy. His meticulous work with telescopes led to groundbreaking discoveries about the moons of Jupiter and the phases of Venus. Beyond the realm of astronomy, his observations and experiments laid the foundation for classical mechanics. One of Galileo’s lesser-known achievements is his development of the Three Laws of Motion, which were critical in advancing the study of kinematics and dynamics. These laws articulate the principles of inertia, the relationship between force and motion, and the law of action and reaction, providing a comprehensive framework for understanding moving bodies. His work on pendulums also contributed substantially to timekeeping and horology, as he discovered that pendulums of different lengths oscillate at predictable periods, a principle still applied in modern clocks. Galileo’s interdisciplinary approach enabled him to synthesize knowledge from various fields, which allowed new theories to emerge, reshaping the scientific landscape of his time and beyond. Notably, his support of the heliocentric model of the solar system earned him both acclaim and censure, highlighting the tension between scientific inquiry and established doctrine. In contrast to Galen’s biological studies and Newton’s later contributions, Galileo’s articulation of the Three Laws of Motion was pivotal in the transition from Aristotelian physics to Newtonian mechanics. His contributions remain a testament to the interplay of observation, theory, and experimentation in scientific progress.
"""

query = "Which law was Galileo Galilei responsible for describing?"

print("="*80)
print("Copy-Paste-Prompting Comparison (with Baseline)")
print("="*80)
print(f"\nQuery: {query}")
print(f"Context length: {len(context)} characters")
print("\n" + "="*80 + "\n")

# Initialize results dictionary
results = {}

# Test baseline first (direct LLM call without pipeline)
print(f"Testing BASELINE (direct LLM call)...")
start_time = time.time()
try:
    # Create a simple RAG prompt
    baseline_prompt = f"""Context: {context}

Question: {query}"""

    baseline_response_text = client.llm_backend.call(
        baseline_prompt,
        agent_name="Baseline"
    )

    # Calculate extractiveness metrics
    metrics = ExtractivenessMetrics()
    coverage, density, score = metrics.calculate_extractiveness_score(context, baseline_response_text)

    # Create a mock response object similar to CopyPasteResponse
    from dataclasses import dataclass
    from typing import Dict, Any

    @dataclass
    class BaselineResponse:
        content: str
        extractiveness_score: float
        extractiveness_coverage: float
        extractiveness_density: float
        pipeline: str
        processing_time: float
        extra: Dict[str, Any]

        def render_heatmap(self, context: str, show_legend: bool = True) -> str:
            return metrics.render_extractiveness_heatmap(context, self.content, show_legend)

        def get_fragments(self, context: str, min_length: int = 2, sort_by_length: bool = True):
            fragments, _ = metrics.get_extractive_fragments(context, self.content, descending=sort_by_length, include_min_len=min_length)
            return fragments

    baseline_response = BaselineResponse(
        content=baseline_response_text,
        extractiveness_score=score,
        extractiveness_coverage=coverage,
        extractiveness_density=density,
        pipeline="baseline",
        processing_time=time.time() - start_time,
        extra={}
    )

    results["baseline"] = baseline_response
    print(f"  ✓ Completed in {baseline_response.processing_time:.2f}s")
    print(f"    Extractiveness: {baseline_response.extractiveness_score:.3f}")
    print(f"    Response length: {len(baseline_response.content)} chars\n")

except Exception as e:
    print(f"  ✗ Failed: {str(e)}\n")

# Test all three pipelines
pipelines = ["cp-order", "cp-link", "cp-refine"]

for pipeline_name in pipelines:
    print(f"Testing {pipeline_name.upper()} pipeline...")

    start_time = time.time()
    try:
        response = client.responses.create(
            context=context,
            query=query,
            pipeline=pipeline_name
        )
        results[pipeline_name] = response
        print(f"  ✓ Completed in {response.processing_time:.2f}s")
        print(f"    Extractiveness: {response.extractiveness_score:.3f}")
        print(f"    Response length: {len(response.content)} chars\n")
    except Exception as e:
        print(f"  ✗ Failed: {str(e)}\n")

# Display comparison with heatmaps
print("="*80)
print("Pipeline Comparison Results")
print("="*80)

for pipeline_name, response in results.items():
    print(f"\n{'='*80}")
    print(f"{pipeline_name.upper()} Pipeline")
    print(f"{'='*80}")

    # Metrics
    print(f"\nExtractiveness Score: {response.extractiveness_score:.3f}")
    print(f"  Coverage: {response.extractiveness_coverage:.3f}")
    print(f"  Density: {response.extractiveness_density:.3f}")
    print(f"Processing Time: {response.processing_time:.2f} seconds")
    print(f"Response Length: {len(response.content)} characters")

    # Show pipeline-specific info
    if response.extra:
        print(f"\nPipeline-specific Info:")
        for key, value in response.extra.items():
            if key == "num_extracted_sentences":
                print(f"  Extracted sentences: {value}")
            elif key == "num_core_sentences":
                print(f"  Core sentences: {value}")
            elif key == "num_transitions":
                print(f"  Transitions generated: {value}")
            elif key == "iteration_count":
                print(f"  Refinement iterations: {value}")

    # Heatmap visualization
    print(f"\n{'─'*80}")
    print("Response with Copy-Paste Heatmap:")
    print(f"{'─'*80}")
    print(response.render_heatmap(context, show_legend=True))

    # Extracted fragments
    fragments = response.get_fragments(context, min_length=2, sort_by_length=True)
    print(f"\nExtracted {len(fragments)} fragments:")
    for i, frag in enumerate(fragments[:10], 1):
        print(f"  {i:2d}. ({len(frag):3d} chars) {frag[:100]}{'...' if len(frag) > 100 else ''}")
    if len(fragments) > 10:
        print(f"  ... and {len(fragments) - 10} more fragments")

    print()  # Extra spacing between pipelines

# Summary table
print("\n" + "="*80)
print("Summary Table")
print("="*80)
print(f"{'Pipeline':<15} {'Extractiveness':<15} {'Coverage':<12} {'Density':<12} {'Time (s)':<10} {'Length':<10}")
print("-" * 90)

for pipeline_name, response in results.items():
    print(f"{pipeline_name:<15} {response.extractiveness_score:<15.3f} "
          f"{response.extractiveness_coverage:<12.3f} {response.extractiveness_density:<12.3f} "
          f"{response.processing_time:<10.2f} {len(response.content):<10}")

# Recommendations
print("\n" + "="*80)
print("Recommendations")
print("="*80)

# Best extractiveness
best_pipeline = max(results.items(), key=lambda x: x[1].extractiveness_score)
print(f"\n🏆 Best Extractiveness: {best_pipeline[0].upper()} "
      f"(score: {best_pipeline[1].extractiveness_score:.3f})")
print(f"   → Use {best_pipeline[0].upper()} for maximum factual accuracy")

# Fastest
fastest_pipeline = min(results.items(), key=lambda x: x[1].processing_time)
print(f"\n⚡ Fastest: {fastest_pipeline[0].upper()} "
      f"({fastest_pipeline[1].processing_time:.2f}s)")
print(f"   → Use {fastest_pipeline[0].upper()} for quick responses")

# Best coverage
best_coverage = max(results.items(), key=lambda x: x[1].extractiveness_coverage)
print(f"\n📊 Best Coverage: {best_coverage[0].upper()} "
      f"({best_coverage[1].extractiveness_coverage:.3f})")
print(f"   → Use {best_coverage[0].upper()} for most comprehensive extraction")
