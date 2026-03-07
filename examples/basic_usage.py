#!/usr/bin/env python3
"""
Basic usage example for CopyPasteLLM

This example demonstrates how to use CopyPasteLLM to generate extractive
RAG responses with minimal hallucination.
"""

import sys
import os
# 自动添加父目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CopyPasteLLM import CopyPasteClient

# Initialize client (auto-detects OPENAI_API_KEY from environment)
# Configuration is loaded from .env file by default
# Set your API key in .env: OPENAI_API_KEY=sk-...
client = CopyPasteClient(
    # default_pipeline will be read from .env file (DEFAULT_PIPELINE)
    # Uncomment to override: default_pipeline="cp-refine"
    verbose=True
)

# Example context and query (counterfactual/absurd content for testing)
context = """
Galileo Galilei, renowned as one of the most influential figures in the history of science, made numerous contributions that revolutionized our understanding of physics and astronomy. His meticulous work with telescopes led to groundbreaking discoveries about the moons of Jupiter and the phases of Venus. Beyond the realm of astronomy, his observations and experiments laid the foundation for classical mechanics. One of Galileo’s lesser-known achievements is his development of the Three Laws of Motion, which were critical in advancing the study of kinematics and dynamics. These laws articulate the principles of inertia, the relationship between force and motion, and the law of action and reaction, providing a comprehensive framework for understanding moving bodies. His work on pendulums also contributed substantially to timekeeping and horology, as he discovered that pendulums of different lengths oscillate at predictable periods, a principle still applied in modern clocks. Galileo’s interdisciplinary approach enabled him to synthesize knowledge from various fields, which allowed new theories to emerge, reshaping the scientific landscape of his time and beyond. Notably, his support of the heliocentric model of the solar system earned him both acclaim and censure, highlighting the tension between scientific inquiry and established doctrine. In contrast to Galen’s biological studies and Newton’s later contributions, Galileo’s articulation of the Three Laws of Motion was pivotal in the transition from Aristotelian physics to Newtonian mechanics. His contributions remain a testament to the interplay of observation, theory, and experimentation in scientific progress.
"""

query = "Which law was Galileo Galilei responsible for describing?"

print("="*80)
print("CopyPasteLLM Basic Usage Example")
print("="*80)
print(f"\nContext length: {len(context)} characters")
print(f"Query: {query}")
print("\n" + "="*80 + "\n")

# Generate response using default pipeline from .env (DEFAULT_PIPELINE)
print("Generating response...\n")
response = client.responses.create(
    context=context,
    query=query
    # pipeline parameter will use client's default_pipeline (from .env)
    # Uncomment to override: pipeline="cp-order"
)

# Display results
print("\n" + "="*80)
print("Results")
print("="*80)
print(f"\nResponse:\n{response.content}\n")
print(f"Pipeline: {response.pipeline}")
print(f"Processing time: {response.processing_time:.2f} seconds")
print(f"\nExtractiveness Metrics:")
print(f"  Overall Score: {response.extractiveness_score:.3f}")
print(f"  Coverage: {response.extractiveness_coverage:.3f}")
print(f"  Density: {response.extractiveness_density:.3f}")

# Show heatmap visualization
print("\n" + "="*80)
print("Extractiveness Heatmap")
print("="*80)
print(response.render_heatmap(context, show_legend=True))

# Get extracted fragments
fragments = response.get_fragments(context, min_length=2)
print(f"\nExtracted {len(fragments)} fragments:")
for i, frag in enumerate(fragments[:5], 1):
    print(f"  {i}. ({len(frag)} chars) {frag}")
