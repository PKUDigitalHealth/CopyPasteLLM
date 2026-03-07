#!/usr/bin/env python3
"""
Batch processing example for CopyPasteLLM

This example demonstrates how to process multiple samples efficiently
and analyze the results.
"""

import sys
import os
import time
from typing import List, Dict, Any
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CopyPasteLLM import CopyPasteClient

# Initialize client
# Configuration will be loaded from .env file (DEFAULT_MODEL, DEFAULT_PIPELINE, etc.)
client = CopyPasteClient(
    # Uncomment to override: model="gpt-4o-mini", default_pipeline="cp-order"
    verbose=False
)

# Sample dataset (in practice, load from file)
samples = [
    {
        "context": """
The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in
Paris, France. It is named after the engineer Gustave Eiffel, whose company
designed and built the tower. Constructed from 1887 to 1889, it was the
entrance to the 1889 World's Fair. The tower is 330 metres (1,083 ft) tall,
about the same height as an 81-storey building, and the tallest structure
in Paris. Its base is square, measuring 125 metres (410 ft) on each side.
""",
        "query": "When was the Eiffel Tower built and how tall is it?"
    },
    {
        "context": """
Photosynthesis is a process used by plants and other organisms to convert
light energy into chemical energy that, through cellular respiration, can
later be released to fuel the organism's activities. Some of this chemical
energy is stored in carbohydrate molecules, such as sugars and starches,
which are synthesized from carbon dioxide and water. The name comes from
the Greek phōs, "light", and synthesis, "putting together". In most cases,
oxygen is also released as a waste product. Most plants, most algae, and
cyanobacteria perform photosynthesis; such organisms are called
photoautotrophs. Photosynthesis is largely responsible for producing and
maintaining the oxygen content of the Earth's atmosphere, and supplies most
of the energy necessary for life on Earth.
""",
        "query": "What is photosynthesis and why is it important?"
    },
    {
        "context": """
The Industrial Revolution was the transition to new manufacturing processes
in Great Britain, continental Europe, and the United States, that occurred
during the period from around 1760 to about 1820-1840. This transition
included going from hand production methods to machines, new chemical
manufacturing and iron production processes, the increasing use of steam
power and water power, the development of machine tools and the rise of the
factory system. The Industrial Revolution also led to an unprecedented rise
in the rate of population growth, and began the anthropogenic climate
change crisis which continues to the present day.
""",
        "query": "What were the main changes during the Industrial Revolution?"
    },
    {
        "context": """
The moon is Earth's only natural satellite. It is the fifth-largest
satellite in the Solar System and the largest among planetary satellites
relative to the size of the planet that it orbits. The Moon is a
planetary-mass object that formed a differentiated rocky body, making it a
satellite planet under the geophysical definitions of the term. It is
second brightest object in the sky after the Sun. Its surface is actually
dark, but compared to the night sky it appears very bright. The Moon's
orbit around Earth takes approximately 27.3 days and is used to track months
in calendars.
""",
        "query": "How long does it take for the moon to orbit Earth?"
    },
    {
        "context": """
A blockchain is a distributed ledger with growing lists of records (blocks)
that are securely linked together via cryptographic hashes. Each block
contains a cryptographic hash of the previous block, a timestamp, and
transaction data (generally represented as a Merkle tree, where data nodes
are represented by leaves). The timestamp proves that the transaction data
existed when the block was created. Since each block contains information
about the block previous to it, they effectively form a chain (as linked
in the title "blockchain"), with each additional block linking to the ones
before it. Consequently, blockchain transactions are irreversible in that,
once they are recorded, the data in any given block cannot be altered
retroactively without altering all subsequent blocks.
""",
        "query": "How does blockchain ensure data integrity?"
    }
]

print("="*80)
print("CopyPasteLLM Batch Processing Example")
print("="*80)
print(f"\nProcessing {len(samples)} samples...")
print("\n" + "="*80 + "\n")

# Process all samples
results = []
total_start_time = time.time()

for i, sample in enumerate(samples, 1):
    print(f"[{i}/{len(samples)}] Processing: {sample['query'][:50]}...")

    start_time = time.time()
    try:
        response = client.responses.create(
            context=sample['context'],
            query=sample['query']
            # Use client's default_pipeline from .env, or uncomment to override:
            # pipeline="cp-order"
        )

        results.append({
            "sample_id": i,
            "query": sample['query'],
            "response": response.content,
            "extractiveness_score": response.extractiveness_score,
            "coverage": response.extractiveness_coverage,
            "density": response.extractiveness_density,
            "processing_time": response.processing_time,
            "response_length": len(response.content)
        })

        elapsed = time.time() - start_time
        print(f"  ✓ Completed in {elapsed:.2f}s (extractiveness: {response.extractiveness_score:.3f})")

    except Exception as e:
        print(f"  ✗ Failed: {str(e)}")
        results.append({
            "sample_id": i,
            "query": sample['query'],
            "error": str(e)
        })

total_time = time.time() - total_start_time

# Calculate statistics
successful_results = [r for r in results if "error" not in r]
print("\n" + "="*80)
print("Batch Processing Statistics")
print("="*80)

print(f"\nTotal samples: {len(samples)}")
print(f"Successful: {len(successful_results)}")
print(f"Failed: {len(results) - len(successful_results)}")
print(f"Total time: {total_time:.2f} seconds")
print(f"Average time per sample: {total_time/len(samples):.2f} seconds")

if successful_results:
    # Extractiveness statistics
    extractiveness_scores = [r['extractiveness_score'] for r in successful_results]
    coverage_scores = [r['coverage'] for r in successful_results]
    density_scores = [r['density'] for r in successful_results]

    print(f"\nExtractiveness Statistics:")
    print(f"  Mean: {sum(extractiveness_scores)/len(extractiveness_scores):.3f}")
    print(f"  Min: {min(extractiveness_scores):.3f}")
    print(f"  Max: {max(extractiveness_scores):.3f}")

    print(f"\nCoverage Statistics:")
    print(f"  Mean: {sum(coverage_scores)/len(coverage_scores):.3f}")
    print(f"  Min: {min(coverage_scores):.3f}")
    print(f"  Max: {max(coverage_scores):.3f}")

    print(f"\nDensity Statistics:")
    print(f"  Mean: {sum(density_scores)/len(density_scores):.3f}")
    print(f"  Min: {min(density_scores):.3f}")
    print(f"  Max: {max(density_scores):.3f}")

    # Response length statistics
    response_lengths = [r['response_length'] for r in successful_results]
    print(f"\nResponse Length Statistics:")
    print(f"  Mean: {sum(response_lengths)/len(response_lengths):.0f} characters")
    print(f"  Min: {min(response_lengths)} characters")
    print(f"  Max: {max(response_lengths)} characters")

# Detailed results
print("\n" + "="*80)
print("Detailed Results")
print("="*80)

for result in successful_results:
    print(f"\nSample {result['sample_id']}:")
    print(f"  Query: {result['query']}")
    print(f"  Extractiveness: {result['extractiveness_score']:.3f}")
    print(f"  Coverage: {result['coverage']:.3f}")
    print(f"  Density: {result['density']:.3f}")
    print(f"  Response: {result['response'][:100]}...")

# Recommendations
print("\n" + "="*80)
print("Analysis & Recommendations")
print("="*80)

avg_extractiveness = sum(extractiveness_scores) / len(extractiveness_scores)
if avg_extractiveness > 0.8:
    print("\n✓ Excellent extractiveness!")
    print("  Your responses are highly grounded in the context with minimal hallucination.")
elif avg_extractiveness > 0.6:
    print("\n✓ Good extractiveness.")
    print("  Responses are mostly grounded, but there may be some generated content.")
else:
    print("\n⚠ Moderate extractiveness.")
    print("  Consider using CP-Order or CP-Refine for higher extractiveness.")
    print("  Lower extractiveness may indicate more generation/hallucination.")

print("\n💡 Tip: Use CP-Refine for highest quality, CP-Order for speed, CP-Link for flow.")
