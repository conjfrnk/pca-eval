#!/usr/bin/env python3
"""Run all PCA benchmark evaluations.

Evaluates the proof-carrying answer verification pipeline on 7 benchmarks:
SciFact, FEVER, HAGRID, QASPER, FActScore, AttributionBench, FACTS Grounding.
"""


def main() -> None:
    print("PCA Benchmark Evaluation Suite")
    print("=" * 50)
    print()
    print("Benchmarks:")
    print("  1. SciFact    - Claim verification against scientific abstracts")
    print("  2. FEVER      - Fact verification against Wikipedia")
    print("  3. HAGRID     - Hallucination detection in RAG systems")
    print("  4. QASPER     - Question answering with evidence on scientific papers")
    print("  5. FActScore  - Atomic fact verification in biographical text")
    print("  6. AttributionBench - Cross-domain attribution verification")
    print("  7. FACTS Grounding  - Document-grounded response verification")
    print()
    print("Usage:")
    print("  python -m benchmarks.run <benchmark> --tier nli-only")
    print()
    print("See README.md for benchmark details and results.")


if __name__ == "__main__":
    main()
