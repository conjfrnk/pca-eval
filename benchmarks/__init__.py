"""
Academic benchmark suites for PCA evaluation pipeline.

Benchmarks:
    SciFact  - Claim verification against scientific abstracts
    FEVER    - Fact verification against Wikipedia
    QASPER   - Question answering on scientific papers with evidence
    HAGRID   - Hallucination detection in RAG systems
    AIS      - Attributable to Identified Sources metric

Execution tiers (for cost-efficient iteration):
    dry-run  - Load data, validate format, print stats.  $0, seconds.
    nli-only - Run local DeBERTa NLI model only.          $0, minutes.
    sample   - Full pipeline on N examples.                ~$5-20.
    full     - Full pipeline on entire dataset.            $70-500.
"""
