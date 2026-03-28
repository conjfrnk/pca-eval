"""
Academic benchmark suites for PCA evaluation pipeline.

Benchmarks:
    SciFact         - Claim verification against scientific abstracts
    FEVER           - Fact verification against Wikipedia
    QASPER          - Question answering on scientific papers with evidence
    HAGRID          - Hallucination detection in RAG systems
    FActScore       - Atomic fact verification in biographical text
    AttributionBench - Cross-domain attribution verification
    AIS             - Attributable to Identified Sources metric

Execution tiers (all run locally, $0):
    dry-run    - Load data, validate format, print stats.     Seconds.
    nli-only   - Run local DeBERTa NLI model on gold evidence. Minutes.
    nli-abstract - Run NLI against full abstracts (SciFact).  Minutes.
    sweep      - Threshold sweep to find optimal F1.          Minutes.
    calibrate  - Temperature + threshold calibration.         Minutes.
    ais        - AIS attribution metric.                      Minutes.
"""
