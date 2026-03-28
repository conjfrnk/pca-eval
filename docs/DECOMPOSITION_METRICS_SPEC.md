# Claim Decomposition Metrics — Technical Specification

## Module Architecture

New module: `benchmarks/decomposition.py` with supporting integration in `benchmarks/run.py`.

```
benchmarks/
├── decomposition.py          # All decomposition metrics (NEW)
│   ├── AtomicityScorer       # Pattern + LLM-based atomicity
│   ├── CompletenessEvaluator # Recall + drop-rate metrics
│   ├── TypeAccuracyEvaluator # Type assignment validation
│   ├── DownstreamTracer      # Pipeline impact correlation
│   └── helpers/              # Utilities (LLM calls, caching, matching)
├── run.py                     # CLI updated: --eval-decomposition flag
└── [existing benchmarks]      # No changes needed
```

---

## 1. AtomicityScorer

### Purpose
Measure how well decomposed claims represent atomic facts (indivisible propositions).

### Interface

```python
from benchmarks.decomposition import AtomicityScorer

scorer = AtomicityScorer(llm_model="gpt-4-mini", cache_dir=".cache/decomposition")

# Method 1: Pattern-based (fast)
violations = scorer.detect_coordination_violations(claim="X and Y and Z")
# → True (multiple AND operators)

# Method 2: Full scoring
result = scorer.score_atomicity(
    claims=["X was founded in 1995", "X and Y are competitors"],
    method="semantic"  # or "pattern" or "hybrid"
)
# → {
#     "overall_score": 0.92,
#     "per_claim": [
#         {"claim": "X was founded...", "score": 0.98, "method": "semantic"},
#         {"claim": "X and Y...", "score": 0.85, "violation": True, "explanation": "..."}
#     ],
#     "statistics": {"mean": 0.92, "std": 0.08, "violations_count": 1}
# }
```

### Metrics

#### 1.1 Pattern-Based (No LLM Cost)

```python
def detect_coordination_violations(claim: str) -> dict:
    """
    Detects syntactic patterns indicating non-atomic claims.

    Returns:
        {
            "is_violation": bool,
            "patterns": [str],  # e.g., ["AND at position 15", "OR at position 45"]
            "confidence": float  # 0.5–1.0 (higher = more certain it's a violation)
        }
    """
    # Regex patterns to detect:
    # 1. Top-level AND/OR (not in quotes, parentheses)
    # 2. Negation chains (NOT NOT, NOT... NOT)
    # 3. Conditional + assertion ("IF X THEN Y, AND Y is true")
    # 4. Multiple subjects/verbs (e.g., "X is Y, Z is W")
```

**Implementation detail**: Use `nltk.sent_tokenize()` + dependency parsing to avoid false positives in quoted strings or proper names.

#### 1.2 Semantic Atomicity (LLM-Based)

```python
def score_semantic_atomicity(
    claims: list[str],
    original_answer: str = None,
    model: str = "gpt-4-mini"
) -> dict:
    """
    Uses LLM to judge if each claim is atomic, mostly atomic, or composite.

    Args:
        claims: List of decomposed claims
        original_answer: (Optional) Full answer text for context
        model: LLM to use (gpt-4-mini recommended; fallback to local model)

    Returns:
        {
            "overall_score": 0.92,  # (A + 0.5*B) / total
            "per_claim": [
                {"claim": "...", "label": "A", "score": 1.0, "explanation": "..."},
                {"claim": "...", "label": "B", "score": 0.5, "explanation": "..."}
            ],
            "label_distribution": {"A": 0.70, "B": 0.20, "C": 0.10}
        }
    """
    # Prompt:
    # "Is this claim atomic (A), mostly atomic (B), or composite (C)?
    #  Atomic: single indivisible fact.
    #  Mostly atomic: core fact is singular, with minor adjuncts.
    #  Composite: 2+ independent propositions."
```

**Caching**: Responses cached in SQLite (reuse existing `benchmarks/cache.py`).

#### 1.3 Hybrid Score

```python
def coordination_index(claims: list[str]) -> float:
    """
    Quick metric: (Total Claims - Pattern Violations) / Total Claims

    Returns score in [0, 1]; > 0.95 is excellent.
    """
```

---

## 2. CompletenessEvaluator

### Purpose
Measure if all substantive claims from the original answer are captured in the decomposition.

### Interface

```python
from benchmarks.decomposition import CompletenessEvaluator

evaluator = CompletenessEvaluator(matching_method="soft")

# Compare decomposed claims against gold reference
result = evaluator.evaluate(
    original_text="Smith founded TechCorp in 2010 and was CEO for 8 years.",
    decomposed_claims=["Smith founded TechCorp in 2010", "Smith was CEO for 8 years"],
    gold_reference_claims=[
        "Smith founded TechCorp",
        "This happened in 2010",
        "Smith served as CEO",
        "Service duration was 8 years"
    ]
)
# → {
#     "recall": 1.0,  # All gold claims covered
#     "precision": 0.67,  # 3 out of 4 decomposed claims match gold
#     "coverage_by_type": {"E": 1.0, "I": 0.8, "S": 0.75},
#     "dropped_claims": [],
#     "drop_rate": 0.0
# }
```

### Metrics

#### 2.1 Claim Recall

```python
def claim_recall(
    decomposed_claims: list[str],
    gold_claims: list[str],
    matching_method: str = "soft"  # or "exact" or "semantic"
) -> float:
    """
    Fraction of gold claims covered by decomposition.

    Matching methods:
    - exact: Token-level exact match
    - soft: Fuzzy string match (Levenshtein ratio > 0.8)
    - semantic: Embedding similarity (cosine > 0.8)

    Returns:
        recall in [0, 1]
    """
```

**Implementation**: For each gold claim, find best match in decomposed claims using chosen method. Count matches.

#### 2.2 Drop Rate

```python
def drop_rate(
    decomposed_claims: list[str],
    original_answer: str,
    entity_type: str = None
) -> dict:
    """
    Identifies claims present in original but missing from decomposition.

    Args:
        decomposed_claims: Output of claim decomposition
        original_answer: Input text
        entity_type: (Optional) Filter for E/I/S types

    Returns:
        {
            "drop_rate": 0.05,  # % of original claims not in decomposition
            "dropped_claims": ["...", "..."],
            "drops_by_type": {"E": 0.02, "I": 0.08, "S": 0.15}
        }
    """
    # Method: Use heuristics to extract candidate claims from original answer.
    # For each candidate, check if it appears (soft match) in decomposed_claims.
    # If not, mark as dropped.
```

---

## 3. TypeAccuracyEvaluator

### Purpose
Validate that claims are assigned correct types (Extractive, Attributed, Synthesis).

### Interface

```python
from benchmarks.decomposition import TypeAccuracyEvaluator

evaluator = TypeAccuracyEvaluator()

result = evaluator.evaluate(
    claims=["X was founded in 1995", "X is financially stable", "X and Y are leaders"],
    predicted_types=["E", "I", "S"],
    gold_types=["E", "I", "S"]
)
# → {
#     "overall_accuracy": 0.81,
#     "per_type": {
#         "E": {"precision": 0.903, "recall": 0.935, "f1": 0.918},
#         "I": {"precision": 0.726, "recall": 0.690, "f1": 0.707},
#         "S": {"precision": 0.770, "recall": 0.671, "f1": 0.717}
#     },
#     "confusion_matrix": [[187, 12, 1], [18, 69, 13], [2, 21, 47]],
#     "kappa": 0.76  # Cohen's kappa
# }
```

### Metrics

```python
def type_accuracy(
    predicted_types: list[str],
    gold_types: list[str]
) -> dict:
    """
    Multi-class accuracy with confusion matrix and per-type metrics.

    Returns:
        {
            "overall_accuracy": float,
            "per_type": {type: {"precision", "recall", "f1"}},
            "confusion_matrix": 3x3 array,
            "kappa": Cohen's kappa (inter-annotator agreement if applicable)
        }
    """
```

**Note**: Types are E, I, S (3-class). Use sklearn.metrics for confusion matrix + f1.

---

## 4. DownstreamTracer

### Purpose
Measure how decomposition quality affects retrieval recall and verification accuracy.

### Interface

```python
from benchmarks.decomposition import DownstreamTracer

tracer = DownstreamTracer(benchmark="scifact")

# Instrument full pipeline
result = tracer.trace_full_pipeline(
    examples=[...],  # SciFact dev set
    decompose_fn=lambda x: ...,
    retrieve_fn=lambda c: ...,
    verify_fn=lambda c, e: ...
)
# → {
#     "correlations": {
#         "atomicity_vs_retrieval_recall": {"pearson": 0.58, "spearman": 0.61, "p": 0.001},
#         "completeness_vs_verification_f1": {"pearson": 0.72, ...}
#     },
#     "per_claim_metrics": [
#         {
#             "claim_id": "scifact_1",
#             "atomicity_score": 0.94,
#             "completeness": True,
#             "type_accuracy": True,
#             "retrieval_recall@5": 0.8,
#             "nli_f1": 0.92
#         },
#         ...
#     ],
#     "ablation": {
#         "oracle": 0.953,
#         "gold_decomposition": 0.879,
#         "full_pipeline": 0.841,
#         "gap_analysis": {
#             "decomposition_impact": 0.038,
#             "retrieval_impact": 0.074
#         }
#     }
# }
```

### Metrics

```python
def correlate_decomposition_to_downstream(
    metrics_df: pd.DataFrame
) -> dict:
    """
    Compute Pearson + Spearman correlations between:
    - Decomposition quality (atomicity, completeness, type_accuracy)
    - Retrieval quality (recall@k)
    - Verification quality (nli_f1)

    Args:
        metrics_df: DataFrame with columns:
            - atomicity_score, completeness, type_accuracy
            - retrieval_recall@1, @5, @10
            - nli_f1, nli_accuracy

    Returns:
        {
            "metric_pair": {
                "pearson": float,
                "spearman": float,
                "p_value": float
            }
        }
    """
```

---

## 5. Helpers & Utilities

### 5.1 LLM Client Wrapper

```python
# benchmarks/decomposition_llm_client.py
class DecompositionLLMClient:
    """
    Unified interface for LLM-based atomicity and type inference.
    Supports OpenAI + local models (via together.ai or ollama).
    """

    def __init__(self, model: str, cache_dir: str, fallback_model: str = None):
        """
        Args:
            model: "gpt-4-mini" or "mistral-7b" etc.
            cache_dir: Where to cache responses
            fallback_model: Used if primary model fails
        """

    def score_atomicity(self, claim: str, context: str = None) -> dict:
        """Score semantic atomicity; cache result."""

    def infer_type(self, claim: str, answer: str) -> str:
        """Infer claim type (E/I/S); cache result."""
```

### 5.2 Matching Utilities

```python
# benchmarks/decomposition_matching.py
from difflib import SequenceMatcher

def soft_match(text1: str, text2: str, threshold: float = 0.8) -> bool:
    """Fuzzy string match using Levenshtein-like ratio."""

def semantic_match(text1: str, text2: str, model: str = "sentence-transformers/all-minilm-l6-v2") -> bool:
    """Embedding-based similarity match."""

def best_match_index(query: str, candidates: list[str], method: str = "soft") -> int:
    """Find best-matching candidate for a query."""
```

### 5.3 Statistics & CI

```python
# benchmarks/decomposition_stats.py
def bootstrap_ci(values: list[float], n_bootstrap: int = 10000) -> tuple:
    """Compute 95% CI using percentile method."""
    return (ci_lower, ci_upper)

def mcnemar_test(contingency_table) -> tuple:
    """Compare two methods on paired binary data."""
    return (stat, p_value)

def plot_correlations(correlation_dict: dict) -> matplotlib.figure.Figure:
    """Heatmap of correlations."""
```

---

## 6. CLI Integration

### New flags in `benchmarks/run.py`

```bash
# Evaluate decomposition on any benchmark
python -m benchmarks.run scifact --eval-decomposition

# Specify evaluation layers
python -m benchmarks.run scifact --eval-decomposition \
    --decomposition-methods pattern semantic hybrid

# Run downstream impact analysis
python -m benchmarks.run scifact --eval-decomposition --trace-downstream

# Use manual gold annotations for type accuracy
python -m benchmarks.run scifact --eval-decomposition \
    --gold-annotations benchmarks/gold_decomposition_annotations.json

# Batch across all benchmarks
python -m benchmarks.run all --eval-decomposition --output results/decomposition_eval/
```

### Output Files

```
results/decomposition_eval/
├── scifact/
│   ├── atomicity_scores.json       # Pattern + semantic scores
│   ├── completeness_report.json    # Recall + drop-by-type
│   ├── type_accuracy.json          # Confusion matrix + per-type metrics
│   └── downstream_correlation.json # Impact analysis
├── qasper/
│   ├── ...
├── summary.json                    # Aggregated results across benchmarks
└── summary.html                    # Human-readable report
```

---

## 7. Testing Strategy

### Unit Tests (`tests/test_decomposition.py`)

```python
def test_coordination_detection():
    """Test pattern matching for AND/OR violations."""
    assert detect_coordination_violations("X and Y") == True
    assert detect_coordination_violations("X is Y") == False

def test_soft_matching():
    """Test fuzzy claim matching."""
    recall = claim_recall(
        ["Smith founded TechCorp"],
        ["Smith founded TechCorp in 2010", "Founded TechCorp"]
    )
    assert recall >= 0.8

def test_confusion_matrix():
    """Test type accuracy computation."""
    result = type_accuracy(
        predicted=["E", "I", "S"],
        gold=["E", "I", "S"]
    )
    assert result["overall_accuracy"] == 1.0
    assert result["kappa"] == 1.0
```

### Integration Tests

```python
def test_end_to_end_scifact():
    """Test full evaluation pipeline on SciFact sample."""
    # Load SciFact dev set (50 claims)
    # Run decomposition → evaluate
    # Verify output JSON format
    # Check that recall is in [0, 1]
```

---

## 8. Expected Output Example

### Table Format (from metrics output)

```
Atomicity Results (All Benchmarks)
─────────────────────────────────────────────
Benchmark         N Claims  Violations  Semantic  Status
─────────────────────────────────────────────
SciFact           1,850     2.3%        0.94      🟡
QASPER            2,100     5.7%        0.89      🔴
HAGRID            1,500     3.1%        0.92      🟡
FActScore         3,500     1.2%        0.96      🟢
SAFE              1,200     4.5%        0.91      🟡
─────────────────────────────────────────────
Overall           10,150    3.4%        0.92      🟡
```

### JSON Output Example

```json
{
  "benchmark": "scifact",
  "timestamp": "2026-03-28T10:30:00Z",
  "decomposition_eval": {
    "atomicity": {
      "overall_score": 0.94,
      "coordination_violations": 0.023,
      "semantic_score": 0.94,
      "per_claim_sample": [
        {"claim": "...", "score": 0.98},
        {"claim": "...", "score": 0.85}
      ]
    },
    "completeness": {
      "recall": 0.942,
      "drop_rate": 0.058,
      "drops_by_type": {"E": 0.032, "I": 0.081, "S": 0.124}
    },
    "type_accuracy": {
      "overall": 0.814,
      "confusion_matrix": [[187, 12, 1], [18, 69, 13], [2, 21, 47]]
    },
    "downstream_impact": {
      "correlations": {
        "atomicity_vs_retrieval_recall": 0.58
      },
      "ablation_gap": 0.038
    }
  }
}
```

---

## 9. Dependencies

### New Packages (if needed)

```
# In pyproject.toml [dependencies]
sentence-transformers>=2.2.0      # For semantic matching (optional)
scikit-learn>=1.0.0                # For confusion matrix + metrics
scipy>=1.8.0                       # For statistical tests (McNemar, etc.)
```

### Reuse Existing

- `benchmarks/cache.py` — SQLite caching for LLM responses
- `benchmarks/nli.py` — NLI model (can reuse for type inference if needed)
- `benchmarks/stats.py` — Bootstrap CI logic

---

## 10. Backward Compatibility

- All new code is isolated in `benchmarks/decomposition.py`
- Existing benchmarks (SciFact, FEVER, etc.) unchanged
- New CLI flags are optional; existing commands still work
- No changes to data loading or existing metrics

---

## Timeline

- **Week 1**: Implement AtomicityScorer + CompletenessEvaluator (unit tested)
- **Week 2**: TypeAccuracyEvaluator + DownstreamTracer + CLI integration
- **Week 3**: Run evaluation on all benchmarks; generate results JSON
- **Week 4**: Generate tables + figures; integrate into paper
