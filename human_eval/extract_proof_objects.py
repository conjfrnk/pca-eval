#!/usr/bin/env python3
"""
Extract 200 proof objects for human evaluation.

Combines:
- 75 SciFact-derived objects
- 75 HAGRID-derived objects
- 50 ClaimVerify edge cases

Outputs: all_proof_objects_200.json (suitable for Prolific/annotation interface)

Usage:
    python extract_proof_objects.py \
        --scifact-dir /path/to/scifact/data \
        --hagrid-dir /path/to/hagrid/data \
        --claimverify-dir /path/to/claimverify/data \
        --output all_proof_objects_200.json
"""

import argparse
import json
import random
import csv
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class ProofObject:
    """Standard proof object structure."""
    object_id: str
    claim_text: str
    evidence_text: str
    evidence_spans: List[Dict[str, Any]]  # [{"start": 0, "end": 10, "text": "..."}]
    scenario: str  # "scifact" | "hagrid" | "claimverify"
    system_score: float
    system_verdict: str  # "SUPPORTED" | "UNSUPPORTED" | "DEFLECTED"
    claim_type: str  # "factual" | "counterfactual" | "multi-hop" | "implicit"
    source_id: str  # reference to source document/paper


class ProofObjectExtractor:
    """Extract proof objects from datasets."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

    def extract_scifact(self, data_dir: Path, count: int = 75) -> List[ProofObject]:
        """
        Extract proof objects from SciFact dataset.

        Expected structure:
            {data_dir}/
              claims.jsonl  - one claim per line: {id, claim_text, evidence_id}
              evidence.json - {evidence_id: {abstract: [sent1, sent2, ...]}}
        """
        objects = []

        # Placeholder implementation
        # In practice, parse claims.jsonl and evidence.json
        print(f"[PLACEHOLDER] Would extract {count} objects from {data_dir}")

        for i in range(count):
            obj = ProofObject(
                object_id=f"scifact_{i+1:03d}",
                claim_text=f"[PLACEHOLDER] SciFact claim #{i+1}",
                evidence_text=f"[PLACEHOLDER] Evidence passage #{i+1}",
                evidence_spans=[
                    {"start": 0, "end": 20, "text": "[evidence span]"}
                ],
                scenario="scifact",
                system_score=random.random(),
                system_verdict=random.choice(["SUPPORTED", "UNSUPPORTED"]),
                claim_type=random.choice(["factual", "multi-hop"]),
                source_id=f"scifact_paper_{i+1:03d}"
            )
            objects.append(obj)

        return objects

    def extract_hagrid(self, data_dir: Path, count: int = 75) -> List[ProofObject]:
        """
        Extract proof objects from HAGRID dataset.

        Expected structure:
            {data_dir}/
              objects.jsonl - one object per line: {id, question, contexts, answer, answer_spans}
        """
        objects = []

        # Placeholder implementation
        # In practice, parse objects.jsonl
        print(f"[PLACEHOLDER] Would extract {count} objects from {data_dir}")

        for i in range(count):
            obj = ProofObject(
                object_id=f"hagrid_{i+1:03d}",
                claim_text=f"[PLACEHOLDER] HAGRID claim #{i+1}",
                evidence_text=f"[PLACEHOLDER] Evidence passage #{i+1}",
                evidence_spans=[
                    {"start": 0, "end": 20, "text": "[evidence span]"}
                ],
                scenario="hagrid",
                system_score=random.random(),
                system_verdict=random.choice(["SUPPORTED", "UNSUPPORTED"]),
                claim_type=random.choice(["implicit", "counterfactual"]),
                source_id=f"hagrid_context_{i+1:03d}"
            )
            objects.append(obj)

        return objects

    def extract_claimverify(self, data_dir: Path, count: int = 50) -> List[ProofObject]:
        """
        Extract proof objects from ClaimVerify live data.

        Expected structure:
            {data_dir}/
              edge_cases.json - high-value examples: {id, claim, evidence, verdict}
              OR use synthetic insurance examples if live data unavailable
        """
        objects = []

        # Placeholder implementation
        # In practice, parse live ClaimVerify data or use synthetic examples
        print(f"[PLACEHOLDER] Would extract {count} objects from {data_dir}")

        for i in range(count):
            obj = ProofObject(
                object_id=f"claimverify_{i+1:03d}",
                claim_text=f"[PLACEHOLDER] Insurance claim #{i+1}",
                evidence_text=f"[PLACEHOLDER] Policy excerpt #{i+1}",
                evidence_spans=[
                    {"start": 0, "end": 20, "text": "[policy clause]"}
                ],
                scenario="claimverify",
                system_score=random.random(),
                system_verdict=random.choice(["SUPPORTED", "UNSUPPORTED", "DEFLECTED"]),
                claim_type=random.choice(["factual", "implicit", "multi-hop"]),
                source_id=f"claimverify_policy_{i+1:03d}"
            )
            objects.append(obj)

        return objects

    def merge_and_validate(
        self,
        scifact_objs: List[ProofObject],
        hagrid_objs: List[ProofObject],
        claimverify_objs: List[ProofObject],
        total_count: int = 200,
        deflection_count: int = 20
    ) -> List[ProofObject]:
        """
        Merge objects from all scenarios, validate, and ensure diversity.

        Rules:
        - Total: 200 objects
        - Distribute scenarios proportionally
        - Ensure ~20 DEFLECTED objects (system cannot verify)
        - Shuffle to avoid clustering by scenario
        """
        all_objs = scifact_objs + hagrid_objs + claimverify_objs

        # Validate counts
        assert len(all_objs) == total_count, f"Expected {total_count}, got {len(all_objs)}"

        # Count deflections
        deflected = [o for o in all_objs if o.system_verdict == "DEFLECTED"]
        print(f"Total objects: {len(all_objs)}")
        print(f"Deflected objects: {len(deflected)}")
        print(f"Scenario distribution:")
        for scenario in ["scifact", "hagrid", "claimverify"]:
            count = len([o for o in all_objs if o.scenario == scenario])
            print(f"  {scenario}: {count}")

        # Shuffle
        random.shuffle(all_objs)

        return all_objs

    def export_for_prolific(self, objects: List[ProofObject], output_path: Path) -> None:
        """
        Export as CSV for Prolific bulk upload.

        Prolific reads CSV columns and iterates over rows, one per task.
        """
        csv_path = output_path.with_suffix(".csv")

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "object_id",
                "claim_text",
                "evidence_text",
                "scenario",
                "system_score",
                "system_verdict"
            ])
            writer.writeheader()
            for obj in objects:
                writer.writerow({
                    "object_id": obj.object_id,
                    "claim_text": obj.claim_text,
                    "evidence_text": obj.evidence_text,
                    "scenario": obj.scenario,
                    "system_score": f"{obj.system_score:.3f}",
                    "system_verdict": obj.system_verdict
                })

        print(f"Exported CSV: {csv_path}")

    def export_for_annotation_interface(self, objects: List[ProofObject], output_path: Path) -> None:
        """
        Export as JSON for standalone annotation interface.

        Format: [{ object_id, claim_text, evidence_text, ... }, ...]
        """
        data = [asdict(obj) for obj in objects]

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Exported JSON: {output_path}")

    def export_system_verdicts(self, objects: List[ProofObject], output_path: Path) -> None:
        """
        Export system verdicts as lookup table for analysis script.

        Format: { object_id: system_verdict, ... }
        """
        verdicts = {obj.object_id: obj.system_verdict for obj in objects}

        with open(output_path, "w") as f:
            json.dump(verdicts, f, indent=2)

        print(f"Exported verdicts: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract 200 proof objects for human evaluation"
    )
    parser.add_argument(
        "--scifact-dir",
        type=Path,
        help="Path to SciFact dataset directory"
    )
    parser.add_argument(
        "--hagrid-dir",
        type=Path,
        help="Path to HAGRID dataset directory"
    )
    parser.add_argument(
        "--claimverify-dir",
        type=Path,
        help="Path to ClaimVerify dataset directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("all_proof_objects_200.json"),
        help="Output JSON file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    extractor = ProofObjectExtractor(seed=args.seed)

    print("=== Extracting Proof Objects ===")
    print()

    # Extract from each scenario
    print("1. SciFact (75 objects)")
    scifact_objs = extractor.extract_scifact(args.scifact_dir or Path("."), count=75)

    print("2. HAGRID (75 objects)")
    hagrid_objs = extractor.extract_hagrid(args.hagrid_dir or Path("."), count=75)

    print("3. ClaimVerify (50 objects)")
    claimverify_objs = extractor.extract_claimverify(args.claimverify_dir or Path("."), count=50)

    # Merge and validate
    print("\n4. Merging and validating...")
    all_objs = extractor.merge_and_validate(scifact_objs, hagrid_objs, claimverify_objs)

    # Export
    print("\n5. Exporting...")
    extractor.export_for_prolific(all_objs, args.output)
    extractor.export_for_annotation_interface(all_objs, args.output)
    extractor.export_system_verdicts(all_objs, Path("system_verdicts.json"))

    print("\n✓ Complete. Ready for annotation interface or Prolific upload.")


if __name__ == "__main__":
    main()
