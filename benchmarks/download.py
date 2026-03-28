"""
Download benchmark datasets.

All datasets are fetched from HuggingFace Hub and saved as JSONL
in benchmarks/data/<name>/. No HuggingFace account needed.

Usage:
    python -m benchmarks.download              # Download all
    python -m benchmarks.download scifact      # Download one
    python -m benchmarks.download --list       # Show available
"""

import argparse
import json
import logging
import sys
import tarfile
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


def _count_lines(path: Path) -> int:
    """Count lines in a file safely."""
    with open(path) as f:
        return sum(1 for _ in f)

# HuggingFace datasets API base
HF_API = "https://datasets-server.huggingface.co/rows"

# Direct download URLs
SCIFACT_TARBALL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
FEVER_DEV_URL = "https://fever.ai/download/fever/shared_task_dev.jsonl"
FEVER_WIKI_URL = "https://fever.ai/download/fever/wiki-pages.zip"


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file with progress indicator."""
    try:
        logger.info(f"Downloading {desc or url}")
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 // total
                    print(f"\r  {desc}: {pct}% ({downloaded // 1024}KB)", end="", flush=True)
        print()
        logger.info(f"Saved to {dest}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def download_hf_dataset(dataset: str, config: str, split: str, dest: Path, max_rows: int = 0) -> bool:
    """Download a dataset split from HuggingFace datasets server API."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    offset = 0
    page_size = 100
    all_rows = []

    desc = f"{dataset}/{config}/{split}" if config else f"{dataset}/{split}"
    logger.info(f"Downloading {desc} from HuggingFace")

    while True:
        params = {
            "dataset": dataset,
            "split": split,
            "offset": offset,
            "length": page_size,
        }
        if config:
            params["config"] = config

        try:
            resp = requests.get(HF_API, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"HF API error at offset {offset}: {e}")
            break

        rows = data.get("rows", [])
        if not rows:
            break

        for row in rows:
            all_rows.append(row.get("row", row))

        offset += len(rows)
        print(f"\r  {desc}: {len(all_rows)} rows", end="", flush=True)

        if max_rows and len(all_rows) >= max_rows:
            all_rows = all_rows[:max_rows]
            break

        if len(rows) < page_size:
            break

    print()

    if not all_rows:
        logger.error(f"No data fetched for {desc}")
        return False

    with open(dest, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    logger.info(f"Saved {len(all_rows)} rows to {dest}")
    return True


def download_scifact() -> bool:
    """
    Download SciFact dataset.

    SciFact: 1,409 expert-annotated claims verified against 5,183 abstracts.
    Labels: SUPPORT, CONTRADICT, NOT_ENOUGH_INFO

    Downloads the official tarball from Allen AI's S3 and extracts it.
    """
    out_dir = DATA_DIR / "scifact"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    if (out_dir / "corpus.jsonl").exists() and (out_dir / "claims_dev.jsonl").exists():
        corpus_count = _count_lines(out_dir / "corpus.jsonl")
        claims_count = _count_lines(out_dir / "claims_dev.jsonl")
        logger.info(f"SciFact already downloaded: {corpus_count} abstracts, {claims_count} dev claims")
        return True

    # Download tarball from Allen AI S3
    tarball_path = out_dir / "data.tar.gz"
    if not download_file(SCIFACT_TARBALL, tarball_path, desc="SciFact tarball"):
        # Fallback: HuggingFace API
        logger.info("Tarball download failed, trying HuggingFace API")
        ok1 = download_hf_dataset("allenai/scifact", "corpus", "train", out_dir / "corpus_hf.jsonl")
        ok2 = download_hf_dataset("allenai/scifact", "claims", "validation", out_dir / "claims_dev_hf.jsonl")
        return ok1 and ok2

    # Extract tarball
    logger.info("Extracting SciFact tarball...")
    try:
        with tarfile.open(tarball_path, "r:gz") as tar:
            tar.extractall(path=out_dir, filter="data")

        # The tarball extracts to data/ subdirectory; move files up
        extracted_dir = out_dir / "data"
        if extracted_dir.exists():
            for f in extracted_dir.iterdir():
                dest = out_dir / f.name
                if not dest.exists():
                    f.rename(dest)

        # Clean up
        tarball_path.unlink(missing_ok=True)
        if extracted_dir.exists():
            import shutil
            shutil.rmtree(extracted_dir, ignore_errors=True)

        corpus_count = _count_lines(out_dir / "corpus.jsonl")
        claims_count = _count_lines(out_dir / "claims_dev.jsonl")
        logger.info(f"SciFact: {corpus_count} abstracts, {claims_count} dev claims")
        return True

    except Exception as e:
        logger.error(f"Failed to extract SciFact tarball: {e}")
        tarball_path.unlink(missing_ok=True)
        return False


def download_fever() -> bool:
    """
    Download FEVER dataset.

    FEVER: 185K claims verified against Wikipedia. We download the
    pre-processed version with gold evidence sentences included.
    For NLI-only evaluation we use the labelled claim + evidence pairs.

    We download a manageable subset (dev set, ~20K examples).
    """
    out_dir = DATA_DIR / "fever"
    out_dir.mkdir(parents=True, exist_ok=True)

    # FEVER shared task dev set with evidence
    if download_file(FEVER_DEV_URL, out_dir / "shared_task_dev.jsonl", desc="FEVER dev set"):
        count = _count_lines(out_dir / "shared_task_dev.jsonl")
        logger.info(f"FEVER: {count} dev examples")
        return True

    # Fallback: HuggingFace
    logger.info("Direct download failed, trying HuggingFace API")

    # fever dataset on HF
    ok = download_hf_dataset("fever/fever", "v1.0", "labelled_dev", out_dir / "dev_hf.jsonl")
    if not ok:
        # Try alternate config names
        ok = download_hf_dataset("fever/fever", "default", "validation", out_dir / "dev_hf.jsonl")
    return ok


def download_fever_wiki() -> bool:
    """
    Download processed Wikipedia pages for FEVER evidence lookup.

    This is large (~4GB). Skipped by default -- use --include-wiki flag.
    For NLI-only evaluation, we extract evidence text from the annotations.
    """
    out_dir = DATA_DIR / "fever"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("FEVER Wikipedia dump is ~4GB. Downloading...")
    return download_file(FEVER_WIKI_URL, out_dir / "wiki-pages.zip", desc="FEVER Wikipedia")


def download_qasper() -> bool:
    """
    Download QASPER dataset.

    QASPER: ~5K questions on 1,585 NLP papers with evidence annotations.
    Each answer has extractive/abstractive answer + evidence paragraphs.

    Downloads v0.3 tarballs from Allen AI's S3 bucket (the official source
    used by the HuggingFace dataset loading script).
    """
    out_dir = DATA_DIR / "qasper"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if (out_dir / "qasper_dev.json").exists() or (out_dir / "qasper_test.json").exists():
        logger.info("QASPER already downloaded")
        return True

    # Official S3 URLs from the HuggingFace dataset loading script (qasper.py)
    train_dev_url = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
    test_url = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz"

    ok = False

    # Download and extract train/dev tarball (~10MB)
    train_dev_tarball = out_dir / "qasper-train-dev-v0.3.tgz"
    if download_file(train_dev_url, train_dev_tarball, desc="QASPER train+dev"):
        try:
            with tarfile.open(train_dev_tarball, "r:gz") as tar:
                tar.extractall(path=out_dir, filter="data")
            # Rename extracted files to canonical names
            for src, dest in [
                ("qasper-dev-v0.3.json", "qasper_dev.json"),
                ("qasper-train-v0.3.json", "qasper_train.json"),
            ]:
                src_path = out_dir / src
                if src_path.exists():
                    src_path.rename(out_dir / dest)
            ok = True
        except Exception as e:
            logger.error(f"Failed to extract QASPER train/dev tarball: {e}")
        finally:
            train_dev_tarball.unlink(missing_ok=True)

    # Download and extract test tarball (~4MB)
    test_tarball = out_dir / "qasper-test-and-evaluator-v0.3.tgz"
    if download_file(test_url, test_tarball, desc="QASPER test"):
        try:
            with tarfile.open(test_tarball, "r:gz") as tar:
                tar.extractall(path=out_dir, filter="data")
            src_path = out_dir / "qasper-test-v0.3.json"
            if src_path.exists():
                src_path.rename(out_dir / "qasper_test.json")
            ok = True
        except Exception as e:
            logger.error(f"Failed to extract QASPER test tarball: {e}")
        finally:
            test_tarball.unlink(missing_ok=True)

    if ok:
        return True

    # Fallback: HuggingFace datasets-server API (config is "qasper", not "default")
    logger.info("S3 download failed, trying HuggingFace datasets-server API")
    return download_hf_dataset("allenai/qasper", "qasper", "validation", out_dir / "dev_hf.jsonl")


def download_hagrid() -> bool:
    """
    Download HAGRID dataset.

    HAGRID: Hallucination-Annotated Generation for RAG Informative Dialogue.
    Contains queries, passages, answers, and attribution annotations.

    The HAGRID dataset uses a custom loading script on HuggingFace, so the
    datasets-server API does not support it. We download the JSONL files
    directly from the repository instead.
    """
    out_dir = DATA_DIR / "hagrid"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Direct download from HuggingFace repo (datasets-server API is not
    # supported for this dataset because it uses a custom loading script)
    base = "https://huggingface.co/datasets/miracl/hagrid/resolve/main/hagrid-v1.0-en"
    ok1 = download_file(f"{base}/dev.jsonl", out_dir / "dev.jsonl", desc="HAGRID dev")
    ok2 = download_file(f"{base}/train.jsonl", out_dir / "train.jsonl", desc="HAGRID train")
    return ok1 and ok2


def download_attribution_bench() -> bool:
    """
    Download AttributionBench dataset.

    AttributionBench (OSU NLP, ACL 2024 Findings): ~26K examples aggregated
    from 7 attribution datasets. Binary labels (attributable / not attributable).
    Includes HAGRID as an out-of-distribution test set.

    Data: https://huggingface.co/datasets/osunlp/AttributionBench
    """
    out_dir = DATA_DIR / "attribution_bench"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if (out_dir / "test_id.jsonl").exists() and (out_dir / "test_ood.jsonl").exists():
        id_count = _count_lines(out_dir / "test_id.jsonl")
        ood_count = _count_lines(out_dir / "test_ood.jsonl")
        logger.info(f"AttributionBench already downloaded: {id_count} ID test, {ood_count} OOD test")
        return True

    # Download from HuggingFace datasets-server API
    # AttributionBench uses "full_data" config with train/dev/test splits
    # The test split contains both ID and OOD examples with src_dataset metadata
    splits = {
        "train": "train",
        "dev": "validation",
        "test": "test",
    }

    ok = True
    for local_name, hf_split in splits.items():
        dest = out_dir / f"{local_name}.jsonl"
        if dest.exists():
            continue
        result = download_hf_dataset(
            "osunlp/AttributionBench", "full_data", hf_split, dest,
        )
        if not result:
            # Try without config
            result = download_hf_dataset(
                "osunlp/AttributionBench", "", hf_split, dest,
            )
        ok = ok and result

    # Split test set into ID and OOD based on src_dataset metadata
    test_path = out_dir / "test.jsonl"
    if test_path.exists():
        _split_test_by_domain(out_dir, test_path)

    if ok:
        for f in out_dir.glob("*.jsonl"):
            count = _count_lines(f)
            logger.info(f"  {f.name}: {count} examples")

    return ok


def _split_test_by_domain(out_dir: Path, test_path: Path) -> None:
    """Split test set into in-distribution and out-of-distribution subsets."""
    # OOD datasets per the AttributionBench paper
    ood_datasets = {"BEGIN", "HAGRID", "AttrEval-GenSearch"}

    id_examples = []
    ood_examples = []

    with open(test_path) as f:
        for line in f:
            item = json.loads(line)
            src = item.get("src_dataset", "")
            if src in ood_datasets:
                ood_examples.append(line)
            else:
                id_examples.append(line)

    if id_examples:
        with open(out_dir / "test_id.jsonl", "w") as f:
            f.writelines(id_examples)
        logger.info(f"  test_id: {len(id_examples)} examples")

    if ood_examples:
        with open(out_dir / "test_ood.jsonl", "w") as f:
            f.writelines(ood_examples)
        logger.info(f"  test_ood: {len(ood_examples)} examples")


def download_factscore() -> bool:
    """
    Download FActScore labeled data from Google Drive.

    FActScore (Min et al., EMNLP 2023): Human-annotated atomic facts from
    LLM-generated biographies. Each fact is labeled as Supported (S),
    Not-supported (NS), or Irrelevant (IR).

    Data: https://github.com/shmsw25/FActScore
    """
    from .factscore import FActScore
    suite = FActScore()
    try:
        suite.download()
        return True
    except Exception as e:
        logger.error(f"FActScore download failed: {e}")
        return False


DOWNLOADERS = {
    "scifact": download_scifact,
    "fever": download_fever,
    "qasper": download_qasper,
    "hagrid": download_hagrid,
    "attribution_bench": download_attribution_bench,
    "factscore": download_factscore,
}


def download_all(include_wiki: bool = False) -> dict[str, bool]:
    """Download all benchmark datasets."""
    results = {}
    for name, fn in DOWNLOADERS.items():
        logger.info(f"\n--- Downloading {name} ---")
        results[name] = fn()

    if include_wiki:
        results["fever_wiki"] = download_fever_wiki()

    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument("datasets", nargs="*", help="Specific datasets to download (default: all)")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--include-wiki", action="store_true", help="Download FEVER Wikipedia dump (~4GB)")
    args = parser.parse_args()

    if args.list:
        print("Available benchmark datasets:")
        for name in DOWNLOADERS:
            print(f"  {name}")
        return 0

    if args.datasets:
        for name in args.datasets:
            if name not in DOWNLOADERS:
                print(f"Unknown dataset: {name}. Available: {', '.join(DOWNLOADERS.keys())}")
                return 1
            DOWNLOADERS[name]()
    else:
        results = download_all(include_wiki=args.include_wiki)
        print("\nDownload summary:")
        for name, ok in results.items():
            status = "OK" if ok else "FAILED"
            print(f"  {name}: {status}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
