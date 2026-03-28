# Third-Party Notices

This project uses the following third-party datasets and models. All datasets are downloaded at runtime from their original sources and are not included in this repository.

## Benchmark Datasets

| Dataset | Source | License | Citation |
|---------|--------|---------|----------|
| **SciFact** | [Allen AI](https://github.com/allenai/scifact) | CC BY-NC 2.0 | Wadden et al., EMNLP 2020 |
| **FEVER** | [fever.ai](https://fever.ai/) | CC BY-SA 3.0 | Thorne et al., NAACL 2018 |
| **QASPER** | [Allen AI](https://allenai.org/data/qasper) | CC BY 4.0 | Dasigi et al., NAACL 2021 |
| **HAGRID** | [miracl/hagrid](https://huggingface.co/datasets/miracl/hagrid) | Apache 2.0 | Kamalloo et al., 2023 |
| **FActScore** | [shmsw25/FActScore](https://github.com/shmsw25/FActScore) | MIT | Min et al., EMNLP 2023 |
| **AttributionBench** | [osunlp/AttributionBench](https://huggingface.co/datasets/osunlp/AttributionBench) | CC BY 4.0 | Li et al., ACL 2024 Findings |

## Pre-trained Models

| Model | Source | License |
|-------|--------|---------|
| DeBERTa-v3-base/large | [Microsoft](https://huggingface.co/microsoft/deberta-v3-base) | MIT |
| cross-encoder/nli-deberta-v3-base | [sentence-transformers](https://huggingface.co/cross-encoder/nli-deberta-v3-base) | Apache 2.0 |
| MiniCheck-DeBERTa-v3-Large | [lytang](https://huggingface.co/lytang/MiniCheck-DeBERTa-v3-Large) | MIT |
| FactCG-DeBERTa-v3-Large | [yaxili96](https://huggingface.co/yaxili96/FactCG-DeBERTa-v3-Large) | Apache 2.0 |
| MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli | [MoritzLaurer](https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli) | MIT |

## Software Dependencies

This project depends on open-source libraries including PyTorch (BSD), Hugging Face Transformers (Apache 2.0), sentence-transformers (Apache 2.0), scikit-learn (BSD), NumPy (BSD), and SciPy (BSD).
