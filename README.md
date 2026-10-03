# Updated Legal Assistant Interface

The original `rag.py` and `intent_eval.py` remain unchanged. The new files are in the same directory as the original files:

| File | Purpose |
| --- | --- |
| `rag_backend.py` | Copy of the RAG backend; loads `rag_frontend.html` for the homepage |
| `intent_eval_backend.py` | Copy of the intent classification backend; adds a homepage that loads `intent_frontend.html` |
| `rag_frontend.html` | Legal question-and-answer page |
| `intent_frontend.html` | Intent classification and question rewriting page |

## Getting Started

### Download models from Hugging Face / 从 Hugging Face 下载模型

Model weights are distributed separately from the GitHub source code. The `models/` directory is ignored by Git. Download the model repository into `models/` at the project root, alongside `rag/`, before starting the services.

模型文件不随 GitHub 代码提交。请先下载模型，并确保 `models/` 与 `rag/` 位于同一级目录。

Hugging Face model repository / 模型仓库：[MichaeY/LawHelper-models](https://huggingface.co/MichaeY/LawHelper-models/tree/main).

Install the download tool:

```powershell
python -m pip install --upgrade huggingface_hub
```

Run the following from the project root (the directory containing `README.md`, `rag_backend.py`, and `rag/`). For the existing Windows checkout:

```powershell
cd D:\Python\CS336\LargeLawHelper
hf download MichaeY/LawHelper-models --repo-type model --local-dir ./models
```

On another computer, change into your own project directory first. `--local-dir ./models` creates the destination directory and downloads the actual weights there; no manual Git LFS setup is required. For a private or gated repository, first run `hf auth login` with an HF token that has read access, and obtain access to the model if required. Public, ungated repositories can normally be downloaded without logging in.

以上命令假设 Hugging Face 仓库根目录直接包含 `qwen3-8b/`、`sft_lora/` 等子目录（即上传了本地 `models/` 的内容）。下载后应为以下结构，**不要多套一层 `models/models/`，也不要放到 `rag/models/` 中**：

```text
LargeLawHelper/
├── README.md
├── rag_backend.py
├── intent_eval_backend.py
├── models/
│   ├── qwen3-8b/
│   ├── qwen3-0.6b/
│   ├── bge-large-zh-v1.5/
│   ├── bge-reranker-large/
│   ├── sft_lora/
│   │   └── checkpoint-18000/
│   └── dpo_lora/
│       └── checkpoint-150/
└── rag/
    ├── bm25_index.pkl
    ├── faiss_chunks.pkl
    └── faiss_index.bin
```

The current local model collection is approximately 26 GiB; allow sufficient disk space and download time. The `rag/` retrieval indexes are separate files supplied through the GitHub repository's Git LFS configuration; this model download does not fetch those indexes.

Downloading the models does not change paths hard-coded in the application. In particular, `intent_eval.py` and `intent_eval_backend.py` currently load their tokenizer from `/root/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B`. When running on another machine, adjust that tokenizer path to the downloaded `./models/qwen3-0.6b` directory, or provide the original path in your runtime environment.

Reference: [Hugging Face CLI documentation](https://huggingface.co/docs/huggingface_hub/en/guides/cli).

### Start the services

Run the following from this directory in the original model runtime environment:

```bash
python rag_backend.py
```

Open http://127.0.0.1:6008/ in your browser. The original command-line options, environment variables, and default values still apply.

To use intent classification and question rewriting independently:

```bash
python intent_eval_backend.py
```

Open http://127.0.0.1:6006/ in your browser. The two services can be used independently; RAG does not require the intent classification HTTP service to be started first.

Each frontend is served by its corresponding backend; do not open the HTML files directly by double-clicking them. No installation of Node, frontend frameworks, or additional Python dependencies is required, and no external fonts or CDNs are used. The RAG page is loaded at startup, so restart the service after editing it. The intent classification page is loaded each time the homepage is requested.


## Experimental Results

### Intent Classification

**Table 1. Results on the Inference400 Dataset**

| Class / Average | Qwen3:0.6b Precision | Qwen3:0.6b Recall | Qwen3:0.6b F1-score | Qwen3:1.7b Precision | Qwen3:1.7b Recall | Qwen3:1.7b F1-score | Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Legal | 74.91 | **100.00** | 85.65 | 95.24 | **100.00** | **97.56** | 200 |
| Policy-violating | **100.00** | 33.00 | 49.62 | **100.00** | 94.00 | 96.91 | 100 |
| Chitchat | 95.00 | 95.00 | **95.00** | **100.00** | 95.00 | 97.44 | 100 |
| Micro average | 82.00 | 82.00 | 82.00 | 97.49 | 97.25 | 97.37 | 400 |
| Macro average | 89.97 | 76.00 | 76.76 | 98.41 | 96.33 | 97.30 | 400 |
| Weighted average | 86.20 | 82.00 | 78.98 | 97.62 | 97.25 | 97.37 | 400 |

**Table 2. Results on the Intent_part2000 Dataset**

| Class / Average | Qwen3:0.6b Precision | Qwen3:0.6b Recall | Qwen3:0.6b F1-score | Qwen3:1.7b Precision | Qwen3:1.7b Recall | Qwen3:1.7b F1-score | Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Legal | 73.42 | **100.00** | 84.67 | 90.74 | **100.00** | 95.15 | 1000 |
| Policy-violating | **100.00** | 26.80 | 42.27 | **100.00** | 84.00 | 91.30 | 500 |
| Chitchat | 98.02 | 98.80 | **98.41** | 99.15 | 93.80 | **96.40** | 500 |
| Micro average | 81.40 | 81.40 | 81.40 | 94.69 | 94.45 | 94.57 | 2000 |
| Macro average | 90.48 | 75.20 | 75.12 | 96.63 | 92.60 | 94.28 | 2000 |
| Weighted average | 86.21 | 81.40 | 77.51 | 95.14 | 94.45 | 94.50 | 2000 |

### Baseline Answer Evaluation

**Table 3. 0/1 Evaluation on the Qa_testset_500 Dataset**

Judge model: Qwen3:32b.

| Metric | Qwen3:0.6b | Qwen3:1.7b | Support |
| --- | ---: | ---: | ---: |
| Consistency | **71.40** | **80.40** | 500 |

### Retrieval and Reranking Evaluation

**Table 4. Retrieval Evaluation on the Qa_testset_100 Dataset**

Judge model: Qwen3:32b.

| Metric | BGE-large | BM25 | Union | Weighted Fusion | RRF |
| --- | ---: | ---: | ---: | ---: | ---: |
| Recall | **59.81** | **46.62** | **69.87** | **68.87** | **66.62** |

**Table 5. Reranking Evaluation on the Qa_testset_100 Dataset**

Judge model: Qwen3:8b.

| Metric | RRF + BGE-base | RRF + BGE-large |
| --- | ---: | ---: |
| Recall | **55.87** | **61.53** |

### Query Rewriting and RAG Answer Evaluation

**Table 6. 0/1 Evaluation on the Qa_testset_500 Dataset**

Judge model: Qwen3:32b.

| Model | Baseline | RAG | Rewrite + RAG |
| --- | ---: | ---: | ---: |
| Qwen3:4b | 86.00 | 89.60 | 87.80 |
| Qwen3:8b | **89.60** | **91.80** | **92.40** |

### Supervised Fine-Tuning (SFT) Evaluation

**Table 7. Qwen3:8b SFT 0/1 Evaluation on the Qa_val_500 Dataset**

Judge model: Qwen3:32b.

| Experiment | Consistency | Win Rate | Epochs | Rank |
| --- | ---: | ---: | ---: | ---: |
| 1 | 89.00 | 65.24 | 0.5 | 8 |
| 2 | 90.40 | 66.87 | 0.5 | 16 |
| 3 | 87.20 | 65.30 | 1 | 8 |
| 4 | 88.60 | 65.86 | 1 | 16 |
| 5 | 87.98 | 67.06 | 2 | 8 |
| 6 | 90.20 | 65.86 | 2 | 16 |
| 7 | **91.80** | 67.06 | 3 | 8 |
| 8 | 91.18 | **68.69** | 3 | 16 |

### Direct Preference Optimization (DPO) Evaluation

**Table 8. Qwen3:8b DPO 0/1 Evaluation on the Qa_val_500 Dataset**

Judge model: Qwen3:32b.

| Experiment | Consistency | Win Rate | Epochs | Rank |
| --- | ---: | ---: | ---: | ---: |
| 1 | 92.00 | 87.07 | 0.5 | 8 |
| 2 | **95.00** | 87.27 | 0.5 | 16 |
| 3 | 94.40 | 85.25 | 1 | 8 |
| 4 | 92.60 | 87.27 | 1 | 16 |
| 5 | **95.00** | 87.68 | 2 | 8 |
| 6 | **95.00** | **88.87** | 2 | 16 |
| 7 | 93.40 | 86.87 | 3 | 8 |
| 8 | 94.40 | 87.88 | 3 | 16 |
