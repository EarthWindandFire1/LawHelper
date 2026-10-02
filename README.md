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

