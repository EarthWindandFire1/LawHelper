# Updated Legal Assistant Interface

The original `rag.py` and `intent_eval.py` remain unchanged. The new files are in the same directory as the original files:

| File | Purpose |
| --- | --- |
| `rag_backend.py` | Copy of the RAG backend; loads `rag_frontend.html` for the homepage |
| `intent_eval_backend.py` | Copy of the intent classification backend; adds a homepage that loads `intent_frontend.html` |
| `rag_frontend.html` | Legal question-and-answer page |
| `intent_frontend.html` | Intent classification and question rewriting page |

## Getting Started

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

## Unchanged Behavior

Both backends are copies of the original code. The only changes are the HTML source used by RAG and the addition of a homepage for intent classification. Model loading, all prompts, generation parameters, retrieval, reranking, classification, rewriting, locks, random seeds, error handling, and legal question saving behavior are preserved. RAG's default `--intent-file` still points to the original `intent_eval.py`, preserving the original parameter; the new copy does not automatically change this default.

The interfaces continue to use `POST /ask` (RAG) and `GET /intent?question=...` (intent classification); the other original endpoints are also preserved. The pages do not add conversation context, append prompts, or modify answers or reference materials returned by the backend. The intent classification page only displays classification and rewriting results; it does not generate legal answers.

The local model paths, dependencies, and index requirements in the original code also remain unchanged, including the original absolute path for the intent classification tokenizer. Start the services in a model environment where the original services can run.

`frontend_original_hashes.json` records the SHA-256 hashes of the two original files before the copies were created and can be used to verify that the original files have not been modified.
