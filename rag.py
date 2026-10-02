"""Place beside intent_eval.py; run this file instead of intent_eval.py.

python rag.py
Place the answer base model in models/qwen3-8b, or pass --base-model /path/to/base.
Defaults: models/sft_lora/checkpoint-18000 + models/dpo_lora/checkpoint-150.
All bundled paths are resolved relative to this script, not the working directory.
Default sft_dpo mode assumes a NEW DPO adapter trained on a merged SFT base.
If DPO continued training the SFT adapter, use dpo mode (avoid applying SFT twice).
Dependencies: existing intent_eval dependencies plus faiss-cpu, jieba,
rank-bm25, sentence-transformers, numpy, accelerate.
Only load trusted local pickle indexes. No evaluation JSON file is needed.
"""
from __future__ import annotations

import argparse
import os
from typing import List
import importlib.util
import json
import logging
import pickle
import re
import threading
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger("law-helper")

HTML = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>法律问答助手</title><style>
body{font:16px/1.7 system-ui,sans-serif;background:#f4f6fa;color:#17243b;margin:0}
main{max-width:850px;margin:40px auto;padding:24px}section{background:white;padding:24px;border-radius:14px;margin:18px 0}
textarea{box-sizing:border-box;width:100%;min-height:130px;padding:14px;font:inherit;border:1px solid #bbc5d4;border-radius:8px}
button{background:#245fce;color:white;border:0;border-radius:8px;padding:10px 24px;font:inherit;cursor:pointer}
button:disabled{opacity:.6}#answer,.context{white-space:pre-wrap;overflow-wrap:anywhere}.muted{color:#62718a}h1,h2{line-height:1.3}
</style><main><h1>法律问答助手</h1><p class="muted">输入问题，查看改写、回答和参考资料。</p>
<section><form id="form"><textarea id="question" maxlength="4000" required placeholder="请输入你的法律问题"></textarea>
<button id="submit">生成答案</button></form><p id="status" role="status" aria-live="polite"></p></section>
<section id="result" hidden><p id="rewrite"></p><p id="keywords" class="muted"></p><h2>回答</h2><div id="answer"></div>
<details id="sources"><summary>查看参考资料</summary><div id="contexts"></div></details></section></main>
<script>
const $=id=>document.getElementById(id);
$('form').addEventListener('submit',async event=>{event.preventDefault();
const question=$('question').value.trim();if(!question)return;
$('submit').disabled=true;$('result').hidden=true;$('status').textContent='正在识别问题、检索资料并生成答案，请稍候…';
try{const response=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question})});
const data=await response.json();if(!response.ok)throw new Error(data.error||'请求失败');
$('rewrite').textContent=data.rewritten?'改写问题：'+data.rewritten:'';
$('keywords').textContent=data.keywords?'关键词：'+data.keywords:'';
$('answer').textContent=data.LLM_answer||data.message||'';
$('contexts').replaceChildren();(data.contexts||[]).forEach((text,index)=>{const p=document.createElement('p');p.className='context';p.textContent=(index+1)+'. '+text;$('contexts').appendChild(p)});
$('sources').hidden=!(data.contexts||[]).length;$('result').hidden=false;$('status').textContent='处理完成';
}catch(error){$('status').textContent='处理失败：'+error.message}finally{$('submit').disabled=false}});
</script></html>"""


DEFAULT_PROMPT = """你是一名法律咨询助手。

你的任务是：
针对用户提出的法律相关问题，提供法律原则说明和建议。

请严格遵守以下规则：
1. 简明解释法律原则、制度，然后给出常见处理思路，不要编造具体事实。
2. 回答避免冗长，不要超过150词。

例子：
输入："我在餐厅吃饭，滑倒摔断了腿，餐厅地板确实很滑且没放提示牌，我可以索赔吗？",
输出："根据《民法典》，宾馆、商场、餐馆等经营场所的经营者负有安全保障义务。未尽义务导致他人损害的，应当承担侵权责任。您可以主张医药费、护理费、误工费等赔偿。"

用户问题：
{query}

请按照以上要求，给出你的回答：
"""

DEFAULT_RAG_PROMPT = """你是一名法律咨询助手。

你的任务是：
针对用户提出的法律相关问题，提供法律原则说明和建议。

【参考资料】
{contexts}

请严格遵守以下规则：
1. 简明解释法律原则、制度，然后给出常见处理思路，不要编造具体事实。
2. 回答避免冗长，不要超过500词。

例子：
输入："我在餐厅吃饭，滑倒摔断了腿，餐厅地板确实很滑且没放提示牌，我可以索赔吗？",
输出："根据《民法典》，宾馆、商场、餐馆等经营场所的经营者负有安全保障义务。未尽义务导致他人损害的，应当承担侵权责任。您可以主张医药费、护理费、误工费等赔偿。"

用户问题：
{query}

请按照以上要求，给出你的回答：
"""

DTYPE = os.getenv("DTYPE", "float16")

USE_CHAT_TEMPLATE = os.getenv("USE_CHAT_TEMPLATE", "1") == "1"

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "768"))

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

TOP_P = float(os.getenv("TOP_P", "0.8"))

DO_SAMPLE = os.getenv("DO_SAMPLE", "1") == "1"

USE_RAG = True

def format_contexts(ctxs: List[str]) -> str:
    lines = []
    for i, c in enumerate(ctxs, start=1):
        c = (c or "").strip()
        if not c:
            continue
        lines.append(f"{i}. {c}")
    return "\n".join(lines) if lines else "(无)"

def build_model_input(tokenizer, prompt_tpl: str, query: str, ctxs: List[str] = None,
                      use_chat_template: bool = True) -> str:
    if ctxs and USE_RAG:
        contexts_text = format_contexts(ctxs)
        if "{contexts}" in prompt_tpl:
            user_text = prompt_tpl.format(query=query, contexts=contexts_text)
        else:
            user_text = f"{prompt_tpl.format(query=query)}\n\n【参考资料】\n{contexts_text}"
    else:
        user_text = prompt_tpl.format(query=query)

    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            pass
    return user_text

def generate_one(model, tokenizer, input_text: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    import torch
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[-1]
    gen_ids = outputs[0][input_len:]
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
    return answer

def _torch_dtype(dtype: str):
    import torch
    d = (dtype or "").lower()
    if d in ("bf16", "bfloat16"):
        return torch.bfloat16
    if d in ("fp16", "float16"):
        return torch.float16
    return torch.float32


def load_answer_prompt():
    # Same file-first selection as infer; relative paths follow this script.
    p = Path(os.getenv("PROMPT_FILE", "baseline_qa.txt"))
    if not p.is_absolute():
        p = ROOT / p
    if p.exists():
        txt = p.read_text(encoding="utf-8").strip()
        if "{query}" not in txt:
            raise ValueError(f"Prompt 文件必须包含 '{{query}}' 占位符：{p}")
        return txt
    return DEFAULT_RAG_PROMPT


def load_answer_model(base_path):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_path, trust_remote_code=True,
        torch_dtype=_torch_dtype(DTYPE) if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None)
    return tokenizer, model


class Retriever:
    def __init__(self, args):
        import faiss
        import jieba
        import numpy as np
        from sentence_transformers import SentenceTransformer, CrossEncoder
        self.np, self.jieba, self.args = np, jieba, args
        self.index = faiss.read_index(str(args.rag_dir / "faiss_index.bin"))
        with (args.rag_dir / "faiss_chunks.pkl").open("rb") as f:
            self.chunks = pickle.load(f)
        with (args.rag_dir / "bm25_index.pkl").open("rb") as f:
            self.bm25, self.bm25_chunks, _ = pickle.load(f)
        if not self.index.ntotal or self.index.ntotal != len(self.chunks):
            raise ValueError("Faiss 索引为空或与 faiss_chunks 数量不一致")
        if not self.bm25_chunks or self.bm25.corpus_size != len(self.bm25_chunks):
            raise ValueError("BM25 索引为空或与片段数量不一致")
        for chunk in list(self.chunks) + list(self.bm25_chunks):
            if not isinstance(chunk, dict) or not isinstance(chunk.get("text"), str):
                raise ValueError("检索片段必须为包含 text 字符串的字典")
        self.embedder = SentenceTransformer(args.embedding_model, device=args.retrieval_device)
        if self.embedder.get_sentence_embedding_dimension() != self.index.d:
            raise ValueError("Embedding 模型维度与 Faiss 索引不匹配")
        self.reranker = None if args.no_rerank else CrossEncoder(
            args.reranker_model, device=args.retrieval_device)

    def search(self, query):
        np = self.np
        vec = self.embedder.encode([query], normalize_embeddings=not self.args.no_normalize,
                                   show_progress_bar=False)
        _, ids = self.index.search(np.asarray(vec, dtype=np.float32),
                                   min(self.args.candidate_k, self.index.ntotal))
        bge = [self.chunks[int(i)]["text"] for i in ids[0] if 0 <= i < len(self.chunks)]
        scores = self.bm25.get_scores(list(self.jieba.cut(query)))
        bm25 = [self.bm25_chunks[int(i)]["text"] for i in
                np.argsort(scores)[::-1][:self.args.candidate_k]]
        # Fuse by text identity, not array position: index orders may differ.
        fused = defaultdict(float)
        for ranking in (bge, bm25):
            seen = set()
            for rank, text in enumerate(ranking, 1):
                text = text.strip()
                if text and text not in seen:
                    fused[text] += 1.0 / (60 + rank)
                    seen.add(text)
        candidates = sorted(fused, key=fused.get, reverse=True)[:self.args.fusion_k]
        if self.reranker is not None and candidates:
            scores = self.reranker.predict([(query, text) for text in candidates],
                                           batch_size=8, show_progress_bar=False)
            candidates = [text for text, _ in sorted(zip(candidates, scores),
                                                     key=lambda pair: float(pair[1]), reverse=True)]
        return candidates[:self.args.context_k]


def load_causal_model(base_path, adapter=None, tokenizer_path=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or base_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_path, trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    return tokenizer, model.eval()


def normalize_adapter_keys(weights):
    """Remove only redundant PEFT wrapper prefixes; never discard tensors."""
    normalized = {}
    changed = 0
    for key, tensor in weights.items():
        original = key
        while key.startswith("base_model.model.base_model.model."):
            key = key[len("base_model.model."):]
        if key in normalized:
            raise ValueError(f"Adapter 权重名称转换后发生冲突：{key}")
        normalized[key] = tensor
        changed += key != original
    return normalized, changed


def load_checked_adapter(base, path, label):
    from peft import PeftConfig, get_peft_model
    from peft.utils.save_and_load import (
        load_peft_weights, get_peft_model_state_dict, set_peft_model_state_dict)
    LOG.info("正在加载 %s adapter：%s", label, path)
    config = PeftConfig.from_pretrained(path)
    model = get_peft_model(base, config)
    weights, changed = normalize_adapter_keys(load_peft_weights(path, device="cpu"))
    expected = get_peft_model_state_dict(model)
    missing = sorted(set(expected) - set(weights))
    unexpected = sorted(set(weights) - set(expected))
    mismatched = [key for key in set(expected) & set(weights)
                  if expected[key].shape != weights[key].shape]
    if missing or unexpected or mismatched:
        raise ValueError(f"{label} 权重不匹配，停止启动：missing={len(missing)}, "
                         f"unexpected={len(unexpected)}, shape_mismatch={len(mismatched)}; "
                         f"示例：{(missing + unexpected + mismatched)[:5]}")
    result = set_peft_model_state_dict(model, weights, adapter_name="default")
    adapter_missing = [key for key in result.missing_keys if "lora_" in key]
    if adapter_missing or result.unexpected_keys:
        raise ValueError(f"{label} 加载后仍存在未匹配权重："
                         f"{(adapter_missing + list(result.unexpected_keys))[:5]}")
    LOG.info("%s adapter 加载完成：%d 个张量，修正 %d 个重复前缀", label, len(weights), changed)
    return model.eval()


class Pipeline:
    def __init__(self, args):
        from peft import PeftConfig, PeftModel
        self.args = args
        self.lock = threading.Lock()
        # Import classification/rewrite functions without starting the old server.
        spec = importlib.util.spec_from_file_location("intent_backend", args.intent_file)
        self.intent = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.intent)
        adapter = None
        intent_base = args.intent_model
        if (Path(intent_base) / "adapter_config.json").exists():
            adapter = intent_base
            intent_base = args.intent_base_model or PeftConfig.from_pretrained(adapter).base_model_name_or_path
        self.intent_tokenizer, self.intent_model = load_causal_model(
            intent_base, adapter, args.intent_tokenizer)
        self.retriever = Retriever(args)
        self.prompt_tpl = load_answer_prompt()
        self.tokenizer, self.model = load_answer_model(args.base_model)
        if args.mode == "sft_dpo":
            self.model = load_checked_adapter(self.model, args.sft_lora, "SFT").merge_and_unload()
        self.model = load_checked_adapter(self.model, args.dpo_lora, "DPO").eval()

    def generate(self, query, contexts):
        input_text = build_model_input(
            self.tokenizer, self.prompt_tpl, query, ctxs=contexts,
            use_chat_template=USE_CHAT_TEMPLATE)
        return generate_one(self.model, self.tokenizer, input_text,
                            max_new_tokens=self.args.max_new_tokens)

    def ask(self, question):
        # Both rewrite and generation are protected against concurrent GPU calls.
        with self.lock:
            label = self.intent.classify_question(self.intent_tokenizer, self.intent_model, question)
            payload = {"question": question, "classification": label}
            if label == "闲聊类":
                return dict(payload, result=1, message="请输入需要咨询的法律问题。")
            if label == "违规类":
                return dict(payload, result=2, message="无法提供违法行为的实施指导，可以咨询相关法律后果。")
            if label != "法律类":
                raise ValueError("未能识别问题类别，请换一种表达后重试")
            rewrite = self.intent.rewrite_question(self.intent_tokenizer, self.intent_model, question)
            query = rewrite.get("rewritten", "").strip() or question
            # The rewritten question is used for BOTH retrieval and answering.
            contexts = [text[:self.args.context_chars] for text in self.retriever.search(query)]
            answer = self.generate(query, contexts)
            return dict(payload, result=0, rewritten=query, keywords=rewrite.get("keywords", ""),
                        LLM_answer=answer, contexts=contexts)


def create_handler(pipeline):
    class Handler(BaseHTTPRequestHandler):
        def send_body(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status, data):
            self.send_body(status, json.dumps(data, ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")

        def respond(self, question):
            if not isinstance(question, str) or not question.strip() or len(question) > 4000:
                self.send_json(400, {"error": "question 必须为 1–4000 字符的非空字符串"})
                return
            try:
                result = pipeline.ask(question.strip())
            except ValueError as error:
                self.send_json(422, {"error": str(error)})
                return
            except Exception:
                LOG.exception("答案生成失败")
                self.send_json(500, {"error": "答案生成失败，请查看服务端日志"})
                return
            self.send_json(200, result)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                self.send_body(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif url.path == "/intent":
                self.respond(parse_qs(url.query).get("question", [None])[0])
            else:
                self.send_json(404, {"error": "route not found"})

        def do_POST(self):
            if urlparse(self.path).path not in ("/ask", "/intent"):
                self.send_json(404, {"error": "route not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError()
                data = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(data, dict):
                    raise ValueError()
            except (ValueError, UnicodeError):
                self.send_json(400, {"error": "请发送有效的 JSON 对象，大小不超过 64KB"})
                return
            self.respond(data.get("question"))

        def log_message(self, fmt, *args):
            # Do not print the question from GET query strings into access logs.
            LOG.info("HTTP request from %s", self.client_address[0])
    return Handler


def parse_args():
    parser = argparse.ArgumentParser(description="意图识别 → 改写 → RAG → LoRA/DPO 回答 → 网页")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6008)
    parser.add_argument("--intent-file", type=Path, default=ROOT / "intent_eval.py")
    parser.add_argument("--intent-model", default=str(ROOT / "models/qwen3-0.6b"))
    parser.add_argument("--intent-base-model", help="意图模型为 adapter 时覆盖其底模路径")
    parser.add_argument("--intent-tokenizer", help="默认使用意图底模中的 tokenizer")
    parser.add_argument("--base-model", default=str(ROOT / "models/qwen3-8b"), help="答案模型底模路径")
    parser.add_argument("--dpo-lora", default=str(ROOT / "models/dpo_lora/checkpoint-150"), help="DPO adapter 路径")
    parser.add_argument("--sft-lora", default=str(ROOT / "models/sft_lora/checkpoint-18000"), help="仅 sft_dpo 模式使用")
    parser.add_argument("--mode", choices=["dpo", "sft_dpo"], default="sft_dpo")
    parser.add_argument("--rag-dir", type=Path, default=ROOT / "rag")
    parser.add_argument("--embedding-model", default=str(ROOT / "models/bge-large-zh-v1.5"))
    parser.add_argument("--reranker-model", default=str(ROOT / "models/bge-reranker-large"))
    parser.add_argument("--retrieval-device", default="cpu", help="cpu 或 cuda:0 等")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--no-normalize", action="store_true", help="仅当建库时没有归一化时使用")
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--fusion-k", type=int, default=20)
    parser.add_argument("--context-k", type=int, default=5)
    parser.add_argument("--context-chars", type=int, default=1800)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()
    if args.mode == "sft_dpo" and not args.sft_lora:
        parser.error("sft_dpo 模式需要 --sft-lora")
    for name in ("candidate_k", "fusion_k", "context_k", "context_chars", "max_new_tokens"):
        if getattr(args, name) < 1:
            parser.error(f"{name} 必须大于零")
    for path in [args.intent_file] + [args.rag_dir / name for name in
            ("faiss_index.bin", "faiss_chunks.pkl", "bm25_index.pkl")]:
        if not path.is_file():
            parser.error(f"找不到文件：{path}")
    return args


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    LOG.info("正在加载意图模型、检索模型和答案模型，请稍候")
    pipeline = Pipeline(args)
    server = ThreadingHTTPServer((args.host, args.port), create_handler(pipeline))
    LOG.info("网页已启动：http://%s:%s/", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
