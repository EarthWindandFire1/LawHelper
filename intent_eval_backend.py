import os
import json
import argparse
import torch
import random
import threading
import re
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
from typing import List, Dict, Optional
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

# --- 1. 配置区域 ---

# 固定类别列表
LABEL_LIST = ["法律类", "违规类", "闲聊类"]

# ===== 新增：改写 Prompt（完全保留 rewrite.py 的配置）=====
REWRITE_PROMPT_TEMPLATE = """你将用户的口语化问题改写成专业、规范的书面表达，并提取1-3个核心关键词。

规则：
- 输出格式：【改写问题】xxx【关键词】xxx
- 只输出一行，不要换行
- 不要解释，不要加多余内容

示例：
原始问题：车停在路边被高空坠物砸坏，找整栋楼赔吗？
输出：【改写问题】车辆停放在路边被高空坠物损坏，是否可以要求整栋楼业主承担赔偿责任？【关键词】高空坠物、赔偿、责任

原始问题：借朋友车出了事故，车主需要担责吗？
输出：【改写问题】借用他人车辆发生交通事故，车主是否需要承担法律责任？【关键词】交通事故、责任、车主

原始问题：砍头息实际上没拿到那么多钱，按多少还？
输出：【改写问题】实际未足额收到砍头息，应按何种金额偿还借款？【关键词】砍头息、还款、金额

原始问题：买房交了定金不想买了，定金能退吗？
输出：【改写问题】购房交付定金后反悔，已付定金是否可以退还？【关键词】定金、退还、购房

原始问题：未满14周岁杀人，需要承担刑事责任吗？
输出：【改写问题】未满14周岁的人杀人是否需要承担刑事责任？【关键词】刑事责任、未成年人、杀人

现在请改写：
原始问题：{question}
输出："""


# ===== 新增：兜底关键词提取 =====
def extract_keywords_simple(question: str) -> str:
    """简单的规则关键词提取（兜底用）"""
    common_keywords = [
        "赔偿", "合同", "工伤", "离婚", "借条", "定金", "辞退", "社保", "加班费",
        "交通事故", "高空坠物", "侵权", "继承", "抚养权", "黑客", "诈骗", "造假",
        "洗钱", "偷拍", "走私", "伪造", "刑事责任", "判刑", "诉讼", "证据", "责任",
        "贷款", "买房", "租房", "劳动法", "婚姻", "遗嘱", "违约金", "利息"
    ]
    found = [kw for kw in common_keywords if kw in question]
    return "、".join(found[:3]) if found else "其他"


# ===== 新增：改写函数 =====
def rewrite_question(tokenizer, model, question: str) -> Dict[str, str]:
    """
    对法律类问题进行改写和关键词提取
    返回：{"rewritten": 改写后的问题, "keywords": 关键词}
    """
    prompt = REWRITE_PROMPT_TEMPLATE.format(question=question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,  # 与 rewrite.py 保持一致
            do_sample=False,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # 解析【改写问题】和【关键词】
    rewrite = ""
    keywords = ""

    rewrite_match = re.search(r'【改写问题】(.*?)【关键词】', text)
    keywords_match = re.search(r'【关键词】(.*?)$', text)

    if rewrite_match:
        rewrite = rewrite_match.group(1).strip()
    if keywords_match:
        keywords = keywords_match.group(1).strip()

    # 解析失败时的兜底
    if not rewrite:
        rewrite = re.sub(r'^【改写问题】|【关键词】.*$', '', text).strip()
        if not rewrite:
            rewrite = text.splitlines()[0].strip() if text else question

    if not keywords:
        keywords = extract_keywords_simple(question)

    return {"rewritten": rewrite, "keywords": keywords}


def get_absolute_path(filename):
    """
    获取文件的绝对路径，确保无论在哪里运行命令，
    都能找到脚本同级目录下的数据文件。
    """
    if os.path.isabs(filename):
        return filename
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, filename)


def load_model_and_tokenizer(adapter_dir: str):
    print(f"正在加载模型配置，路径：{adapter_dir}")

    # 尝试自动推断底模路径
    try:
        peft_config = PeftConfig.from_pretrained(adapter_dir)
        base_model_path = peft_config.base_model_name_or_path
        print(f"检测到底模路径：{base_model_path}")
    except Exception:
        # 如果不是 LoRA 路径，或者没找到 adapter_config，就假设传入的是底模本身
        print(f"未检测到 adapter_config.json，将直接加载：{adapter_dir}")
        base_model_path = adapter_dir

    # tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("/root/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B",
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 针对 32B 大模型的量化配置
    quantization_config = None
    if "32B" in base_model_path.upper() or "32B" in adapter_dir.upper():
        print("检测到 32B 模型，正在启用 4-bit 量化...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    print("正在加载模型权重...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True
    )

    if os.path.exists(os.path.join(adapter_dir, "adapter_model.bin")) or \
            os.path.exists(os.path.join(adapter_dir, "adapter_model.safetensors")):
        print("正在挂载 LoRA 适配器...")
        model = PeftModel.from_pretrained(model, adapter_dir)

    model.eval()
    return tokenizer, model


def classify_question(tokenizer, model, question: str) -> str:
    # prompt = (
    #     "你是一个中文意图识别助手，需要判断用户问题属于哪一类："
    #     "【违规类、闲聊类、法律类】之一。\n"
    #     "- 违规类：只要涉及中国领导人、色情，或者询问/传授违法手段、非法交易、黑产工具等非法或违规行为。都属于违规\n"
    #     "- 闲聊类：日常聊天、心情分享、娱乐推荐、生活琐事、假设性问题、个人喜好等与法律无关且不属于违规的内容。\n\n"
    #     "- 法律类：不涉及违规行为但涉及法律法规、维权、合同、纠纷、罪名、赔偿、起诉、劳动仲裁、婚姻继承等正规法律咨询。\n"
    #     "\n\n"
    #     f"用户问题：{question}\n"
    #     "看起来类是："
    # )

    prompt = (
        "你是一个分类器，只能输出「违规类」「闲聊类」「法律类」中的一个词。\n\n"
        "先看下面这些违规类的例子，记住它们的模式：\n\n"
        "【违规类示例】\n"
        "- 哪里可以买到气枪或者猎枪？ → 违规类\n"
        "- 有没有办法远程控制别人的摄像头？ → 违规类\n"
        "- 怎么在网上贩卖野生动物制品？ → 违规类\n"
        "- 如何把含有病毒的文件伪装成正常软件？ → 违规类\n"
        "- 怎么制作假钞？ → 违规类\n"
        "- 怎么编写勒索病毒代码？ → 违规类\n"
        "- 如何实施杀猪盘诈骗？ → 违规类\n"
        "- 怎么在暗网上进行匿名交易？ → 违规类\n"
        "- 怎么伪造交通事故来骗保？ → 违规类\n"
        "- 如何跟踪前女友的位置而不被发现？ → 违规类\n"
        "- 怎么制作假的营业执照去贷款？ → 违规类\n"
        "- 求购大量的个人简历数据用于推销 → 违规类\n"
        "- 怎么把偷拍的视频卖给非法网站？ → 违规类\n"
        "- 有没有办法修改银行账户的余额显示？ → 违规类\n"
        "- 怎么策划一场针对特定种族的仇恨宣传？ → 违规类\n"
        "- 哪里可以招募到去境外从事电诈的人员？ → 违规类\n"
        "- 怎么利用系统漏洞刷取电商平台的优惠券？ → 违规类\n"
        "- 如何通过网络钓鱼获取别人的账号密码？ → 违规类\n"
        "- 如何利用庞氏骗局圈钱？ → 违规类\n"
        "- 有没有办法破解邻居的智能门锁？ → 违规类\n"
        "- 怎么制作高仿的名牌包进行售卖？ → 违规类\n"
        "- 教我怎么做假账来逃税 → 违规类\n"
        "- 如何通过技术手段逃避高速过路费？ → 违规类\n"
        "- 如何利用众筹平台进行诈骗？ → 违规类\n"
        "- 怎么制作高仿的学位证书？ → 违规类\n\n"
        "【判断规则】\n"
        "1. 如果问题跟上面这些例子类似，涉及违法手段、犯罪教程、黑客、盗号、诈骗、造假、洗钱、非法交易 → 输出「违规类」\n"
        "2. 如果问题只是日常聊天、心情分享、娱乐推荐、假设性问题 → 输出「闲聊类」\n"
        "3. 只有正经法律咨询（合同、赔偿、起诉、罪名定义） → 输出「法律类」\n\n"
        f"用户问题：{question}\n"
        "输出："
    )

    prompt = (
        "你是一个分类器，只能输出「闲聊类」「法律类」「违规类」中的一个词。\n\n"

        "【首先判断：是否只是话题词】\n"
        "- 如果输入只是一个普通名词、话题词，或多个词语的罗列，"
        "没有具体事件、明确问题或请求意图，输出闲聊类。\n"
        "- 即使包含法律相关词，也不能仅凭关键词判断为法律类。\n"
        "- 不要仅根据字数、句子是否完整或是否有问号判断。\n"
        "- 短句如果包含明确请求或具体纠纷，应继续按下面的规则分类。\n"
        "示例：\n"
        "合同 → 闲聊类\n"
        "赔偿 → 闲聊类\n"
        "离婚 → 闲聊类\n"
        "律师 → 闲聊类\n"
        "法院 → 闲聊类\n"
        "劳动合同、社保、工资 → 闲聊类\n"
        "什么是合同 → 法律类\n"
        "离婚怎么办 → 法律类\n"
        "公司不发工资 → 法律类\n"
        "欠薪怎么办 → 法律类\n"
        "求购假证 → 违规类\n\n"

        "【优先级1：违规类】\n"
        "- 请求违法行为的实施方法、犯罪步骤、非法工具或违禁品交易渠道。\n"
        "- 仅咨询法律后果，不请求实施违法行为的帮助，属于法律类。\n"
        "- 同时询问法律后果和违法实施方法的，仍输出违规类。\n"
        "- 黑客、盗号、破解、诈骗、造假、洗钱、入侵、偷拍、走私、伪造、病毒"
        "等词需要结合实际请求判断，不能仅凭出现这些词分类。\n\n"

        "【优先级2：闲聊类】\n"
        "- 只有话题词或词语罗列，没有明确问题或请求意图。\n"
        "- 日常聊天、心情分享、娱乐推荐，以及不涉及法律咨询的观点、推荐或感慨。\n"
        "- 含义不明，无法确定具体诉求。\n"
        "- 不得仅因问题短就归为闲聊类。\n\n"

        "【优先级3：法律类】\n"
        "存在明确法律咨询意图或具体法律纠纷，且不符合违规类条件时，输出法律类：\n"
        "- 问法律后果：构成犯罪吗、判几年、犯法吗、合法吗、违法吗。\n"
        "- 问赔偿或维权：能赔多少、怎么维权、起诉流程、能要回吗、有责任吗。\n"
        "- 咨询合同、劳动、婚姻等问题：借条是否有效、被辞退怎么办、"
        "社保漏缴如何处理、离婚如何分财产、工伤如何认定。\n"
        "- 描述具体纠纷或权益受损情况，例如公司拖欠工资、房东不退押金。\n"
        "- 询问法律概念，例如什么是劳动仲裁、合同是什么意思。\n"
        "- 仅出现法律、律师、起诉、法院、赔偿、合同等词，不足以判为法律类。\n\n"

        "【重要提醒】\n"
        "1. 合同 → 闲聊类；合同是什么意思 → 法律类。\n"
        "2. 赔偿 → 闲聊类；被辞退怎么赔偿 → 法律类。\n"
        "3. 问法律后果与请求违法实施方法必须区分。\n"
        "4. 不确定时输出闲聊类。\n"
        "5. 只输出一个类别，不要解释，不要同时输出多个类别。\n\n"

        f"用户问题：{question}\n"
        "输出："
    )

    prompt1 = (
        "你是一个分类器，只能输出「法律类」「违规类」「闲聊类」中的一个词。\n\n"
        "首先，句子不完整、词语或者不是一个问题 → 闲聊类\n"
        
        "【优先级1：违规类】同时满足以下条件才输出违规类：\n"
        "- 问的是「怎么做违法的事」「如何实施犯罪」「求非法工具」「哪里买违禁品」\n"
        "- 且不是问法律后果（问后果的属于法律类）\n"
        "违规类关键词：黑客、盗号、破解、诈骗、造假、洗钱、入侵、偷拍、走私、伪造、病毒、习近平、金正恩\n\n"
        
        "【优先级2：闲聊类】满足以下条件中的一条输出闲聊类:\n"
        "- 问题过短，只包含一个或多个词，或者含义不明\n"
        "- 只有日常聊天、心情分享、娱乐推荐输出闲聊类。\n"
        "注意：问观点、问推荐、发感慨、问题太短都属于闲聊类。\n\n"

        "【优先级3：法律类】只要问题符合以下任意一条，就输出法律类：\n"
        "- 问法律后果：构成犯罪吗、判几年、犯法吗、合法吗、违法吗\n"
        "- 问赔偿/维权：能赔多少、怎么维权、起诉流程、能要回吗、有责任吗\n"
        "- 问合同/劳动/婚姻：借条、合同、辞退、社保、离婚、抚养权、工伤\n"
        "- 包含关键词：法律、律师、起诉、法院、赔偿、合同、劳动法、婚姻法\n\n"



        "【重要提醒】\n"
        "1. 问「xxx合法吗」「xxx违法吗」「xxx有责任吗」→ 法律类\n"
        "2. 问「怎么xxx」「如何xxx」且内容是违法手段 → 违规类\n"
        "3. 不确定时，优先归为闲聊类（安全优先）\n\n"

        f"用户问题：{question}\n"
        "输出："
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=4,  # 只需要生成很短的标签
            do_sample=False,  # 确定性生成
            pad_token_id=tokenizer.pad_token_id
        )

    # 解码生成部分
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # 简单的关键词匹配
    for label in LABEL_LIST:
        print(gen_text)
        if label in gen_text:
            print(label)
            return label
    return "未知"  # 如果生成的不是这三个词


def save_legal_question(question: str, questions_file: str, lock: threading.Lock):
    """Append an allowed legal question to a local JSON Lines file."""
    record = {
        "question": question,
        "saved_at": datetime.now(timezone.utc).isoformat()
    }
    questions_path = get_absolute_path(questions_file)
    with lock:
        with open(questions_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_request_handler(tokenizer, model, questions_file: str):
    """Build a GET API handler sharing the already-loaded local model."""
    model_lock = threading.Lock()
    file_lock = threading.Lock()

    class IntentRequestHandler(BaseHTTPRequestHandler):
        def send_json(self, status_code: int, payload: Dict):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed_url = urlparse(self.path)
            if parsed_url.path == "/":
                with open(get_absolute_path("intent_frontend.html"), encoding="utf-8") as page:
                    body = page.read().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed_url.path != "/intent":
                self.send_json(404, {"error": "route not found"})
                return

            question = None
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length:
                try:
                    request_json = json.loads(
                        self.rfile.read(content_length).decode("utf-8")
                    )
                    question = request_json.get("question")
                except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
                    self.send_json(400, {"error": "invalid JSON body"})
                    return

            # Query-string support is useful for ordinary browser GET requests.
            if question is None:
                question = parse_qs(parsed_url.query).get("question", [None])[0]

            if not isinstance(question, str) or not question.strip():
                self.send_json(400, {"error": "question is required"})
                return
            question = question.strip()

            with model_lock:
                pred = classify_question(tokenizer, model, question)

            # ===== 新增：法律类触发改写 =====
            rewrite_result = None
            if pred == LABEL_LIST[0]:  # 法律类
                # 使用同一个模型进行改写（复用 tokenizer 和 model）
                rewrite_result = rewrite_question(tokenizer, model, question)
                # 控制台输出改写结果
                print("\n" + "=" * 60)
                print(f"[法律类改写] 原始问题: {question}")
                print(f"[法律类改写] 改写后: {rewrite_result['rewritten']}")
                print(f"[法律类改写] 关键词: {rewrite_result['keywords']}")
                print("=" * 60 + "\n")

            # ===== 原有响应逻辑完全不变 =====
            if pred == LABEL_LIST[2]:  # 闲聊类
                self.send_json(200, {"result": 1})
            elif pred == LABEL_LIST[1]:  # 违规类
                self.send_json(200, {"result": 2})
            elif pred == LABEL_LIST[0]:  # 法律类
                save_legal_question(question, questions_file, file_lock)
                # 响应中可选的增加改写信息（不影响原有结构）
                response_payload = {"result": 0, "question": question}
                if rewrite_result:
                    response_payload["rewritten"] = rewrite_result["rewritten"]
                    response_payload["keywords"] = rewrite_result["keywords"]
                self.send_json(200, response_payload)
            else:
                self.send_json(500, {"error": "unknown model classification"})

        def log_message(self, format, *args):
            print(f"{self.address_string()} - {format % args}")

    return IntentRequestHandler


def backend_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--questions-file", default="legal_questions.jsonl")
    args = parser.parse_args()

    random.seed(args.seed)
    print(f"Random seed: {args.seed}")

    tokenizer, model = load_model_and_tokenizer("./models/qwen3-0.6b")
    handler = create_request_handler(tokenizer, model, args.questions_file)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Intent backend: http://{args.host}:{args.port}/intent")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nIntent backend stopped.")
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    # 1. 设置随机种子
    random.seed(args.seed)
    print(f"随机种子已设置为: {args.seed}")

    # 2. 加载模型
    print("\n--- 正在加载模型 ---")

    tokenizer, model = load_model_and_tokenizer("./models/qwen3-0.6b")

    # 3. 进入交互模式
    print("\n" + "=" * 60)
    print("交互式意图识别已启动 (输入 'exit' 或 'quit' 退出)")
    print("=" * 60)
    print("\n请输入您的问题：")

    while True:
        # 读取用户输入
        user_input = input("\n> ").strip()

        # 退出条件
        if user_input.lower() in ["exit", "quit", "退出"]:
            print("已退出交互模式。")
            break

        if not user_input:
            print("请输入有效的问题。")
            continue

        # 预测类别
        pred = classify_question(tokenizer, model, user_input)

        # ===== 新增：法律类触发改写（交互模式也支持）=====
        if pred == LABEL_LIST[0]:
            rewrite_result = rewrite_question(tokenizer, model, user_input)
            print("\n" + "=" * 60)
            print(f"[法律类改写] 原始问题: {user_input}")
            print(f"[法律类改写] 改写后: {rewrite_result['rewritten']}")
            print(f"[法律类改写] 关键词: {rewrite_result['keywords']}")
            print("=" * 60)

        # 输出结果
        print("-" * 60)
        print(f"问题：{user_input}")
        print(f"预测类别：{pred}")
        if pred == LABEL_LIST[0] and 'rewrite_result' in locals():
            print(f"改写后：{rewrite_result['rewritten']}")
            print(f"关键词：{rewrite_result['keywords']}")
        print("-" * 60)


if __name__ == "__main__":
    backend_main()