"""
消防问答模型推理脚本 (PC/服务器端)
支持: 单次问答 / 批量推理 / 交互式对话
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

SYSTEM_PROMPT = (
    "你是一个专业的消防安全助手，熟悉中国消防法律法规、"
    "消防安全技术规范及火灾预防知识。请根据相关法规条款给出准确、"
    "权威的回答。回答应简洁、专业，必要时注明依据的法规条款。"
)

DEFAULT_GEN_KWARGS = dict(
    max_new_tokens=256,
    do_sample=True,
    temperature=0.3,
    top_p=0.9,
    repetition_penalty=1.1,
)


class FireSafetyInference:
    def __init__(self, model_path: str, device: str = "auto"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=device,
        )
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_prompt(self, question: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.inference_mode()
    def answer(self, question: str, stream: bool = False, **gen_kwargs) -> str:
        prompt = self._build_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        kwargs = {**DEFAULT_GEN_KWARGS, **gen_kwargs}

        if stream:
            streamer = TextStreamer(self.tokenizer, skip_prompt=True)
            self.model.generate(**inputs, streamer=streamer, **kwargs)
            return ""

        output_ids = self.model.generate(**inputs, **kwargs)
        # 只取新生成部分
        new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    @torch.inference_mode()
    def batch_answer(self, questions: list[str], **gen_kwargs) -> list[str]:
        prompts = [self._build_prompt(q) for q in questions]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.model.device)

        kwargs = {**DEFAULT_GEN_KWARGS, **gen_kwargs}
        output_ids = self.model.generate(**inputs, **kwargs)

        results = []
        for i, out in enumerate(output_ids):
            new_ids = out[inputs["input_ids"].shape[-1]:]
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            results.append(text)
        return results

    def interactive(self):
        """交互式命令行对话"""
        print("=" * 50)
        print("消防安全问答助手 (输入 'exit' 退出)")
        print("=" * 50)
        while True:
            question = input("\n问题: ").strip()
            if question.lower() in ("exit", "quit", "q"):
                break
            if not question:
                continue
            print("\n回答: ", end="", flush=True)
            self.answer(question, stream=True)
            print()


# ─────────────────────────────────────────────
# 示例
# ─────────────────────────────────────────────
if __name__ == "__main__":
    MODEL_PATH = "./fire_safety_model/checkpoint-4400"  # 微调后的模型路径

    infer = FireSafetyInference(MODEL_PATH)

    # 单次问答
    test_questions = [
        "民用木结构建筑与另外一个木结构建筑相距5m以内",
        "厂房未按照规定悬挂消防疏散指示灯",
        "未按照规定悬挂消防疏散指示灯",
        "未按照规定悬挂消防疏散指示灯都有哪些相关条例？",
        "消防安全标志》GB 13495和《建筑灯光设计标准》GB 50031中关于悬挂消防疏散指示灯的规定"
    ]

    print("─── 单次问答测试 ───")
    for q in test_questions:
        print(f"\n❓ {q}")
        answer = infer.answer(q)
        print(f"✅ {answer}")

    # 交互式
    # infer.interactive()
