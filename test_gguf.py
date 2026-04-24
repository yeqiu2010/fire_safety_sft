"""
GGUF 模型推理测试脚本
用于测试导出的 fire_safety_q4_k_m.gguf 模型
依赖: pip install llama-cpp-python
"""

import json
import os
from llama_cpp import Llama

# 加载推理配置
CONFIG_PATH = "./export/inference_config.json"
MODEL_PATH = "./export/fire_safety_q4_k_m.gguf"


class GGUFInference:
    def __init__(self, model_path: str = MODEL_PATH, config_path: str = CONFIG_PATH):
        # 加载配置
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = {
                "system_prompt": (
                    "你是一个专业的消防安全助手，熟悉中国消防法律法规、"
                    "消防安全技术规范及火灾预防知识。请根据相关法规条款给出准确、"
                    "权威的回答。"
                ),
                "max_new_tokens": 256,
                "temperature": 0.3,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
            }

        # 加载 GGUF 模型
        print(f"加载模型: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=512,          # 上下文长度
            n_gpu_layers=0,     # GPU 层数 (0=纯CPU)
            verbose=False,
        )
        print("模型加载完成")

    def _build_prompt(self, question: str) -> str:
        """构建对话 prompt"""
        return f"<|im_start|>system\n{self.config['system_prompt']}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

    def answer(self, question: str) -> str:
        """单次问答"""
        prompt = self._build_prompt(question)
        response = self.llm(
            prompt,
            max_tokens=self.config.get("max_new_tokens", 256),
            temperature=self.config.get("temperature", 0.3),
            top_p=self.config.get("top_p", 0.9),
            repeat_penalty=self.config.get("repetition_penalty", 1.1),
            stop=["<|im_end|>", "\n<|im_start|>"],
        )
        return response["choices"][0]["text"].strip()

    def interactive(self):
        """交互式对话"""
        print("=" * 50)
        print("消防安全问答助手 (GGUF 版) - 输入 'exit' 退出")
        print("=" * 50)
        while True:
            question = input("\n问题: ").strip()
            if question.lower() in ("exit", "quit", "q"):
                break
            if not question:
                continue
            print("\n回答: ", end="", flush=True)
            answer = self.answer(question)
            print(answer)


# 测试问题集
TEST_QUESTIONS = [
    "民用木结构建筑与另外一个木结构建筑相距5m以内",
    "厂房未按照规定悬挂消防疏散指示灯",
    "未按照规定悬挂消防疏散指示灯都有哪些相关条例？",
    "消防安全标志 GB 13495 关于消防疏散指示灯的规定",
    "高层民用建筑防火分区最大允许建筑面积是多少？",
]


def run_tests():
    """运行测试"""
    infer = GGUFInference()

    print("\n" + "=" * 50)
    print("开始测试 GGUF 模型")
    print("=" * 50)

    results = []
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[测试 {i}] 问题: {question}")
        answer = infer.answer(question)
        print(f"[测试 {i}] 回答: {answer}")
        results.append({"question": question, "answer": answer})

    # 保存测试结果
    output_path = "./export/test_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n测试结果已保存: {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GGUF 模型测试")
    parser.add_argument("--model", default=MODEL_PATH, help="GGUF 模型路径")
    parser.add_argument("--interactive", action="store_true", help="进入交互模式")
    args = parser.parse_args()

    infer = GGUFInference(model_path=args.model)

    if args.interactive:
        infer.interactive()
    else:
        run_tests()