"""
消防问答数据集微调脚本
模型: Qwen/Qwen3.5-0.8B
数据集: sdzjoy/fire-safety-sft-dataset
框架: HuggingFace Transformers + PyTorch
"""
import os
os.environ["USE_TORCH"] = "True"

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from datasets import load_dataset, Dataset
import torch
from typing import Optional, Dict, List
from dataclasses import dataclass, field
import logging
import json



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. 配置
# ─────────────────────────────────────────────
@dataclass
class FireSafetyConfig:
    # 模型与数据
    model_name: str = "/home/ubuntu/.cache/modelscope/hub/models/Qwen/Qwen3.5-0.8B"
    dataset_name: str = "sdzjoy/fire-safety-sft-dataset"
    output_dir: str = "./fire_safety_model"

    # 训练超参数
    max_length: int = 512           # 最大序列长度 (手机端推理建议 ≤512)
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4   # 等效 batch_size = 16
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01

    # 精度与优化
    bf16: bool = True              # A100/H100 上开启; 普通 GPU 用 fp16
    fp16: bool = False
    gradient_checkpointing: bool = True   # 显存不足时开启

    # 日志与保存
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"

    # 数据划分
    test_size: float = 0.05
    seed: int = 42


CONFIG = FireSafetyConfig()


# ─────────────────────────────────────────────
# 2. Prompt 模板
# ─────────────────────────────────────────────
SYSTEM_PROMPT = (
    "你是一个专业的消防安全助手，熟悉中国消防法律法规、"
    "消防安全技术规范及火灾预防知识。请根据相关法规条款给出准确、"
    "权威的回答。回答应简洁、专业，必要时注明依据的法规条款。"
)


def build_prompt(instruction: str, output: str = "", tokenizer=None) -> str:
    """
    使用 Qwen 的 ChatML 格式构建 prompt。
    训练时 output 非空；推理时 output 为空。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": instruction},
    ]
    if output:
        # 训练: 需要完整的对话 (包含 assistant 回复)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt += output + tokenizer.eos_token
    else:
        # 推理: 只到 <|im_start|>assistant\n
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt


# ─────────────────────────────────────────────
# 3. 数据集加载与预处理
# ─────────────────────────────────────────────
def load_and_preprocess(tokenizer, config: FireSafetyConfig):
    """加载数据集并 tokenize"""
    logger.info(f"加载数据集: {config.dataset_name}")
    raw = load_dataset(config.dataset_name)

    # 探测字段名 (不同数据集字段名不同)
    sample = raw["train"][0]
    logger.info(f"数据集字段: {list(sample.keys())}")
    logger.info(
        f"样本示例: {json.dumps(sample, ensure_ascii=False, indent=2)[:500]}")

    # ── 根据实际字段名调整以下映射 ──
    # 常见字段组合:
    #   instruction / output
    #   question / answer
    #   input / output (有时含 instruction)
    def get_instruction_output(example: Dict) -> tuple:
        """统一提取 instruction 和 output"""
        # 优先尝试 instruction/output
        if "instruction" in example and "output" in example:
            instruction = example["instruction"]
            if example.get("input"):
                instruction = instruction + "\n" + example["input"]
            return instruction, example["output"]
        # question/answer
        if "question" in example:
            return example["question"], example.get("answer", "")
        # messages 格式 (ShareGPT)
        if "messages" in example:
            msgs = example["messages"]
            user_msg = next((m["content"]
                            for m in msgs if m["role"] == "user"), "")
            asst_msg = next((m["content"]
                            for m in msgs if m["role"] == "assistant"), "")
            return user_msg, asst_msg
        # 兜底
        keys = list(example.keys())
        return example[keys[0]], example[keys[1]] if len(keys) > 1 else ""

    def tokenize_fn(examples: Dict) -> Dict:
        input_ids_list, attention_mask_list, labels_list = [], [], []

        batch_size = len(examples[list(examples.keys())[0]])
        for i in range(batch_size):
            example = {k: examples[k][i] for k in examples}
            instruction, output = get_instruction_output(example)
            if not instruction or not output:
                continue

            full_prompt = build_prompt(instruction, output, tokenizer)

            # Tokenize 完整 prompt
            tokenized = tokenizer(
                full_prompt,
                truncation=True,
                max_length=config.max_length,
                padding=False,
                return_tensors=None,
            )
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

            # 计算 instruction 部分长度 (用于 label masking)
            prompt_only = build_prompt(instruction, "", tokenizer)
            prompt_ids = tokenizer(
                prompt_only,
                truncation=True,
                max_length=config.max_length,
                padding=False,
                return_tensors=None,
            )["input_ids"]
            prompt_len = len(prompt_ids)

            # Labels: instruction 部分用 -100 (不计算 loss)
            labels = [-100] * prompt_len + input_ids[prompt_len:]

            # 截断到 max_length
            if len(labels) > config.max_length:
                labels = labels[:config.max_length]

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        return {
            "input_ids":      input_ids_list,
            "attention_mask": attention_mask_list,
            "labels":         labels_list,
        }

    # 划分训练/验证集
    dataset = raw["train"].train_test_split(
        test_size=config.test_size, seed=config.seed
    )
    train_ds = dataset["train"]
    eval_ds = dataset["test"]

    logger.info(f"训练集大小: {len(train_ds)}, 验证集大小: {len(eval_ds)}")

    # Tokenize
    train_ds = train_ds.map(
        tokenize_fn,
        batched=True,
        batch_size=500,
        remove_columns=train_ds.column_names,
        desc="Tokenizing train set",
    )
    eval_ds = eval_ds.map(
        tokenize_fn,
        batched=True,
        batch_size=500,
        remove_columns=eval_ds.column_names,
        desc="Tokenizing eval set",
    )

    # 过滤空样本
    train_ds = train_ds.filter(lambda x: len(x["input_ids"]) > 0)
    eval_ds = eval_ds.filter(lambda x: len(x["input_ids"]) > 0)

    logger.info(f"过滤后 — 训练: {len(train_ds)}, 验证: {len(eval_ds)}")
    return train_ds, eval_ds


# ─────────────────────────────────────────────
# 4. 模型加载
# ─────────────────────────────────────────────
def load_model_and_tokenizer(config: FireSafetyConfig):
    logger.info(f"加载模型: {config.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right",   # Causal LM 用右填充
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        # 根据配置动态选择:
        torch_dtype=(
            torch.bfloat16 if config.bf16
            else torch.float16 if config.fp16
            else torch.float32
        ),
        device_map="auto",      # 自动分配 GPU/CPU
    )

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()   # gradient_checkpointing 需要

    logger.info(
        f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    return model, tokenizer


# ─────────────────────────────────────────────
# 5. 训练
# ─────────────────────────────────────────────
def train(config: FireSafetyConfig = CONFIG):
    # 加载模型和 tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # 加载数据集
    train_ds, eval_ds = load_and_preprocess(tokenizer, config)

    # Data collator (动态 padding)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        weight_decay=config.weight_decay,
        fp16=config.fp16,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=False,
        report_to="none",            # 关闭 wandb; 需要时改为 "wandb"
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        seed=config.seed,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    logger.info("开始训练...")
    trainer.train()

    # 保存最终模型
    logger.info(f"保存模型到: {config.output_dir}/final")
    trainer.save_model(f"{config.output_dir}/final")
    tokenizer.save_pretrained(f"{config.output_dir}/final")

    logger.info("训练完成!")
    return trainer


# ─────────────────────────────────────────────
# 6. 入口
# ─────────────────────────────────────────────
if __name__ == "__main__":
    train()
