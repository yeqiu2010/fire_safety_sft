"""
将微调后的模型导出为适合 Android 部署的格式。
支持两种方案:
  方案A: 转换为 GGUF 格式 (通过 llama.cpp 在 Android 运行)
  方案B: 转换为 MNN 格式 (阿里云 MNN 框架 Android 部署)

推荐方案A, 因为 llama.cpp 对 Qwen 系列有较好支持。
"""

import os
import shutil
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = "./fire_safety_model/final"     # 微调后模型
OUTPUT_DIR = "./export"


# ═══════════════════════════════════════════════════════
# 方案A: 导出为 GGUF 格式 (推荐)
# 依赖: pip install llama-cpp-python; 并克隆 llama.cpp 仓库
# ═══════════════════════════════════════════════════════
def export_to_gguf(
    model_path: str = MODEL_PATH,
    output_dir: str = OUTPUT_DIR,
    quantization: str = "q4_k_m",  # 推荐量化精度: q4_k_m / q5_k_m / q8_0
    llama_cpp_path: str = "./llama.cpp",  # llama.cpp 本地路径
):
    """
    步骤:
      1. 将 HuggingFace 模型转换为 fp16 GGUF
      2. 使用 llama.cpp quantize 进行 INT4 量化
      3. 最终文件用于 Android llama.cpp JNI
    """
    os.makedirs(output_dir, exist_ok=True)
    gguf_fp16 = os.path.join(output_dir, "fire_safety_fp16.gguf")
    gguf_q4   = os.path.join(output_dir, f"fire_safety_{quantization}.gguf")

    logger.info("步骤1: HuggingFace → GGUF (fp16)")
    convert_script = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")
    cmd_convert = [
        "python", convert_script,
        model_path,
        "--outfile", gguf_fp16,
        "--outtype", "f16",
    ]
    logger.info(f"执行: {' '.join(cmd_convert)}")
    subprocess.run(cmd_convert, check=True)

    logger.info(f"步骤2: 量化为 {quantization.upper()}")
    quantize_bin = os.path.join(llama_cpp_path, "llama-quantize")
    if not os.path.exists(quantize_bin):
        quantize_bin = os.path.join(llama_cpp_path, "build", "bin", "llama-quantize")
    cmd_quantize = [
        quantize_bin,
        gguf_fp16,
        gguf_q4,
        quantization.upper(),
    ]
    logger.info(f"执行: {' '.join(cmd_quantize)}")
    subprocess.run(cmd_quantize, check=True)

    size_mb = os.path.getsize(gguf_q4) / 1024 / 1024
    logger.info(f"导出完成: {gguf_q4} ({size_mb:.1f} MB)")
    logger.info("将此文件复制到 Android assets 目录即可使用。")
    return gguf_q4


# ═══════════════════════════════════════════════════════
# 方案B: 导出为 MNN 格式
# 依赖: pip install MNN; 并安装 MNNConvert 工具
# ═══════════════════════════════════════════════════════
def export_to_mnn(
    model_path: str = MODEL_PATH,
    output_dir: str = OUTPUT_DIR,
):
    """
    步骤:
      1. 将 HuggingFace 模型导出为 ONNX
      2. 使用 MNNConvert 转换为 .mnn 文件
    """
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "fire_safety.onnx")
    mnn_path  = os.path.join(output_dir, "fire_safety.mnn")

    # 步骤1: 导出 ONNX
    logger.info("步骤1: 导出 ONNX")
    _export_onnx(model_path, onnx_path)

    # 步骤2: ONNX → MNN
    logger.info("步骤2: 转换 MNN")
    cmd = [
        "MNNConvert",
        "-f", "ONNX",
        "--modelFile", onnx_path,
        "--MNNModel", mnn_path,
        "--bizCode", "MNN",
    ]
    subprocess.run(cmd, check=True)
    logger.info(f"MNN 模型已导出: {mnn_path}")
    return mnn_path


def _export_onnx(model_path: str, onnx_path: str):
    """通过 optimum 库导出 ONNX"""
    try:
        from optimum.exporters.onnx import main_export
        main_export(
            model_name_or_path=model_path,
            output=os.path.dirname(onnx_path),
            task="text-generation-with-past",
            opset=17,
            trust_remote_code=True,
        )
        logger.info(f"ONNX 导出完成: {onnx_path}")
    except ImportError:
        logger.error("请安装: pip install optimum[exporters]")
        raise


# ═══════════════════════════════════════════════════════
# 方案C: 直接打包 tokenizer 配置 (与 GGUF 一起使用)
# ═══════════════════════════════════════════════════════
def export_tokenizer_config(
    model_path: str = MODEL_PATH,
    output_dir: str = OUTPUT_DIR,
):
    """
    将 tokenizer 文件单独导出, 供 Android 端加载。
    llama.cpp 的 GGUF 已内嵌 tokenizer, 此步骤仅供 MNN 方案使用。
    """
    import json
    from transformers import AutoTokenizer

    os.makedirs(output_dir, exist_ok=True)
    tok_dir = os.path.join(output_dir, "tokenizer")
    os.makedirs(tok_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.save_pretrained(tok_dir)
    logger.info(f"Tokenizer 已保存: {tok_dir}")

    # 额外保存消防场景的 system prompt
    config = {
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
    with open(os.path.join(output_dir, "inference_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    logger.info("推理配置已保存: inference_config.json")


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="消防模型 Android 导出工具")
    parser.add_argument("--method", choices=["gguf", "mnn"], default="gguf",
                        help="导出格式: gguf (推荐) 或 mnn")
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--quantization", default="q4_k_m",
                        help="GGUF 量化类型: q4_k_m / q5_k_m / q8_0")
    parser.add_argument("--llama_cpp_path", default="./llama.cpp")
    args = parser.parse_args()

    if args.method == "gguf":
        logger.info("使用方案A: GGUF + llama.cpp")
        export_to_gguf(
            model_path=args.model_path,
            output_dir=args.output_dir,
            quantization=args.quantization,
            llama_cpp_path=args.llama_cpp_path,
        )
    else:
        logger.info("使用方案B: MNN")
        export_to_mnn(
            model_path=args.model_path,
            output_dir=args.output_dir,
        )

    export_tokenizer_config(
        model_path=args.model_path,
        output_dir=args.output_dir,
    )

    logger.info("\n✅ 导出完成!")
    logger.info(f"   输出目录: {args.output_dir}")
    logger.info("   下一步: 将 *.gguf 文件复制到 Android 项目的 assets/ 目录")
