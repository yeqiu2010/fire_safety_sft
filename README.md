# 消防问答模型微调 & Android 部署

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 微调训练
```bash
python train.py
# 模型保存至: ./fire_safety_model/checkpoint-4400/
```

### 3. PC 端推理测试
```bash
# HuggingFace 模型推理
python inference.py

# GGUF 模型测试 (Android 部署前验证)
python test_gguf.py
# 交互模式
python test_gguf.py --interactive
```

### 4. 导出为 GGUF (Android 部署)
```bash
# 自动克隆并编译 llama.cpp，导出 GGUF 模型
python export_for_android.py --method gguf --quantization q4_k_m
# 输出: ./export/fire_safety_q4_k_m.gguf (~530MB)
```

### 5. Android 集成
```
1. 将 fire_safety_q4_k_m.gguf 复制到 Android 项目 app/src/main/assets/
2. 集成 llama.cpp Android JNI (.aar 或源码编译 libllama.so)
3. 将 FireSafetyActivity.kt 加入项目
4. 配置 build.gradle:
   - minSdk 26 (Android 8.0+)
   - abiFilters "arm64-v8a"  (现代手机均为 arm64)
```

## 模型说明

| 项目 | 说明 |
|------|------|
| 基础模型 | Qwen3.5-0.8B |
| 训练数据 | sdzjoy/fire-safety-sft-dataset |
| 训练方式 | Full fine-tuning (SFT) |
| 上下文长度 | 512 tokens |
| 量化格式 | GGUF Q4_K_M (~530MB) |
| Android 框架 | llama.cpp JNI |

## 项目结构

```
fire_safety_sft/
├── train.py                    # 模型微调脚本
├── inference.py                # PC端 HuggingFace 推理
├── test_gguf.py                # GGUF 模型测试脚本
├── export_for_android.py       # Android端导出工具
├── llama.cpp                   # llama.cpp 源码 (自动克隆)
├── fire_safety_model/          # 微调后模型
│   └── checkpoint-4400/        # 推荐 checkpoint
├── export/                     # 导出输出目录
│   ├── fire_safety_q4_k_m.gguf # 量化模型
│   ├── inference_config.json   # 推理配置
│   └── tokenizer/              # tokenizer 文件
└── rag/                        # RAG 检索增强模块
```

## 显存需求

| 阶段 | 需求 |
|------|------|
| 训练 (fp16) | ≥8GB VRAM |
| 训练 (开启 gradient_checkpointing) | ~6GB VRAM |
| PC 推理 (fp16) | ~2GB VRAM |
| Android 推理 (Q4_K_M) | ~500MB RAM |

## 常见问题

**Q: 数据集字段名不对?**
A: 运行 `train.py` 后查看日志中 `数据集字段:` 一行, 修改 `get_instruction_output()` 函数。

**Q: 显存不足?**
A: 在 `FireSafetyConfig` 中减小 `per_device_train_batch_size`, 增大 `gradient_accumulation_steps`。

**Q: 导出 GGUF 时报错 "cmake not found"?**
A: 需安装编译工具: `sudo apt install cmake build-essential`

**Q: Android 模型加载慢?**
A: 首次运行需从 assets 解压到 files 目录, 后续直接加载。也可预置到 SD 卡路径。

**Q: 想用 LoRA 减少训练参数?**
A: 在 `train.py` 中引入 peft 库并用 `get_peft_model()` 包装模型, 只微调约 1% 参数。
