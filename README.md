# 消防问答模型微调 & Android 部署

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 微调训练
```bash
python train.py
# 模型保存至: ./fire_safety_model/final/
```

### 3. PC 端推理测试
```bash
python inference.py
```

### 4. 导出为 GGUF (Android 部署)
```bash
# 先克隆并编译 llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
cd ../..

# 导出模型
python export_for_android.py --method gguf --llama_cpp_path ./llama.cpp
# 输出: ./export/fire_safety_q4_k_m.gguf (~350MB)
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
| 基础模型 | Qwen2.5-0.5B-Instruct (Qwen3.5-0.8B 可用后替换) |
| 训练数据 | sdzjoy/fire-safety-sft-dataset |
| 训练方式 | Full fine-tuning (SFT) |
| 上下文长度 | 512 tokens |
| 量化格式 | GGUF Q4_K_M (~350MB) |
| Android 框架 | llama.cpp JNI |

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

**Q: Android 模型加载慢?**
A: 首次运行需从 assets 解压到 files 目录, 后续直接加载。也可预置到 SD 卡路径。

**Q: 想用 LoRA 减少训练参数?**
A: 在 `train.py` 中引入 peft 库并用 `get_peft_model()` 包装模型, 只微调约 1% 参数。
