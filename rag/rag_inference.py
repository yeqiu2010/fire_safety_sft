"""
消防问答 RAG 推理引擎
流程: 用户问题 → 向量检索召回法规原文 → 注入 Prompt → 微调模型生成回答
支持多路召回：向量检索 + BM25关键词检索
"""

import re
import json
import logging
import pickle
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, TextStreamer

# BM25 相关
import jieba
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 分词配置（用于 BM25）
# ─────────────────────────────────────────────
# 添加消防领域自定义词汇
CUSTOM_WORDS = [
    "消防车道", "消防车登高操作场地", "消防救援口", "防火门", "防火窗",
    "防火卷帘", "消火栓", "喷淋", "喷头", "报警阀", "应急照明",
    "疏散指示", "防火封堵", "耐火极限", "燃烧性能", "防火墙",
    "防火隔墙", "楼梯间", "前室", "避难层", "储瓶间",
    "气体灭火", "自动喷水灭火", "消防给水", "防烟排烟",
    "火灾自动报警", "消防电梯", "消防控制室",
]

for word in CUSTOM_WORDS:
    jieba.add_word(word)


# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
# 获取脚本所在目录，用于构建绝对路径
_SCRIPT_DIR = Path(__file__).parent.resolve()


@dataclass
class RAGConfig:
    # 微调后的生成模型路径
    gen_model_path: str = str(_SCRIPT_DIR.parent / "fire_safety_model" / "checkpoint-4400")
    # gen_model_path: str = "/home/ubuntu/.cache/modelscope/hub/models/Qwen/Qwen3.5-0.8B"
    # gen_model_path: str = "/home/ubuntu/.cache/modelscope/hub/models/Qwen/Qwen3.5-2B"
    # gen_model_path: str = "/home/ubuntu/.cache/modelscope/hub/models/google/gemma-4-E2B-it"

    # 向量库路径
    vectordb_dir: str = str(_SCRIPT_DIR / "vectordb")

    # BM25 索引路径
    bm25_index_path: str = str(_SCRIPT_DIR / "vectordb" / "bm25_index.pkl")

    # Embedding 模型 (与构建时一致)
    embed_model_path: str = "/home/ubuntu/.cache/modelscope/hub/models/BAAI/bge-small-zh-v1.5"

    # 检索参数
    top_k: int = 20              # 增加召回数量（多路融合需要更多候选）
    min_score: float = 0.2       # 放宽阈值
    max_context_len: int = 3000  # 上下文长度

    # 多路召回参数
    use_hybrid: bool = True      # 是否使用多路召回
    bm25_top_k: int = 20         # BM25 召回数量
    rrf_k: int = 60              # RRF 融合参数

    # 生成参数
    max_new_tokens: int = 512    # 增加生成长度
    temperature: float = 0.3     # 提高温度避免停滞
    top_p: float = 0.9
    repetition_penalty: float = 1.1

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────
# RAG Prompt 模板 (简化版，适合小模型)
# ─────────────────────────────────────────────
SYSTEM_PROMPT = (
    "你是一个专业的消防安全法规查询助手。你的唯一任务是依据提供的法规原文回答问题。\n"
    "【核心规则】\n"
    "1. 只使用【参考法规】中明确列出的条款编号，禁止创造或猜测任何条款号。\n"
    "2. 若参考法规中没有用户询问的条款，必须回答'未在提供的法规中找到相关条款'。\n"
    "3. 所有技术术语（如净宽度、净高度等）的含义必须严格依据法规原文定义。\n"
    "4. 回答格式：先说明依据的条款编号和法规名称，再给出具体内容。\n"
    "5. 禁止编造、推测或添加任何法规原文中没有的信息。"
)

RAG_PROMPT_TEMPLATE = """\
【参考法规】（以下是你唯一可以引用的内容）
{context}

【用户问题】
{question}

【回答要求】
- 必须从上述参考法规中找到对应的条款编号
- 禁止创造不存在的条款号（如6.1.2等），只能使用参考法规中明确出现的编号
- 如果参考法规中没有相关条款，必须明确说明'未找到'
- 回答时先标注来源：【法规名称】第X.X.X条

请根据以上要求回答："""

NO_CONTEXT_PROMPT_TEMPLATE = """\
【用户问题】
{question}

注意：未能从法规库中检索到直接相关内容。
请回答：'根据现有法规库，未找到与该问题直接相关的条款规定。建议查阅完整的消防法规文件或咨询专业消防工程师。'
禁止编造任何法规条款或内容。"""


# ─────────────────────────────────────────────
# Embedding 模型 (复用 build_vectordb.py 的逻辑)
# ─────────────────────────────────────────────
class EmbeddingModel:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32)
        self.model.to(device)
        self.model.eval()
        self.device = device

    @torch.inference_mode()
    def encode_query(self, query: str) -> np.ndarray:
        prefixed = f"为这个句子生成表示以用于检索相关文章：{query}"
        encoded = self.tokenizer(
            prefixed, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        output = self.model(**encoded)
        emb = output.last_hidden_state[:, 0, :]
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy().astype(np.float32)


# ─────────────────────────────────────────────
# 核心 RAG 引擎
# ─────────────────────────────────────────────
class FireSafetyRAG:
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self._load_components()

    def _load_components(self):
        cfg = self.config

        # 1. 加载向量库
        logger.info("加载向量库...")
        self.index = faiss.read_index(f"{cfg.vectordb_dir}/index.faiss")
        with open(f"{cfg.vectordb_dir}/chunks.pkl", "rb") as f:
            self.chunks: List[Dict] = pickle.load(f)
        logger.info(f"  向量库: {self.index.ntotal} 条法规片段")

        # 2. 加载 BM25 索引（如果启用多路召回）
        if cfg.use_hybrid:
            logger.info("加载 BM25 索引...")
            try:
                with open(cfg.bm25_index_path, "rb") as f:
                    bm25_data = pickle.load(f)
                self.bm25_corpus = bm25_data['corpus']
                self.bm25 = BM25Okapi(self.bm25_corpus)
                logger.info(f"  BM25 索引: {len(self.bm25_corpus)} 个文档")
            except FileNotFoundError:
                logger.warning("BM25 索引文件不存在，仅使用向量检索")
                self.bm25 = None

        # 3. 加载 Embedding 模型
        logger.info("加载 Embedding 模型...")
        self.embed_model = EmbeddingModel(cfg.embed_model_path, cfg.device)

        # 4. 加载生成模型 (微调后)
        logger.info(f"加载生成模型: {cfg.gen_model_path}")
        self.gen_tokenizer = AutoTokenizer.from_pretrained(
            cfg.gen_model_path, trust_remote_code=True
        )
        self.gen_model = AutoModelForCausalLM.from_pretrained(
            cfg.gen_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.gen_model.eval()
        logger.info("所有组件加载完成")

    # ──────────────────────────────────────────
    # BM25 检索
    # ──────────────────────────────────────────
    def _tokenize_query(self, query: str) -> List[str]:
        """使用 jieba 分词"""
        # 移除特殊字符
        query = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', query)
        tokens = jieba.lcut(query)
        # 过滤停用词和单字符
        tokens = [t for t in tokens if len(t) > 1 and t not in ['的', '了', '是', '在', '和', '与', '等', '应', '可', '宜']]
        return tokens

    def bm25_search(self, query: str, top_k: int = None) -> List[Tuple[Dict, float]]:
        """BM25 关键词检索"""
        if self.bm25 is None:
            return []

        top_k = top_k or self.config.bm25_top_k
        tokens = self._tokenize_query(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]

        results = []
        for idx in top_indices:
            score = scores[idx]
            if score > 0:  # BM25 分数可能为负
                results.append((self.chunks[idx], float(score)))

        return results

    # ──────────────────────────────────────────
    # RRF 融合
    # ──────────────────────────────────────────
    def rrf_fusion(self, vector_results: List[Tuple[Dict, float]],
                   bm25_results: List[Tuple[Dict, float]],
                   k: int = 60) -> List[Tuple[Dict, float]]:
        """
        Reciprocal Rank Fusion 融合算法
        score = sum(1 / (k + rank)) for each result list
        """
        scores = {}

        # 向量检索结果
        for rank, (chunk, _) in enumerate(vector_results):
            chunk_id = id(chunk)  # 使用对象 id 作为唯一标识
            if chunk_id not in scores:
                scores[chunk_id] = {'chunk': chunk, 'score': 0}
            scores[chunk_id]['score'] += 1 / (k + rank + 1)

        # BM25 检索结果
        for rank, (chunk, _) in enumerate(bm25_results):
            chunk_id = id(chunk)
            if chunk_id not in scores:
                scores[chunk_id] = {'chunk': chunk, 'score': 0}
            scores[chunk_id]['score'] += 1 / (k + rank + 1)

        # 按融合分数排序
        sorted_items = sorted(scores.items(), key=lambda x: -x[1]['score'])
        return [(item[1]['chunk'], item[1]['score']) for item in sorted_items]

    # ──────────────────────────────────────────
    # 向量检索
    # ──────────────────────────────────────────
    def vector_search(self, query: str, top_k: int = None) -> List[Tuple[Dict, float]]:
        """向量检索"""
        top_k = top_k or self.config.top_k
        query_vec = self.embed_model.encode_query(query)
        query_vec = query_vec.reshape(1, -1)

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if score < self.config.min_score:
                continue
            results.append((self.chunks[idx], float(score)))

        return results

    # ──────────────────────────────────────────
    # 检索（多路召回）
    # ──────────────────────────────────────────
    def retrieve(self, query: str) -> List[Tuple[Dict, float]]:
        """检索最相关的 Top-K 法规片段, 返回 [(chunk, score)]"""
        if self.config.use_hybrid and self.bm25 is not None:
            # 多路召回
            vector_results = self.vector_search(query, self.config.top_k)
            bm25_results = self.bm25_search(query, self.config.bm25_top_k)

            # RRF 融合
            merged_results = self.rrf_fusion(vector_results, bm25_results, self.config.rrf_k)

            # 返回 top_k 个结果
            return merged_results[:self.config.top_k]
        else:
            # 仅向量检索
            return self.vector_search(query)

    # ──────────────────────────────────────────
    # Prompt 构建
    # ──────────────────────────────────────────
    def _build_rag_prompt(self, question: str, retrieved: List[Tuple[Dict, float]]) -> Tuple[str, bool]:
        """
        构建注入了法规原文的 prompt。
        返回 (prompt_text, has_context)
        """
        if not retrieved:
            user_content = NO_CONTEXT_PROMPT_TEMPLATE.format(question=question)
            has_context = False
        else:
            # 拼接检索到的法规片段，明确标注条款编号
            context_parts = []
            total_len = 0
            for chunk, score in retrieved:
                # 构建带条款编号的引用格式
                source_info = chunk['source']
                article_id = chunk.get('article_id', '')
                section_title = chunk.get('section_title', '')
                is_table = chunk.get('is_table', False)

                # 清晰标注条款来源
                if article_id:
                    header = f"【{source_info} 第{article_id}条】"
                elif is_table:
                    header = f"【{source_info} - 表格内容】"
                else:
                    header = f"【{source_info} - {section_title}】"

                snippet = f"{header}\n{chunk['text']}"
                if total_len + len(snippet) > self.config.max_context_len:
                    # 截断避免超出上下文长度
                    remaining = self.config.max_context_len - total_len
                    if remaining > 100:
                        snippet = snippet[:remaining] + "...(内容截断)"
                        context_parts.append(snippet)
                    break
                context_parts.append(snippet)
                total_len += len(snippet)

            context = "\n\n".join(context_parts)
            user_content = RAG_PROMPT_TEMPLATE.format(
                context=context, question=question
            )
            has_context = True

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]
        prompt = self.gen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt, has_context

    # ──────────────────────────────────────────
    # 条款号验证
    # ──────────────────────────────────────────
    def _validate_article_ids(self, answer: str, retrieved: List[Tuple[Dict, float]]) -> Dict:
        """
        验证回答中提到的条款号是否在检索结果中真实存在。
        返回: {"valid_ids": [], "invalid_ids": [], "warning": str}
        """
        # 从检索结果中提取所有真实的条款号
        valid_ids = set()
        for chunk, score in retrieved:
            if chunk.get('article_id'):
                valid_ids.add(chunk['article_id'])

        # 从回答中提取条款号（匹配 X.X.X 格式）
        import re
        mentioned_ids = re.findall(r'\d+\.\d+\.\d+', answer)

        invalid_ids = [id for id in mentioned_ids if id not in valid_ids]

        result = {
            "valid_ids": list(valid_ids),
            "invalid_ids": invalid_ids,
        }

        if invalid_ids:
            result["warning"] = f"⚠️ 回答中包含检索结果未出现的条款号: {invalid_ids}"

        return result

    # ──────────────────────────────────────────
    # 生成
    # ──────────────────────────────────────────
    @torch.inference_mode()
    def answer(
        self,
        question: str,
        stream: bool = False,
        return_sources: bool = True,
    ) -> Dict:
        """
        完整的 RAG 问答。
        返回:
          {
            "answer": str,
            "sources": [{"source": ..., "page": ..., "score": ..., "text": ...}],
            "has_context": bool,
          }
        """
        # 1. 检索
        retrieved = self.retrieve(question)

        # print("检索到条款片段：", retrieved)

        # 2. 构建 Prompt
        prompt, has_context = self._build_rag_prompt(question, retrieved)

        # 3. Tokenize (不截断，让模型处理完整prompt)
        inputs = self.gen_tokenizer(
            prompt, return_tensors="pt", truncation=False
        ).to(self.gen_model.device)

        # 检查prompt长度，如果超长则警告并截断
        prompt_len = inputs["input_ids"].shape[-1]
        if prompt_len > 4096:
            logger.warning(f"Prompt过长({prompt_len} tokens)，截断至4096")
            inputs = self.gen_tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=4096
            ).to(self.gen_model.device)

        # 4. 生成
        gen_kwargs = dict(
            max_new_tokens=self.config.max_new_tokens,
            do_sample=True,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            repetition_penalty=self.config.repetition_penalty,
        )

        if stream:
            streamer = TextStreamer(self.gen_tokenizer, skip_prompt=True)
            self.gen_model.generate(**inputs, streamer=streamer, **gen_kwargs)
            answer_text = ""   # 流式模式下直接打印, 不返回
        else:
            output_ids = self.gen_model.generate(**inputs, **gen_kwargs)
            new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
            answer_text = self.gen_tokenizer.decode(
                new_ids, skip_special_tokens=True
            ).strip()

        # 5. 整理来源
        sources = []
        if return_sources:
            for chunk, score in retrieved:
                sources.append({
                    "source": chunk["source"],
                    "page":   chunk["page"],
                    "score":  round(score, 4),
                    "article_id": chunk.get("article_id", ""),
                    "text":   chunk["text"][:150] + "...",
                })

        # 6. 验证条款号（防止幻觉）
        validation = self._validate_article_ids(answer_text, retrieved)

        return {
            "answer":      answer_text,
            "sources":     sources,
            "has_context": has_context,
            "validation":  validation,
        }

    def interactive(self):
        """交互式命令行"""
        print("=" * 60)
        print("消防安全 RAG 问答系统 (输入 'exit' 退出, 's:问题' 跳过检索)")
        print("=" * 60)
        while True:
            query = input("\n❓ 问题: ").strip()
            if query.lower() in ("exit", "q"):
                break
            if not query:
                continue

            print("\n🔍 检索中...")
            result = self.answer(query, stream=False)

            print(f"\n✅ 回答:\n{result['answer']}")

            # 显示条款号验证警告
            if result.get("validation") and result["validation"].get("warning"):
                print(f"\n{result['validation']['warning']}")

            if result["sources"]:
                print("\n📚 参考来源:")
                for i, src in enumerate(result["sources"], 1):
                    article_id = src.get("article_id", "")
                    if article_id:
                        print(f"  [{i}] {src['source']} 第{article_id}条 "
                              f"(相似度: {src['score']:.3f})")
                    else:
                        print(f"  [{i}] {src['source']} (相似度: {src['score']:.3f})")
                    print(f"      {src['text']}")
            else:
                print("\n⚠️  未检索到相关法规，回答基于模型训练知识")


# ─────────────────────────────────────────────
# 测试
# ─────────────────────────────────────────────
if __name__ == "__main__":
    rag = FireSafetyRAG()

    # 1、不符合《建筑防火通用规范》GB55037-2022第12.0.2条规定
    # 2、不符合《建筑设计防火规范》GB50016-2014（2018年版）第7.2.2条规定
    # 3、不符合《建筑防火通用规范》GB55037-2022第2.2.3条规定
    # 4、不符合《建筑设计防火规范》GB50016-2014（2018年版）第6.1.5条规定
    # 5、不符合《建筑防火通用规范》GB55037-2022第7.1.5条规定
    # 6、不符合《建筑设计防火规范》GB50016-2014（2018年版）第6.2.9、6.3.5条规定。
    # 7、不符合《消防给水及消火栓系统技术规范》GB50974-2014第12.3.10条规定
    # 8、不符合《建筑内部装修设计防火规范》GB50222-2017第4.0.2条规定
    # 9、不符合《自动喷水灭火系统设计规范》GB50084-2017第6.1.2条规定
    # 10、不符合《建筑防火通用规范》GB55037-2022第2.2.4条规定



    test_questions = [
        "1、该工程消防车道和消防登高操作场地未设置明显的标识和不得占用、阻塞的警示标志",
        "2、该工程T5#楼消防登高操作场地净宽度不足10m",
        "3、该工程消防救援口未设置可在室内和室外识别的永久性明显标志",
        "4、该工程S1#楼内使用功能、平面布置、内部装修、消防设施等与设计文件不符；S1#楼与车库区域防火墙上存在洞口（防火门上洞口及其他洞口）",
        "5、该工程T5#楼三十二层等部分疏散通道净高度不足2.1m",
        "6、该工程部分管道、桥架、风管等水平穿墙处未封堵严密",
        "7、该工程T3#楼首层消火栓箱箱门开启不足120°",
        "8、该工程部分消火栓箱门未设置明显颜色标志或发光标志",
        "9、该工程部分厨房未设置93℃喷头",
        "10、该工程T1#楼地下室楼梯间常闭式应急排烟排热窗未设置手动和联动开启功能，T1#楼楼梯间顶部顶常闭式应急排烟排热窗未设置手动开启功能"
    ]

    print("\n─── RAG 问答测试 ───\n")
    for q in test_questions:
        print(f"❓ {q}")
        result = rag.answer(q)
        print(f"✅ {result['answer']}")
        if result["sources"]:
            print(f"📚 来源: {result['sources'][0]['source']}")
        print()

    # 交互模式
    # rag.interactive()