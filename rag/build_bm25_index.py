"""
构建 BM25 关键词索引
用于多路召回：向量检索 + 关键词检索
"""

import pickle
import json
import re
from pathlib import Path
from typing import List, Dict

import jieba
from rank_bm25 import BM25Okapi

# 添加消防领域自定义词汇
CUSTOM_WORDS = [
    "消防车道", "消防车登高操作场地", "消防救援口", "防火门", "防火窗",
    "防火卷帘", "消火栓", "喷淋", "喷头", "报警阀", "应急照明",
    "疏散指示", "防火封堵", "耐火极限", "燃烧性能", "防火墙",
    "防火隔墙", "楼梯间", "前室", "避难层", "储瓶间",
    "气体灭火", "自动喷水灭火", "消防给水", "防烟排烟",
    "火灾自动报警", "消防电梯", "消防控制室",
    # 规范名称关键词
    "建筑防火通用规范", "建筑设计防火规范", "消防给水及消火栓系统技术规范",
    "自动喷水灭火系统设计规范", "建筑内部装修设计防火规范",
    "防火卷帘防火门防火窗施工及验收规范", "气体灭火系统设计规范",
    "GB55037", "GB50016", "GB50974", "GB50084", "GB50222", "GB50877", "GB50370",
]

# 注册自定义词汇
for word in CUSTOM_WORDS:
    jieba.add_word(word)


def load_chunks(vectordb_dir: str) -> List[Dict]:
    """加载向量库中的 chunks"""
    chunks_path = Path(vectordb_dir) / "chunks.pkl"
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    print(f"加载 {len(chunks)} 个 chunks")
    return chunks


def preprocess_text(text: str) -> str:
    """预处理文本"""
    # 移除表格标签
    text = re.sub(r'<table>|</table>|<tr>|</tr>|<td>|</td>', ' ', text)
    # 移除特殊字符
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
    # 合并空格
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    """使用 jieba 分词"""
    processed = preprocess_text(text)
    tokens = jieba.lcut(processed)
    # 过滤停用词和单字符
    tokens = [t for t in tokens if len(t) > 1 and t not in ['的', '了', '是', '在', '和', '与', '等', '应', '可', '宜']]
    return tokens


def build_bm25_index(chunks: List[Dict]) -> tuple:
    """构建 BM25 索引"""
    print("正在构建 BM25 索引...")

    # 对每个 chunk 进行分词
    corpus = []
    for i, chunk in enumerate(chunks):
        text = chunk.get('text', '')
        # 增加条款编号作为关键词
        article_id = chunk.get('article_id', '')
        source = chunk.get('source', '')

        # 构建增强文本
        enhanced_text = f"{article_id} {source} {text}"

        tokens = tokenize(enhanced_text)
        corpus.append(tokens)

        if (i + 1) % 1000 == 0:
            print(f"  处理进度: {i + 1}/{len(chunks)}")

    # 创建 BM25 索引
    bm25 = BM25Okapi(corpus)
    print(f"BM25 索引构建完成: {len(corpus)} 个文档")

    return bm25, corpus


def save_bm25_index(bm25: BM25Okapi, corpus: List[List[str]], chunks: List[Dict], output_dir: str):
    """保存 BM25 索引"""
    output_path = Path(output_dir)

    # 保存 BM25 索引数据（corpus 和 chunks 索引映射）
    bm25_data = {
        'corpus': corpus,  # 分词后的文档列表
        'doc_lengths': [len(doc) for doc in corpus],
        'chunk_count': len(chunks),
    }

    with open(output_path / "bm25_index.pkl", "wb") as f:
        pickle.dump(bm25_data, f)

    print(f"BM25 索引已保存到: {output_path / 'bm25_index.pkl'}")


def main():
    vectordb_dir = "rag/vectordb"

    # 加载 chunks
    chunks = load_chunks(vectordb_dir)

    # 构建 BM25 索引
    bm25, corpus = build_bm25_index(chunks)

    # 保存索引
    save_bm25_index(bm25, corpus, chunks, vectordb_dir)

    # 测试检索效果
    print("\n=== 测试 BM25 检索 ===")
    test_queries = [
        "消防车道 未设置 标识",
        "储瓶间 应急照明",
        "防火门 间隙 9mm",
    ]

    for query in test_queries:
        tokens = tokenize(query)
        scores = bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:5]

        print(f"\n查询: {query}")
        for idx in top_indices:
            chunk = chunks[idx]
            print(f"  [{idx}] 第{chunk.get('article_id', '?')}条 {chunk.get('source', '')[:50]}")
            print(f"      分数: {scores[idx]:.4f}")


if __name__ == "__main__":
    main()