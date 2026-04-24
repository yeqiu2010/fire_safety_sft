"""
离线构建消防法规向量数据库 (USearch 版本)
支持: PDF / TXT / DOCX / MD 格式的法规文件
向量模型: BAAI/bge-small-zh-v1.5 (中文效果最好)
向量库: USearch (本地文件, 无需服务器, 更高效)
"""

import os
import json
import logging
import pickle
import re
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from usearch.index import Index

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
class VectorDBConfig:
    # 法规文档目录 (放置所有 .pdf/.txt/.docx/.md 文件)
    docs_dir: str = "./regulations"

    # 向量模型 (bge-small-zh-v1.5)
    embed_model: str = "/home/ubuntu/.cache/modelscope/hub/models/BAAI/bge-small-zh-v1.5"

    # 输出目录 (独立存放，不和 faiss 放在一起)
    output_dir: str = "./vectordb_usearch"

    # 分块参数 (法律条款优化)
    chunk_size: int = 512        # 法律条款通常较长，增加块大小
    chunk_overlap: int = 64      # 相邻块重叠，保留上下文
    min_chunk_len: int = 50      # 提高最小长度，避免碎片

    # 法律条款专用参数
    preserve_articles: bool = True   # 条款完整性优先模式
    max_sub_articles: int = 10       # 单条款最大子条款数

    batch_size: int = 32         # Embedding 批次大小
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CONFIG = VectorDBConfig()


# ─────────────────────────────────────────────
# 1. 文档加载
# ─────────────────────────────────────────────
def load_documents(docs_dir: str) -> List[Dict]:
    """加载目录中所有支持格式的文档, 返回 [{text, source, page}]"""
    docs = []
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        docs_path.mkdir(parents=True)
        logger.warning(f"已创建目录 {docs_dir}, 请将法规文件放入其中")
        return docs

    for fpath in docs_path.rglob("*"):
        suffix = fpath.suffix.lower()
        try:
            if suffix in [".txt", ".md"]:
                chunks = _load_txt(fpath)
            elif suffix == ".pdf":
                chunks = _load_pdf(fpath)
            elif suffix in (".docx", ".doc"):
                chunks = _load_docx(fpath)
            else:
                continue
            docs.extend(chunks)
            logger.info(f"  加载 {fpath.name}: {len(chunks)} 段")
        except Exception as e:
            logger.warning(f"  跳过 {fpath.name}: {e}")

    logger.info(f"共加载文档段落: {len(docs)}")
    return docs


def _load_txt(fpath: Path) -> List[Dict]:
    text = fpath.read_text(encoding="utf-8", errors="ignore")
    return [{"text": text, "source": fpath.name, "page": 1}]


def _load_pdf(fpath: Path) -> List[Dict]:
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("请安装: pip install pdfplumber")

    chunks = []
    with pdfplumber.open(fpath) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                chunks.append({"text": text, "source": fpath.name, "page": i + 1})
    return chunks


def _load_docx(fpath: Path) -> List[Dict]:
    try:
        from docx import Document
    except ImportError:
        raise ImportError("请安装: pip install python-docx")

    doc = Document(fpath)
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [{"text": text, "source": fpath.name, "page": 1}]


# ─────────────────────────────────────────────
# 2. 文本分块 (法律条款专用分割器)
# ─────────────────────────────────────────────
class RegulationChunker:
    """
    法律条款专用分割器
    核心原则：条款完整性优先，保留层级元数据
    """

    # 章节标题正则：匹配 # X.X 章节名称 或 # X 章节名称
    SECTION_PATTERN = re.compile(r'^#\s+(\d+(?:\.\d+)?)\s+([^\n【]+)', re.MULTILINE)

    # 条款编号正则：匹配 X.X.X 格式（如 3.2.13）
    ARTICLE_PATTERN = re.compile(r'^(\d+\.\d+\.\d+)\s+', re.MULTILINE)

    # 子条款正则：匹配条款下的 1、2、3 子项（独立成行）
    SUBARTICLE_PATTERN = re.compile(r'^(\d+)\s+[^\d\s]', re.MULTILINE)

    # 表格正则：匹配 <table>...</table>
    TABLE_PATTERN = re.compile(r'<table>.*?</table>', re.DOTALL)

    def __init__(self, config: VectorDBConfig):
        self.config = config
        self._max_encode_length = 512

    def _count_tokens(self, text: str, tokenizer) -> int:
        """安全计算 token 数，避免超长文本触发警告"""
        if len(text) < 2000:
            return len(tokenizer.encode(text, add_special_tokens=False, truncation=True))
        return int(len(text) / 1.5)

    def chunk(self, docs: List[Dict], tokenizer) -> List[Dict]:
        """对法律条款文档进行语义分割"""
        chunks = []

        for doc in docs:
            text = doc["text"].strip()
            if not text:
                continue

            # 1. 提取表格（表格作为独立片段）
            tables, text_without_tables = self._extract_tables(text)

            for table in tables:
                if len(table) >= self.config.min_chunk_len:
                    chunks.append({
                        "text": table,
                        "source": doc["source"],
                        "page": doc["page"],
                        "section_title": "",
                        "article_id": "",
                        "is_table": True
                    })

            # 2. 提取章节边界
            sections = self._identify_sections(text_without_tables)

            # 3. 按条款分割
            article_chunks = self._split_by_articles(text_without_tables, sections, doc, tokenizer)
            chunks.extend(article_chunks)

        logger.info(f"法律条款分割完成: {len(docs)} 段 → {len(chunks)} 块")
        return chunks

    def _extract_tables(self, text: str) -> Tuple[List[str], str]:
        """提取表格，返回 (表格列表, 去除表格后的文本)"""
        tables = self.TABLE_PATTERN.findall(text)
        text_without_tables = self.TABLE_PATTERN.sub('\n[表格内容已单独提取]\n', text)
        return tables, text_without_tables

    def _identify_sections(self, text: str) -> List[Dict]:
        """提取章节标题和边界位置"""
        sections = []
        matches = list(self.SECTION_PATTERN.finditer(text))

        for i, match in enumerate(matches):
            section_id = match.group(1)
            section_title = match.group(2).strip()
            start_pos = match.start()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections.append({
                "id": section_id,
                "title": section_title,
                "start": start_pos,
                "end": end_pos
            })

        if not sections:
            sections.append({
                "id": "",
                "title": "全文",
                "start": 0,
                "end": len(text)
            })

        return sections

    def _split_by_articles(self, text: str, sections: List[Dict], doc: Dict, tokenizer) -> List[Dict]:
        """按条款编号分割文本，保留层级信息"""
        chunks = []

        for section in sections:
            section_text = text[section["start"]:section["end"]]
            article_matches = list(self.ARTICLE_PATTERN.finditer(section_text))

            if not article_matches:
                para_chunks = self._split_by_paragraphs(section_text, section, doc, tokenizer)
                chunks.extend(para_chunks)
                continue

            for i, match in enumerate(article_matches):
                article_id = match.group(1)
                start_pos = match.start()
                end_pos = article_matches[i + 1].start() if i + 1 < len(article_matches) else len(section_text)

                article_text = section_text[start_pos:end_pos].strip()
                article_tokens = self._count_tokens(article_text, tokenizer)

                if article_tokens <= self.config.chunk_size:
                    if len(article_text) >= self.config.min_chunk_len:
                        chunks.append({
                            "text": article_text,
                            "source": doc["source"],
                            "page": doc["page"],
                            "section_title": section["title"],
                            "article_id": article_id,
                            "is_table": False
                        })
                else:
                    sub_chunks = self._split_by_subarticles(article_text, article_id, section, doc, tokenizer)
                    chunks.extend(sub_chunks)

        return chunks

    def _split_by_subarticles(self, article_text: str, article_id: str, section: Dict, doc: Dict, tokenizer) -> List[Dict]:
        """按子条款分割过长条款"""
        chunks = []
        sub_matches = list(self.SUBARTICLE_PATTERN.finditer(article_text))

        if not sub_matches:
            return self._force_split(article_text, article_id, section, doc, tokenizer)

        preamble = article_text[:sub_matches[0].start()].strip()
        current_chunk = preamble
        current_tokens = self._count_tokens(current_chunk, tokenizer)

        for i, match in enumerate(sub_matches):
            sub_id = match.group(1)
            start_pos = match.start()
            end_pos = sub_matches[i + 1].start() if i + 1 < len(sub_matches) else len(article_text)

            sub_text = article_text[start_pos:end_pos].strip()
            sub_tokens = self._count_tokens(sub_text, tokenizer)

            if current_tokens + sub_tokens <= self.config.chunk_size:
                current_chunk += "\n" + sub_text
                current_tokens += sub_tokens
            else:
                if len(current_chunk) >= self.config.min_chunk_len:
                    chunks.append({
                        "text": current_chunk,
                        "source": doc["source"],
                        "page": doc["page"],
                        "section_title": section["title"],
                        "article_id": article_id,
                        "is_table": False
                    })

                current_chunk = preamble + "\n" + sub_text
                current_tokens = self._count_tokens(current_chunk, tokenizer)

        if len(current_chunk) >= self.config.min_chunk_len:
            chunks.append({
                "text": current_chunk,
                "source": doc["source"],
                "page": doc["page"],
                "section_title": section["title"],
                "article_id": article_id,
                "is_table": False
            })

        return chunks

    def _split_by_paragraphs(self, text: str, section: Dict, doc: Dict, tokenizer) -> List[Dict]:
        """按段落分割无条款编号的文本"""
        chunks = []
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        current_chunk = ""
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self._count_tokens(para, tokenizer)

            if current_tokens + para_tokens <= self.config.chunk_size:
                current_chunk += "\n\n" + para if current_chunk else para
                current_tokens += para_tokens
            else:
                if len(current_chunk) >= self.config.min_chunk_len:
                    chunks.append({
                        "text": current_chunk,
                        "source": doc["source"],
                        "page": doc["page"],
                        "section_title": section["title"],
                        "article_id": "",
                        "is_table": False
                    })
                current_chunk = para
                current_tokens = para_tokens

        if len(current_chunk) >= self.config.min_chunk_len:
            chunks.append({
                "text": current_chunk,
                "source": doc["source"],
                "page": doc["page"],
                "section_title": section["title"],
                "article_id": "",
                "is_table": False
            })

        return chunks

    def _force_split(self, text: str, article_id: str, section: Dict, doc: Dict, tokenizer) -> List[Dict]:
        """强制按长度分割过长文本"""
        chunks = []
        sentences = re.split(r'([。！？\n])', text)
        sentences = [''.join(pair) for pair in zip(sentences[::2], sentences[1::2] or [''])]

        current_chunk = ""
        current_tokens = 0

        for sent in sentences:
            if not sent.strip():
                continue
            sent_tokens = self._count_tokens(sent, tokenizer)

            if current_tokens + sent_tokens <= self.config.chunk_size:
                current_chunk += sent
                current_tokens += sent_tokens
            else:
                if len(current_chunk) >= self.config.min_chunk_len:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "source": doc["source"],
                        "page": doc["page"],
                        "section_title": section["title"],
                        "article_id": article_id,
                        "is_table": False
                    })
                current_chunk = sent
                current_tokens = sent_tokens

        if len(current_chunk) >= self.config.min_chunk_len:
            chunks.append({
                "text": current_chunk.strip(),
                "source": doc["source"],
                "page": doc["page"],
                "section_title": section["title"],
                "article_id": article_id,
                "is_table": False
            })

        return chunks


# ─────────────────────────────────────────────
# 3. Embedding 模型
# ─────────────────────────────────────────────
class EmbeddingModel:
    def __init__(self, model_path: str, device: str = "cpu"):
        logger.info(f"加载 Embedding 模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32)
        self.model.to(device)
        self.model.eval()
        self.device = device

    @torch.inference_mode()
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """批量编码文本为向量"""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            output = self.model(**encoded)
            embeddings = output.last_hidden_state[:, 0, :]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

            if (i // batch_size) % 10 == 0:
                logger.info(f"  Embedding 进度: {min(i+batch_size, len(texts))}/{len(texts)}")

        return np.vstack(all_embeddings).astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """编码查询 (加 BGE 查询前缀)"""
        prefixed = f"为这个句子生成表示以用于检索相关文章：{query}"
        return self.encode([prefixed])[0]


# ─────────────────────────────────────────────
# 4. 构建 USearch 索引
# ─────────────────────────────────────────────
def build_usearch_index(embeddings: np.ndarray) -> Index:
    """
    构建 USearch 索引
    USearch 使用余弦相似度 (Cosine Similarity)
    """
    dim = embeddings.shape[1]
    count = embeddings.shape[0]

    # 创建 USearch 索引
    # 使用余弦相似度，向量已经 L2 归一化
    index = Index(
        ndim=dim,
        metric="cos",  # 余弦相似度
        dtype="f32",   # 32位浮点
        connectivity=16,  # 连接数，影响召回率
        expansion_add=40, # 添加时的扩展因子
        expansion_search=16,  # 搜索时的扩展因子
    )

    # 添加向量
    keys = np.arange(count, dtype=np.uint64)
    index.add(keys, embeddings)

    logger.info(f"USearch 索引构建完成: {index.size} 条向量, 维度 {dim}")
    return index


# ─────────────────────────────────────────────
# 5. 保存与加载
# ─────────────────────────────────────────────
def save_vectordb(index: Index, chunks: List[Dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # 保存 USearch 索引
    index.save(os.path.join(output_dir, "index.usearch"))

    # 保存 chunks 数据
    with open(os.path.join(output_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    # 同时保存 JSON 版本便于查阅
    with open(os.path.join(output_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"向量库已保存: {output_dir}")


def load_vectordb(output_dir: str):
    """加载 USearch 向量库"""
    index = Index(os.path.join(output_dir, "index.usearch"))
    with open(os.path.join(output_dir, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    logger.info(f"向量库已加载: {index.size} 条")
    return index, chunks


# ─────────────────────────────────────────────
# 6. 主流程
# ─────────────────────────────────────────────
def build(config: VectorDBConfig = CONFIG):
    # 加载 Embedding 模型
    embed_model = EmbeddingModel(config.embed_model, config.device)

    # 加载文档
    docs = load_documents(config.docs_dir)
    if not docs:
        logger.error(f"未找到文档，请将法规文件放入 {config.docs_dir} 目录")
        return

    # 分块
    chunker = RegulationChunker(config)
    chunks = chunker.chunk(docs, embed_model.tokenizer)

    # Embedding
    texts = [c["text"] for c in chunks]
    logger.info(f"开始 Embedding {len(texts)} 个文本块...")
    embeddings = embed_model.encode(texts, batch_size=config.batch_size)

    # 构建索引
    index = build_usearch_index(embeddings)

    # 保存
    save_vectordb(index, chunks, config.output_dir)
    logger.info("USearch 向量库构建完成!")


if __name__ == "__main__":
    build()