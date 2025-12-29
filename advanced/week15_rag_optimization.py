"""
Week 15: RAG Pipeline优化
包括：Embedding模型对比、Chunking策略优化、重排序技术、混合检索

本模块实现RAG系统的多种优化技术，提升检索质量和答案准确性。
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import re
from collections import Counter


class ChunkingStrategy:
    """文本分块策略基类"""
    
    def chunk(self, text: str) -> List[str]:
        """分块接口"""
        raise NotImplementedError


class FixedSizeChunking(ChunkingStrategy):
    """固定大小分块策略"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str) -> List[str]:
        """固定大小分块"""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.overlap
        
        return chunks


class SentenceChunking(ChunkingStrategy):
    """句子级别分块策略"""
    
    def __init__(self, sentences_per_chunk: int = 3):
        self.sentences_per_chunk = sentences_per_chunk
    
    def chunk(self, text: str) -> List[str]:
        """按句子分块"""
        # 简单的句子分割（中英文）
        sentences = re.split(r'[。！？.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        for i in range(0, len(sentences), self.sentences_per_chunk):
            chunk = '。'.join(sentences[i:i + self.sentences_per_chunk])
            if chunk:
                chunks.append(chunk)
        
        return chunks


class SemanticChunking(ChunkingStrategy):
    """语义分块策略（基于相似度）"""
    
    def __init__(self, similarity_threshold: float = 0.7, max_chunk_size: int = 500):
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度（简单的词重叠）"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def chunk(self, text: str) -> List[str]:
        """基于语义相似度分块"""
        sentences = re.split(r'[。！？.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = [sentences[0]]
        current_size = len(sentences[0])
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            # 检查与当前块的最后一句的相似度
            similarity = self._compute_similarity(current_chunk[-1], sentence)
            
            # 如果相似度高且不超过最大长度，加入当前块
            if similarity >= self.similarity_threshold and current_size + len(sentence) <= self.max_chunk_size:
                current_chunk.append(sentence)
                current_size += len(sentence)
            else:
                # 开始新的块
                chunks.append('。'.join(current_chunk))
                current_chunk = [sentence]
                current_size = len(sentence)
        
        # 添加最后一个块
        if current_chunk:
            chunks.append('。'.join(current_chunk))
        
        return chunks


class RecursiveChunking(ChunkingStrategy):
    """递归分块策略（按结构层次）"""
    
    def __init__(self, chunk_size: int = 500, separators: List[str] = None):
        self.chunk_size = chunk_size
        self.separators = separators or ['\n\n', '\n', '。', ' ']
    
    def chunk(self, text: str) -> List[str]:
        """递归分块"""
        return self._recursive_split(text, self.separators)
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """递归分割文本"""
        if not separators:
            # 没有分隔符了，直接按固定大小分割
            return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # 按当前分隔符分割
        parts = text.split(separator)
        chunks = []
        
        for part in parts:
            if len(part) <= self.chunk_size:
                if part.strip():
                    chunks.append(part)
            else:
                # 部分太长，使用下一个分隔符递归分割
                sub_chunks = self._recursive_split(part, remaining_separators)
                chunks.extend(sub_chunks)
        
        return chunks


class EmbeddingModel:
    """Embedding模型基类"""
    
    def embed(self, text: str) -> np.ndarray:
        """文本向量化"""
        raise NotImplementedError


class TFIDFEmbedding(EmbeddingModel):
    """TF-IDF向量化（简化版）"""
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 384):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.vocab = {}
        self.idf = {}
    
    def fit(self, documents: List[str]):
        """训练词汇表和IDF"""
        # 构建词汇表
        word_counts = Counter()
        doc_word_counts = []
        
        for doc in documents:
            words = doc.split()
            word_counts.update(words)
            doc_word_counts.append(set(words))
        
        # 选择最常见的词
        most_common = word_counts.most_common(self.vocab_size)
        self.vocab = {word: idx for idx, (word, _) in enumerate(most_common)}
        
        # 计算IDF
        num_docs = len(documents)
        for word in self.vocab:
            doc_freq = sum(1 for doc_words in doc_word_counts if word in doc_words)
            self.idf[word] = np.log(num_docs / (1 + doc_freq))
    
    def embed(self, text: str) -> np.ndarray:
        """TF-IDF向量化"""
        words = text.split()
        word_count = Counter(words)
        
        # 创建稀疏向量
        vector = np.zeros(self.embedding_dim)
        
        for word, count in word_count.items():
            if word in self.vocab:
                idx = self.vocab[word] % self.embedding_dim
                tf = count / len(words) if words else 0
                idf = self.idf.get(word, 0)
                vector[idx] += tf * idf
        
        # 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector


class Word2VecEmbedding(EmbeddingModel):
    """Word2Vec风格的向量化（简化版）"""
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 300):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.word_vectors = {}
        self.vocab = {}
    
    def fit(self, documents: List[str]):
        """训练词向量（简化版：随机初始化）"""
        # 构建词汇表
        word_counts = Counter()
        for doc in documents:
            words = doc.split()
            word_counts.update(words)
        
        most_common = word_counts.most_common(self.vocab_size)
        self.vocab = {word: idx for idx, (word, _) in enumerate(most_common)}
        
        # 随机初始化词向量（实际应该用skip-gram或CBOW训练）
        np.random.seed(42)
        for word in self.vocab:
            self.word_vectors[word] = np.random.randn(self.embedding_dim)
            # 归一化
            self.word_vectors[word] /= np.linalg.norm(self.word_vectors[word])
    
    def embed(self, text: str) -> np.ndarray:
        """平均词向量"""
        words = text.split()
        vectors = []
        
        for word in words:
            if word in self.word_vectors:
                vectors.append(self.word_vectors[word])
        
        if not vectors:
            return np.zeros(self.embedding_dim)
        
        # 平均池化
        avg_vector = np.mean(vectors, axis=0)
        
        # 归一化
        norm = np.linalg.norm(avg_vector)
        if norm > 0:
            avg_vector = avg_vector / norm
        
        return avg_vector


class TransformerEmbedding(EmbeddingModel):
    """Transformer风格的向量化（模拟）"""
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 768):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.vocab = {}
        self.token_embeddings = None
    
    def fit(self, documents: List[str]):
        """训练（模拟）"""
        # 构建词汇表
        word_counts = Counter()
        for doc in documents:
            words = doc.split()
            word_counts.update(words)
        
        most_common = word_counts.most_common(self.vocab_size)
        self.vocab = {word: idx for idx, (word, _) in enumerate(most_common)}
        
        # 随机初始化token embeddings
        np.random.seed(42)
        self.token_embeddings = np.random.randn(self.vocab_size, self.embedding_dim).astype(np.float32)
        
        # 归一化
        for i in range(self.vocab_size):
            norm = np.linalg.norm(self.token_embeddings[i])
            if norm > 0:
                self.token_embeddings[i] /= norm
    
    def embed(self, text: str) -> np.ndarray:
        """Transformer-style embedding（简化：平均池化）"""
        words = text.split()
        token_ids = [self.vocab.get(word, 0) for word in words if word in self.vocab]
        
        if not token_ids:
            return np.zeros(self.embedding_dim)
        
        # 获取token embeddings
        embeddings = self.token_embeddings[token_ids]
        
        # 平均池化（实际Transformer会加attention）
        avg_embedding = np.mean(embeddings, axis=0)
        
        # 归一化
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
        
        return avg_embedding


class BM25Retriever:
    """BM25检索器（稀疏检索）"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_freqs = Counter()
        self.idf = {}
        self.avg_doc_len = 0
    
    def fit(self, documents: List[str]):
        """训练BM25"""
        self.documents = documents
        doc_lens = []
        
        # 计算文档频率
        for doc in documents:
            words = set(doc.split())
            doc_lens.append(len(doc.split()))
            for word in words:
                self.doc_freqs[word] += 1
        
        self.avg_doc_len = np.mean(doc_lens)
        
        # 计算IDF
        num_docs = len(documents)
        for word, freq in self.doc_freqs.items():
            self.idf[word] = np.log((num_docs - freq + 0.5) / (freq + 0.5) + 1.0)
    
    def search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """BM25搜索"""
        query_words = query.split()
        scores = []
        
        for doc_idx, doc in enumerate(self.documents):
            doc_words = doc.split()
            doc_len = len(doc_words)
            word_counts = Counter(doc_words)
            
            score = 0.0
            for word in query_words:
                if word in word_counts:
                    tf = word_counts[word]
                    idf = self.idf.get(word, 0)
                    
                    # BM25公式
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                    score += idf * numerator / denominator
            
            scores.append((doc_idx, score))
        
        # 排序并返回Top-K
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


class CrossEncoderReranker:
    """交叉编码器重排序器（模拟）"""
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        # 模拟的权重矩阵
        np.random.seed(42)
        self.W = np.random.randn(embedding_dim * 2, 1).astype(np.float32) * 0.01
    
    def score(self, query: str, document: str) -> float:
        """计算query-document相关性分数"""
        # 简化的特征提取
        query_vec = self._simple_embed(query)
        doc_vec = self._simple_embed(document)
        
        # 拼接特征
        combined = np.concatenate([query_vec, doc_vec])
        
        # 线性打分（实际应该是复杂的神经网络）
        score = float(np.dot(combined, self.W))
        
        # Sigmoid激活
        return 1.0 / (1.0 + np.exp(-score))
    
    def _simple_embed(self, text: str) -> np.ndarray:
        """简单的文本向量化"""
        words = text.split()
        vector = np.zeros(self.embedding_dim)
        
        for i, word in enumerate(words[:self.embedding_dim]):
            vector[i] = hash(word) % 100 / 100.0
        
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[int, float]]:
        """重排序"""
        scores = []
        for idx, doc in enumerate(documents):
            score = self.score(query, doc)
            scores.append((idx, score))
        
        # 排序
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class HybridRetriever:
    """混合检索器（BM25 + Dense）"""
    
    def __init__(self, 
                 bm25_weight: float = 0.5,
                 dense_weight: float = 0.5):
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.bm25 = BM25Retriever()
        self.embedder = None
        self.doc_vectors = []
        self.documents = []
    
    def fit(self, documents: List[str], embedder: EmbeddingModel):
        """训练混合检索器"""
        self.documents = documents
        self.embedder = embedder
        
        # 训练BM25
        self.bm25.fit(documents)
        
        # 计算文档向量
        print(f"   计算 {len(documents)} 个文档的向量...")
        self.doc_vectors = [embedder.embed(doc) for doc in documents]
    
    def search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """混合检索"""
        # BM25检索
        bm25_results = self.bm25.search(query, k=len(self.documents))
        bm25_scores = {idx: score for idx, score in bm25_results}
        
        # Dense检索
        query_vec = self.embedder.embed(query)
        dense_scores = {}
        for idx, doc_vec in enumerate(self.doc_vectors):
            similarity = np.dot(query_vec, doc_vec)
            dense_scores[idx] = float(similarity)
        
        # 归一化分数
        if bm25_scores:
            max_bm25 = max(bm25_scores.values())
            if max_bm25 > 0:
                bm25_scores = {k: v / max_bm25 for k, v in bm25_scores.items()}
        
        if dense_scores:
            max_dense = max(dense_scores.values())
            min_dense = min(dense_scores.values())
            if max_dense > min_dense:
                dense_scores = {k: (v - min_dense) / (max_dense - min_dense) 
                               for k, v in dense_scores.items()}
        
        # 混合分数
        hybrid_scores = []
        for idx in range(len(self.documents)):
            bm25_score = bm25_scores.get(idx, 0.0)
            dense_score = dense_scores.get(idx, 0.0)
            hybrid_score = self.bm25_weight * bm25_score + self.dense_weight * dense_score
            hybrid_scores.append((idx, hybrid_score))
        
        # 排序
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        return hybrid_scores[:k]


def demonstrate_chunking_strategies():
    """演示不同的分块策略"""
    print("\n" + "="*70)
    print("📝 演示：分块策略对比")
    print("="*70)
    
    # 示例文本
    text = """
    人工智能（AI）是计算机科学的一个分支。它企图了解智能的实质。
    机器学习是AI的核心技术之一。深度学习则是机器学习的子领域。
    神经网络模拟人脑的工作方式。卷积神经网络特别适合图像处理。
    自然语言处理让计算机理解人类语言。大语言模型展现了惊人的能力。
    RAG技术结合了检索和生成。它能显著提升答案的准确性。
    向量数据库是RAG系统的关键组件。高效的索引机制至关重要。
    """.strip()
    
    print(f"\n原始文本长度: {len(text)} 字符")
    print(f"文本内容:\n{text[:200]}...\n")
    
    strategies = [
        ("固定大小分块", FixedSizeChunking(chunk_size=100, overlap=20)),
        ("句子级别分块", SentenceChunking(sentences_per_chunk=2)),
        ("语义分块", SemanticChunking(similarity_threshold=0.3, max_chunk_size=150)),
        ("递归分块", RecursiveChunking(chunk_size=100)),
    ]
    
    for name, strategy in strategies:
        chunks = strategy.chunk(text)
        print(f"\n【{name}】")
        print(f"   块数量: {len(chunks)}")
        print(f"   平均长度: {np.mean([len(c) for c in chunks]):.1f} 字符")
        print(f"   前2块预览:")
        for i, chunk in enumerate(chunks[:2], 1):
            print(f"      块{i}: {chunk[:60]}...")


def demonstrate_embedding_comparison():
    """演示不同的Embedding模型对比"""
    print("\n" + "="*70)
    print("🔢 演示：Embedding模型对比")
    print("="*70)
    
    # 示例文档
    documents = [
        "机器学习是人工智能的核心技术",
        "深度学习使用多层神经网络",
        "自然语言处理让计算机理解文本",
        "计算机视觉处理图像和视频",
        "强化学习通过奖励优化策略",
    ]
    
    query = "什么是深度学习"
    
    print(f"\n文档数量: {len(documents)}")
    print(f"查询: '{query}'")
    
    models = [
        ("TF-IDF", TFIDFEmbedding(vocab_size=1000, embedding_dim=128)),
        ("Word2Vec", Word2VecEmbedding(vocab_size=1000, embedding_dim=300)),
        ("Transformer", TransformerEmbedding(vocab_size=1000, embedding_dim=768)),
    ]
    
    for name, model in models:
        print(f"\n【{name} Embedding】")
        
        # 训练
        model.fit(documents)
        
        # 向量化
        query_vec = model.embed(query)
        doc_vecs = [model.embed(doc) for doc in documents]
        
        print(f"   向量维度: {len(query_vec)}")
        
        # 计算相似度
        similarities = [np.dot(query_vec, doc_vec) for doc_vec in doc_vecs]
        
        # 排序
        ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
        
        print(f"   Top-3检索结果:")
        for rank, (idx, score) in enumerate(ranked[:3], 1):
            print(f"      {rank}. (相似度: {score:.3f}) {documents[idx]}")


def demonstrate_reranking():
    """演示重排序技术"""
    print("\n" + "="*70)
    print("🔄 演示：重排序技术")
    print("="*70)
    
    documents = [
        "机器学习是人工智能的核心",
        "深度学习使用神经网络",
        "Python是流行的编程语言",
        "卷积神经网络处理图像",
        "自然语言处理分析文本",
    ]
    
    query = "神经网络和深度学习"
    
    print(f"\n查询: '{query}'")
    print(f"候选文档数: {len(documents)}")
    
    # 初始检索（使用简单的TF-IDF）
    embedder = TFIDFEmbedding(embedding_dim=128)
    embedder.fit(documents)
    
    query_vec = embedder.embed(query)
    initial_scores = []
    for idx, doc in enumerate(documents):
        doc_vec = embedder.embed(doc)
        score = np.dot(query_vec, doc_vec)
        initial_scores.append((idx, score))
    
    initial_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n【初始检索结果】")
    for rank, (idx, score) in enumerate(initial_scores, 1):
        print(f"   {rank}. (分数: {score:.3f}) {documents[idx]}")
    
    # 重排序
    reranker = CrossEncoderReranker(embedding_dim=128)
    top_k_docs = [documents[idx] for idx, _ in initial_scores[:5]]
    reranked = reranker.rerank(query, top_k_docs, top_k=3)
    
    print(f"\n【重排序后结果】")
    for rank, (idx, score) in enumerate(reranked, 1):
        original_idx = initial_scores[idx][0]
        print(f"   {rank}. (分数: {score:.3f}) {top_k_docs[idx]}")


def demonstrate_hybrid_retrieval():
    """演示混合检索"""
    print("\n" + "="*70)
    print("🔀 演示：混合检索（BM25 + Dense）")
    print("="*70)
    
    documents = [
        "机器学习是人工智能的重要分支，包括监督学习和无监督学习",
        "深度学习使用多层神经网络，能够学习复杂的特征表示",
        "自然语言处理研究计算机如何理解和生成人类语言",
        "计算机视觉让机器能够识别和理解图像中的内容",
        "强化学习通过与环境交互来学习最优策略",
        "卷积神经网络在图像识别任务中表现出色",
        "循环神经网络适合处理序列数据如文本和时间序列",
        "注意力机制让模型关注输入中最重要的部分",
    ]
    
    query = "深度学习神经网络"
    
    print(f"\n文档数量: {len(documents)}")
    print(f"查询: '{query}'")
    
    # 准备Embedding模型
    embedder = TFIDFEmbedding(embedding_dim=256)
    embedder.fit(documents)
    
    # 创建混合检索器
    hybrid = HybridRetriever(bm25_weight=0.5, dense_weight=0.5)
    hybrid.fit(documents, embedder)
    
    # BM25单独检索
    print(f"\n【BM25检索】")
    bm25_results = hybrid.bm25.search(query, k=3)
    for rank, (idx, score) in enumerate(bm25_results, 1):
        print(f"   {rank}. (分数: {score:.3f}) {documents[idx][:50]}...")
    
    # Dense单独检索
    print(f"\n【Dense检索】")
    query_vec = embedder.embed(query)
    dense_scores = []
    for idx, doc_vec in enumerate(hybrid.doc_vectors):
        score = np.dot(query_vec, doc_vec)
        dense_scores.append((idx, score))
    dense_scores.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (idx, score) in enumerate(dense_scores[:3], 1):
        print(f"   {rank}. (分数: {score:.3f}) {documents[idx][:50]}...")
    
    # 混合检索
    print(f"\n【混合检索】")
    hybrid_results = hybrid.search(query, k=3)
    for rank, (idx, score) in enumerate(hybrid_results, 1):
        print(f"   {rank}. (分数: {score:.3f}) {documents[idx][:50]}...")


def run_week15_demo():
    """运行Week 15完整演示"""
    print("\n" + "="*70)
    print("🚀 Week 15: RAG Pipeline优化 - 完整演示")
    print("="*70)
    
    # 1. 分块策略
    demonstrate_chunking_strategies()
    
    input("\n按Enter继续查看Embedding模型对比...")
    
    # 2. Embedding模型对比
    demonstrate_embedding_comparison()
    
    input("\n按Enter继续查看重排序技术...")
    
    # 3. 重排序
    demonstrate_reranking()
    
    input("\n按Enter继续查看混合检索...")
    
    # 4. 混合检索
    demonstrate_hybrid_retrieval()
    
    print("\n" + "="*70)
    print("✅ Week 15演示完成！")
    print("="*70)
    print("\n核心收获：")
    print("  1. 掌握了4种分块策略：固定、句子、语义、递归")
    print("  2. 对比了3种Embedding：TF-IDF、Word2Vec、Transformer")
    print("  3. 理解了重排序技术的重要性")
    print("  4. 学会了混合检索（BM25 + Dense）")


if __name__ == "__main__":
    run_week15_demo()
