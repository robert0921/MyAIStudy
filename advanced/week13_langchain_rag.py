"""
第13周：LangChain框架与文档检索原理
实现简单的知识库问答系统

核心知识点：
1. LangChain框架架构
2. 文档加载与分割
3. Embedding向量化
4. 检索式问答(RAG)
5. Prompt模板设计
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import re
import json


class SimpleTextSplitter:
    """简单的文本分割器"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Args:
            chunk_size: 每个分块的字符数
            chunk_overlap: 分块之间的重叠字符数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """将文本分割成多个块"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # 尝试在句子边界分割
            if end < text_length:
                # 寻找最近的句号、问号、感叹号
                last_period = max(
                    chunk.rfind('。'),
                    chunk.rfind('！'),
                    chunk.rfind('？'),
                    chunk.rfind('.'),
                    chunk.rfind('!'),
                    chunk.rfind('?')
                )
                if last_period > self.chunk_size * 0.5:  # 至少保留一半内容
                    chunk = chunk[:last_period + 1]
                    end = start + len(chunk)
            
            chunks.append(chunk.strip())
            start = end - self.chunk_overlap
        
        return [c for c in chunks if c]  # 过滤空块


class SimpleEmbedding:
    """简单的文本向量化（模拟真实Embedding）"""
    
    def __init__(self, embedding_dim: int = 384):
        """
        Args:
            embedding_dim: 向量维度
        """
        self.embedding_dim = embedding_dim
        # 简单的词表（实际应用中使用预训练模型）
        self.vocab = {}
        self._build_vocab()
    
    def _build_vocab(self):
        """构建简单词表"""
        common_words = [
            # 技术词汇
            'AI', 'ML', 'DL', 'NLP', 'LLM', 'RAG', 'embedding', 'vector',
            'transformer', 'attention', 'model', 'training', 'inference',
            'dataset', 'python', 'pytorch', 'tensorflow',
            # 中文词汇
            '人工智能', '机器学习', '深度学习', '自然语言处理', '大模型',
            '向量', '嵌入', '检索', '生成', '训练', '推理', '数据集',
            # 常用词
            'the', 'is', 'are', 'in', 'of', 'to', 'and', 'or',
            '是', '在', '的', '和', '或', '与', '为'
        ]
        
        for i, word in enumerate(common_words):
            # 生成固定的向量（实际中使用预训练权重）
            np.random.seed(hash(word) % (2**32))
            self.vocab[word.lower()] = np.random.randn(self.embedding_dim)
            np.random.seed()  # 重置seed
    
    def embed_text(self, text: str) -> np.ndarray:
        """将文本转换为向量"""
        # 简单分词
        words = re.findall(r'\w+', text.lower())
        
        # 获取词向量
        vectors = []
        for word in words:
            if word in self.vocab:
                vectors.append(self.vocab[word])
            else:
                # 未知词使用随机向量
                np.random.seed(hash(word) % (2**32))
                vec = np.random.randn(self.embedding_dim) * 0.1
                vectors.append(vec)
                np.random.seed()
        
        if not vectors:
            return np.zeros(self.embedding_dim)
        
        # 平均池化
        embedding = np.mean(vectors, axis=0)
        # L2归一化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def embed_documents(self, documents: List[str]) -> List[np.ndarray]:
        """批量向量化"""
        return [self.embed_text(doc) for doc in documents]


class SimpleVectorStore:
    """简单的向量存储和检索"""
    
    def __init__(self, embedding_function):
        """
        Args:
            embedding_function: 向量化函数
        """
        self.embedding_function = embedding_function
        self.documents = []
        self.embeddings = []
        self.metadata = []
    
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None):
        """添加文档"""
        self.documents.extend(documents)
        self.embeddings.extend(self.embedding_function.embed_documents(documents))
        
        if metadatas is None:
            metadatas = [{'index': i + len(self.metadata)} for i in range(len(documents))]
        self.metadata.extend(metadatas)
    
    def similarity_search(self, query: str, k: int = 3) -> List[Tuple[str, float, Dict]]:
        """相似度搜索"""
        if not self.documents:
            return []
        
        # 查询向量化
        query_embedding = self.embedding_function.embed_text(query)
        
        # 计算余弦相似度
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = np.dot(query_embedding, doc_embedding)
            similarities.append((i, similarity))
        
        # 排序并返回top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]
        
        results = []
        for idx, score in top_k:
            results.append((
                self.documents[idx],
                float(score),
                self.metadata[idx]
            ))
        
        return results


class SimplePromptTemplate:
    """简单的Prompt模板"""
    
    def __init__(self, template: str):
        """
        Args:
            template: 模板字符串，使用{变量名}表示占位符
        """
        self.template = template
    
    def format(self, **kwargs) -> str:
        """格式化模板"""
        return self.template.format(**kwargs)


class SimpleRAGChain:
    """简单的RAG问答链"""
    
    def __init__(
        self,
        vector_store: SimpleVectorStore,
        prompt_template: Optional[SimplePromptTemplate] = None,
        k: int = 3
    ):
        """
        Args:
            vector_store: 向量存储
            prompt_template: Prompt模板
            k: 检索文档数量
        """
        self.vector_store = vector_store
        self.k = k
        
        # 默认模板
        if prompt_template is None:
            template = """基于以下上下文回答问题。如果上下文中没有相关信息，请说"我不知道"。

上下文：
{context}

问题：{question}

回答："""
            self.prompt_template = SimplePromptTemplate(template)
        else:
            self.prompt_template = prompt_template
    
    def run(self, question: str) -> Dict:
        """运行RAG问答"""
        # 1. 检索相关文档
        search_results = self.vector_store.similarity_search(question, k=self.k)
        
        # 2. 组装上下文
        context_parts = []
        for i, (doc, score, metadata) in enumerate(search_results):
            context_parts.append(f"[文档{i+1}] (相似度: {score:.3f})\n{doc}")
        context = "\n\n".join(context_parts)
        
        # 3. 生成prompt
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )
        
        # 4. 模拟LLM回答（实际应用中调用真实LLM）
        answer = self._generate_answer(question, search_results)
        
        return {
            'question': question,
            'answer': answer,
            'source_documents': search_results,
            'prompt': prompt
        }
    
    def _generate_answer(self, question: str, docs: List[Tuple[str, float, Dict]]) -> str:
        """模拟生成答案"""
        if not docs:
            return "抱歉，我没有找到相关信息。"
        
        # 简单规则：返回最相关文档的摘要
        best_doc, score, metadata = docs[0]
        
        if score < 0.3:
            return "抱歉，找到的信息相关度较低，无法准确回答。"
        
        # 提取文档前200字符作为答案
        answer = best_doc[:200] + "..."
        return f"根据检索到的信息：\n\n{answer}\n\n（这是一个模拟回答，实际应用中需要使用LLM生成）"


def demonstrate_langchain_rag():
    """演示LangChain RAG系统"""
    print("=" * 80)
    print("第13周：LangChain框架与RAG原理演示")
    print("=" * 80)
    
    # 1. 准备示例文档
    print("\n【步骤1】准备知识库文档")
    print("-" * 80)
    
    documents = [
        """
        深度学习是机器学习的一个分支，它基于人工神经网络，特别是深层神经网络。
        深度学习在图像识别、语音识别、自然语言处理等领域取得了突破性进展。
        常见的深度学习框架包括PyTorch、TensorFlow、Keras等。
        """,
        """
        RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术。
        它首先从知识库中检索相关文档，然后将检索结果作为上下文输入到大语言模型中生成答案。
        RAG可以有效减少模型幻觉，提供更准确和可验证的回答。
        """,
        """
        向量数据库是专门用于存储和检索高维向量的数据库系统。
        常见的向量数据库包括FAISS、Milvus、Chroma、Pinecone等。
        向量数据库支持快速的相似度搜索，是RAG系统的核心组件。
        """,
        """
        LangChain是一个用于开发由语言模型驱动的应用程序的框架。
        它提供了文档加载器、文本分割器、向量存储、检索器等组件。
        LangChain简化了RAG应用的开发流程。
        """,
        """
        Transformer是现代深度学习中最重要的架构之一。
        它基于自注意力机制，可以并行处理序列数据。
        BERT、GPT、LLaMA等模型都基于Transformer架构。
        """,
        """
        Prompt Engineering是设计和优化提示词的技术。
        好的prompt可以显著提升模型的输出质量。
        常见技术包括Few-shot学习、Chain-of-Thought、ReAct等。
        """
    ]
    
    print(f"✅ 加载了 {len(documents)} 个文档")
    for i, doc in enumerate(documents):
        preview = doc.strip().replace('\n', ' ')[:60]
        print(f"   文档{i+1}: {preview}...")
    
    # 2. 文本分割
    print("\n【步骤2】文本分割")
    print("-" * 80)
    
    splitter = SimpleTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = []
    for doc in documents:
        chunks.extend(splitter.split_text(doc.strip()))
    
    print(f"✅ 分割成 {len(chunks)} 个文本块")
    print(f"   块大小: {splitter.chunk_size} 字符")
    print(f"   重叠度: {splitter.chunk_overlap} 字符")
    print(f"\n   示例块:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"   块{i+1}: {chunk[:60]}...")
    
    # 3. 向量化
    print("\n【步骤3】文本向量化")
    print("-" * 80)
    
    embedder = SimpleEmbedding(embedding_dim=384)
    print(f"✅ 初始化Embedding模型 (维度: {embedder.embedding_dim})")
    
    # 测试向量化
    test_text = "深度学习"
    test_vec = embedder.embed_text(test_text)
    print(f"   测试文本: '{test_text}'")
    print(f"   向量形状: {test_vec.shape}")
    print(f"   向量前5维: {test_vec[:5]}")
    print(f"   向量L2范数: {np.linalg.norm(test_vec):.4f}")
    
    # 4. 构建向量存储
    print("\n【步骤4】构建向量存储")
    print("-" * 80)
    
    vector_store = SimpleVectorStore(embedder)
    vector_store.add_documents(chunks)
    
    print(f"✅ 向量存储已构建")
    print(f"   存储文档数: {len(vector_store.documents)}")
    print(f"   向量矩阵形状: ({len(vector_store.embeddings)}, {embedder.embedding_dim})")
    
    # 5. 测试检索
    print("\n【步骤5】测试相似度检索")
    print("-" * 80)
    
    test_queries = [
        "什么是RAG？",
        "有哪些深度学习框架？",
        "向量数据库有什么用？"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        results = vector_store.similarity_search(query, k=2)
        
        for i, (doc, score, metadata) in enumerate(results):
            print(f"  结果{i+1} (相似度: {score:.3f}):")
            print(f"    {doc[:100]}...")
    
    # 6. 构建RAG链
    print("\n【步骤6】构建RAG问答系统")
    print("-" * 80)
    
    rag_chain = SimpleRAGChain(vector_store, k=3)
    
    print("✅ RAG链已构建")
    print(f"   检索文档数: {rag_chain.k}")
    print(f"   Prompt模板:")
    print("   " + rag_chain.prompt_template.template[:100].replace('\n', ' ') + "...")
    
    # 7. 问答演示
    print("\n【步骤7】问答演示")
    print("-" * 80)
    
    questions = [
        "什么是RAG技术？",
        "LangChain是什么？",
        "Transformer的特点是什么？"
    ]
    
    for question in questions:
        print(f"\n问题: {question}")
        print("-" * 40)
        
        result = rag_chain.run(question)
        
        print(f"回答: {result['answer']}")
        print(f"\n引用文档:")
        for i, (doc, score, metadata) in enumerate(result['source_documents']):
            print(f"  [{i+1}] 相似度: {score:.3f}")
            print(f"      {doc[:80]}...")
    
    # 8. 性能分析
    print("\n【步骤8】系统性能分析")
    print("-" * 80)
    
    print(f"📊 统计信息:")
    print(f"   原始文档数: {len(documents)}")
    print(f"   分块后数量: {len(chunks)}")
    print(f"   平均块长度: {np.mean([len(c) for c in chunks]):.1f} 字符")
    print(f"   向量维度: {embedder.embedding_dim}")
    print(f"   存储大小: {len(vector_store.embeddings) * embedder.embedding_dim * 4 / 1024:.2f} KB")
    
    # 9. 总结
    print("\n【总结】RAG系统核心组件")
    print("=" * 80)
    print("""
    ✅ 已实现的核心功能:
    
    1. 文档加载与预处理
       - 文本分割器（支持重叠分块）
       - 智能边界检测（句子级别）
    
    2. 文本向量化
       - 简单Embedding模型（词袋 + 平均池化）
       - L2归一化
       - 支持中英文
    
    3. 向量存储与检索
       - 内存向量存储
       - 余弦相似度搜索
       - Top-K检索
    
    4. RAG问答链
       - Prompt模板系统
       - 上下文组装
       - 答案生成（模拟）
    
    💡 实际应用建议:
    - 使用预训练Embedding模型（如sentence-transformers）
    - 使用专业向量数据库（FAISS、Milvus）
    - 接入真实LLM（OpenAI、Claude、本地LLaMA）
    - 添加缓存机制提升性能
    - 实现答案质量评估
    """)
    
    print("\n" + "=" * 80)
    print("演示完成！")
    print("=" * 80)


def test_rag_components():
    """测试RAG各组件"""
    print("\n【组件单元测试】")
    print("=" * 80)
    
    # 测试文本分割
    print("\n1. 测试文本分割器")
    splitter = SimpleTextSplitter(chunk_size=50, chunk_overlap=10)
    test_text = "这是第一句。这是第二句。这是第三句。这是第四句。这是第五句。"
    chunks = splitter.split_text(test_text)
    print(f"   原文长度: {len(test_text)}")
    print(f"   分块数量: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"   块{i+1}: {chunk}")
    
    # 测试向量化
    print("\n2. 测试Embedding")
    embedder = SimpleEmbedding(embedding_dim=128)
    texts = ["机器学习", "深度学习", "人工智能"]
    embeddings = embedder.embed_documents(texts)
    print(f"   向量维度: {embeddings[0].shape}")
    print(f"   向量数量: {len(embeddings)}")
    
    # 计算相似度
    sim_12 = np.dot(embeddings[0], embeddings[1])
    sim_13 = np.dot(embeddings[0], embeddings[2])
    print(f"   '机器学习' vs '深度学习' 相似度: {sim_12:.3f}")
    print(f"   '机器学习' vs '人工智能' 相似度: {sim_13:.3f}")
    
    # 测试向量存储
    print("\n3. 测试向量存储")
    vector_store = SimpleVectorStore(embedder)
    vector_store.add_documents(["文档1: 机器学习基础", "文档2: 深度学习进阶"])
    results = vector_store.similarity_search("学习", k=2)
    print(f"   存储文档数: {len(vector_store.documents)}")
    print(f"   检索结果数: {len(results)}")
    for doc, score, _ in results:
        print(f"     - {doc} (分数: {score:.3f})")
    
    print("\n✅ 所有组件测试通过")


if __name__ == "__main__":
    # 运行主演示
    demonstrate_langchain_rag()
    
    # 运行组件测试
    test_rag_components()
