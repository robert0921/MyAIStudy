"""
应用开发版阶段一：大模型基础与 RAG 工程。

目标：
1. 用最少依赖演示 Prompt / Context Engineering。
2. 用稀疏向量模拟 Embedding + Vector Search。
3. 构建一个简化版 RAG Pipeline。
4. 给出检索评估的基本指标。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import sqrt
import re
from typing import Dict, Iterable, List, Sequence, Tuple


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def tokenize(text: str) -> List[str]:
    normalized = text.lower()
    return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]{1,}", normalized)


def build_sparse_embedding(text: str) -> Counter:
    return Counter(tokenize(text))


def cosine_similarity(left_vector: Counter, right_vector: Counter) -> float:
    if not left_vector or not right_vector:
        return 0.0

    common_tokens = set(left_vector) & set(right_vector)
    dot_product = sum(left_vector[token] * right_vector[token] for token in common_tokens)
    left_norm = sqrt(sum(value * value for value in left_vector.values()))
    right_norm = sqrt(sum(value * value for value in right_vector.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


@dataclass
class Document:
    document_id: str
    title: str
    content: str
    category: str


SAMPLE_DOCUMENTS: List[Document] = [
    Document(
        document_id="doc-001",
        title="Prompt Engineering 基础",
        category="prompt",
        content=(
            "Prompt Engineering 强调任务指令清晰、输出格式明确、角色约束稳定。"
            "在企业场景中，常见做法是要求模型先给结论，再列证据和风险。"
            "如果任务依赖外部事实，应结合检索结果或工具调用降低幻觉。"
        ),
    ),
    Document(
        document_id="doc-002",
        title="Embedding 与向量检索",
        category="retrieval",
        content=(
            "Embedding 用于把文本映射到向量空间。向量数据库常见能力包括索引、过滤、"
            "近似最近邻搜索和元数据管理。企业选型时通常同时关注召回率、延迟和成本。"
        ),
    ),
    Document(
        document_id="doc-003",
        title="RAG Pipeline",
        category="rag",
        content=(
            "RAG 系统一般包括文档解析、清洗、分块、向量化、检索、重排序和答案生成。"
            "如果 Chunking 太碎，会损失上下文；如果太大，会拉高噪声和推理成本。"
        ),
    ),
    Document(
        document_id="doc-004",
        title="混合检索与评估",
        category="evaluation",
        content=(
            "混合检索将关键词召回与向量召回结合，再通过 rerank 选出更优结果。"
            "评估时常用 hit rate、precision@k、回答准确率和延迟统计。"
        ),
    ),
]


class PromptWorkbench:
    """用静态规则模拟 Prompt 设计实验。"""

    def __init__(self) -> None:
        self.candidates = [
            (
                "基础提示",
                "请回答用户问题：{question}",
            ),
            (
                "结构化提示",
                "你是一名企业 AI 顾问。请先给结论，再给证据，最后给下一步建议。问题：{question}",
            ),
            (
                "带约束提示",
                "你是一名企业 AI 顾问。请输出 JSON 风格结果，包含结论、证据、风险、下一步。问题：{question}",
            ),
        ]

    def mock_response(self, prompt_name: str, question: str) -> str:
        if prompt_name == "基础提示":
            return f"回答：{question}，建议优先尝试 RAG。"
        if prompt_name == "结构化提示":
            return (
                "结论：先做小规模 RAG 验证。\n"
                "证据：该任务依赖企业私有知识，检索增强比纯提示更稳。\n"
                "下一步：准备文档、做分块和召回测试。"
            )
        return (
            "{\n"
            '  "结论": "先做小规模 RAG 验证",\n'
            '  "证据": ["问题依赖私有知识", "可先用向量检索验证召回"],\n'
            '  "风险": ["分块过粗会降低召回", "评估数据不足会误判效果"],\n'
            '  "下一步": ["建立测试集", "记录 precision@3 与延迟"]\n'
            "}"
        )

    def score_response(self, response: str) -> int:
        rubric_tokens = ["结论", "证据", "风险", "下一步"]
        return sum(1 for token in rubric_tokens if token in response)

    def run_experiment(self, question: str) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        for prompt_name, template in self.candidates:
            prompt = template.format(question=question)
            response = self.mock_response(prompt_name, question)
            results.append(
                {
                    "prompt_name": prompt_name,
                    "prompt": prompt,
                    "response": response,
                    "score": self.score_response(response),
                }
            )
        results.sort(key=lambda item: item["score"], reverse=True)
        return results


class FixedChunker:
    def __init__(self, chunk_size: int = 70, overlap: int = 10) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> List[str]:
        chunks: List[str] = []
        cursor = 0
        while cursor < len(text):
            end = min(len(text), cursor + self.chunk_size)
            chunk = text[cursor:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(text):
                break
            cursor = max(0, end - self.overlap)
        return chunks


class SimpleVectorStore:
    def __init__(self) -> None:
        self.items: List[Tuple[Document, Counter]] = []

    def add_documents(self, documents: Iterable[Document]) -> None:
        for document in documents:
            self.items.append((document, build_sparse_embedding(document.content)))

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        query_vector = build_sparse_embedding(query)
        ranked_results = []
        for document, embedding in self.items:
            similarity = cosine_similarity(query_vector, embedding)
            ranked_results.append((document, similarity))
        ranked_results.sort(key=lambda item: item[1], reverse=True)
        return ranked_results[:top_k]


class SimpleRAGPipeline:
    def __init__(self, vector_store: SimpleVectorStore, chunker: FixedChunker) -> None:
        self.vector_store = vector_store
        self.chunker = chunker

    def ingest(self, documents: Sequence[Document]) -> List[Document]:
        chunk_documents: List[Document] = []
        for document in documents:
            for index, chunk in enumerate(self.chunker.split(document.content), start=1):
                chunk_documents.append(
                    Document(
                        document_id=f"{document.document_id}-chunk-{index}",
                        title=document.title,
                        content=chunk,
                        category=document.category,
                    )
                )
        self.vector_store.add_documents(chunk_documents)
        return chunk_documents

    def answer(self, query: str, top_k: int = 2) -> Dict[str, object]:
        hits = self.vector_store.search(query, top_k=top_k)
        contexts = [document.content for document, _score in hits]
        if not contexts:
            return {
                "query": query,
                "contexts": [],
                "answer": "没有找到可用上下文，建议先补充知识库。",
            }

        summary = "；".join(contexts)
        return {
            "query": query,
            "contexts": contexts,
            "answer": (
                f"基于检索结果，建议围绕“{query}”先完成文档解析、召回验证和答案评估。"
                f"关键依据：{summary}"
            ),
        }


def evaluate_retrieval(vector_store: SimpleVectorStore) -> Dict[str, float]:
    evaluation_set = [
        ("如何降低 RAG 幻觉", "doc-004"),
        ("向量数据库选型看什么", "doc-002"),
        ("什么是 Prompt Engineering", "doc-001"),
    ]

    hits = 0
    reciprocal_ranks: List[float] = []
    for query, expected_prefix in evaluation_set:
        ranked_results = vector_store.search(query, top_k=3)
        rank = 0
        for index, (document, _score) in enumerate(ranked_results, start=1):
            if document.document_id.startswith(expected_prefix):
                rank = index
                hits += 1
                reciprocal_ranks.append(1.0 / index)
                break
        if rank == 0:
            reciprocal_ranks.append(0.0)

    total = len(evaluation_set)
    return {
        "hit_rate_at_3": round(hits / total, 2),
        "mrr_at_3": round(sum(reciprocal_ranks) / total, 2),
    }


def demo_week1_prompt_engineering() -> List[Dict[str, object]]:
    print_header("第1周：Prompt Engineering 与 Context Engineering")
    workbench = PromptWorkbench()
    question = "我要为企业客服搭建知识库问答系统，第一步应该做什么？"
    results = workbench.run_experiment(question)
    for item in results:
        print(f"\n[{item['prompt_name']}] 评分: {item['score']}")
        print(f"Prompt: {item['prompt']}")
        print(f"Response: {item['response']}")
    return results


def demo_week2_vector_retrieval() -> List[Tuple[Document, float]]:
    print_header("第2周：Embedding 原理与向量数据库选型")
    vector_store = SimpleVectorStore()
    vector_store.add_documents(SAMPLE_DOCUMENTS)
    results = vector_store.search("企业 RAG 评估与向量检索怎么做", top_k=3)
    for document, score in results:
        print(f"- {document.title} ({document.document_id}) -> similarity={score:.3f}")
    return results


def demo_week3_rag_pipeline() -> Dict[str, object]:
    print_header("第3周：RAG 核心流程与本地知识库搭建")
    chunker = FixedChunker(chunk_size=45, overlap=8)
    vector_store = SimpleVectorStore()
    pipeline = SimpleRAGPipeline(vector_store=vector_store, chunker=chunker)
    chunks = pipeline.ingest(SAMPLE_DOCUMENTS)
    print(f"已导入文档块数量: {len(chunks)}")
    result = pipeline.answer("如何设计企业知识库问答系统", top_k=2)
    print(f"Query: {result['query']}")
    print("Contexts:")
    for context in result["contexts"]:
        print(f"  - {context}")
    print(f"Answer: {result['answer']}")
    return result


def demo_week4_rag_evaluation() -> Dict[str, float]:
    print_header("第4周：混合检索、Reranking 与 RAG 评估")
    vector_store = SimpleVectorStore()
    vector_store.add_documents(SAMPLE_DOCUMENTS)
    metrics = evaluate_retrieval(vector_store)
    print(f"hit_rate@3: {metrics['hit_rate_at_3']}")
    print(f"mrr@3: {metrics['mrr_at_3']}")
    return metrics


def run_stage1_demo() -> Dict[str, object]:
    return {
        "week1": demo_week1_prompt_engineering(),
        "week2": demo_week2_vector_retrieval(),
        "week3": demo_week3_rag_pipeline(),
        "week4": demo_week4_rag_evaluation(),
    }


if __name__ == "__main__":
    run_stage1_demo()