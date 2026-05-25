"""
应用开发版阶段三：模型微调、高并发部署与工程效能。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Dict, List

from application.week1_4_rag_engineering import (
    Document,
    FixedChunker,
    SAMPLE_DOCUMENTS,
    SimpleRAGPipeline,
    SimpleVectorStore,
)
from application.week5_8_agent_workflows import (
    ConversationMemory,
    SimpleReActAgent,
    build_default_registry,
)


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


@dataclass
class FinetuningMethod:
    name: str
    trainable_ratio: float
    memory_ratio: float
    deployment_note: str


class FinetuningPlanner:
    def compare(self) -> List[FinetuningMethod]:
        return [
            FinetuningMethod("Full Fine-Tuning", 100.0, 100.0, "效果最强，但成本最高"),
            FinetuningMethod("LoRA", 0.8, 28.0, "适合业务 POC 和快速迭代"),
            FinetuningMethod("QLoRA", 0.8, 18.0, "显存更低，适合个人设备实验"),
        ]


class DeploymentBenchmarker:
    def compare(self, concurrent_requests: int) -> List[Dict[str, object]]:
        base_profiles = {
            "vLLM": {"ttft_ms": 650, "tokens_per_second": 1850, "memory_gb": 18.5},
            "SGLang": {"ttft_ms": 590, "tokens_per_second": 1720, "memory_gb": 17.8},
            "Ollama": {"ttft_ms": 1100, "tokens_per_second": 860, "memory_gb": 13.5},
        }
        results: List[Dict[str, object]] = []
        for framework, profile in base_profiles.items():
            scale = max(1, concurrent_requests / 8)
            results.append(
                {
                    "framework": framework,
                    "ttft_ms": int(profile["ttft_ms"] * scale),
                    "tokens_per_second": int(profile["tokens_per_second"] / scale),
                    "memory_gb": round(profile["memory_gb"] * (1 + concurrent_requests / 64), 1),
                }
            )
        return results


class SpecCodingAssistant:
    def generate_checklist(self, feature_name: str) -> Dict[str, List[str]]:
        return {
            "feature": [feature_name],
            "functional": [
                "定义输入输出契约",
                "列出异常分支",
                "给出最小验收用例",
            ],
            "non_functional": [
                "明确延迟目标",
                "明确监控指标",
                "明确回滚策略",
            ],
            "tests": [
                "Happy path 冒烟测试",
                "边界输入测试",
                "错误恢复测试",
            ],
        }


class SimpleSQLCopilot:
    def generate_sql(self, question: str) -> str:
        if "订单" in question and "本周" in question:
            return (
                "SELECT order_status, COUNT(*) AS total_orders "
                "FROM orders WHERE created_at >= CURRENT_DATE - INTERVAL '7 days' "
                "GROUP BY order_status ORDER BY total_orders DESC;"
            )
        if "活跃用户" in question:
            return (
                "SELECT DATE(event_time) AS event_date, COUNT(DISTINCT user_id) AS active_users "
                "FROM user_events GROUP BY DATE(event_time) ORDER BY event_date DESC;"
            )
        return "SELECT * FROM business_metrics LIMIT 10;"


class EnterpriseAIAssistant:
    def __init__(self) -> None:
        chunker = FixedChunker(chunk_size=48, overlap=8)
        vector_store = SimpleVectorStore()
        self.pipeline = SimpleRAGPipeline(vector_store=vector_store, chunker=chunker)
        self.pipeline.ingest(
            SAMPLE_DOCUMENTS
            + [
                Document(
                    document_id="doc-101",
                    title="企业部署建议",
                    category="deployment",
                    content=(
                        "企业部署要同时考虑 TTFT、吞吐量、监控指标和回滚策略。"
                        "上线前至少准备健康检查、告警规则和容量估算。"
                    ),
                )
            ]
        )
        self.agent = SimpleReActAgent(build_default_registry(), ConversationMemory())

    def answer(self, request: str) -> Dict[str, object]:
        if "订单" in request or "天气" in request:
            return {"mode": "agent", "result": self.agent.run(request)}
        return {"mode": "rag", "result": self.pipeline.answer(request, top_k=2)}


def demo_week9_finetuning() -> List[FinetuningMethod]:
    print_header("第9周：LoRA / QLoRA / PEFT 微调策略")
    planner = FinetuningPlanner()
    results = planner.compare()
    for item in results:
        print(
            f"- {item.name}: trainable={item.trainable_ratio:.1f}% | "
            f"memory={item.memory_ratio:.1f}% | {item.deployment_note}"
        )
    return results


def demo_week10_deployment() -> List[Dict[str, object]]:
    print_header("第10周：vLLM / SGLang / Ollama 高并发部署")
    benchmarker = DeploymentBenchmarker()
    results = benchmarker.compare(concurrent_requests=16)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return results


def demo_week11_engineering_efficiency() -> Dict[str, object]:
    print_header("第11周：Spec Coding、测试与 ChatBI")
    assistant = SpecCodingAssistant()
    sql_copilot = SimpleSQLCopilot()
    checklist = assistant.generate_checklist("企业知识库客服")
    sql_query = sql_copilot.generate_sql("统计本周订单状态分布")
    result = {"checklist": checklist, "sample_sql": sql_query}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def demo_week12_integrated_project() -> Dict[str, object]:
    print_header("第12周：企业级 RAG + Agent + 部署综合项目")
    assistant = EnterpriseAIAssistant()
    knowledge_response = assistant.answer("怎样规划企业知识库问答系统的上线步骤")
    agent_response = assistant.answer("帮我确认订单状态，并给出客户沟通建议")
    result = {
        "architecture": ["document_ingestion", "retrieval", "agent_tools", "service_layer", "monitoring"],
        "knowledge_response": knowledge_response,
        "agent_response": agent_response,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def run_stage3_demo() -> Dict[str, object]:
    return {
        "week9": demo_week9_finetuning(),
        "week10": demo_week10_deployment(),
        "week11": demo_week11_engineering_efficiency(),
        "week12": demo_week12_integrated_project(),
    }


if __name__ == "__main__":
    run_stage3_demo()