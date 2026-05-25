"""
应用开发版阶段二：Agent 架构与开发框架实战。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable, Dict, List


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


@dataclass
class Tool:
    name: str
    description: str
    handler: Callable[..., Dict[str, object]]


class ToolRegistry:
    def __init__(self) -> None:
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def list_tools(self) -> List[Dict[str, str]]:
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self.tools.values()
        ]

    def call(self, tool_name: str, **kwargs) -> Dict[str, object]:
        tool = self.tools[tool_name]
        return tool.handler(**kwargs)


class ConversationMemory:
    def __init__(self) -> None:
        self.messages: List[str] = []

    def add(self, message: str) -> None:
        self.messages.append(message)

    def summary(self) -> str:
        if not self.messages:
            return "暂无历史上下文。"
        return " | ".join(self.messages[-4:])


class SimpleMCPServer:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def handle(self, request: Dict[str, object]) -> Dict[str, object]:
        method = request["method"]
        if method == "tools/list":
            return {"tools": self.registry.list_tools()}
        if method == "tools/call":
            params = request.get("params", {})
            tool_name = params["name"]
            arguments = params.get("arguments", {})
            return {"result": self.registry.call(tool_name, **arguments)}
        return {"error": f"unsupported method: {method}"}


class SimpleReActAgent:
    def __init__(self, registry: ToolRegistry, memory: ConversationMemory) -> None:
        self.registry = registry
        self.memory = memory

    def run(self, task: str) -> Dict[str, object]:
        steps: List[Dict[str, object]] = []
        self.memory.add(f"user:{task}")

        if "天气" in task:
            weather_result = self.registry.call("weather_lookup", city="上海")
            steps.append({"thought": "任务需要外部天气信息", "action": "weather_lookup", "observation": weather_result})

        if "订单" in task:
            order_result = self.registry.call("order_lookup", order_id="A10086")
            steps.append({"thought": "任务涉及订单状态", "action": "order_lookup", "observation": order_result})

        if "知识库" in task or "RAG" in task or not steps:
            kb_result = self.registry.call("knowledge_search", topic="RAG")
            steps.append({"thought": "需要查询知识库", "action": "knowledge_search", "observation": kb_result})

        final_answer = {
            "summary": "已根据任务调用所需工具，并整理执行结果。",
            "steps": steps,
            "memory": self.memory.summary(),
        }
        self.memory.add(f"assistant:{final_answer['summary']}")
        return final_answer


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()

    def weather_lookup(city: str) -> Dict[str, object]:
        return {"city": city, "temperature": 26, "condition": "多云", "advice": "适合安排线下客户沟通"}

    def order_lookup(order_id: str) -> Dict[str, object]:
        return {"order_id": order_id, "status": "已发货", "eta": "2026-05-27", "risk": "无异常"}

    def knowledge_search(topic: str) -> Dict[str, object]:
        knowledge_base = {
            "RAG": "RAG 适合依赖私有知识的问答与助手类应用，应优先验证召回质量。",
            "MCP": "MCP 提供 Host、Client、Server 的标准化协作方式，适合工具扩展。",
        }
        return {"topic": topic, "summary": knowledge_base.get(topic, "暂无相关知识")}

    registry.register(Tool("weather_lookup", "查询城市天气", weather_lookup))
    registry.register(Tool("order_lookup", "查询订单状态", order_lookup))
    registry.register(Tool("knowledge_search", "查询内部知识库", knowledge_search))
    return registry


def demo_week5_function_calling_and_mcp() -> Dict[str, object]:
    print_header("第5周：Function Calling 与 MCP")
    registry = build_default_registry()
    server = SimpleMCPServer(registry)

    tools_response = server.handle({"method": "tools/list"})
    call_response = server.handle(
        {
            "method": "tools/call",
            "params": {"name": "weather_lookup", "arguments": {"city": "北京"}},
        }
    )

    print("可用工具:")
    print(json.dumps(tools_response, ensure_ascii=False, indent=2))
    print("\nMCP 工具调用结果:")
    print(json.dumps(call_response, ensure_ascii=False, indent=2))
    return {"tools": tools_response, "call": call_response}


def demo_week6_agent_architecture() -> Dict[str, object]:
    print_header("第6周：Agent 规划、记忆与 ReAct")
    registry = build_default_registry()
    memory = ConversationMemory()
    agent = SimpleReActAgent(registry, memory)
    result = agent.run("请帮我整理一个企业知识库客服方案，并确认上海今天适不适合做客户拜访")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def demo_week7_frameworks() -> List[Dict[str, object]]:
    print_header("第7周：LangChain / LlamaIndex / AutoGen 框架定位")
    comparison = [
        {"framework": "LangChain", "strength": "工具编排与链式流程", "best_for": "业务工作流与 Agent"},
        {"framework": "LlamaIndex", "strength": "数据摄取与检索链路", "best_for": "知识库与 RAG"},
        {"framework": "AutoGen", "strength": "多智能体协作", "best_for": "复杂任务拆解与角色分工"},
    ]
    for item in comparison:
        print(f"- {item['framework']}: {item['strength']} | 适用场景: {item['best_for']}")
    return comparison


def demo_week8_low_code_workflow() -> Dict[str, object]:
    print_header("第8周：Coze / Dify 低代码工作流思路")

    def detect_intent(user_message: str) -> str:
        if "订单" in user_message:
            return "order_service"
        if "知识库" in user_message or "文档" in user_message:
            return "knowledge_service"
        return "general_service"

    workflow_context = {
        "user_message": "帮我回答客户关于知识库接入周期的问题",
    }
    workflow_context["intent"] = detect_intent(workflow_context["user_message"])
    workflow_context["response"] = (
        "知识库接入通常先做文档清洗、分块与召回测试，"
        "POC 周期一般控制在 2 到 3 周。"
    )

    print(json.dumps(workflow_context, ensure_ascii=False, indent=2))
    return workflow_context


def run_stage2_demo() -> Dict[str, object]:
    return {
        "week5": demo_week5_function_calling_and_mcp(),
        "week6": demo_week6_agent_architecture(),
        "week7": demo_week7_frameworks(),
        "week8": demo_week8_low_code_workflow(),
    }


if __name__ == "__main__":
    run_stage2_demo()