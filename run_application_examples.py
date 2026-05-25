"""
MyAIStudy 应用开发版学习系统入口 v4.0。

12周应用开发实战计划：
1. 第1-4周：Prompt、Embedding、RAG 工程与评估
2. 第5-8周：Function Calling、MCP、Agent 与低代码工作流
3. 第9-12周：微调、高并发部署、工程效能与综合项目

使用方法：
    python run_application_examples.py
    python run_application_examples.py week1
    python run_application_examples.py week5-8
    python run_application_examples.py quick
    python run_application_examples.py all
    python run_example.py application quick
"""

from __future__ import annotations

import sys
from pathlib import Path


def configure_output_encoding() -> None:
    """Ensure Chinese output renders correctly on Windows terminals."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


configure_output_encoding()

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from application.week1_4_rag_engineering import (  # noqa: E402
    demo_week1_prompt_engineering,
    demo_week2_vector_retrieval,
    demo_week3_rag_pipeline,
    demo_week4_rag_evaluation,
    run_stage1_demo,
)
from application.week5_8_agent_workflows import (  # noqa: E402
    demo_week5_function_calling_and_mcp,
    demo_week6_agent_architecture,
    demo_week7_frameworks,
    demo_week8_low_code_workflow,
    run_stage2_demo,
)
from application.week9_12_delivery import (  # noqa: E402
    demo_week9_finetuning,
    demo_week10_deployment,
    demo_week11_engineering_efficiency,
    demo_week12_integrated_project,
    run_stage3_demo,
)


def print_banner() -> None:
    banner = """
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║          🏢 MyAIStudy 应用开发版 v4.0 - 企业级大模型应用实战          ║
║                                                                       ║
║          从 RAG 工程到 Agent 部署，用 12 周做一条可演示链路           ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_menu() -> None:
    menu = """
📚 应用开发版学习模块：

📌 阶段一：大模型基础与 RAG 工程（第1-4周）
【1】第1周 - Prompt / Context Engineering
【2】第2周 - Embedding 与向量检索
【3】第3周 - RAG Pipeline 搭建
【4】第4周 - 检索评估与效果量化

🤖 阶段二：Agent 架构与开发框架（第5-8周）
【5】第5周 - Function Calling 与 MCP
【6】第6周 - Agent 规划、记忆与 ReAct
【7】第7周 - LangChain / LlamaIndex / AutoGen 定位
【8】第8周 - Coze / Dify 工作流思路

🚀 阶段三：微调、部署与工程效能（第9-12周）
【9】第9周 - LoRA / QLoRA 微调策略
【10】第10周 - vLLM / SGLang / Ollama 部署对比
【11】第11周 - Spec Coding 与 ChatBI
【12】第12周 - 综合项目演示

【13】快速演示 - 每个阶段各看一个代表模块
【14】完整演示 - 运行全部12周内容
【15】退出

═══════════════════════════════════════════════════════════════════════
    """
    print(menu)


def run_quick_demo() -> None:
    demo_week1_prompt_engineering()
    demo_week6_agent_architecture()
    demo_week10_deployment()


def run_all() -> None:
    run_stage1_demo()
    run_stage2_demo()
    run_stage3_demo()


def run_by_argument(argument: str) -> bool:
    normalized = argument.lower()
    mapping = {
        "week1": demo_week1_prompt_engineering,
        "week2": demo_week2_vector_retrieval,
        "week3": demo_week3_rag_pipeline,
        "week4": demo_week4_rag_evaluation,
        "week5": demo_week5_function_calling_and_mcp,
        "week6": demo_week6_agent_architecture,
        "week7": demo_week7_frameworks,
        "week8": demo_week8_low_code_workflow,
        "week9": demo_week9_finetuning,
        "week10": demo_week10_deployment,
        "week11": demo_week11_engineering_efficiency,
        "week12": demo_week12_integrated_project,
        "week1-4": run_stage1_demo,
        "stage1": run_stage1_demo,
        "week5-8": run_stage2_demo,
        "stage2": run_stage2_demo,
        "week9-12": run_stage3_demo,
        "stage3": run_stage3_demo,
        "quick": run_quick_demo,
        "all": run_all,
    }

    if normalized not in mapping:
        return False

    mapping[normalized]()
    return True


def main() -> None:
    if len(sys.argv) > 1:
        argument = sys.argv[1]
        if argument in {"--help", "-h", "help"}:
            print(__doc__)
            return
        if run_by_argument(argument):
            return
        print(f"❌ 未知参数: {argument}")
        print("\n使用方法:")
        print("  python run_application_examples.py              # 交互式选择")
        print("  python run_application_examples.py week1        # 运行第1周")
        print("  python run_application_examples.py week5-8      # 运行阶段二")
        print("  python run_application_examples.py quick        # 快速演示")
        print("  python run_application_examples.py all          # 完整演示")
        return

    print_banner()

    while True:
        print_menu()
        try:
            choice = input("请输入选择 (1-15): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 感谢使用应用开发版学习系统！")
            break

        action_map = {
            "1": demo_week1_prompt_engineering,
            "2": demo_week2_vector_retrieval,
            "3": demo_week3_rag_pipeline,
            "4": demo_week4_rag_evaluation,
            "5": demo_week5_function_calling_and_mcp,
            "6": demo_week6_agent_architecture,
            "7": demo_week7_frameworks,
            "8": demo_week8_low_code_workflow,
            "9": demo_week9_finetuning,
            "10": demo_week10_deployment,
            "11": demo_week11_engineering_efficiency,
            "12": demo_week12_integrated_project,
            "13": run_quick_demo,
            "14": run_all,
        }

        if choice == "15":
            print("\n👋 感谢使用应用开发版学习系统！")
            break

        action = action_map.get(choice)
        if action is None:
            print(f"\n❌ 无效选择: {choice}")
            print("   请输入 1-15 之间的数字\n")
            continue

        action()
        input("\n按 Enter 键返回主菜单...")
        print("\n" * 2)


if __name__ == "__main__":
    main()