"""
MyAIStudy v4.0 项目统一入口。

当前支持四条阶段化路线：
1. 入门版：Python / 数据科学 / 机器学习 / 深度学习入门
2. 进阶版：深度学习原理 / LLM 工程基础 / 训练与推理优化
3. 高级版：RAG / Agent / 科研化输出 / 职业化准备
4. 应用开发版：12 周企业落地冲刺，聚焦 RAG / Agent / 部署 / 工程效能

使用方法：
    python run_example.py                   # 交互式选择
    python run_example.py beginner          # 直接启动入门版
    python run_example.py intermediate      # 直接启动进阶版
    python run_example.py advanced          # 直接启动高级版
    python run_example.py application       # 直接启动应用开发版
    python run_example.py application quick # 参数透传给子入口
    python run_example.py --help            # 查看帮助
"""

import subprocess
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
SYSTEM_NAME = "MyAIStudy"
SYSTEM_VERSION = "4.0"
SYSTEM_TAGLINE = "48周整合主线 + 应用开发12周冲刺"

ROUTES = [
    {
        "menu_key": "1",
        "slug": "beginner",
        "aliases": ["beginner", "b", "1"],
        "label": "入门版",
        "script": "run_beginner_examples.py",
        "weeks": "第1-12周",
        "summary": [
            "第1-4周：Python与数据科学基础",
            "第5-8周：机器学习基础",
            "第9-12周：深度学习入门",
        ],
        "audience": "编程基础，AI零基础",
        "outcome": "数据分析报告、分类模型、深度学习入门项目",
    },
    {
        "menu_key": "2",
        "slug": "intermediate",
        "aliases": ["intermediate", "i", "2"],
        "label": "进阶版",
        "script": "run_intermediate_examples.py",
        "weeks": "第13-24周",
        "summary": [
            "第13-16周：深度学习数学内核",
            "第17-20周：工程级训练与压缩",
            "第21-24周：LLM架构、Prompt、推理优化",
        ],
        "audience": "想深入理解原理与工程实现的学习者",
        "outcome": "手写反向传播、LoRA微调、KV Cache、训练系统",
    },
    {
        "menu_key": "3",
        "slug": "advanced",
        "aliases": ["advanced", "a", "3"],
        "label": "高级版",
        "script": "run_advanced_examples.py",
        "weeks": "第25-36周",
        "summary": [
            "第25-30周：RAG、Agent、服务化与监控",
            "第31-36周：科研化输出、知识管理、职业化准备",
        ],
        "audience": "想形成系统化 AI 工程能力的学习者",
        "outcome": "知识库问答、论文工具、技术白皮书、项目展示",
    },
    {
        "menu_key": "4",
        "slug": "application",
        "aliases": ["application", "app", "4"],
        "label": "应用开发版",
        "script": "run_application_examples.py",
        "weeks": "12周企业级大模型应用实战",
        "summary": [
            "第1-4周：Prompt、Embedding、RAG 工程",
            "第5-8周：MCP、Agent、框架与工作流",
            "第9-12周：微调、部署、工程效能、综合项目",
        ],
        "audience": "已有基础，想快速做出企业 AI 应用原型",
        "outcome": "轻量 RAG + Agent + 部署演示链路",
    },
]


def build_alias_map():
    alias_map = {}
    for route in ROUTES:
        for alias in route["aliases"]:
            alias_map[alias] = route
    return alias_map


ALIAS_MAP = build_alias_map()


def print_banner():
    """打印欢迎横幅"""
    banner = f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║          🎓 {SYSTEM_NAME} v{SYSTEM_VERSION} - {SYSTEM_TAGLINE}          ║
║                                                                       ║
║      从 Python 基础到企业级 AI 应用交付的完整学习与演示入口           ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_menu():
    """打印主菜单"""
    print("\n📚 请选择学习路线：\n")
    for route in ROUTES:
        print(f"【{route['menu_key']}】{route['label']} - {route['weeks']}")
        for item in route["summary"]:
            print(f"    ├─ {item}")
        print("")
        print(f"    适合人群：{route['audience']}")
        print(f"    输出成果：{route['outcome']}\n")

    print("【5】查看项目信息")
    print("【6】退出\n")
    print("═══════════════════════════════════════════════════════════════════════")


def show_project_info():
    """显示项目信息"""
    route_names = " / ".join(route["label"] for route in ROUTES)
    info = f"""
📊 项目信息
═══════════════════════════════════════════════════════════════════════

🧩 系统版本：
  - {SYSTEM_NAME} v{SYSTEM_VERSION}
  - 总入口定位：{SYSTEM_TAGLINE}
  - 支持路线：{route_names}

📁 核心目录：
  ├── beginner/                    # 入门版代码
  ├── intermediate/                # 进阶版代码
  ├── advanced/                    # 高级版代码
  ├── application/                 # 应用开发版代码
  ├── run_example.py               # 统一入口
  ├── run_beginner_examples.py     # 入门版入口
  ├── run_intermediate_examples.py # 进阶版入口
  ├── run_advanced_examples.py     # 高级版入口
  └── run_application_examples.py  # 应用开发版入口

📖 学习文档：
  - README.md
  - README_beginner.md
  - README_intermediate.md
  - README_advanced.md
  - README_application.md
  - AI学习48周实战计划表（整合版）.md

🔧 主要技术栈：
  - Python 3.8+
  - NumPy / Pandas / Matplotlib
  - Scikit-Learn / PyTorch
  - LangChain / FastAPI / 向量检索 / 推理部署（进阶使用）

🧭 使用建议：
  - 想按一条主线长期推进：优先看整合版 48 周计划
  - 想快速做业务原型：直接运行应用开发版
  - 想从统一入口直达具体演示：使用参数透传，例如 python run_example.py application quick

🔗 快速链接：
  - GitHub: https://github.com/robert0921/MyAIStudy
  - 主文档: ./README.md

═══════════════════════════════════════════════════════════════════════
    """
    print(info)


def run_route(route, forwarded_args=None):
    """运行指定路线。"""
    forwarded_args = forwarded_args or []

    print(f"\n🚀 启动{route['label']}学习系统...")
    print("═══════════════════════════════════════════════════════════════════════")

    script_path = PROJECT_ROOT / route["script"]
    if not script_path.exists():
        print(f"❌ 错误：找不到 {route['script']}")
        print(f"   请确保文件存在于: {script_path}")
        return False

    try:
        command = [sys.executable, str(script_path), *forwarded_args]
        result = subprocess.run(command, cwd=str(PROJECT_ROOT))
        return result.returncode == 0
    except Exception as error:
        print(f"❌ 运行{route['label']}时出错: {error}")
        print("\n💡 提示：你也可以直接运行:")
        print(f"   python {route['script']} {' '.join(forwarded_args)}".rstrip())
        return False


def print_usage():
    print("\n使用方法:")
    print("  python run_example.py                   # 交互式选择")
    for route in ROUTES:
        print(f"  python run_example.py {route['slug']:<12} # 直接启动{route['label']}")
    print("  python run_example.py application quick # 参数透传到子入口")
    print("  python run_example.py --help            # 查看帮助")


def main():
    """主函数"""
    if len(sys.argv) > 1:
        argument = sys.argv[1].lower()

        if argument in ["--help", "-h", "help"]:
            print(__doc__)
            return

        route = ALIAS_MAP.get(argument)
        if route is not None:
            print_banner()
            run_route(route, sys.argv[2:])
            return

        print(f"❌ 未知参数: {argument}")
        print_usage()
        return

    print_banner()

    while True:
        print_menu()

        try:
            choice = input("请输入选择 (1-6): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 感谢使用！")
            break

        route = ALIAS_MAP.get(choice)
        if route is not None:
            success = run_route(route)
            if success:
                print(f"\n✅ {route['label']}运行完成")
            input("\n按Enter键返回主菜单...")
            print("\n" * 2)
            continue

        if choice == "5":
            show_project_info()
            input("\n按Enter键返回主菜单...")
            print("\n" * 2)
            continue

        if choice == "6":
            print(f"\n👋 感谢使用 {SYSTEM_NAME} v{SYSTEM_VERSION} 学习系统！")
            print("   继续推进你的 AI 工程主线。\n")
            break

        print(f"\n❌ 无效选择: {choice}")
        print("   请输入 1-6 之间的数字\n")


if __name__ == "__main__":
    main()
