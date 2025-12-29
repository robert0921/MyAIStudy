"""
MyAIStudy 项目统一入口
选择运行入门版、进阶版或高级版学习系统

项目结构：
- 入门版（第1-12周）：Python基础 → 数据科学 → 机器学习 → 深度学习入门
- 进阶版（第1-12周）：深度学习理论 → 工程实践 → LLM专项训练
- 高级版（第13-18周）：LangChain/RAG → 向量数据库 → AI Agent → 服务化部署 🆕

使用方法：
    python run_example.py              # 交互式选择
    python run_example.py beginner     # 直接启动入门版
    python run_example.py intermediate # 直接启动进阶版
    python run_example.py advanced     # 直接启动高级版
    python run_example.py --help       # 查看帮助
"""

import sys
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def print_banner():
    """打印欢迎横幅"""
    banner = """
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║               🎓 MyAIStudy - AI学习24周实战计划                       ║
║                                                                       ║
║   从零开始，系统学习Python、数据科学、机器学习与深度学习               ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_menu():
    """打印主菜单"""
    menu = """
📚 请选择学习路线：

【1】入门版 - 12周基础实战计划
    ├─ 第1-4周：Python与数据科学基础
    ├─ 第5-8周：机器学习基础 (Scikit-Learn)
    └─ 第9-12周：深度学习入门 (PyTorch)
    
    适合人群：有编程基础，想系统学习AI的初学者
    输出成果：MNIST、CIFAR-10、文本分类等实战项目
    
【2】进阶版 - 12周深度提升计划
    ├─ 第1-4周：深度学习数学内核
    ├─ 第5-8周：工程级模型与高效训练
    └─ 第9-12周：LLM微调与推理优化
    
    适合人群：掌握深度学习基础，想深入理解原理的学习者
    输出成果：手写反向传播、模型压缩、LoRA微调、推理优化

【3】高级版 - RAG与智能体系统 (第13-18周) 🆕
    ├─ 第13-14周：LangChain与向量数据库
    ├─ 第15-16周：RAG优化与AI Agent
    └─ 第17-18周：服务化部署与系统监控
    
    适合人群：完成进阶版，想构建企业级AI应用的学习者
    输出成果：知识库问答系统、AI Agent、生产级API
    
【4】查看项目信息
【5】退出

═══════════════════════════════════════════════════════════════════════
    """
    print(menu)


def show_project_info():
    """显示项目信息"""
    info = """
📊 项目信息
═══════════════════════════════════════════════════════════════════════

📁 项目结构：
  ├── beginner/                      # 入门版代码（6个模块）
  ├── intermediate/                  # 进阶版代码（26个模块）
  ├── run_beginner_examples.py       # 入门版入口
  ├── run_intermediate_examples.py   # 进阶版入口
  └── docs/                          # 学习计划文档

📖 学习文档：
  - README.md                # 项目总览
  - README_BEGINNER.md       # 入门版详细指南
  - README_Intermediate.md   # 进阶版详细指南
  - docs/AI学习12周实战计划表（入门版）.md
  - docs/AI学习12周实战计划表（进阶版）.md

🔧 技术栈：
  - Python 3.8+
  - NumPy, Pandas, Matplotlib
  - Scikit-Learn
  - PyTorch

📊 项目统计：
  - 入门版：6个模块，覆盖12周内容
  - 进阶版：26个模块，覆盖12周内容
  - 总代码量：~220KB+
  - 测试覆盖率：88.9%

🔗 快速链接：
  - GitHub: https://github.com/robert0921/MyAIStudy
  - 文档: ./README.md

═══════════════════════════════════════════════════════════════════════
    """
    print(info)


def run_beginner():
    """运行入门版"""
    print("\n🚀 启动入门版学习系统...")
    print("═══════════════════════════════════════════════════════════════════════")
    
    beginner_script = PROJECT_ROOT / "run_beginner_examples.py"
    if not beginner_script.exists():
        print("❌ 错误：找不到 run_beginner_examples.py")
        print(f"   请确保文件存在于: {beginner_script}")
        return False
    
    # 使用subprocess运行
    try:
        result = subprocess.run(
            [sys.executable, str(beginner_script)],
            cwd=str(PROJECT_ROOT)
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 运行入门版时出错: {e}")
        print("\n💡 提示：你也可以直接运行:")
        print(f"   python run_beginner_examples.py")
        return False


def run_intermediate():
    """运行进阶版"""
    print("\n🚀 启动进阶版学习系统...")
    print("═══════════════════════════════════════════════════════════════════════")
    
    intermediate_script = PROJECT_ROOT / "run_intermediate_examples.py"
    if not intermediate_script.exists():
        print("❌ 错误：找不到 run_intermediate_examples.py")
        print(f"   请确保文件存在于: {intermediate_script}")
        return False
    
    # 使用subprocess运行
    try:
        result = subprocess.run(
            [sys.executable, str(intermediate_script)],
            cwd=str(PROJECT_ROOT)
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 运行进阶版时出错: {e}")
        print("\n💡 提示：你也可以直接运行:")
        print(f"   python run_intermediate_examples.py")
        return False


def run_advanced():
    """运行高级版"""
    print("\n🚀 启动高级版学习系统...")
    print("═══════════════════════════════════════════════════════════════════════")
    
    advanced_script = PROJECT_ROOT / "run_advanced_examples.py"
    if not advanced_script.exists():
        print("❌ 错误：找不到 run_advanced_examples.py")
        print(f"   请确保文件存在于: {advanced_script}")
        return False
    
    # 使用subprocess运行
    try:
        result = subprocess.run(
            [sys.executable, str(advanced_script)],
            cwd=str(PROJECT_ROOT)
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 运行高级版时出错: {e}")
        print("\n💡 提示：你也可以直接运行:")
        print(f"   python run_advanced_examples.py")
        return False


def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ['--help', '-h', 'help']:
            print(__doc__)
            return
        
        elif arg in ['beginner', 'b', '1']:
            print_banner()
            run_beginner()
            return
        
        elif arg in ['intermediate', 'i', 'adv', '2']:
            print_banner()
            run_intermediate()
            return
        
        elif arg in ['advanced', 'a', 'rag', '3']:
            print_banner()
            run_advanced()
            return
        
        else:
            print(f"❌ 未知参数: {arg}")
            print("\n使用方法:")
            print("  python run_example.py              # 交互式选择")
            print("  python run_example.py beginner     # 直接启动入门版")
            print("  python run_example.py intermediate # 直接启动进阶版")
            print("  python run_example.py advanced     # 直接启动高级版")
            print("  python run_example.py --help       # 查看帮助")
            return
    
    # 交互式菜单
    print_banner()
    
    while True:
        print_menu()
        
        try:
            choice = input("请输入选择 (1-4): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 感谢使用！")
            break
        
        if choice == '1':
            success = run_beginner()
            if success:
                print("\n✅ 入门版运行完成")
            input("\n按Enter键返回主菜单...")
            print("\n" * 2)
        
        elif choice == '2':
            success = run_intermediate()
            if success:
                print("\n✅ 进阶版运行完成")
            input("\n按Enter键返回主菜单...")
            print("\n" * 2)
        
        elif choice == '3':
            success = run_advanced()
            if success:
                print("\n✅ 高级版运行完成")
            input("\n按Enter键返回主菜单...")
            print("\n" * 2)
        
        elif choice == '4':
            show_project_info()
            input("\n按Enter键返回主菜单...")
            print("\n" * 2)
        
        elif choice == '5':
            print("\n👋 感谢使用MyAIStudy学习系统！")
            print("   继续努力，成为AI专家！🚀\n")
            break
        
        else:
            print(f"\n❌ 无效选择: {choice}")
            print("   请输入1-5之间的数字\n")


if __name__ == "__main__":
    main()
