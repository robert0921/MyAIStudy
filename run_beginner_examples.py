"""
MyAIStudy 入门版学习系统入口 v4.0
对应整合版第1-12周：Python、数据科学与机器学习基础

运行方式:
    python run_beginner_examples.py           # 交互菜单
    python run_beginner_examples.py week1     # 运行第1周
    python run_beginner_examples.py stage1    # 运行阶段1（第1-4周）
    python run_beginner_examples.py all       # 运行全部
"""
import sys
import time


def configure_output_encoding() -> None:
    """Ensure Chinese output renders correctly on Windows terminals."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


configure_output_encoding()

# 检查依赖
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("Warning: numpy not available. Please install: pip install numpy")
    NUMPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    print("Warning: pandas not available. Please install: pip install pandas")
    PANDAS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib not available. Please install: pip install matplotlib")
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    print("Warning: seaborn not available. Please install: pip install seaborn")
    SEABORN_AVAILABLE = False

try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available. Please install: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not available. Please install: pip install torch")
    TORCH_AVAILABLE = False

# 导入各周模块
try:
    from beginner.week1_python_basics import demonstrate_python_basics
    WEEK1_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Week 1 module not available: {e}")
    WEEK1_AVAILABLE = False

try:
    from beginner.week2_numpy_operations import (
        demonstrate_numpy_basics, 
        demonstrate_image_processing,
        demonstrate_matrix_operations
    )
    WEEK2_AVAILABLE = NUMPY_AVAILABLE
except ImportError:
    WEEK2_AVAILABLE = False

try:
    from beginner.week3_pandas_analysis import (
        demonstrate_pandas_basics,
        demonstrate_data_cleaning,
        analyze_ecommerce_data
    )
    WEEK3_AVAILABLE = PANDAS_AVAILABLE and NUMPY_AVAILABLE
except ImportError:
    WEEK3_AVAILABLE = False

try:
    from beginner.week4_visualization import (
        demonstrate_matplotlib_basics,
        demonstrate_sales_visualization,
        demonstrate_seaborn_advanced
    )
    WEEK4_AVAILABLE = MATPLOTLIB_AVAILABLE and SEABORN_AVAILABLE and PANDAS_AVAILABLE
except ImportError:
    WEEK4_AVAILABLE = False

try:
    from beginner.week5_8_machine_learning import (
        week5_classification,
        week6_regression,
        week7_clustering,
        week8_model_tuning
    )
    WEEK5_8_AVAILABLE = SKLEARN_AVAILABLE and NUMPY_AVAILABLE
except ImportError:
    WEEK5_8_AVAILABLE = False

try:
    from beginner.week9_12_deep_learning import (
        week9_mnist,
        week10_cifar10,
        week11_text_classification,
        week12_comprehensive_project
    )
    WEEK9_12_AVAILABLE = TORCH_AVAILABLE and NUMPY_AVAILABLE
except ImportError:
    WEEK9_12_AVAILABLE = False


def print_banner():
    """打印欢迎横幅"""
    print("\n" + "="*70)
    print("🎓 初学者AI学习路线 - 12周实战计划 v1.0")
    print("="*70)
    print("\n📚 学习路线概览:")
    print("  阶段1 (第1-4周): Python与数据科学基础")
    print("  阶段2 (第5-8周): 机器学习基础与Scikit-Learn")
    print("  阶段3 (第9-12周): 深度学习入门与PyTorch")
    print("\n💡 目标: 从编程基础到AI实战，系统掌握核心技能")
    print("="*70)


def print_module_status():
    """打印模块可用性状态"""
    print("\n📦 模块依赖状态:")
    modules = [
        ("NumPy", NUMPY_AVAILABLE),
        ("Pandas", PANDAS_AVAILABLE),
        ("Matplotlib", MATPLOTLIB_AVAILABLE),
        ("Seaborn", SEABORN_AVAILABLE),
        ("Scikit-Learn", SKLEARN_AVAILABLE),
        ("PyTorch", TORCH_AVAILABLE),
    ]
    
    for name, available in modules:
        status = "✓ 已安装" if available else "✗ 未安装"
        print(f"  {name:<15} {status}")
    
    print("\n📋 课程模块状态:")
    weeks = [
        ("第1周 (Python基础)", WEEK1_AVAILABLE),
        ("第2周 (NumPy)", WEEK2_AVAILABLE),
        ("第3周 (Pandas)", WEEK3_AVAILABLE),
        ("第4周 (可视化)", WEEK4_AVAILABLE),
        ("第5-8周 (机器学习)", WEEK5_8_AVAILABLE),
        ("第9-12周 (深度学习)", WEEK9_12_AVAILABLE),
    ]
    
    for name, available in weeks:
        status = "✓ 可用" if available else "✗ 不可用"
        print(f"  {name:<25} {status}")


def run_week1():
    """运行第1周：Python基础"""
    if not WEEK1_AVAILABLE:
        print("❌ 第1周模块不可用")
        return
    
    print("\n" + "="*70)
    print("开始第1周学习...")
    print("="*70)
    demonstrate_python_basics()


def run_week2():
    """运行第2周：NumPy"""
    if not WEEK2_AVAILABLE:
        print("❌ 第2周模块不可用，请安装: pip install numpy")
        return
    
    print("\n" + "="*70)
    print("开始第2周学习...")
    print("="*70)
    demonstrate_numpy_basics()
    demonstrate_image_processing()
    demonstrate_matrix_operations()


def run_week3():
    """运行第3周：Pandas"""
    if not WEEK3_AVAILABLE:
        print("❌ 第3周模块不可用，请安装: pip install pandas numpy")
        return
    
    print("\n" + "="*70)
    print("开始第3周学习...")
    print("="*70)
    demonstrate_pandas_basics()
    demonstrate_data_cleaning()
    analyze_ecommerce_data()


def run_week4():
    """运行第4周：可视化"""
    if not WEEK4_AVAILABLE:
        print("❌ 第4周模块不可用，请安装: pip install matplotlib seaborn pandas")
        return
    
    print("\n" + "="*70)
    print("开始第4周学习...")
    print("="*70)
    demonstrate_matplotlib_basics()
    demonstrate_sales_visualization()
    demonstrate_seaborn_advanced()


def run_week5_8():
    """运行第5-8周：机器学习"""
    if not WEEK5_8_AVAILABLE:
        print("❌ 第5-8周模块不可用，请安装: pip install scikit-learn numpy")
        return
    
    print("\n" + "="*70)
    print("开始第5-8周学习...")
    print("="*70)
    week5_classification()
    week6_regression()
    week7_clustering()
    week8_model_tuning()


def run_week9_12():
    """运行第9-12周：深度学习"""
    if not WEEK9_12_AVAILABLE:
        print("❌ 第9-12周模块不可用，请安装: pip install torch numpy")
        return
    
    print("\n" + "="*70)
    print("开始第9-12周学习...")
    print("="*70)
    week9_mnist()
    week10_cifar10()
    week11_text_classification()
    week12_comprehensive_project()


def run_stage1():
    """运行阶段1（第1-4周）"""
    print("\n" + "="*70)
    print("🚀 阶段1: Python与数据科学基础（第1-4周）")
    print("="*70)
    run_week1()
    run_week2()
    run_week3()
    run_week4()
    print("\n🎉 阶段1完成!")


def run_stage2():
    """运行阶段2（第5-8周）"""
    print("\n" + "="*70)
    print("🚀 阶段2: 机器学习基础与Scikit-Learn（第5-8周）")
    print("="*70)
    run_week5_8()
    print("\n🎉 阶段2完成!")


def run_stage3():
    """运行阶段3（第9-12周）"""
    print("\n" + "="*70)
    print("🚀 阶段3: 深度学习入门与PyTorch（第9-12周）")
    print("="*70)
    run_week9_12()
    print("\n🎉 阶段3完成!")


def run_all():
    """运行全部12周内容"""
    print("\n" + "="*70)
    print("🎯 开始完整的12周学习路线")
    print("="*70)
    
    start_time = time.time()
    
    run_stage1()
    run_stage2()
    run_stage3()
    
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print(f"🎓 恭喜! 12周学习路线全部完成!")
    print(f"⏱️  总用时: {elapsed:.2f}秒")
    print("="*70)


def interactive_menu():
    """交互式菜单"""
    print_banner()
    print_module_status()
    
    print("\n" + "="*70)
    print("请选择要运行的内容:")
    print("="*70)
    print("\n📖 按周学习:")
    print("  1. 第1周 - Python基础语法与面向对象")
    print("  2. 第2周 - NumPy数组操作与矩阵运算")
    print("  3. 第3周 - Pandas数据处理与分析")
    print("  4. 第4周 - Matplotlib/Seaborn可视化")
    print("  5. 第5-8周 - 机器学习基础（完整）")
    print("  6. 第9-12周 - 深度学习入门（完整）")
    
    print("\n📚 按阶段学习:")
    print("  7. 阶段1 - Python与数据科学基础（第1-4周）")
    print("  8. 阶段2 - 机器学习基础（第5-8周）")
    print("  9. 阶段3 - 深度学习入门（第9-12周）")
    
    print("\n🚀 其他选项:")
    print("  10. 运行全部12周内容")
    print("  0. 退出")
    
    try:
        choice = input("\n请输入选项 (0-10): ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n\n再见!")
        return
    
    # 执行选择
    actions = {
        '1': run_week1,
        '2': run_week2,
        '3': run_week3,
        '4': run_week4,
        '5': run_week5_8,
        '6': run_week9_12,
        '7': run_stage1,
        '8': run_stage2,
        '9': run_stage3,
        '10': run_all,
        '0': lambda: print("\n再见!"),
    }
    
    action = actions.get(choice)
    if action:
        try:
            action()
        except KeyboardInterrupt:
            print("\n\n操作已取消")
        except Exception as e:
            print(f"\n❌ 执行出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n❌ 无效选项: {choice}")


def main():
    """主函数"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        commands = {
            'week1': run_week1,
            'week2': run_week2,
            'week3': run_week3,
            'week4': run_week4,
            'week5-8': run_week5_8,
            'week9-12': run_week9_12,
            'stage1': run_stage1,
            'stage2': run_stage2,
            'stage3': run_stage3,
            'all': run_all,
            'status': lambda: (print_banner(), print_module_status()),
            'help': lambda: print(__doc__),
        }
        
        if command in commands:
            print_banner()
            try:
                commands[command]()
            except Exception as e:
                print(f"\n❌ 执行出错: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"❌ 未知命令: {command}")
            print(__doc__)
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
