"""
测试run_example.py的主要功能
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试所有导入是否正常"""
    print("测试导入模块...")
    try:
        import run_example
        print("✓ run_example导入成功")
        
        # 测试关键类和函数
        assert hasattr(run_example, 'PromptDebugger'), "缺少PromptDebugger类"
        assert hasattr(run_example, 'FewShotManager'), "缺少FewShotManager类"
        assert hasattr(run_example, 'demonstrate_fundamentals'), "缺少demonstrate_fundamentals函数"
        assert hasattr(run_example, 'demonstrate_llm_architecture'), "缺少demonstrate_llm_architecture函数"
        assert hasattr(run_example, 'demonstrate_snn_performance'), "缺少demonstrate_snn_performance函数"
        assert hasattr(run_example, 'run_dashboard'), "缺少run_dashboard函数"
        assert hasattr(run_example, 'run_deep_learning_training'), "缺少run_deep_learning_training函数"
        assert hasattr(run_example, 'demonstrate_prompt_engineering'), "缺少demonstrate_prompt_engineering函数"
        
        print("✓ 所有关键函数和类都存在")
        return True
    except Exception as e:
        print(f"✗ 导入测试失败: {e}")
        return False

def test_prompt_engineering():
    """测试Prompt Engineering功能"""
    print("\n测试Prompt Engineering功能...")
    try:
        from run_example import PromptDebugger, FewShotManager
        
        # 测试FewShotManager
        fewshot = FewShotManager()
        fewshot.add_example("测试示例1")
        fewshot.add_example("测试示例2")
        assert len(fewshot.get_examples()) == 2, "Few-shot示例数量不正确"
        
        # 测试自动生成
        auto_examples = fewshot.auto_generate_examples("测试任务", n=3)
        assert len(auto_examples) == 3, "自动生成示例数量不正确"
        
        # 测试PromptDebugger
        debugger = PromptDebugger()
        result = debugger.test_prompt("测试prompt")
        assert 'prompt' in result, "test_prompt返回格式不正确"
        assert 'output' in result, "test_prompt缺少output字段"
        
        # 测试批量测试
        batch_results = debugger.batch_test(["prompt1", "prompt2"])
        assert len(batch_results) == 2, "批量测试结果数量不正确"
        
        # 测试优化
        opt_result = debugger.optimize_prompt("测试", "目标")
        assert 'optimized_prompt' in opt_result, "optimize_prompt返回格式不正确"
        
        print("✓ Prompt Engineering功能测试通过")
        return True
    except Exception as e:
        print(f"✗ Prompt Engineering测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_system_info():
    """测试系统信息打印"""
    print("\n测试系统信息打印...")
    try:
        from run_example import print_system_info
        print_system_info()
        print("✓ 系统信息打印成功")
        return True
    except Exception as e:
        print(f"✗ 系统信息打印失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("="*60)
    print("运行run_example.py测试套件")
    print("="*60)
    
    results = []
    
    # 运行各项测试
    results.append(("导入测试", test_imports()))
    results.append(("Prompt Engineering", test_prompt_engineering()))
    results.append(("系统信息", test_system_info()))
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    passed = 0
    failed = 0
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("\n🎉 所有测试通过!")
        return 0
    else:
        print(f"\n⚠️  有 {failed} 个测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
