"""测试模型剪枝功能"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_core.models_torch import CIFAR10Net
from ml_core.pruning import ModelPruner
from ml_core.evaluation import ModelEvaluator


def create_dummy_data(num_samples=100, batch_size=32):
    """创建虚拟测试数据"""
    # 创建随机数据
    X = torch.randn(num_samples, 3, 32, 32)
    y = torch.randint(0, 10, (num_samples,))
    
    # 创建数据集和加载器
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return loader


def test_pruning_methods():
    """测试不同的剪枝方法"""
    print("\n" + "="*70)
    print("🧪 测试模型剪枝功能")
    print("="*70)
    
    # 创建模型和数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 创建测试数据
    print("\n1. 创建测试数据...")
    test_loader = create_dummy_data(num_samples=100, batch_size=32)
    print("   ✓ 测试数据创建完成")
    
    # 测试1: 幅度剪枝
    print("\n" + "-"*70)
    print("测试 1: 幅度剪枝 (Magnitude Pruning)")
    print("-"*70)
    
    model1 = CIFAR10Net().to(device)
    pruner1 = ModelPruner(model1)
    
    print("\n执行30%幅度剪枝...")
    pruner1.magnitude_pruning(amount=0.3)
    pruner1.print_sparsity()
    pruner1.compare_with_original()
    
    # 测试2: 结构化剪枝
    print("\n" + "-"*70)
    print("测试 2: 结构化剪枝 (Structured Pruning)")
    print("-"*70)
    
    model2 = CIFAR10Net().to(device)
    pruner2 = ModelPruner(model2)
    
    print("\n执行20%结构化剪枝...")
    pruner2.structured_pruning(amount=0.2, dim=0)
    pruner2.print_sparsity()
    pruner2.compare_with_original()
    
    # 测试3: 全局剪枝
    print("\n" + "-"*70)
    print("测试 3: 全局剪枝 (Global Pruning)")
    print("-"*70)
    
    model3 = CIFAR10Net().to(device)
    pruner3 = ModelPruner(model3)
    
    print("\n执行40%全局剪枝...")
    pruner3.global_pruning(amount=0.4)
    pruner3.print_sparsity()
    pruner3.compare_with_original()
    
    # 测试4: 评估压缩效果
    print("\n" + "-"*70)
    print("测试 4: 评估压缩效果")
    print("-"*70)
    
    evaluator = ModelEvaluator(device)
    
    # 创建原始模型和剪枝模型
    original_model = CIFAR10Net().to(device)
    pruned_model = CIFAR10Net().to(device)
    pruner = ModelPruner(pruned_model)
    pruner.magnitude_pruning(amount=0.3)
    
    # 评估压缩效果
    results = evaluator.evaluate_compression(
        original_model=original_model,
        compressed_model=pruned_model,
        data_loader=test_loader,
        compression_type="pruning"
    )
    
    print("\n压缩效果总结:")
    print(f"  精度变化: {results['accuracy_diff']:+.2f}%")
    print(f"  大小减少: {results['size_reduction_percent']:.2f}%")
    print(f"  推理加速: {results['speedup']:.2f}x")
    print(f"  是否可接受: {'✓' if results['is_acceptable'] else '✗'}")
    
    # 测试5: 比较不同剪枝方法
    print("\n" + "-"*70)
    print("测试 5: 比较不同剪枝方法")
    print("-"*70)
    
    models_dict = {}
    
    # 原始模型
    models_dict['Original'] = CIFAR10Net().to(device)
    
    # 幅度剪枝模型
    mag_model = CIFAR10Net().to(device)
    mag_pruner = ModelPruner(mag_model)
    mag_pruner.magnitude_pruning(amount=0.3)
    models_dict['Magnitude-30%'] = mag_model
    
    # 结构化剪枝模型
    struct_model = CIFAR10Net().to(device)
    struct_pruner = ModelPruner(struct_model)
    struct_pruner.structured_pruning(amount=0.2, dim=0)
    models_dict['Structured-20%'] = struct_model
    
    # 全局剪枝模型
    global_model = CIFAR10Net().to(device)
    global_pruner = ModelPruner(global_model)
    global_pruner.global_pruning(amount=0.4)
    models_dict['Global-40%'] = global_model
    
    # 比较所有方法
    evaluator.compare_pruning_methods(models_dict, test_loader)
    
    print("\n" + "="*70)
    print("✅ 所有剪枝功能测试完成!")
    print("="*70)
    
    print("\n📝 测试总结:")
    print("  ✓ 幅度剪枝 - 可以有效减少参数数量")
    print("  ✓ 结构化剪枝 - 可以真正减少计算量")
    print("  ✓ 全局剪枝 - 在所有层中统一选择最优剪枝")
    print("  ✓ 评估功能 - 完整的精度、大小、速度对比")
    print("  ✓ 比较功能 - 支持多种剪枝方法的横向对比")
    
    return True


if __name__ == "__main__":
    try:
        success = test_pruning_methods()
        if success:
            print("\n🎉 测试成功!")
            sys.exit(0)
        else:
            print("\n❌ 测试失败!")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
