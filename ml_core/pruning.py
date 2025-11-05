"""模型剪枝工具模块"""
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, List, Tuple, Optional
import copy
import numpy as np


class ModelPruner:
    """模型剪枝工具类，支持多种剪枝策略"""
    
    def __init__(self, model: nn.Module):
        """
        初始化剪枝器
        
        Args:
            model: 要剪枝的PyTorch模型
        """
        self.model = model
        self.original_model = None
        self.pruning_history: List[Dict] = []
        
    def magnitude_pruning(
        self, 
        amount: float = 0.3,
        layers_to_prune: Optional[List[str]] = None
    ) -> None:
        """
        幅度剪枝（Magnitude Pruning）
        移除权重绝对值最小的连接
        
        Args:
            amount: 剪枝比例（0-1之间）
            layers_to_prune: 要剪枝的层名称列表，None表示所有卷积和线性层
        """
        print(f"\n🔪 执行幅度剪枝 (剪枝比例: {amount*100:.1f}%)")
        
        # 保存原始模型
        if self.original_model is None:
            self.original_model = copy.deepcopy(self.model)
        
        pruned_count = 0
        total_params = 0
        
        for name, module in self.model.named_modules():
            # 只对卷积层和全连接层剪枝
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if layers_to_prune is None or name in layers_to_prune:
                    # 对权重进行L1非结构化剪枝
                    prune.l1_unstructured(module, name='weight', amount=amount)
                    
                    # 统计参数
                    total_params += module.weight.nelement()
                    pruned_count += int(module.weight.nelement() * amount)
                    
                    print(f"  ✓ 剪枝层: {name} - 参数: {module.weight.nelement():,}")
        
        # 记录剪枝历史
        self.pruning_history.append({
            'type': 'magnitude',
            'amount': amount,
            'pruned_params': pruned_count,
            'total_params': total_params
        })
        
        print(f"\n  总参数数: {total_params:,}")
        print(f"  剪枝参数: {pruned_count:,} ({pruned_count/total_params*100:.2f}%)")
        
    def structured_pruning(
        self,
        amount: float = 0.3,
        dim: int = 0,
        layers_to_prune: Optional[List[str]] = None
    ) -> None:
        """
        结构化剪枝（Structured Pruning）
        移除整个通道或滤波器
        
        Args:
            amount: 剪枝比例（0-1之间）
            dim: 剪枝维度（0表示输出通道，1表示输入通道）
            layers_to_prune: 要剪枝的层名称列表
        """
        print(f"\n🔪 执行结构化剪枝 (剪枝比例: {amount*100:.1f}%, 维度: {dim})")
        
        if self.original_model is None:
            self.original_model = copy.deepcopy(self.model)
        
        pruned_count = 0
        total_params = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                if layers_to_prune is None or name in layers_to_prune:
                    # 对整个通道进行剪枝
                    prune.ln_structured(
                        module, 
                        name='weight', 
                        amount=amount, 
                        n=2,  # L2 norm
                        dim=dim
                    )
                    
                    total_params += module.weight.nelement()
                    pruned_count += int(module.weight.nelement() * amount)
                    
                    print(f"  ✓ 剪枝层: {name} - 形状: {tuple(module.weight.shape)}")
        
        self.pruning_history.append({
            'type': 'structured',
            'amount': amount,
            'dim': dim,
            'pruned_params': pruned_count,
            'total_params': total_params
        })
        
        print(f"\n  总参数数: {total_params:,}")
        print(f"  剪枝参数: {pruned_count:,} ({pruned_count/total_params*100:.2f}%)")
        
    def global_pruning(
        self,
        amount: float = 0.3,
        pruning_method: str = 'l1'
    ) -> None:
        """
        全局剪枝（Global Pruning）
        在所有层中统一剪枝权重最小的部分
        
        Args:
            amount: 剪枝比例（0-1之间）
            pruning_method: 剪枝方法 ('l1' 或 'random')
        """
        print(f"\n🔪 执行全局剪枝 (剪枝比例: {amount*100:.1f}%, 方法: {pruning_method})")
        
        if self.original_model is None:
            self.original_model = copy.deepcopy(self.model)
        
        # 收集所有要剪枝的参数
        parameters_to_prune = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
        
        # 执行全局剪枝
        if pruning_method == 'l1':
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=amount,
            )
        elif pruning_method == 'random':
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.RandomUnstructured,
                amount=amount,
            )
        else:
            raise ValueError(f"不支持的剪枝方法: {pruning_method}")
        
        # 统计参数
        total_params = sum(p[0].weight.nelement() for p in parameters_to_prune)
        pruned_count = int(total_params * amount)
        
        self.pruning_history.append({
            'type': 'global',
            'amount': amount,
            'method': pruning_method,
            'pruned_params': pruned_count,
            'total_params': total_params
        })
        
        print(f"\n  剪枝层数: {len(parameters_to_prune)}")
        print(f"  总参数数: {total_params:,}")
        print(f"  剪枝参数: {pruned_count:,} ({pruned_count/total_params*100:.2f}%)")
        
    def iterative_pruning(
        self,
        target_amount: float = 0.5,
        steps: int = 5,
        pruning_type: str = 'magnitude'
    ) -> None:
        """
        迭代剪枝（Iterative Pruning）
        逐步增加剪枝比例，允许模型在每次剪枝后进行微调
        
        Args:
            target_amount: 最终目标剪枝比例
            steps: 迭代步数
            pruning_type: 剪枝类型 ('magnitude', 'structured', 'global')
        """
        print(f"\n🔪 执行迭代剪枝 (目标: {target_amount*100:.1f}%, 步数: {steps})")
        
        if self.original_model is None:
            self.original_model = copy.deepcopy(self.model)
        
        # 计算每步的剪枝比例
        step_amount = 1 - (1 - target_amount) ** (1 / steps)
        
        for step in range(steps):
            print(f"\n  --- 迭代步骤 {step+1}/{steps} (当前剪枝比例: {step_amount*100:.1f}%) ---")
            
            if pruning_type == 'magnitude':
                self.magnitude_pruning(amount=step_amount)
            elif pruning_type == 'structured':
                self.structured_pruning(amount=step_amount)
            elif pruning_type == 'global':
                self.global_pruning(amount=step_amount)
            else:
                raise ValueError(f"不支持的剪枝类型: {pruning_type}")
            
            # 提示用户在实际使用中应该在此处进行微调
            print(f"  💡 提示: 在实际应用中，应在此处对模型进行微调")
        
    def remove_pruning(self) -> None:
        """
        移除剪枝掩码，使剪枝永久生效
        剪枝的权重将被真正删除
        """
        print("\n🔧 移除剪枝掩码，使剪枝永久生效...")
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                try:
                    prune.remove(module, 'weight')
                    print(f"  ✓ 移除 {name} 的剪枝掩码")
                except ValueError:
                    # 该层没有被剪枝
                    pass
                    
    def get_sparsity(self) -> Dict[str, float]:
        """
        计算模型的稀疏度
        
        Returns:
            Dict: 包含每层和全局稀疏度的字典
        """
        sparsity_dict = {}
        total_zeros = 0
        total_params = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight.data
                zeros = (weight == 0).sum().item()
                total = weight.nelement()
                
                sparsity = 100.0 * zeros / total
                sparsity_dict[name] = sparsity
                
                total_zeros += zeros
                total_params += total
        
        # 计算全局稀疏度
        global_sparsity = 100.0 * total_zeros / total_params
        sparsity_dict['global'] = global_sparsity
        
        return sparsity_dict
        
    def print_sparsity(self) -> None:
        """打印模型各层的稀疏度"""
        sparsity_dict = self.get_sparsity()
        
        print("\n📊 模型稀疏度统计:")
        print("=" * 60)
        
        for name, sparsity in sparsity_dict.items():
            if name != 'global':
                print(f"  {name:40s}: {sparsity:6.2f}%")
        
        print("=" * 60)
        print(f"  {'全局稀疏度':40s}: {sparsity_dict['global']:6.2f}%")
        print("=" * 60)
        
    def get_model_size(self) -> Tuple[int, float]:
        """
        获取模型大小
        
        Returns:
            Tuple[int, float]: (参数数量, 模型大小MB)
        """
        param_count = sum(p.numel() for p in self.model.parameters())
        
        # 计算模型大小（假设每个参数4字节）
        model_size_mb = param_count * 4 / (1024 * 1024)
        
        return param_count, model_size_mb
        
    def compare_with_original(self) -> Dict[str, float]:
        """
        与原始模型比较
        
        Returns:
            Dict: 包含参数减少比例和大小减少的字典
        """
        if self.original_model is None:
            print("⚠️ 没有保存原始模型，无法比较")
            return {}
        
        # 原始模型
        orig_params, orig_size = self._get_model_size(self.original_model)
        
        # 当前模型
        curr_params, curr_size = self.get_model_size()
        
        # 计算稀疏度
        sparsity_dict = self.get_sparsity()
        global_sparsity = sparsity_dict['global']
        
        # 实际参数减少（考虑稀疏性）
        effective_params = curr_params * (1 - global_sparsity / 100)
        param_reduction = (orig_params - effective_params) / orig_params * 100
        
        print("\n📊 模型压缩比较:")
        print("=" * 60)
        print(f"原始模型:")
        print(f"  参数数量: {orig_params:,}")
        print(f"  模型大小: {orig_size:.2f} MB")
        print(f"\n剪枝后模型:")
        print(f"  参数数量: {curr_params:,}")
        print(f"  有效参数: {int(effective_params):,} ({100-global_sparsity:.2f}%)")
        print(f"  模型大小: {curr_size:.2f} MB")
        print(f"  全局稀疏度: {global_sparsity:.2f}%")
        print(f"\n压缩效果:")
        print(f"  参数减少: {param_reduction:.2f}%")
        print(f"  大小减少: {(orig_size-curr_size)/orig_size*100:.2f}%")
        print("=" * 60)
        
        return {
            'original_params': orig_params,
            'current_params': curr_params,
            'effective_params': effective_params,
            'sparsity': global_sparsity,
            'param_reduction': param_reduction,
            'size_reduction': (orig_size-curr_size)/orig_size*100
        }
        
    def _get_model_size(self, model: nn.Module) -> Tuple[int, float]:
        """获取指定模型的大小"""
        param_count = sum(p.numel() for p in model.parameters())
        model_size_mb = param_count * 4 / (1024 * 1024)
        return param_count, model_size_mb
        
    def restore_original(self) -> None:
        """恢复到原始模型"""
        if self.original_model is None:
            print("⚠️ 没有保存的原始模型")
            return
        
        # 先移除所有剪枝掩码
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                try:
                    prune.remove(module, 'weight')
                except ValueError:
                    # 该层没有被剪枝，跳过
                    pass
        
        # 然后加载原始模型权重
        self.model.load_state_dict(self.original_model.state_dict())
        self.pruning_history.clear()
        print("✓ 已恢复到原始模型")


def demonstrate_pruning():
    """演示模型剪枝功能"""
    print("\n" + "="*70)
    print("🔪 模型剪枝功能演示")
    print("="*70)
    
    # 创建一个简单的示例模型
    from ml_core.models_torch import CIFAR10Net
    
    model = CIFAR10Net()
    pruner = ModelPruner(model)
    
    # 显示原始模型信息
    print("\n📊 原始模型信息:")
    param_count, size_mb = pruner.get_model_size()
    print(f"  参数数量: {param_count:,}")
    print(f"  模型大小: {size_mb:.2f} MB")
    
    # 1. 幅度剪枝
    print("\n" + "-"*70)
    print("1️⃣ 幅度剪枝演示")
    print("-"*70)
    pruner.magnitude_pruning(amount=0.3)
    pruner.print_sparsity()
    
    # 2. 结构化剪枝
    print("\n" + "-"*70)
    print("2️⃣ 结构化剪枝演示")
    print("-"*70)
    pruner.restore_original()
    pruner.structured_pruning(amount=0.2, dim=0)
    pruner.print_sparsity()
    
    # 3. 全局剪枝
    print("\n" + "-"*70)
    print("3️⃣ 全局剪枝演示")
    print("-"*70)
    pruner.restore_original()
    pruner.global_pruning(amount=0.4)
    pruner.print_sparsity()
    
    # 4. 比较压缩效果
    print("\n" + "-"*70)
    print("4️⃣ 压缩效果对比")
    print("-"*70)
    pruner.compare_with_original()
    
    print("\n" + "="*70)
    print("✅ 模型剪枝演示完成!")
    print("="*70)
    print("\n💡 提示:")
    print("  • 幅度剪枝适合快速压缩，但可能影响精度")
    print("  • 结构化剪枝可以真正减少计算量")
    print("  • 全局剪枝在所有层中统一选择要剪枝的权重")
    print("  • 迭代剪枝配合微调可以获得最佳效果")
    print("  • 实际使用时应该在剪枝后进行微调以恢复精度")


if __name__ == "__main__":
    demonstrate_pruning()
