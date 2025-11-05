"""模型评估和精度比较工具"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import numpy as np
import time

class ModelEvaluator:
    def __init__(self, device: torch.device):
        self.device = device
        self.results: Dict[str, float] = {}

    def evaluate_model(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        model_name: str,
        criterion: Optional[nn.Module] = None
    ) -> float:
        """评估模型的Top-1准确率和损失

        Args:
            model: 要评估的模型
            data_loader: 数据加载器
            model_name: 模型名称（用于结果存储）
            criterion: 损失函数（可选）

        Returns:
            float: Top-1准确率
        """
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                
                # 计算准确率
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # 如果提供了损失函数，计算损失
                if criterion is not None:
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()

        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(data_loader) if criterion is not None else None

        # 存储结果
        self.results[model_name] = accuracy

        print(f"\n{model_name} 评估结果:")
        print(f"Top-1 准确率: {accuracy:.2f}%")
        if avg_loss is not None:
            print(f"平均损失: {avg_loss:.4f}")

        return accuracy

    def compare_models(self, baseline_name: str, compressed_name: str) -> bool:
        """比较压缩前后的模型精度

        Args:
            baseline_name: 基准模型名称
            compressed_name: 压缩模型名称

        Returns:
            bool: 如果压缩模型在允许的精度范围内（±1%）则返回True
        """
        if baseline_name not in self.results or compressed_name not in self.results:
            raise ValueError("未找到模型评估结果")

        baseline_acc = self.results[baseline_name]
        compressed_acc = self.results[compressed_name]
        diff = compressed_acc - baseline_acc

        print("\n模型精度比较:")
        print(f"基准模型 ({baseline_name}): {baseline_acc:.2f}%")
        print(f"压缩模型 ({compressed_name}): {compressed_acc:.2f}%")
        print(f"精度变化: {diff:+.2f}%")

        # 检查精度变化是否在±1%范围内
        is_acceptable = abs(diff) <= 1.0
        if is_acceptable:
            print("✓ 压缩后的精度变化在可接受范围内 (±1%)")
        else:
            print("✗ 压缩后的精度变化超出可接受范围 (±1%)")

        return is_acceptable

    def get_results(self) -> Dict[str, float]:
        """获取所有评估结果"""
        return self.results.copy()

    def clear_results(self):
        """清除存储的评估结果"""
        self.results.clear()

    def evaluate_compression(
        self,
        original_model: nn.Module,
        compressed_model: nn.Module,
        data_loader: DataLoader,
        compression_type: str = "pruning",
        criterion: Optional[nn.Module] = None
    ) -> Dict[str, any]:
        """
        评估模型压缩效果（量化、剪枝等）
        
        Args:
            original_model: 原始模型
            compressed_model: 压缩后的模型
            data_loader: 测试数据加载器
            compression_type: 压缩类型 ("pruning" 或 "quantization")
            criterion: 损失函数
            
        Returns:
            Dict: 包含精度、速度、模型大小等比较结果
        """
        print(f"\n{'='*70}")
        print(f"📊 {compression_type.upper()} 压缩效果评估")
        print(f"{'='*70}")
        
        results = {}
        
        # 1. 评估原始模型
        print("\n1️⃣ 评估原始模型...")
        orig_acc = self.evaluate_model(
            original_model, 
            data_loader, 
            f"original_{compression_type}",
            criterion
        )
        results['original_accuracy'] = orig_acc
        
        # 2. 评估压缩模型
        print(f"\n2️⃣ 评估压缩模型 ({compression_type})...")
        comp_acc = self.evaluate_model(
            compressed_model,
            data_loader,
            f"compressed_{compression_type}",
            criterion
        )
        results['compressed_accuracy'] = comp_acc
        
        # 3. 计算精度变化
        acc_diff = comp_acc - orig_acc
        results['accuracy_diff'] = acc_diff
        
        # 4. 比较模型大小
        print("\n3️⃣ 比较模型大小...")
        orig_size = self._get_model_size(original_model)
        comp_size = self._get_model_size(compressed_model)
        size_reduction = (orig_size - comp_size) / orig_size * 100
        
        results['original_size_mb'] = orig_size
        results['compressed_size_mb'] = comp_size
        results['size_reduction_percent'] = size_reduction
        
        print(f"  原始模型大小: {orig_size:.2f} MB")
        print(f"  压缩模型大小: {comp_size:.2f} MB")
        print(f"  大小减少: {size_reduction:.2f}%")
        
        # 5. 比较推理速度
        print("\n4️⃣ 比较推理速度...")
        orig_time = self._measure_inference_time(original_model, data_loader)
        comp_time = self._measure_inference_time(compressed_model, data_loader)
        speedup = orig_time / comp_time if comp_time > 0 else 0
        
        results['original_time_ms'] = orig_time
        results['compressed_time_ms'] = comp_time
        results['speedup'] = speedup
        
        print(f"  原始模型推理时间: {orig_time:.2f} ms/batch")
        print(f"  压缩模型推理时间: {comp_time:.2f} ms/batch")
        print(f"  加速比: {speedup:.2f}x")
        
        # 6. 输出总结
        print(f"\n{'='*70}")
        print("📈 压缩效果总结:")
        print(f"{'='*70}")
        print(f"  精度变化: {acc_diff:+.2f}% (原始: {orig_acc:.2f}% → 压缩: {comp_acc:.2f}%)")
        print(f"  模型大小减少: {size_reduction:.2f}%")
        print(f"  推理速度提升: {speedup:.2f}x")
        
        # 检查是否在可接受范围内
        is_acceptable = abs(acc_diff) <= 1.0
        if is_acceptable:
            print(f"  ✅ 精度变化在可接受范围内 (±1%)")
        else:
            print(f"  ⚠️  精度变化超出可接受范围 (±1%)")
        
        results['is_acceptable'] = is_acceptable
        print(f"{'='*70}\n")
        
        return results
    
    def _get_model_size(self, model: nn.Module) -> float:
        """
        计算模型大小（MB）
        
        Args:
            model: PyTorch模型
            
        Returns:
            float: 模型大小（MB）
        """
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        return size_mb
    
    def _measure_inference_time(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        num_batches: int = 50
    ) -> float:
        """
        测量模型推理时间
        
        Args:
            model: PyTorch模型
            data_loader: 数据加载器
            num_batches: 测试的批次数
            
        Returns:
            float: 平均推理时间（ms/batch）
        """
        model.eval()
        times = []
        
        with torch.no_grad():
            for i, (inputs, _) in enumerate(data_loader):
                if i >= num_batches:
                    break
                
                inputs = inputs.to(self.device)
                
                # 预热（第一次运行可能较慢）
                if i == 0:
                    _ = model(inputs)
                    continue
                
                # 测量时间
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.time()
                _ = model(inputs)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # 转换为毫秒
        
        avg_time = np.mean(times) if times else 0
        return avg_time
    
    def compare_pruning_methods(
        self,
        models: Dict[str, nn.Module],
        data_loader: DataLoader,
        criterion: Optional[nn.Module] = None
    ) -> Dict[str, Dict]:
        """
        比较不同剪枝方法的效果
        
        Args:
            models: 模型字典 {方法名: 模型}
            data_loader: 测试数据加载器
            criterion: 损失函数
            
        Returns:
            Dict: 各方法的评估结果
        """
        print(f"\n{'='*70}")
        print("🔍 比较不同剪枝方法")
        print(f"{'='*70}\n")
        
        results = {}
        
        for method_name, model in models.items():
            print(f"评估 {method_name}...")
            acc = self.evaluate_model(model, data_loader, method_name, criterion)
            size = self._get_model_size(model)
            inf_time = self._measure_inference_time(model, data_loader)
            
            results[method_name] = {
                'accuracy': acc,
                'size_mb': size,
                'inference_time_ms': inf_time
            }
        
        # 打印比较表格
        print(f"\n{'='*70}")
        print("📊 剪枝方法比较:")
        print(f"{'='*70}")
        print(f"{'方法':<20} {'准确率':<12} {'大小(MB)':<12} {'推理时间(ms)':<15}")
        print("-" * 70)
        
        for method_name, res in results.items():
            print(f"{method_name:<20} {res['accuracy']:>10.2f}% {res['size_mb']:>10.2f} {res['inference_time_ms']:>13.2f}")
        
        print(f"{'='*70}\n")
        
        return results
