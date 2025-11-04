"""模型评估和精度比较工具"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
import numpy as np

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
