"""训练性能监控工具"""
import torch
from typing import Dict, List, Optional
import time
from dataclasses import dataclass
import psutil
import json
from pathlib import Path
import numpy as np

@dataclass
class TrainingMetrics:
    """训练指标数据类"""
    epoch: int
    iteration: int
    loss: float
    accuracy: float
    learning_rate: float
    gpu_memory_used: Optional[float] = None  # MB
    gpu_utilization: Optional[float] = None  # %
    batch_time: Optional[float] = None  # ms
    
    def to_dict(self) -> Dict:
        return {
            'epoch': self.epoch,
            'iteration': self.iteration,
            'loss': self.loss,
            'accuracy': self.accuracy,
            'learning_rate': self.learning_rate,
            'gpu_memory_used': self.gpu_memory_used,
            'gpu_utilization': self.gpu_utilization,
            'batch_time': self.batch_time
        }

class TrainingMonitor:
    """训练监控器"""
    def __init__(self, save_dir: str, log_freq: int = 100):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_freq = log_freq
        
        self.metrics: List[TrainingMetrics] = []
        self.best_accuracy = 0.0
        self.start_time = time.time()
        
        # 初始显存基准
        self.baseline_memory = (
            torch.cuda.memory_allocated()
            if torch.cuda.is_available()
            else 0
        )
    
    def update(
        self,
        epoch: int,
        iteration: int,
        loss: float,
        accuracy: float,
        learning_rate: float
    ):
        """更新训练指标"""
        # 计算GPU指标
        gpu_memory = None
        gpu_util = None
        if torch.cuda.is_available():
            gpu_memory = (
                torch.cuda.memory_allocated() - self.baseline_memory
            ) / 1024 / 1024  # 转换为MB
            
            # GPU利用率（需要nvidia-smi或类似工具）
            try:
                gpu_util = torch.cuda.utilization()
            except:
                gpu_util = None
        
        # 计算批处理时间
        current_time = time.time()
        batch_time = (current_time - self.start_time) * 1000  # 转换为毫秒
        self.start_time = current_time
        
        # 创建指标记录
        metrics = TrainingMetrics(
            epoch=epoch,
            iteration=iteration,
            loss=loss,
            accuracy=accuracy,
            learning_rate=learning_rate,
            gpu_memory_used=gpu_memory,
            gpu_utilization=gpu_util,
            batch_time=batch_time
        )
        
        self.metrics.append(metrics)
        
        # 更新最佳准确率
        self.best_accuracy = max(self.best_accuracy, accuracy)
        
        # 定期保存
        if iteration % self.log_freq == 0:
            self.save_metrics()
            self._print_status(metrics)
    
    def save_metrics(self):
        """保存训练指标到文件"""
        metrics_file = self.save_dir / 'training_metrics.json'
        
        # 转换为可序列化的格式
        metrics_data = [m.to_dict() for m in self.metrics]
        
        # 保存到文件
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def _print_status(self, metrics: TrainingMetrics):
        """打印当前状态"""
        print(f"\nEpoch {metrics.epoch} [{metrics.iteration}]:")
        print(f"Loss: {metrics.loss:.4f}")
        print(f"Accuracy: {metrics.accuracy:.2f}%")
        print(f"Learning Rate: {metrics.learning_rate:.6f}")
        
        if metrics.gpu_memory_used is not None:
            print(f"GPU Memory: {metrics.gpu_memory_used:.1f}MB")
        if metrics.gpu_utilization is not None:
            print(f"GPU Utilization: {metrics.gpu_utilization:.1f}%")
        if metrics.batch_time is not None:
            print(f"Batch Time: {metrics.batch_time:.2f}ms")
    
    def get_summary(self) -> Dict:
        """获取训练摘要"""
        if not self.metrics:
            return {}
        
        # 计算平均值和最佳值
        avg_loss = np.mean([m.loss for m in self.metrics])
        avg_accuracy = np.mean([m.accuracy for m in self.metrics])
        avg_batch_time = np.mean([m.batch_time for m in self.metrics if m.batch_time])
        
        # GPU指标
        gpu_metrics = {}
        if any(m.gpu_memory_used for m in self.metrics):
            max_memory = max(
                m.gpu_memory_used
                for m in self.metrics
                if m.gpu_memory_used is not None
            )
            avg_memory = np.mean([
                m.gpu_memory_used
                for m in self.metrics
                if m.gpu_memory_used is not None
            ])
            gpu_metrics.update({
                'max_gpu_memory_mb': max_memory,
                'avg_gpu_memory_mb': avg_memory
            })
        
        if any(m.gpu_utilization for m in self.metrics):
            avg_util = np.mean([
                m.gpu_utilization
                for m in self.metrics
                if m.gpu_utilization is not None
            ])
            gpu_metrics['avg_gpu_utilization'] = avg_util
        
        return {
            'best_accuracy': self.best_accuracy,
            'average_loss': avg_loss,
            'average_accuracy': avg_accuracy,
            'average_batch_time_ms': avg_batch_time,
            **gpu_metrics
        }
    
    def plot_metrics(self, save_path: Optional[str] = None):
        """绘制训练指标图表"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("警告：需要matplotlib来绘制图表")
            return
        
        # 提取数据
        epochs = [m.epoch for m in self.metrics]
        losses = [m.loss for m in self.metrics]
        accuracies = [m.accuracy for m in self.metrics]
        learning_rates = [m.learning_rate for m in self.metrics]
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        ax1.plot(epochs, losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        
        # 准确率曲线
        ax2.plot(epochs, accuracies)
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        
        # 学习率曲线
        ax3.plot(epochs, learning_rates)
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        
        # GPU内存使用（如果有）
        if any(m.gpu_memory_used for m in self.metrics):
            memories = [
                m.gpu_memory_used
                for m in self.metrics
                if m.gpu_memory_used is not None
            ]
            ax4.plot(range(len(memories)), memories)
            ax4.set_title('GPU Memory Usage')
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Memory (MB)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close()
