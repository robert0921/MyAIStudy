"""性能监控工具"""
import time
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    throughput: float  # 样本/秒
    batch_time: float  # 毫秒/批次
    gpu_memory: Optional[float] = None  # MB
    gpu_utilization: Optional[float] = None  # %

class PerformanceMonitor:
    def __init__(self):
        self.metrics_history: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.start_time: Optional[float] = None
        self.batch_size: int = 0
        self.running_samples: int = 0
        
    def start_batch(self, batch_size: int):
        """开始批次计时"""
        self.start_time = time.perf_counter()
        self.batch_size = batch_size
        
    def end_batch(self, config_name: str):
        """结束批次计时并记录性能指标"""
        if self.start_time is None:
            raise RuntimeError("必须先调用start_batch")
            
        # 计算批次时间
        batch_time = (time.perf_counter() - self.start_time) * 1000  # 转换为毫秒
        self.running_samples += self.batch_size
        
        # 计算吞吐量 (样本/秒)
        throughput = self.batch_size / (batch_time / 1000)  # 转换回秒
        
        # 获取GPU指标
        gpu_memory = None
        gpu_utilization = None
        if torch.cuda.is_available():
            # 当前GPU的内存使用量
            gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # 转换为MB
            torch.cuda.reset_peak_memory_stats()
            
            # GPU利用率需要使用nvidia-smi工具获取，这里仅作示例
            # 实际项目中可以使用pynvml库获取更详细的GPU信息
        
        metrics = PerformanceMetrics(
            throughput=throughput,
            batch_time=batch_time,
            gpu_memory=gpu_memory,
            gpu_utilization=gpu_utilization
        )
        
        self.metrics_history[config_name].append(metrics)
        
    def get_summary(self, config_name: str) -> Dict[str, float]:
        """获取性能指标摘要"""
        if not self.metrics_history[config_name]:
            return {}
            
        metrics_list = self.metrics_history[config_name]
        
        # 计算平均值，跳过前几个批次的预热时间
        warmup = 5
        if len(metrics_list) > warmup:
            metrics_list = metrics_list[warmup:]
            
        avg_throughput = np.mean([m.throughput for m in metrics_list])
        avg_batch_time = np.mean([m.batch_time for m in metrics_list])
        
        summary = {
            "平均吞吐量 (样本/秒)": avg_throughput,
            "平均批次时间 (ms)": avg_batch_time,
        }
        
        # 添加GPU指标（如果可用）
        if metrics_list[0].gpu_memory is not None:
            avg_gpu_memory = np.mean([m.gpu_memory for m in metrics_list])
            summary["平均GPU内存使用 (MB)"] = avg_gpu_memory
            
        if metrics_list[0].gpu_utilization is not None:
            avg_gpu_util = np.mean([m.gpu_utilization for m in metrics_list])
            summary["平均GPU利用率 (%)"] = avg_gpu_util
            
        return summary
        
    def print_summary(self, config_name: str):
        """打印性能指标摘要"""
        summary = self.get_summary(config_name)
        if not summary:
            print(f"\n{config_name} - 无性能数据")
            return
            
        print(f"\n{config_name} - 性能指标摘要:")
        for metric_name, value in summary.items():
            print(f"{metric_name}: {value:.2f}")
            
    def compare_configs(self, base_config: str, test_config: str):
        """比较两种配置的性能差异"""
        base_summary = self.get_summary(base_config)
        test_summary = self.get_summary(test_config)
        
        if not base_summary or not test_summary:
            print("无法比较性能：缺少基准数据")
            return
            
        print(f"\n性能比较 ({test_config} vs {base_config}):")
        for metric in base_summary.keys():
            base_value = base_summary[metric]
            test_value = test_summary[metric]
            diff_pct = ((test_value - base_value) / base_value) * 100
            
            print(f"{metric}:")
            print(f"  {base_config}: {base_value:.2f}")
            print(f"  {test_config}: {test_value:.2f}")
            print(f"  变化: {diff_pct:+.2f}%")
