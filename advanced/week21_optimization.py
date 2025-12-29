"""
Week 21: GPU性能优化与成本评估工具
包括：GPU成本计算器、性能分析、模型压缩效果对比

本模块提供系统优化和成本评估的实用工具。
"""

import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import numpy as np


@dataclass
class GPUConfig:
    """GPU配置"""
    name: str
    memory_gb: float  # 显存(GB)
    tflops: float  # 计算性能(TFLOPS)
    price_per_hour: float  # 每小时价格(USD)
    power_watts: int  # 功耗(W)


# 常见GPU配置
GPU_CONFIGS = {
    "A100-80GB": GPUConfig("A100-80GB", 80, 312, 3.06, 400),
    "A100-40GB": GPUConfig("A100-40GB", 40, 312, 2.21, 400),
    "V100-32GB": GPUConfig("V100-32GB", 32, 125, 2.48, 300),
    "V100-16GB": GPUConfig("V100-16GB", 16, 125, 0.90, 300),
    "T4": GPUConfig("T4", 16, 65, 0.35, 70),
    "RTX 4090": GPUConfig("RTX 4090", 24, 82.6, 0.00, 450),  # 本地卡，价格为0
    "RTX 3090": GPUConfig("RTX 3090", 24, 35.6, 0.00, 350),
}


class GPUCostCalculator:
    """GPU成本计算器"""
    
    def __init__(self):
        self.gpu_configs = GPU_CONFIGS
    
    def calculate_training_cost(
        self,
        gpu_name: str,
        training_hours: float,
        num_gpus: int = 1
    ) -> Dict:
        """计算训练成本"""
        
        if gpu_name not in self.gpu_configs:
            raise ValueError(f"未知的GPU: {gpu_name}")
        
        gpu = self.gpu_configs[gpu_name]
        
        # 计算成本
        compute_cost = gpu.price_per_hour * training_hours * num_gpus
        
        # 计算能耗成本（假设电费0.12 USD/kWh）
        energy_kwh = (gpu.power_watts * training_hours * num_gpus) / 1000
        energy_cost = energy_kwh * 0.12
        
        total_cost = compute_cost + energy_cost
        
        return {
            'gpu': gpu_name,
            'num_gpus': num_gpus,
            'training_hours': training_hours,
            'compute_cost': compute_cost,
            'energy_cost': energy_cost,
            'total_cost': total_cost,
            'cost_per_hour': total_cost / training_hours,
            'tflops_total': gpu.tflops * num_gpus
        }
    
    def compare_gpus(
        self,
        gpu_names: List[str],
        training_hours: float,
        num_gpus: int = 1
    ) -> List[Dict]:
        """比较不同GPU的成本"""
        
        results = []
        for gpu_name in gpu_names:
            try:
                cost_info = self.calculate_training_cost(gpu_name, training_hours, num_gpus)
                results.append(cost_info)
            except ValueError as e:
                print(f"警告: {e}")
        
        # 按总成本排序
        results.sort(key=lambda x: x['total_cost'])
        return results
    
    def estimate_inference_cost(
        self,
        gpu_name: str,
        requests_per_second: float,
        avg_latency_ms: float,
        hours_per_day: float = 24
    ) -> Dict:
        """估算推理成本"""
        
        if gpu_name not in self.gpu_configs:
            raise ValueError(f"未知的GPU: {gpu_name}")
        
        gpu = self.gpu_configs[gpu_name]
        
        # 计算GPU利用率
        utilization = (avg_latency_ms / 1000) * requests_per_second
        
        # 估算所需GPU数量
        num_gpus_needed = max(1, int(np.ceil(utilization)))
        
        # 每天成本
        daily_cost = gpu.price_per_hour * hours_per_day * num_gpus_needed
        
        # 每月成本
        monthly_cost = daily_cost * 30
        
        # 每百万次请求成本
        requests_per_day = requests_per_second * hours_per_day * 3600
        cost_per_million = (daily_cost / requests_per_day) * 1_000_000 if requests_per_day > 0 else 0
        
        return {
            'gpu': gpu_name,
            'requests_per_second': requests_per_second,
            'avg_latency_ms': avg_latency_ms,
            'estimated_gpus': num_gpus_needed,
            'daily_cost': daily_cost,
            'monthly_cost': monthly_cost,
            'cost_per_million_requests': cost_per_million
        }
    
    def calculate_roi(
        self,
        cloud_gpu: str,
        local_gpu: str,
        monthly_hours: float
    ) -> Dict:
        """计算本地GPU投资回报"""
        
        # 云端成本
        cloud_monthly = self.gpu_configs[cloud_gpu].price_per_hour * monthly_hours
        
        # 本地GPU假设价格
        local_prices = {
            "RTX 4090": 1599,
            "RTX 3090": 999,
            "A100-80GB": 15000,
        }
        
        local_price = local_prices.get(local_gpu, 2000)
        
        # 计算回本时间
        months_to_roi = local_price / cloud_monthly if cloud_monthly > 0 else float('inf')
        
        return {
            'cloud_gpu': cloud_gpu,
            'local_gpu': local_gpu,
            'cloud_monthly_cost': cloud_monthly,
            'local_purchase_cost': local_price,
            'months_to_roi': months_to_roi,
            'break_even': months_to_roi < 24  # 2年内回本
        }


class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self, name: str):
        self.name = name
        self.metrics = {
            'latency': [],
            'throughput': [],
            'memory_usage': [],
            'gpu_utilization': []
        }
        self.start_time = None
    
    def start(self):
        """开始分析"""
        self.start_time = time.time()
    
    def record_latency(self, latency_ms: float):
        """记录延迟"""
        self.metrics['latency'].append(latency_ms)
    
    def record_throughput(self, tokens_per_second: float):
        """记录吞吐量"""
        self.metrics['throughput'].append(tokens_per_second)
    
    def record_memory(self, memory_mb: float):
        """记录显存使用"""
        self.metrics['memory_usage'].append(memory_mb)
    
    def record_gpu_util(self, utilization_percent: float):
        """记录GPU利用率"""
        self.metrics['gpu_utilization'].append(utilization_percent)
    
    def get_summary(self) -> Dict:
        """获取性能摘要"""
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'p50': np.percentile(values, 50),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99)
                }
        
        return summary
    
    def generate_report(self) -> str:
        """生成性能报告"""
        summary = self.get_summary()
        
        report = f"# {self.name} 性能分析报告\n\n"
        report += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for metric_name, stats in summary.items():
            report += f"## {metric_name.upper()}\n\n"
            report += f"- 平均值: {stats['mean']:.2f}\n"
            report += f"- 标准差: {stats['std']:.2f}\n"
            report += f"- 最小值: {stats['min']:.2f}\n"
            report += f"- 最大值: {stats['max']:.2f}\n"
            report += f"- P50: {stats['p50']:.2f}\n"
            report += f"- P95: {stats['p95']:.2f}\n"
            report += f"- P99: {stats['p99']:.2f}\n\n"
        
        return report


class ModelCompressor:
    """模型压缩效果评估"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.baseline = None
        self.compressed_models = []
    
    def set_baseline(
        self,
        params_millions: float,
        memory_mb: float,
        latency_ms: float,
        accuracy: float
    ):
        """设置基线模型"""
        self.baseline = {
            'name': 'Baseline',
            'params_millions': params_millions,
            'memory_mb': memory_mb,
            'latency_ms': latency_ms,
            'accuracy': accuracy
        }
    
    def add_compressed_model(
        self,
        name: str,
        params_millions: float,
        memory_mb: float,
        latency_ms: float,
        accuracy: float,
        compression_method: str
    ):
        """添加压缩后的模型"""
        
        if not self.baseline:
            raise ValueError("请先设置基线模型")
        
        # 计算压缩比和性能变化
        model_info = {
            'name': name,
            'method': compression_method,
            'params_millions': params_millions,
            'memory_mb': memory_mb,
            'latency_ms': latency_ms,
            'accuracy': accuracy,
            'params_ratio': params_millions / self.baseline['params_millions'],
            'memory_ratio': memory_mb / self.baseline['memory_mb'],
            'speedup': self.baseline['latency_ms'] / latency_ms,
            'accuracy_drop': self.baseline['accuracy'] - accuracy
        }
        
        self.compressed_models.append(model_info)
    
    def get_comparison(self) -> List[Dict]:
        """获取对比结果"""
        results = [self.baseline] if self.baseline else []
        results.extend(self.compressed_models)
        return results
    
    def find_best_tradeoff(self, max_accuracy_drop: float = 0.02) -> Optional[Dict]:
        """找到最佳压缩方案"""
        
        candidates = [
            m for m in self.compressed_models
            if m['accuracy_drop'] <= max_accuracy_drop
        ]
        
        if not candidates:
            return None
        
        # 按speedup排序
        candidates.sort(key=lambda x: x['speedup'], reverse=True)
        return candidates[0]
    
    def generate_report(self) -> str:
        """生成压缩效果报告"""
        report = f"# {self.model_name} 模型压缩报告\n\n"
        
        if not self.baseline:
            return report + "错误: 未设置基线模型\n"
        
        report += "## 基线模型\n\n"
        report += f"- 参数量: {self.baseline['params_millions']:.1f}M\n"
        report += f"- 显存占用: {self.baseline['memory_mb']:.1f}MB\n"
        report += f"- 延迟: {self.baseline['latency_ms']:.2f}ms\n"
        report += f"- 精度: {self.baseline['accuracy']:.4f}\n\n"
        
        report += "## 压缩方案对比\n\n"
        report += "| 方案 | 方法 | 参数比 | 显存比 | 加速比 | 精度损失 | 推荐 |\n"
        report += "|------|------|--------|--------|--------|----------|------|\n"
        
        best = self.find_best_tradeoff()
        
        for model in self.compressed_models:
            is_best = best and model['name'] == best['name']
            recommend = "⭐" if is_best else ""
            
            report += f"| {model['name']} | {model['method']} | "
            report += f"{model['params_ratio']:.2f}x | {model['memory_ratio']:.2f}x | "
            report += f"{model['speedup']:.2f}x | {model['accuracy_drop']:.4f} | {recommend} |\n"
        
        return report


class LatencyBenchmark:
    """延迟基准测试"""
    
    @staticmethod
    def measure_inference_latency(
        model_func: callable,
        input_data: any,
        warmup: int = 10,
        iterations: int = 100
    ) -> Dict:
        """测量推理延迟"""
        
        print(f"预热中... ({warmup}次)")
        for _ in range(warmup):
            model_func(input_data)
        
        print(f"测试中... ({iterations}次)")
        latencies = []
        
        for i in range(iterations):
            start = time.time()
            model_func(input_data)
            end = time.time()
            latencies.append((end - start) * 1000)  # 转换为毫秒
            
            if (i + 1) % 20 == 0:
                print(f"  进度: {i+1}/{iterations}")
        
        return {
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'min': np.min(latencies),
            'max': np.max(latencies),
            'p50': np.percentile(latencies, 50),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99)
        }
    
    @staticmethod
    def compare_batch_sizes(
        model_func: callable,
        batch_sizes: List[int],
        seq_length: int = 512
    ) -> Dict:
        """比较不同batch size的性能"""
        
        results = {}
        
        for bs in batch_sizes:
            print(f"\n测试 batch_size={bs}")
            
            # 模拟输入数据
            input_data = np.random.randn(bs, seq_length, 768)
            
            # 测量延迟
            latency_stats = LatencyBenchmark.measure_inference_latency(
                model_func, input_data, warmup=5, iterations=50
            )
            
            # 计算吞吐量
            throughput = bs / (latency_stats['mean'] / 1000)  # samples/second
            
            results[bs] = {
                'latency_ms': latency_stats['mean'],
                'throughput': throughput,
                'latency_per_sample': latency_stats['mean'] / bs
            }
        
        return results


def demonstrate_cost_calculation():
    """演示成本计算"""
    print("\n" + "="*70)
    print("💰 演示：GPU成本计算")
    print("="*70)
    
    calculator = GPUCostCalculator()
    
    # 训练成本比较
    print("\n【训练成本比较】")
    print("场景: 训练7B模型，预计100小时")
    
    gpus_to_compare = ["A100-80GB", "A100-40GB", "V100-32GB", "T4"]
    results = calculator.compare_gpus(gpus_to_compare, training_hours=100, num_gpus=4)
    
    print(f"\n{'GPU':<15} {'GPU数量':<10} {'计算成本':<12} {'能耗成本':<12} {'总成本':<12} {'TFLOPS':<10}")
    print("-" * 85)
    
    for result in results:
        print(f"{result['gpu']:<15} {result['num_gpus']:<10} "
              f"${result['compute_cost']:<11.2f} ${result['energy_cost']:<11.2f} "
              f"${result['total_cost']:<11.2f} {result['tflops_total']:<10.1f}")
    
    # 推理成本估算
    print("\n【推理成本估算】")
    print("场景: 100 QPS，平均延迟50ms")
    
    inference_cost = calculator.estimate_inference_cost(
        "T4",
        requests_per_second=100,
        avg_latency_ms=50
    )
    
    print(f"\nGPU: {inference_cost['gpu']}")
    print(f"所需GPU数量: {inference_cost['estimated_gpus']}")
    print(f"每日成本: ${inference_cost['daily_cost']:.2f}")
    print(f"每月成本: ${inference_cost['monthly_cost']:.2f}")
    print(f"每百万次请求成本: ${inference_cost['cost_per_million_requests']:.2f}")
    
    # ROI计算
    print("\n【投资回报分析】")
    print("场景: 云端vs本地GPU")
    
    roi = calculator.calculate_roi("A100-80GB", "RTX 4090", monthly_hours=200)
    
    print(f"\n云端GPU: {roi['cloud_gpu']}")
    print(f"每月成本: ${roi['cloud_monthly_cost']:.2f}")
    print(f"\n本地GPU: {roi['local_gpu']}")
    print(f"采购成本: ${roi['local_purchase_cost']:.2f}")
    print(f"回本时间: {roi['months_to_roi']:.1f}个月")
    print(f"是否推荐购买: {'✅ 是' if roi['break_even'] else '❌ 否'}")


def demonstrate_performance_profiling():
    """演示性能分析"""
    print("\n" + "="*70)
    print("⚡ 演示：性能分析")
    print("="*70)
    
    profiler = PerformanceProfiler("GPT-2 推理")
    profiler.start()
    
    print("\n【模拟推理测试】")
    
    # 模拟100次推理
    for i in range(100):
        # 模拟延迟和吞吐量
        latency = 45 + np.random.randn() * 5  # 平均45ms
        throughput = 20 + np.random.randn() * 2  # 平均20 tokens/s
        memory = 2048 + np.random.randn() * 100  # 平均2GB
        gpu_util = 75 + np.random.randn() * 10  # 平均75%
        
        profiler.record_latency(latency)
        profiler.record_throughput(throughput)
        profiler.record_memory(memory)
        profiler.record_gpu_util(gpu_util)
        
        if (i + 1) % 25 == 0:
            print(f"  已完成 {i+1}/100 次推理")
        
        time.sleep(0.01)
    
    # 生成报告
    print("\n【性能报告】")
    summary = profiler.get_summary()
    
    print("\n延迟 (ms):")
    print(f"  平均: {summary['latency']['mean']:.2f}ms")
    print(f"  P50: {summary['latency']['p50']:.2f}ms")
    print(f"  P95: {summary['latency']['p95']:.2f}ms")
    print(f"  P99: {summary['latency']['p99']:.2f}ms")
    
    print("\n吞吐量 (tokens/s):")
    print(f"  平均: {summary['throughput']['mean']:.2f}")
    print(f"  最大: {summary['throughput']['max']:.2f}")
    
    print("\n显存使用 (MB):")
    print(f"  平均: {summary['memory_usage']['mean']:.1f}")
    print(f"  峰值: {summary['memory_usage']['max']:.1f}")
    
    print("\nGPU利用率 (%):")
    print(f"  平均: {summary['gpu_utilization']['mean']:.1f}%")


def demonstrate_model_compression():
    """演示模型压缩效果对比"""
    print("\n" + "="*70)
    print("🗜️  演示：模型压缩效果对比")
    print("="*70)
    
    compressor = ModelCompressor("BERT-Base")
    
    # 设置基线
    compressor.set_baseline(
        params_millions=110,
        memory_mb=1200,
        latency_ms=45,
        accuracy=0.920
    )
    
    print("\n【基线模型】")
    print("BERT-Base: 110M参数，1200MB显存，45ms延迟，92.0%精度")
    
    # 添加压缩方案
    print("\n【测试压缩方案】")
    
    # 量化
    compressor.add_compressed_model(
        name="INT8量化",
        params_millions=110,
        memory_mb=320,
        latency_ms=22,
        accuracy=0.917,
        compression_method="INT8 Quantization"
    )
    print("  ✅ INT8量化")
    
    # 蒸馏
    compressor.add_compressed_model(
        name="知识蒸馏",
        params_millions=66,
        memory_mb=720,
        latency_ms=28,
        accuracy=0.912,
        compression_method="Knowledge Distillation"
    )
    print("  ✅ 知识蒸馏")
    
    # 剪枝
    compressor.add_compressed_model(
        name="结构化剪枝",
        params_millions=77,
        memory_mb=840,
        latency_ms=32,
        accuracy=0.915,
        compression_method="Structured Pruning"
    )
    print("  ✅ 结构化剪枝")
    
    # 蒸馏+量化
    compressor.add_compressed_model(
        name="蒸馏+量化",
        params_millions=66,
        memory_mb=180,
        latency_ms=14,
        accuracy=0.908,
        compression_method="Distillation + Quantization"
    )
    print("  ✅ 蒸馏+量化")
    
    # 生成报告
    print("\n【压缩效果对比】")
    report = compressor.generate_report()
    print(report)
    
    # 推荐方案
    best = compressor.find_best_tradeoff(max_accuracy_drop=0.02)
    if best:
        print(f"\n【推荐方案】")
        print(f"方法: {best['method']}")
        print(f"加速比: {best['speedup']:.2f}x")
        print(f"显存节省: {(1-best['memory_ratio'])*100:.1f}%")
        print(f"精度损失: {best['accuracy_drop']:.4f} (可接受)")


def demonstrate_latency_benchmark():
    """演示延迟基准测试"""
    print("\n" + "="*70)
    print("📊 演示：延迟基准测试")
    print("="*70)
    
    # 模拟模型推理函数
    def mock_model_inference(input_data):
        # 模拟计算时间
        time.sleep(0.001 * len(input_data))  # 1ms per sample
        return np.random.randn(*input_data.shape)
    
    print("\n【测试不同batch size】")
    
    batch_sizes = [1, 4, 8, 16, 32]
    results = LatencyBenchmark.compare_batch_sizes(
        mock_model_inference,
        batch_sizes,
        seq_length=512
    )
    
    print("\n【结果对比】")
    print(f"\n{'Batch Size':<12} {'延迟(ms)':<15} {'吞吐量(samples/s)':<20} {'单样本延迟(ms)':<20}")
    print("-" * 75)
    
    for bs, stats in results.items():
        print(f"{bs:<12} {stats['latency_ms']:<15.2f} "
              f"{stats['throughput']:<20.1f} {stats['latency_per_sample']:<20.2f}")
    
    print("\n【分析】")
    print("  - Batch size越大，总吞吐量越高")
    print("  - 但单样本延迟也会增加")
    print("  - 需要在延迟和吞吐量之间权衡")


def run_week21_demo():
    """运行Week 21完整演示"""
    print("\n" + "="*70)
    print("🚀 Week 21: GPU性能优化与成本评估 - 完整演示")
    print("="*70)
    
    # 1. 成本计算
    demonstrate_cost_calculation()
    
    input("\n按Enter继续查看性能分析...")
    
    # 2. 性能分析
    demonstrate_performance_profiling()
    
    input("\n按Enter继续查看模型压缩对比...")
    
    # 3. 模型压缩
    demonstrate_model_compression()
    
    input("\n按Enter继续查看延迟基准测试...")
    
    # 4. 延迟测试
    demonstrate_latency_benchmark()
    
    print("\n" + "="*70)
    print("✅ Week 21演示完成！")
    print("="*70)
    print("\n核心收获：")
    print("  1. 掌握了GPU成本计算和ROI分析")
    print("  2. 学会了系统性能分析和监控")
    print("  3. 理解了模型压缩的各种方法和权衡")
    print("  4. 可以进行科学的性能优化决策")


if __name__ == "__main__":
    run_week21_demo()
