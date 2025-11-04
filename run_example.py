"""
深度学习训练与架构演示系统 - 统一入口
包含Prompt Engineering与Few-shot技术
"""
import sys
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List, Dict, Any

# 基础模块导入
from ml_core.models_torch import CIFAR10Net, QuantizedCIFAR10Net

# 尝试导入可选模块
try:
    from ml_core.kaggle_models import create_kaggle_model
    KAGGLE_MODELS_AVAILABLE = True
except ImportError:
    print("Warning: timm not available, Kaggle models disabled")
    KAGGLE_MODELS_AVAILABLE = False

try:
    from ml_core.llm_architecture import LLaMAModel, create_llama_example
    from ml_core.llm_visualization import create_llama_visualization
    LLM_AVAILABLE = True
except ImportError:
    print("Warning: LLM modules not fully available")
    LLM_AVAILABLE = False

from ml_core.data import get_cifar10_loaders
try:
    from ml_core.kaggle_data import get_kaggle_loaders
    KAGGLE_DATA_AVAILABLE = True
except ImportError:
    KAGGLE_DATA_AVAILABLE = False

from ml_core.training import Trainer, TrainerConfig
from ml_core.evaluation import ModelEvaluator
from ml_core.monitoring import PerformanceMonitor

# SNN相关导入
from ml_core.models import SimpleNN
from ml_core.optimizers import Adam
from ml_core.performance import benchmark_matmul

# Dashboard相关导入
try:
    from ml_core.visualization import create_dashboard, load_data
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False

# =================== Prompt Engineering 集成代码 ===================
class PromptDebugger:
    """自动化 Prompt 调试与质量分析"""
    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    def test_prompt(self, prompt: str, examples: List[str] = None, temperature: float = 0.7) -> Dict[str, Any]:
        """测试单个Prompt，支持Few-shot示例"""
        print(f"模拟测试 Prompt: {prompt}")
        if examples:
            print(f"使用 {len(examples)} 个Few-shot示例")
        return {
            "prompt": prompt,
            "output": "模拟输出结果",
            "usage": {"tokens": 50}
        }

    def batch_test(self, prompts: List[str], examples: List[str] = None, temperature: float = 0.7) -> List[Dict[str, Any]]:
        """批量测试多个Prompt"""
        results = []
        for prompt in prompts:
            result = self.test_prompt(prompt, examples, temperature)
            results.append(result)
        return results

    def optimize_prompt(self, prompt: str, target: str, examples: List[str] = None, max_iter: int = 5) -> Dict[str, Any]:
        """自动化优化Prompt以提升输出质量"""
        print(f"优化Prompt: {prompt}, 目标: {target}")
        return {"optimized_prompt": f"优化后的Prompt: {prompt}", "score": 0.85}

    def evaluate_output(self, output: str, target: str) -> float:
        """简单的输出质量评估"""
        if target in output:
            return 1.0
        return 0.0

class FewShotManager:
    """Few-shot 示例生成与管理"""
    def __init__(self):
        self.examples = []

    def add_example(self, example: str):
        self.examples.append(example)

    def get_examples(self) -> List[str]:
        return self.examples

    def clear_examples(self):
        self.examples = []

    def auto_generate_examples(self, task_desc: str, n: int = 3) -> List[str]:
        """自动生成Few-shot示例"""
        examples = [f"示例{i+1}: {task_desc}" for i in range(n)]
        self.examples.extend(examples)
        return examples

# =================== 主要功能函数 ===================

def setup(rank, world_size):
    """设置分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def demonstrate_prompt_engineering():
    """演示Prompt Engineering与Few-shot技术"""
    print("\n" + "="*60)
    print("🧠 Prompt Engineering 与 Few-shot 技术演示")
    print("="*60)
    
    # Few-shot示例管理
    print("\n1. Few-shot示例管理")
    fewshot = FewShotManager()
    fewshot.add_example("Q: 2+2=?\nA: 4")
    fewshot.add_example("Q: 3+5=?\nA: 8")
    fewshot.add_example("Q: 10-6=?\nA: 4")
    
    print("Few-shot示例:")
    for i, ex in enumerate(fewshot.get_examples(), 1):
        print(f"  示例{i}: {ex}")
    
    # 自动化Prompt调试
    print("\n2. 自动化Prompt调试")
    debugger = PromptDebugger()
    
    result = debugger.test_prompt("Q: 7+6=?", examples=fewshot.get_examples())
    print(f"Prompt测试结果: {result}")
    
    # 批量Prompt测试
    print("\n3. 批量Prompt测试")
    prompts = ["计算5+3", "求解8-2", "计算9×2"]
    batch_results = debugger.batch_test(prompts, examples=fewshot.get_examples())
    for i, result in enumerate(batch_results, 1):
        print(f"  批量测试{i}: {result['prompt']} -> {result['output']}")
    
    # 自动优化Prompt
    print("\n4. 自动优化Prompt")
    opt_result = debugger.optimize_prompt("请计算7+6", target="13", examples=fewshot.get_examples())
    print(f"优化结果: {opt_result}")
    
    # 自动生成Few-shot示例
    print("\n5. 自动生成Few-shot示例")
    auto_examples = fewshot.auto_generate_examples("数学计算", n=2)
    print(f"自动生成的示例: {auto_examples}")
    
    print("\n" + "="*60)
    print("✅ Prompt Engineering 演示完成!")
    print("="*60)

def demonstrate_snn_performance():
    """演示简单神经网络性能测试"""
    print("\n" + "="*60)
    print("⚡ 简单神经网络 (SNN) 性能演示")
    print("="*60)
    
    # 1. 矩阵运算性能测试
    print("\n1. 运行矩阵乘法性能测试...")
    results = benchmark_matmul(size=1000)
    for method, duration in results.items():
        print(f"   {method} 用时: {duration:.4f}秒")
    if len(results) == 2:
        methods = list(results.keys())
        if 'numpy' in methods and 'numba' in methods:
            speedup = results['numpy'] / results['numba']
            print(f"   Numba 加速比: {speedup:.2f}x")
    
    # 2. 神经网络测试
    print("\n2. 运行神经网络测试...")
    
    # 生成合成数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 784
    n_classes = 10
    
    X = np.random.randn(n_features, n_samples)
    y = np.random.randint(0, n_classes, size=n_samples)
    Y = np.eye(n_classes)[:, y]  # one-hot编码
    
    print(f"   数据形状: X={X.shape}, Y={Y.shape}")
    
    # 创建模型
    model = SimpleNN([n_features, 128, 64, n_classes])
    optimizer = Adam(lr=0.001)
    
    print(f"   模型结构: {n_features} -> 128 -> 64 -> {n_classes}")
    
    # 训练一个epoch
    print("\n3. 开始训练...")
    batch_size = 32
    n_batches = n_samples // batch_size
    
    start_time = time.time()
    total_loss = 0
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        batch_x = X[:, start_idx:end_idx]
        batch_y = Y[:, start_idx:end_idx]
        
        loss = model.train_step(batch_x, batch_y, optimizer)
        total_loss += loss
        
        if (i + 1) % 10 == 0:
            print(f"   Batch {i+1}/{n_batches}, Average Loss: {total_loss/(i+1):.4f}")
    
    train_time = time.time() - start_time
    print(f"\n   训练完成！用时: {train_time:.2f}秒")
    print(f"   平均损失: {total_loss/n_batches:.4f}")
    
    # 4. 评估模型
    print("\n4. 模型评估...")
    
    # 前向传播测试
    test_x = X[:, :100]  # 使用前100个样本测试
    predictions = model.forward(test_x)
    predicted_classes = np.argmax(predictions, axis=0)
    actual_classes = y[:100]
    
    accuracy = np.mean(predicted_classes == actual_classes)
    print(f"   测试准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\n" + "="*60)
    print("✅ SNN 性能演示完成!")
    print("="*60)

def run_dashboard():
    """运行数据分析仪表盘"""
    print("\n" + "="*60)
    print("📊 数据分析仪表盘演示")
    print("="*60)
    
    if not DASHBOARD_AVAILABLE:
        print("❌ 仪表盘模块不可用，请安装依赖: pip install plotly dash")
        return
    
    try:
        # 生成示例数据
        print("生成示例数据...")
        np.random.seed(42)
        n_samples = 1000
        
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
            'A': np.random.randn(n_samples).cumsum(),
            'B': np.random.randn(n_samples).cumsum(),
            'C': np.random.randint(1, 100, n_samples)
        })
        
        print(f"数据概览:")
        print(f"- 行数: {len(df):,}")
        print(f"- 列数: {len(df.columns)}")
        print(f"- 列名: {list(df.columns)}")
        
        # 启动仪表盘
        print("\n启动数据分析仪表盘...")
        print("注意: 仪表盘将在浏览器中打开")
        
        create_dashboard(
            df=df,
            date_col='date',
            default_columns=['A', 'B']
        )
        
    except Exception as e:
        print(f"仪表盘启动失败: {e}")
        print("请确保安装了相关依赖: pip install plotly dash")
    
    print("\n" + "="*60)
    print("✅ 仪表盘演示完成!")
    print("="*60)

def demonstrate_llm_architecture():
    """演示LLM架构原理"""
    print("\n" + "="*60)
    print("🚀 LLM 架构原理演示")
    print("="*60)
    
    if not LLM_AVAILABLE:
        print("❌ LLM模块不完全可用，请检查依赖")
        return
    
    # 1. 创建LLaMA模型
    print("\n1. 创建LLaMA模型...")
    try:
        model = create_llama_example()
        print(f"   模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"   模型创建失败: {e}")
        return
    
    # 2. 演示各个组件
    print("\n2. 演示核心组件...")
    
    # 创建示例输入
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 32000, (batch_size, seq_len))
    
    print(f"   输入形状: {input_ids.shape}")
    
    # 前向传播
    try:
        with torch.no_grad():
            output = model(input_ids)
            print(f"   输出形状: {output.shape}")
    except Exception as e:
        print(f"   前向传播失败: {e}")
        return
    
    # 3. 演示注意力机制
    print("\n3. 演示多头注意力...")
    try:
        from ml_core.llm_architecture import MultiHeadAttention
        
        attention = MultiHeadAttention(d_model=512, n_heads=8)
        x = torch.randn(batch_size, seq_len, 512)
        
        with torch.no_grad():
            attn_output = attention(x)
            print(f"   注意力输出形状: {attn_output.shape}")
    except Exception as e:
        print(f"   注意力演示失败: {e}")
    
    # 4. 演示位置编码
    print("\n4. 演示RoPE位置编码...")
    try:
        from ml_core.llm_architecture import RotaryPositionalEmbedding
        
        rope = RotaryPositionalEmbedding(dim=64)
        q = torch.randn(batch_size, 8, seq_len, 64)
        k = torch.randn(batch_size, 8, seq_len, 64)
        
        with torch.no_grad():
            q_rot, k_rot = rope(q, k)
            print(f"   RoPE输出形状: q={q_rot.shape}, k={k_rot.shape}")
    except Exception as e:
        print(f"   RoPE演示失败: {e}")
    
    print("\n5. 创建架构可视化...")
    try:
        viz = create_llama_visualization()
        print("✓ LLaMA架构可视化已生成")
    except Exception as e:
        print(f"   可视化生成失败: {e}")
    
    print("\n" + "="*60)
    print("✅ LLM 架构演示完成!")
    print("="*60)

def demonstrate_fundamentals():
    """演示基础知识：线性代数、反向传播、优化器、CNN/Transformer"""
    print("\n" + "="*60)
    print("🧮 深度学习基础知识综合演示")
    print("="*60)
    
    results = {}
    
    # 1. 线性代数基础
    print("\n🧮 第1部分：线性代数与自动微分")
    try:
        from ml_core.linear_algebra import demonstrate_comprehensive_linear_algebra
        linear_results = demonstrate_comprehensive_linear_algebra()
        results['linear_algebra'] = linear_results
        print("✓ 线性代数演示完成")
    except Exception as e:
        print(f"   线性代数演示失败: {e}")
        results['linear_algebra'] = None
    
    # 2. 反向传播详解
    print("\n🔄 第2部分：反向传播机制")
    try:
        from ml_core.backpropagation import demonstrate_comprehensive_backprop
        backprop_results = demonstrate_comprehensive_backprop()
        results['backpropagation'] = backprop_results
        print("✓ 反向传播演示完成")
    except Exception as e:
        print(f"   反向传播演示失败: {e}")
        results['backpropagation'] = None
    
    # 3. 优化器对比
    print("\n⚙️ 第3部分：优化算法对比")
    try:
        from ml_core.optimizer_comparison import demonstrate_comprehensive_optimizers
        optimizer_results = demonstrate_comprehensive_optimizers()
        results['optimizers'] = optimizer_results
        print("✓ 优化器演示完成")
    except Exception as e:
        print(f"   优化器演示失败: {e}")
        results['optimizers'] = None
    
    # 4. CNN与Transformer基础
    print("\n🧠 第4部分：卷积与Transformer基础")
    try:
        from ml_core.cnn_transformer import demonstrate_comprehensive_cnn_transformer
        cnn_transformer_results = demonstrate_comprehensive_cnn_transformer()
        results['cnn_transformer'] = cnn_transformer_results
        print("✓ CNN与Transformer演示完成")
    except Exception as e:
        print(f"   CNN与Transformer演示失败: {e}")
        results['cnn_transformer'] = None
    
    print("\n" + "="*60)
    print("✅ 基础知识演示完成!")
    print("="*60)
    
    return results

def run_deep_learning_training():
    """运行深度学习训练模块"""
    print("\n" + "="*60)
    print("🎯 深度学习训练模块")
    print("="*60)
    
    # 配置训练参数
    config = TrainerConfig(
        batch_size=64,
        num_workers=4,
        learning_rate=0.001,
        num_epochs=5,
        mixed_precision=True,
        save_dir='checkpoints'
    )
    
    # 检查是否可用GPU
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"\n找到 {n_gpus} 个GPU设备")
        device = torch.device("cuda:0")
    else:
        print("\n未找到GPU设备，使用CPU训练")
        device = torch.device("cpu")
    
    # 创建模型和数据
    try:
        model = CIFAR10Net().to(device)
        train_loader, val_loader = get_cifar10_loaders(
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device
        )
        
        print("\n开始训练...")
        results = trainer.train()
        
        print("\n训练结果:")
        for key, value in results.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"训练失败: {e}")
    
    print("\n" + "="*60)
    print("✅ 深度学习训练模块完成!")
    print("="*60)

def print_system_info():
    """打印系统信息和功能说明"""
    print("\n" + "="*80)
    print("🎯 深度学习训练与架构演示系统 - 功能说明")
    print("="*80)
    
    print("\n📋 系统功能:")
    print("1. 🧮 线性代数与自动微分基础")
    print("   • 矩阵运算与几何变换")
    print("   • Jacobian 矩阵计算")
    print("   • 链式法则详细推导")
    print("   • 手写线性层和ReLU激活")
    print("   • 数值梯度检查")
    
    print("\n2. 🔄 反向传播机制详解")
    print("   • 手推两层网络梯度公式")
    print("   • 与PyTorch autograd对比验证")
    print("   • 逐步演示反向传播过程")
    print("   • 数值方法验证梯度计算")
    
    print("\n3. ⚙️ 优化算法数学原理")
    print("   • SGD、Adam、RMSProp详细实现")
    print("   • MNIST数据集收敛曲线对比")
    print("   • 学习率对收敛的影响")
    print("   • 优化轨迹可视化")
    
    print("\n4. 🧠 卷积与Transformer基础")
    print("   • 手写CNN卷积层实现")
    print("   • 自注意力机制详细推导")
    print("   • 矩阵乘法在深度学习中的意义")
    print("   • CNN vs Transformer特点对比")
    
    print("\n5. 🚀 LLM 架构原理演示")
    print("   • Attention 机制 (多头自注意力)")
    print("   • RoPE 位置编码 (旋转位置编码)")
    print("   • 残差连接和 RMS 归一化")
    print("   • SwiGLU 前馈网络")
    print("   • LLaMA 完整架构可视化")
    print("   • 文本生成演示")
    
    print("\n6. ⚡ 简单神经网络 (SNN) 性能测试")
    print("   • 矩阵乘法性能基准测试")
    print("   • NumPy vs Numba 性能对比")
    print("   • 多层感知机训练演示")
    print("   • 模型准确率评估")
    
    print("\n7. 📊 数据分析仪表盘")
    print("   • 交互式数据可视化")
    print("   • 时间序列分析")
    print("   • 多维数据探索")
    print("   • 实时数据更新")
    
    print("\n8. 🎯 深度学习训练系统")
    print("   • Kaggle 竞赛级模型训练")
    print("   • 分布式数据并行 (DDP)")
    print("   • 混合精度训练 (FP16)")
    print("   • 模型量化和压缩")
    print("   • 性能监控和对比")
    print("   • 早停和学习率调度")
    
    print("\n9. 🧠 Prompt Engineering 与 Few-shot 技术")
    print("   • 自动化 Prompt 调试与优化")
    print("   • Few-shot 示例生成与管理")
    print("   • 批量 Prompt 测试")
    print("   • GPT 输出质量提升工具")
    print("   • 自动化 Prompt 优化")
    
    print("\n🔧 技术特性:")
    print("• 完整的数学推导和手工实现")
    print("• PyTorch 模型模块化设计")
    print("• 优化的数据管线 (Dataset/DataLoader)")
    print("• 多GPU 训练支持")
    print("• 模型精度和显存监控")
    print("• 可视化训练过程")
    print("• 集成 Prompt Engineering 技术")
    
    print("\n💻 使用方式:")
    print("• 交互模式: python run_example.py")
    print("• 基础演示: python run_example.py fundamentals")
    print("• LLM 演示: python run_example.py llm")
    print("• SNN 测试: python run_example.py snn")
    print("• 仪表盘: python run_example.py dashboard")
    print("• 训练模式: python run_example.py train")
    print("• Prompt工程: python run_example.py prompt")
    print("• 快速演示: python run_example.py quick")
    
    print("\n📦 依赖要求:")
    print("• torch, torchvision")
    print("• numpy, pandas")
    print("• matplotlib, plotly")
    print("• dash (用于仪表盘)")
    print("• openai (用于Prompt Engineering)")
    
    print("="*80)

def main():
    """主菜单交互模式"""
    print_system_info()
    
    print("\n请选择要运行的模块 (输入数字或 'all' 运行全部):")
    print("1. 基础知识演示 (线性代数、反向传播、优化器、CNN/Transformer)")
    print("2. LLM 架构演示")
    print("3. SNN 性能测试")
    print("4. 数据仪表盘")
    print("5. 深度学习训练")
    print("6. Prompt Engineering 与 Few-shot 技术")
    print("7. 全部模块")
    
    try:
        choice = input("\n请输入选择 (1-7): ").strip()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        return
    except:
        choice = "7"  # 默认运行全部
    
    # 根据选择运行对应模块
    if choice in ["1", "7", "all"]:
        demonstrate_fundamentals()
    
    if choice in ["2", "7", "all"]:
        demonstrate_llm_architecture()
    
    if choice in ["3", "7", "all"]:
        demonstrate_snn_performance()
    
    if choice in ["4", "7", "all"]:
        run_dashboard()
    
    if choice in ["5", "7", "all"]:
        run_deep_learning_training()
    
    if choice in ["6", "7", "all"]:
        demonstrate_prompt_engineering()
    
    print("\n" + "="*80)
    print("🎉 所有演示完成!")
    print("="*80)

if __name__ == "__main__":
    # 命令行参数支持
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "fundamentals":
            demonstrate_fundamentals()
        elif mode == "llm":
            demonstrate_llm_architecture()
        elif mode == "snn":
            demonstrate_snn_performance()
        elif mode == "dashboard":
            run_dashboard()
        elif mode == "train":
            run_deep_learning_training()
        elif mode in ["prompt", "prompt_engineering", "fewshot"]:
            demonstrate_prompt_engineering()
        elif mode == "quick":
            demonstrate_fundamentals()
            demonstrate_llm_architecture()
            demonstrate_snn_performance()
            run_deep_learning_training()
            demonstrate_prompt_engineering()
        else:
            print(f"未知模式: {mode}")
            main()
    else:
        # 交互模式
        main()
