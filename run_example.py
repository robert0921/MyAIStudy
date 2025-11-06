"""
深度学习训练与架构演示系统 - 统一入口
包含Prompt Engineering与Few-shot技术
"""
import sys
import os
import time
from typing import List, Dict, Any

# 尝试导入numpy和pandas
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("Warning: numpy not available")
    NUMPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    print("Warning: pandas not available")
    PANDAS_AVAILABLE = False

# 尝试导入torch相关模块
try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.nn.parallel import DistributedDataParallel as DDP
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not available, some features disabled")
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    dist = None
    mp = None
    DDP = None

# 基础模块导入
if TORCH_AVAILABLE:
    try:
        from ml_core.models_torch import CIFAR10Net, QuantizedCIFAR10Net
        MODELS_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: models_torch not available: {e}")
        MODELS_AVAILABLE = False
else:
    MODELS_AVAILABLE = False

# 尝试导入可选模块
try:
    from ml_core.kaggle_models import create_kaggle_model
    KAGGLE_MODELS_AVAILABLE = True
except ImportError:
    KAGGLE_MODELS_AVAILABLE = False

try:
    from ml_core.llm_architecture import LLaMAModel, create_llama_example
    from ml_core.llm_visualization import create_llama_visualization
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

if TORCH_AVAILABLE:
    try:
        from ml_core.data import get_cifar10_loaders
        DATA_AVAILABLE = True
    except ImportError:
        DATA_AVAILABLE = False
else:
    DATA_AVAILABLE = False

try:
    from ml_core.kaggle_data import get_kaggle_loaders
    KAGGLE_DATA_AVAILABLE = True
except ImportError:
    KAGGLE_DATA_AVAILABLE = False

if TORCH_AVAILABLE:
    try:
        from ml_core.training import Trainer, TrainerConfig
        from ml_core.evaluation import ModelEvaluator
        from ml_core.monitoring import PerformanceMonitor
        TRAINING_AVAILABLE = True
    except ImportError:
        TRAINING_AVAILABLE = False
else:
    TRAINING_AVAILABLE = False

# SNN相关导入
if NUMPY_AVAILABLE:
    try:
        from ml_core.models import SimpleNN
        from ml_core.optimizers import Adam
        from ml_core.performance import benchmark_matmul
        SNN_AVAILABLE = True
    except ImportError:
        SNN_AVAILABLE = False
else:
    SNN_AVAILABLE = False

# Dashboard相关导入
try:
    from ml_core.visualization import create_dashboard, load_data
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False

# Pruning相关导入
try:
    from ml_core.pruning import ModelPruner, demonstrate_pruning
    PRUNING_AVAILABLE = True
except ImportError:
    PRUNING_AVAILABLE = False

# Fine-tuning相关导入
try:
    from ml_core.finetuning import (
        demonstrate_lora_finetuning,
        demonstrate_qlora_comparison,
        demonstrate_peft_methods
    )
    FINETUNING_AVAILABLE = True
except ImportError:
    FINETUNING_AVAILABLE = False

# Inference Optimization相关导入
try:
    from ml_core.inference_optimization import (
        demonstrate_kv_cache,
        demonstrate_batched_inference,
        demonstrate_cache_vs_no_cache
    )
    INFERENCE_OPT_AVAILABLE = True
except ImportError:
    INFERENCE_OPT_AVAILABLE = False

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
    if not TORCH_AVAILABLE or dist is None:
        raise RuntimeError("PyTorch not available")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """清理分布式训练环境"""
    if TORCH_AVAILABLE and dist is not None:
        dist.destroy_process_group()

def demonstrate_model_pruning():
    """演示模型剪枝功能"""
    print("\n" + "="*70)
    print("🔪 模型剪枝与压缩演示")
    print("="*70)
    
    if not PRUNING_AVAILABLE:
        print("❌ 剪枝模块不可用，请检查安装")
        return
    
    if not TORCH_AVAILABLE or not MODELS_AVAILABLE:
        print("❌ PyTorch或模型模块不可用")
        return
    
    # 调用剪枝演示函数
    demonstrate_pruning()
    
    print("\n💡 实际应用提示:")
    print("  1. 在真实场景中，剪枝后需要对模型进行微调")
    print("  2. 可以使用 ModelEvaluator 比较剪枝前后的精度")
    print("  3. 结构化剪枝可以真正减少计算量和内存占用")
    print("  4. 建议使用迭代剪枝策略，逐步提高剪枝比例")

def demonstrate_llm_finetuning():
    """演示大模型微调 (LoRA/QLoRA/PEFT)"""
    print("\n" + "="*70)
    print("🎨 大模型微调演示 (LoRA/QLoRA/PEFT)")
    print("="*70)
    
    if not FINETUNING_AVAILABLE:
        print("❌ 微调模块不可用，请检查安装")
        return
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch不可用")
        return
    
    print("\n选择演示内容:")
    print("1. LoRA 微调演示")
    print("2. QLoRA vs LoRA 对比")
    print("3. PEFT 方法对比")
    print("4. 全部演示")
    
    try:
        choice = input("\n请选择 (1-4, 默认4): ").strip()
        if not choice:
            choice = "4"
    except (KeyboardInterrupt, EOFError):
        print("\n使用默认选项: 4")
        choice = "4"
    
    try:
        if choice in ["1", "4"]:
            demonstrate_lora_finetuning()
        
        if choice in ["2", "4"]:
            demonstrate_qlora_comparison()
        
        if choice in ["3", "4"]:
            demonstrate_peft_methods()
    
    except Exception as e:
        print(f"\n微调演示失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n💡 实际应用提示:")
    print("  1. LoRA适用于7B以下模型的高效微调")
    print("  2. QLoRA可在单卡上微调30B+模型")
    print("  3. 建议从rank=8开始尝试，根据效果调整")
    print("  4. 只需训练0.1-1%的参数量，显著降低成本")
    
    print("\n" + "="*70)
    print("✅ 大模型微调演示完成!")
    print("="*70)

def demonstrate_inference_optimization():
    """演示推理优化 (Batched Inference / KV Cache)"""
    print("\n" + "="*70)
    print("⚡ 推理优化演示 (Batched Inference / KV Cache)")
    print("="*70)
    
    if not INFERENCE_OPT_AVAILABLE:
        print("❌ 推理优化模块不可用，请检查安装")
        return
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch不可用")
        return
    
    print("\n选择演示内容:")
    print("1. KV Cache 原理演示")
    print("2. 批量推理优化")
    print("3. 性能对比 (Cache vs No-Cache)")
    print("4. 全部演示")
    
    try:
        choice = input("\n请选择 (1-4, 默认4): ").strip()
        if not choice:
            choice = "4"
    except (KeyboardInterrupt, EOFError):
        print("\n使用默认选项: 4")
        choice = "4"
    
    try:
        if choice in ["1", "4"]:
            demonstrate_kv_cache()
        
        if choice in ["2", "4"]:
            demonstrate_batched_inference()
        
        if choice in ["3", "4"]:
            demonstrate_cache_vs_no_cache()
    
    except Exception as e:
        print(f"\n推理优化演示失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n💡 实际应用提示:")
    print("  1. KV Cache可将生成速度提升2-10倍")
    print("  2. 批量推理提高吞吐量，但会增加延迟")
    print("  3. 生产环境建议batch_size=4-16")
    print("  4. 注意显存占用：KV Cache = 2×batch×layers×heads×seq_len×head_dim")
    
    print("\n" + "="*70)
    print("✅ 推理优化演示完成!")
    print("="*70)

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
    
    if not SNN_AVAILABLE:
        print("❌ SNN模块不可用，请安装numpy: pip install numpy")
        return
    
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
        # 1. 尝试加载数据
        df = None
        try:
            # 尝试从文件加载
            if os.path.exists('data.csv'):
                df = load_data('data.csv')
                print("✓ 从文件加载数据成功")
        except Exception as e:
            print(f"   从文件加载失败: {e}")
        
        # 如果文件不存在或加载失败，生成示例数据
        if df is None:
            print("\n生成示例数据...")
            np.random.seed(42)
            n_samples = 1000
            
            df = pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
                'A': np.random.randn(n_samples).cumsum(),
                'B': np.random.randn(n_samples).cumsum() * 2 + 1,
                'C': np.random.randint(1, 100, n_samples),
                'D': np.random.exponential(2, n_samples)
            })
            print(f"✓ 生成示例数据完成")
        
        print(f"\n数据概览:")
        print(f"- 行数: {len(df):,}")
        print(f"- 列数: {len(df.columns)}")
        print(f"- 列名: {list(df.columns)}")
        
        # 2. 启动仪表盘
        print("\n启动数据分析仪表盘...")
        print("注意: 仪表盘将在浏览器中打开")
        print("      按 Ctrl+C 停止仪表盘服务")
        
        create_dashboard(
            df=df,
            date_col='date',
            default_columns=['A', 'B']
        )
        
    except KeyboardInterrupt:
        print("\n仪表盘服务已停止")
    except Exception as e:
        print(f"\n仪表盘启动失败: {e}")
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
            outputs = model(input_ids)
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs)
            else:
                logits = outputs
            print(f"   输出logits形状: {logits.shape}")
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
            attn_out = attention(x)
            # 处理可能的元组返回值
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]
            print(f"   注意力输出形状: {attn_out.shape}")
    except Exception as e:
        print(f"   注意力演示失败: {e}")
    
    # 4. 演示位置编码
    print("\n4. 演示RoPE位置编码...")
    try:
        from ml_core.llm_architecture import RotaryPositionalEmbedding
        
        rope = RotaryPositionalEmbedding(dim=64)
        q = torch.randn(batch_size, 8, seq_len, 64)  # [batch, heads, seq, head_dim]
        k = torch.randn(batch_size, 8, seq_len, 64)
        
        with torch.no_grad():
            q_with_rope, k_with_rope = rope(q, k)
            print(f"   RoPE编码后的Q形状: {q_with_rope.shape}")
            print(f"   RoPE编码后的K形状: {k_with_rope.shape}")
    except Exception as e:
        print(f"   RoPE演示失败: {e}")
    
    # 5. 演示残差连接和层归一化
    print("\n5. 演示残差连接和RMSNorm...")
    try:
        from ml_core.llm_architecture import RMSNorm, TransformerBlock
        
        norm = RMSNorm(d_model=512)
        transformer_block = TransformerBlock(d_model=512, n_heads=8, d_ff=2048)
        
        x = torch.randn(batch_size, seq_len, 512)
        
        with torch.no_grad():
            # 原始输入
            print(f"   输入均值: {x.mean():.4f}, 标准差: {x.std():.4f}")
            
            # 经过RMSNorm
            normed_x = norm(x)
            print(f"   RMSNorm后均值: {normed_x.mean():.4f}, 标准差: {normed_x.std():.4f}")
            
            # 经过Transformer块（包含残差连接）
            block_out = transformer_block(x)
            if isinstance(block_out, tuple):
                block_out = block_out[0]
            print(f"   Transformer块输出形状: {block_out.shape}")
    except Exception as e:
        print(f"   Transformer演示失败: {e}")
    
    # 6. 演示文本生成
    print("\n6. 演示文本生成...")
    try:
        # 创建简单的输入序列
        input_sequence = torch.randint(1, 1000, (1, 5))  # 避免使用0（可能是pad token）
        print(f"   输入序列: {input_sequence.tolist()}")
        
        # 检查模型是否有generate方法
        if hasattr(model, 'generate'):
            with torch.no_grad():
                generated = model.generate(
                    input_sequence,
                    max_new_tokens=10,
                    temperature=1.0,
                    do_sample=True
                )
                print(f"   生成序列: {generated.tolist()}")
                print(f"   新生成的token数: {generated.shape[1] - input_sequence.shape[1]}")
        else:
            print("   模型不支持generate方法，跳过文本生成演示")
    except Exception as e:
        print(f"   文本生成失败: {e}")
    
    # 7. 创建架构可视化
    print("\n7. 创建架构可视化...")
    try:
        viz = create_llama_visualization()
        print("   ✓ LLaMA架构可视化已生成")
    except Exception as e:
        print(f"   可视化生成失败: {e}")
        print("   请确保安装了matplotlib: pip install matplotlib")
    
    print("\n" + "="*60)
    print("✅ LLM 架构演示完成!")
    print("="*60)

def demonstrate_fundamentals():
    """演示基础知识：线性代数、反向传播、优化器、CNN/Transformer"""
    print("\n" + "="*60)
    print("🧮 深度学习基础知识综合演示")
    print("="*60)
    
    if not NUMPY_AVAILABLE:
        print("❌ 基础模块不可用，请安装numpy: pip install numpy")
        return None
    
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

def train_kaggle_model(rank, world_size, config):
    """训练Kaggle竞赛优化模型（分布式训练）"""
    if not TORCH_AVAILABLE:
        print("PyTorch not available")
        return
    
    print(f"   在GPU {rank}上训练Kaggle竞赛模型...")
    
    # 设置分布式环境
    setup(rank, world_size)
    
    # 设置设备
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # 创建Kaggle竞赛模型（模块化架构）
    if KAGGLE_MODELS_AVAILABLE:
        kaggle_model = create_kaggle_model(
            model_name="efficientnet_b3",  # 或 "custom_resnet"
            num_classes=10
        ).to(device)
    elif MODELS_AVAILABLE:
        # 使用基础模型
        kaggle_model = CIFAR10Net().to(device)
        if rank == 0:
            print("   使用基础CIFAR10模型替代Kaggle模型")
    
    # 包装为DDP模型
    kaggle_model = DDP(kaggle_model, device_ids=[rank])
    
    # 使用优化的数据加载器
    if KAGGLE_DATA_AVAILABLE:
        train_loader, val_loader = get_kaggle_loaders(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            distributed=True
        )
    else:
        # 使用CIFAR-10作为替代
        train_loader, val_loader = get_cifar10_loaders(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            distributed=True
        )
    
    # 创建训练器
    trainer = Trainer(
        model=kaggle_model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        rank=rank
    )
    
    if rank == 0:
        print("\n   训练Kaggle竞赛模型...")
    results = trainer.train()
    
    if rank == 0:
        print("\n   Kaggle模型训练结果:")
        for key, value in results.items():
            print(f"   {key}: {value}")
    
    # 清理
    cleanup()

def train_distributed(rank, world_size, config):
    """分布式训练函数（用于FP32/FP16对比）"""
    if not TORCH_AVAILABLE or not TRAINING_AVAILABLE:
        print("Training modules not available")
        return
    
    if rank == 0:
        print(f"   在GPU {rank}上运行训练...")
    
    # 设置分布式环境
    setup(rank, world_size)
    
    # 设置设备
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # 创建评估器
    evaluator = ModelEvaluator(device)
    
    # 获取数据加载器（展示数据管线优化）
    train_loader, val_loader = get_cifar10_loaders(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        distributed=True
    )
    
    # 创建损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 训练基准模型（展示模型模块化）
    base_model = CIFAR10Net().to(device)
    base_model = DDP(base_model, device_ids=[rank])
    
    trainer = Trainer(
        model=base_model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        rank=rank
    )
    
    if rank == 0:
        precision_type = "FP16" if config.mixed_precision else "FP32"
        print(f"\n   训练基准模型 ({precision_type})...")
    
    trainer.train()
    
    # 评估基准模型
    if rank == 0:
        print("\n   评估基准模型...")
        model_name = "fp16_baseline" if config.mixed_precision else "fp32_baseline"
        evaluator.evaluate_model(base_model, val_loader, model_name, criterion)
    
    if config.quantize and rank == 0:
        # 创建并训练量化模型
        print("\n   训练量化模型...")
        quant_model = QuantizedCIFAR10Net().to(device)
        quant_model = DDP(quant_model, device_ids=[rank])
        
        trainer = Trainer(
            model=quant_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            rank=rank
        )
        trainer.train()
        
        # 评估量化模型
        print("\n   评估量化模型...")
        evaluator.evaluate_model(quant_model, val_loader, "quantized", criterion)
        
        # 比较模型精度
        evaluator.compare_models("fp32_baseline", "quantized")
    
    # 清理
    cleanup()

def run_deep_learning_training():
    """运行深度学习训练模块"""
    print("\n" + "="*60)
    print("🎯 深度学习训练模块")
    print("="*60)
    
    if not TORCH_AVAILABLE or not TRAINING_AVAILABLE:
        print("❌ 训练模块不可用，请安装PyTorch: pip install torch torchvision")
        return
    
    # 检查是否可用GPU
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"\n找到 {n_gpus} 个GPU设备")
        
        print("\n选择训练模式:")
        print("1. 单GPU基础训练 (快速演示)")
        print("2. Kaggle竞赛模型训练 (分布式)")
        print("3. 性能对比 (FP32 vs FP16)")
        print("4. 完整训练流程")
        
        try:
            train_choice = input("\n请选择 (1-4, 默认1): ").strip()
            if not train_choice:
                train_choice = "1"
        except (KeyboardInterrupt, EOFError):
            print("\n使用默认选项: 1")
            train_choice = "1"
        
        if train_choice == "1":
            # 单GPU快速训练
            print("\n运行单GPU基础训练...")
            config = TrainerConfig(
                batch_size=64,
                num_workers=4,
                learning_rate=0.001,
                num_epochs=3,
                mixed_precision=True,
                save_dir='checkpoints/basic'
            )
            
            device = torch.device("cuda:0")
            model = CIFAR10Net().to(device)
            train_loader, val_loader = get_cifar10_loaders(
                batch_size=config.batch_size,
                num_workers=config.num_workers
            )
            
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
        
        elif train_choice == "2":
            # Kaggle竞赛模型训练
            print("\n运行Kaggle竞赛模型训练...")
            kaggle_config = TrainerConfig(
                batch_size=32,
                num_workers=8,
                learning_rate=0.0003,
                num_epochs=10,
                quantize=False,
                mixed_precision=True,
                dropout_rate=0.3,
                optimizer_type="adamw",
                scheduler_type="cosine",
                warmup_epochs=5,
                early_stopping_patience=10,
                save_dir='checkpoints/kaggle'
            )
            
            mp.spawn(
                train_kaggle_model,
                args=(n_gpus, kaggle_config),
                nprocs=n_gpus,
                join=True
            )
        
        elif train_choice == "3":
            # 性能对比训练
            print("\n运行性能对比 (FP32 vs FP16)...")
            
            # FP32配置
            fp32_config = TrainerConfig(
                batch_size=64,
                num_workers=4,
                learning_rate=0.001,
                num_epochs=5,
                quantize=False,
                mixed_precision=False,
                save_dir='checkpoints/fp32'
            )
            
            # FP16配置
            fp16_config = TrainerConfig(
                batch_size=64,
                num_workers=4,
                learning_rate=0.001,
                num_epochs=5,
                quantize=False,
                mixed_precision=True,
                save_dir='checkpoints/fp16'
            )
            
            # 运行FP32训练
            print("\n1. 运行FP32基准测试...")
            mp.spawn(
                train_distributed,
                args=(n_gpus, fp32_config),
                nprocs=n_gpus,
                join=True
            )
            
            # 运行FP16训练
            print("\n2. 运行FP16混合精度测试...")
            mp.spawn(
                train_distributed,
                args=(n_gpus, fp16_config),
                nprocs=n_gpus,
                join=True
            )
            
            print("\n性能对比完成！")
        
        elif train_choice == "4":
            # 完整训练流程
            print("\n运行完整训练流程...")
            
            # Kaggle模型训练
            print("\n第1步: Kaggle竞赛模型训练")
            kaggle_config = TrainerConfig(
                batch_size=32,
                num_workers=8,
                learning_rate=0.0003,
                num_epochs=10,
                mixed_precision=True,
                save_dir='checkpoints/kaggle'
            )
            mp.spawn(
                train_kaggle_model,
                args=(n_gpus, kaggle_config),
                nprocs=n_gpus,
                join=True
            )
            
            # FP32 vs FP16对比
            print("\n第2步: 性能对比测试")
            fp32_config = TrainerConfig(
                batch_size=64,
                num_workers=4,
                learning_rate=0.001,
                num_epochs=5,
                mixed_precision=False,
                save_dir='checkpoints/fp32'
            )
            mp.spawn(
                train_distributed,
                args=(n_gpus, fp32_config),
                nprocs=n_gpus,
                join=True
            )
            
            fp16_config = TrainerConfig(
                batch_size=64,
                num_workers=4,
                learning_rate=0.001,
                num_epochs=5,
                mixed_precision=True,
                save_dir='checkpoints/fp16'
            )
            mp.spawn(
                train_distributed,
                args=(n_gpus, fp16_config),
                nprocs=n_gpus,
                join=True
            )
            
            print("\n完整训练流程完成！")
        
        else:
            print(f"无效选择: {train_choice}，使用默认单GPU训练")
            train_choice = "1"
    
    else:
        print("\n未找到GPU设备，使用CPU训练")
        config = TrainerConfig(
            batch_size=32,
            num_workers=2,
            learning_rate=0.001,
            num_epochs=2,
            mixed_precision=False,
            save_dir='checkpoints/cpu'
        )
        
        device = torch.device("cpu")
        model = CIFAR10Net().to(device)
        train_loader, val_loader = get_cifar10_loaders(
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )
        
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
    
    print("\n" + "="*60)
    print("✅ 深度学习训练模块完成!")
    print("="*60)

def print_system_info():
    """打印系统信息和功能说明"""
    print("\n" + "="*80)
    print("🎯 深度学习训练与架构演示系统 v2.3")
    print("="*80)
    
    print("\n📋 核心功能模块:")
    print("\n1. 🧮 线性代数与自动微分基础")
    print("   • 矩阵运算与几何变换 • Jacobian 矩阵计算")
    print("   • 链式法则详细推导   • 手写线性层和ReLU激活")
    print("   • 数值梯度检查       • PyTorch autograd对比")
    
    print("\n2. 🔄 反向传播机制详解")
    print("   • 手推两层网络梯度公式 • 逐步演示反向传播过程")
    print("   • 数值方法验证梯度     • 训练损失曲线可视化")
    
    print("\n3. ⚙️ 优化算法数学原理")
    print("   • SGD/Adam/RMSProp实现 • MNIST收敛曲线对比")
    print("   • 学习率影响分析       • 优化轨迹可视化")
    
    print("\n4. 🧠 CNN与Transformer基础")
    print("   • 手写卷积层实现       • 自注意力机制推导")
    print("   • 矩阵乘法几何意义     • CNN vs Transformer对比")
    
    print("\n5. 🚀 LLM架构原理 (LLaMA)")
    print("   • 多头自注意力机制     • RoPE旋转位置编码")
    print("   • 残差连接与RMSNorm    • SwiGLU前馈网络")
    print("   • 完整架构可视化       • 文本生成演示")
    
    print("\n6. ⚡ 简单神经网络性能测试")
    print("   • 矩阵乘法性能基准     • NumPy vs Numba对比")
    print("   • 多层感知机训练       • 准确率评估")
    
    print("\n7. 📊 数据分析仪表盘")
    print("   • 交互式数据可视化     • 时间序列分析")
    print("   • 多维数据探索         • 实时数据更新")
    
    print("\n8. 🎯 深度学习训练系统")
    print("   • Kaggle竞赛级模型     • 分布式训练(DDP)")
    print("   • 混合精度(FP16)       • 模型量化压缩")
    print("   • 性能监控对比         • 早停与学习率调度")
    
    print("\n9. 🧠 Prompt Engineering")
    print("   • 自动化Prompt调试     • Few-shot示例管理")
    print("   • 批量Prompt测试       • 输出质量优化")
    
    print("\n� 模型剪枝与压缩")
    print("   • 幅度剪枝(Magnitude)  • 结构化剪枝(Structured)")
    print("   • 全局剪枝(Global)     • 迭代剪枝(Iterative)")
    print("   • 稀疏度分析           • 压缩效果对比")
    
    print("\n�🔧 技术亮点:")
    print("✓ 完整数学推导          ✓ PyTorch模块化设计")
    print("✓ 优化数据管线          ✓ 多GPU训练支持")
    print("✓ 模型精度监控          ✓ 可视化训练过程")
    
    print("\n💻 快速开始:")
    print("  python run_example.py              # 交互菜单")
    print("  python run_example.py quick        # 快速演示")
    print("  python run_example.py all          # 完整演示")
    print("  python run_example.py help         # 查看帮助")
    
    # 显示模块可用性状态
    print("\n📦 模块状态:")
    status_items = [
        ("PyTorch", TORCH_AVAILABLE),
        ("NumPy", NUMPY_AVAILABLE),
        ("Pandas", PANDAS_AVAILABLE),
        ("LLM架构", LLM_AVAILABLE),
        ("训练模块", TRAINING_AVAILABLE),
        ("SNN模块", SNN_AVAILABLE),
        ("Kaggle模型", KAGGLE_MODELS_AVAILABLE),
        ("数据仪表盘", DASHBOARD_AVAILABLE),
        ("模型剪枝", PRUNING_AVAILABLE),
        ("大模型微调", FINETUNING_AVAILABLE),
        ("推理优化", INFERENCE_OPT_AVAILABLE),
    ]
    for name, available in status_items:
        status = "✓" if available else "✗"
        print(f"  {status} {name}")
    
    print("="*80)

def main():
    """主菜单交互模式"""
    print_system_info()
    
    print("\n请选择要运行的模块:")
    print(" 1. 基础知识演示 (线性代数、反向传播、优化器、CNN/Transformer)")
    print(" 2. LLM 架构演示")
    print(" 3. SNN 性能测试")
    print(" 4. 数据仪表盘")
    print(" 5. 深度学习训练")
    print(" 6. Prompt Engineering 与 Few-shot 技术")
    print(" 7. 模型剪枝与压缩")
    print(" 8. 大模型微调 (LoRA/QLoRA/PEFT)")
    print(" 9. 推理优化 (Batched Inference / KV Cache)")
    print("10. 快速演示 (精简版)")
    print("11. 全部模块 (完整版)")
    
    try:
        choice = input("\n请输入选择 (1-11, 默认10): ").strip()
        if not choice:
            choice = "10"
    except (KeyboardInterrupt, EOFError):
        print("\n程序被用户中断")
        return
    
    # 根据选择运行对应模块
    try:
        if choice in ["1", "11", "all"]:
            demonstrate_fundamentals()
        
        if choice in ["2", "10", "11", "all"]:
            demonstrate_llm_architecture()
        
        if choice in ["3", "10", "11", "all"]:
            demonstrate_snn_performance()
        
        if choice in ["4", "11", "all"]:
            # 仪表盘需要用户确认
            if choice in ["4"]:
                run_dashboard()
            else:
                print("\n是否启动数据仪表盘? (y/n, 默认n): ", end="")
                try:
                    dashboard_choice = input().strip().lower()
                    if dashboard_choice in ['y', 'yes']:
                        run_dashboard()
                    else:
                        print("跳过数据仪表盘")
                except (KeyboardInterrupt, EOFError):
                    print("\n跳过数据仪表盘")
        
        if choice in ["5", "11", "all"]:
            run_deep_learning_training()
        
        if choice in ["6", "10", "11", "all"]:
            demonstrate_prompt_engineering()
        
        if choice in ["7", "11", "all"]:
            demonstrate_model_pruning()
        
        if choice in ["8", "10", "11", "all"]:
            demonstrate_llm_finetuning()
        
        if choice in ["9", "10", "11", "all"]:
            demonstrate_inference_optimization()
        
        print("\n" + "="*80)
        print("🎉 所有演示完成!")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n\n运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 命令行参数支持
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        try:
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
            elif mode in ["pruning", "prune", "compression"]:
                demonstrate_model_pruning()
            elif mode in ["finetuning", "finetune", "lora", "qlora", "peft"]:
                demonstrate_llm_finetuning()
            elif mode in ["inference", "inference_opt", "kv_cache", "batch_inference"]:
                demonstrate_inference_optimization()
            elif mode == "quick":
                # 快速演示模式 - 只运行核心功能
                print("\n" + "="*80)
                print("🚀 快速演示模式")
                print("="*80)
                demonstrate_llm_architecture()
                demonstrate_snn_performance()
                demonstrate_prompt_engineering()
                demonstrate_llm_finetuning()
                demonstrate_inference_optimization()
            elif mode == "all":
                # 完整演示模式
                print("\n" + "="*80)
                print("🎯 完整演示模式")
                print("="*80)
                demonstrate_fundamentals()
                demonstrate_llm_architecture()
                demonstrate_snn_performance()
                demonstrate_prompt_engineering()
                demonstrate_model_pruning()
                demonstrate_llm_finetuning()
                demonstrate_inference_optimization()
                
                print("\n是否运行深度学习训练? (y/n, 默认n): ", end="")
                try:
                    train_choice = input().strip().lower()
                    if train_choice in ['y', 'yes']:
                        run_deep_learning_training()
                except (KeyboardInterrupt, EOFError):
                    print("\n跳过深度学习训练")
            elif mode == "help":
                print("\n可用命令:")
                print("  python run_example.py                    # 交互菜单模式")
                print("  python run_example.py fundamentals       # 基础知识演示")
                print("  python run_example.py llm               # LLM架构演示")
                print("  python run_example.py snn               # 性能测试")
                print("  python run_example.py dashboard         # 数据仪表盘")
                print("  python run_example.py train             # 深度学习训练")
                print("  python run_example.py prompt            # Prompt Engineering")
                print("  python run_example.py pruning           # 模型剪枝")
                print("  python run_example.py finetuning        # 大模型微调 (LoRA/QLoRA/PEFT)")
                print("  python run_example.py inference         # 推理优化 (KV Cache)")
                print("  python run_example.py quick             # 快速演示")
                print("  python run_example.py all               # 完整演示")
                print("  python run_example.py help              # 显示帮助")
            else:
                print(f"未知模式: {mode}")
                print("使用 'python run_example.py help' 查看可用命令")
                main()
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
        except Exception as e:
            print(f"\n\n运行出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        # 交互模式
        main()
