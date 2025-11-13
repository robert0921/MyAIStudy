"""
大模型推理优化：Batched Inference / KV Cache
优化推理性能，降低延迟和显存使用
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import time
from dataclasses import dataclass
import math


@dataclass
class InferenceMetrics:
    """推理性能指标"""
    latency_ms: float  # 延迟（毫秒）
    throughput_tokens_per_sec: float  # 吞吐量（token/秒）
    memory_mb: float  # 显存使用（MB）
    batch_size: int  # 批次大小
    sequence_length: int  # 序列长度


class KVCache:
    """
    KV Cache实现
    缓存Key和Value，避免重复计算
    """
    def __init__(self, max_batch_size: int, max_seq_len: int, n_heads: int, head_dim: int, device: str = 'cpu'):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = device
        
        # 初始化缓存
        self.cache_k = torch.zeros(
            max_batch_size, n_heads, max_seq_len, head_dim,
            device=device
        )
        self.cache_v = torch.zeros(
            max_batch_size, n_heads, max_seq_len, head_dim,
            device=device
        )
        
        self.seq_len = 0  # 当前缓存的序列长度
    
    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新KV缓存
        
        Args:
            k: Key张量 [batch, n_heads, seq_len, head_dim]
            v: Value张量 [batch, n_heads, seq_len, head_dim]
            start_pos: 起始位置
        
        Returns:
            完整的Key和Value缓存
        """
        batch_size, n_heads, seq_len, head_dim = k.shape
        
        # 更新缓存
        self.cache_k[:batch_size, :, start_pos:start_pos+seq_len, :] = k
        self.cache_v[:batch_size, :, start_pos:start_pos+seq_len, :] = v
        
        self.seq_len = start_pos + seq_len
        
        # 返回完整缓存（从0到当前位置）
        return (
            self.cache_k[:batch_size, :, :self.seq_len, :],
            self.cache_v[:batch_size, :, :self.seq_len, :]
        )
    
    def reset(self):
        """重置缓存"""
        self.seq_len = 0
        self.cache_k.zero_()
        self.cache_v.zero_()
    
    def get_memory_usage(self) -> float:
        """获取缓存显存使用（MB）"""
        k_size = self.cache_k.numel() * self.cache_k.element_size()
        v_size = self.cache_v.numel() * self.cache_v.element_size()
        return (k_size + v_size) / 1024**2


class AttentionWithKVCache(nn.Module):
    """
    带KV Cache的注意力层
    """
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        max_batch_size: int = 32,
        max_seq_len: int = 2048,
        device: str = 'cpu'
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Q, K, V投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        # KV Cache
        self.kv_cache = KVCache(
            max_batch_size, max_seq_len, n_heads, self.head_dim, device
        )
        
        self.use_cache = False
    
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        use_cache: bool = False
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入 [batch, seq_len, d_model]
            start_pos: KV Cache起始位置
            use_cache: 是否使用KV Cache
        
        Returns:
            输出 [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V投影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape为多头
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 使用KV Cache
        if use_cache:
            k, v = self.kv_cache.update(k, v, start_pos)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Causal mask（自回归生成时需要）
        if use_cache and start_pos > 0:
            # 只计算新token与所有历史token的注意力
            pass
        else:
            # 完整的causal mask
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            scores = scores.masked_fill(mask.to(scores.device), float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        
        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.o_proj(output)
        
        return output
    
    def reset_cache(self):
        """重置KV Cache"""
        self.kv_cache.reset()


class BatchedInferenceEngine:
    """
    批量推理引擎
    支持动态批处理和KV Cache
    """
    def __init__(
        self,
        model: nn.Module,
        max_batch_size: int = 32,
        max_seq_len: int = 512,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.device = device
        
        self.model.eval()
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        use_cache: bool = True,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> Tuple[torch.Tensor, InferenceMetrics]:
        """
        生成文本
        
        Args:
            input_ids: 输入token IDs [batch, seq_len]
            max_new_tokens: 最大生成token数
            use_cache: 是否使用KV Cache
            temperature: 温度参数
            top_k: Top-K采样
        
        Returns:
            生成的token IDs和性能指标
        """
        batch_size, prompt_len = input_ids.shape
        input_ids = input_ids.to(self.device)
        
        # 记录开始时间和显存
        start_time = time.time()
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated()
        
        # 生成循环
        generated = input_ids
        
        for i in range(max_new_tokens):
            # 前向传播
            if use_cache and i > 0:
                # 只需要传入最后一个token
                logits = self.model(generated[:, -1:])
            else:
                # 传入完整序列
                logits = self.model(generated)
            
            # 取最后一个位置的logits
            logits = logits[:, -1, :] / temperature
            
            # Top-K采样
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # 采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 拼接
            generated = torch.cat([generated, next_token], dim=1)
            
            # 检查是否达到最大长度
            if generated.shape[1] >= self.max_seq_len:
                break
        
        # 计算性能指标
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        total_tokens = generated.shape[0] * generated.shape[1]
        throughput = total_tokens / (end_time - start_time)
        
        if self.device == 'cuda':
            peak_mem = torch.cuda.max_memory_allocated()
            memory_mb = (peak_mem - start_mem) / 1024**2
        else:
            memory_mb = 0.0
        
        metrics = InferenceMetrics(
            latency_ms=latency_ms,
            throughput_tokens_per_sec=throughput,
            memory_mb=memory_mb,
            batch_size=batch_size,
            sequence_length=generated.shape[1]
        )
        
        return generated, metrics
    
    @torch.no_grad()
    def benchmark_batch_sizes(
        self,
        vocab_size: int = 10000,
        prompt_len: int = 32,
        max_new_tokens: int = 50,
        batch_sizes: List[int] = [1, 2, 4, 8, 16],
        use_cache: bool = True
    ) -> List[InferenceMetrics]:
        """
        基准测试不同批次大小
        
        Returns:
            各批次大小的性能指标列表
        """
        results = []
        
        print(f"\n🔬 批次大小基准测试")
        print(f"  提示长度: {prompt_len}")
        print(f"  生成token数: {max_new_tokens}")
        print(f"  使用KV Cache: {use_cache}")
        print(f"\n  {'Batch':<8} {'延迟(ms)':<12} {'吞吐量':<15} {'显存(MB)':<12}")
        print(f"  {'-'*50}")
        
        for batch_size in batch_sizes:
            if batch_size > self.max_batch_size:
                print(f"  {batch_size:<8} 跳过（超出最大批次）")
                continue
            
            # 生成随机输入
            input_ids = torch.randint(0, vocab_size, (batch_size, prompt_len))
            
            # 运行推理
            try:
                _, metrics = self.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    use_cache=use_cache
                )
                
                results.append(metrics)
                
                print(f"  {batch_size:<8} {metrics.latency_ms:<12.2f} "
                      f"{metrics.throughput_tokens_per_sec:<15.2f} "
                      f"{metrics.memory_mb:<12.2f}")
            
            except RuntimeError as e:
                print(f"  {batch_size:<8} OOM (内存不足)")
                break
        
        return results


def demonstrate_kv_cache():
    """演示KV Cache原理"""
    print("\n" + "="*70)
    print("🔑 KV Cache 原理演示")
    print("="*70)
    
    device = 'cpu'
    
    print("\n1. 创建带KV Cache的注意力层")
    d_model = 256
    n_heads = 4
    max_batch_size = 8
    max_seq_len = 128
    
    attn = AttentionWithKVCache(
        d_model=d_model,
        n_heads=n_heads,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        device=device
    )
    
    print(f"  模型维度: {d_model}")
    print(f"  注意力头数: {n_heads}")
    print(f"  最大批次: {max_batch_size}")
    print(f"  最大序列长度: {max_seq_len}")
    
    # 不使用KV Cache
    print("\n2. 标准推理（不使用KV Cache）")
    batch_size = 4
    seq_len = 32
    x = torch.randn(batch_size, seq_len, d_model)
    
    start = time.time()
    for step in range(10):
        # 每次都重新计算完整序列
        _ = attn(x, use_cache=False)
    no_cache_time = (time.time() - start) / 10
    
    print(f"  平均耗时: {no_cache_time*1000:.4f} ms")
    
    # 使用KV Cache
    print("\n3. 使用KV Cache的推理")
    attn.reset_cache()
    
    start = time.time()
    # 初始前向传播（处理提示）
    _ = attn(x[:, :16, :], start_pos=0, use_cache=True)
    
    # 自回归生成（每次只处理一个新token）
    for step in range(16):
        _ = attn(x[:, step:step+1, :], start_pos=16+step, use_cache=True)
    
    cache_time = time.time() - start
    
    print(f"  总耗时: {cache_time*1000:.4f} ms")
    print(f"  加速比: {no_cache_time*10/cache_time:.2f}x")
    
    # KV Cache显存占用
    print("\n4. KV Cache显存占用")
    cache_mem = attn.kv_cache.get_memory_usage()
    print(f"  缓存大小: {cache_mem:.4f} MB")
    print(f"  计算方式: 2 * batch * n_heads * seq_len * head_dim * sizeof(float)")
    print(f"  = 2 * {max_batch_size} * {n_heads} * {max_seq_len} * {d_model//n_heads} * 4 bytes")
    
    print("\n" + "="*70)
    print("✅ KV Cache演示完成!")
    print("="*70)


def demonstrate_batched_inference():
    """演示批量推理"""
    print("\n" + "="*70)
    print("📦 批量推理优化演示")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 创建简单的语言模型
    print("\n1. 创建简化的语言模型")
    
    class SimpleGenerativeModel(nn.Module):
        def __init__(self, vocab_size=10000, d_model=256, n_layers=2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.layers = nn.ModuleList([
                AttentionWithKVCache(d_model, n_heads=4, device=str(device))
                for _ in range(n_layers)
            ])
            self.lm_head = nn.Linear(d_model, vocab_size)
        
        def forward(self, x):
            x = self.embedding(x)
            for layer in self.layers:
                x = layer(x, use_cache=False)
            return self.lm_head(x)
    
    model = SimpleGenerativeModel().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数: {total_params:,}")
    
    # 创建推理引擎
    print("\n2. 创建批量推理引擎")
    engine = BatchedInferenceEngine(
        model=model,
        max_batch_size=32,
        max_seq_len=256,
        device=str(device)
    )
    
    # 基准测试
    print("\n3. 运行批次大小基准测试")
    results = engine.benchmark_batch_sizes(
        vocab_size=10000,
        prompt_len=32,
        max_new_tokens=32,
        batch_sizes=[1, 2, 4, 8],
        use_cache=False
    )
    
    # 分析结果
    if results:
        print("\n4. 性能分析")
        print(f"  批次大小从1增加到{results[-1].batch_size}:")
        print(f"  - 延迟增加: {results[-1].latency_ms / results[0].latency_ms:.2f}x")
        print(f"  - 吞吐量提升: {results[-1].throughput_tokens_per_sec / results[0].throughput_tokens_per_sec:.2f}x")
        
        if device.type == 'cuda':
            print(f"  - 显存使用: {results[-1].memory_mb:.2f} MB")
    
    print("\n" + "="*70)
    print("✅ 批量推理演示完成!")
    print("="*70)


def demonstrate_cache_vs_no_cache():
    """对比使用和不使用KV Cache的性能"""
    print("\n" + "="*70)
    print("⚖️ KV Cache 性能对比")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    class SimpleGenerativeModel(nn.Module):
        def __init__(self, vocab_size=5000, d_model=256):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.attn = AttentionWithKVCache(d_model, n_heads=4, device=str(device))
            self.lm_head = nn.Linear(d_model, vocab_size)
        
        def forward(self, x):
            x = self.embedding(x)
            x = self.attn(x, use_cache=False)
            return self.lm_head(x)
    
    model = SimpleGenerativeModel().to(device)
    
    # 创建两个推理引擎
    engine_no_cache = BatchedInferenceEngine(model, device=str(device))
    engine_with_cache = BatchedInferenceEngine(model, device=str(device))
    
    # 测试配置
    batch_size = 4
    prompt_len = 32
    max_new_tokens = 50
    
    print(f"\n测试配置:")
    print(f"  批次大小: {batch_size}")
    print(f"  提示长度: {prompt_len}")
    print(f"  生成token数: {max_new_tokens}")
    
    input_ids = torch.randint(0, 5000, (batch_size, prompt_len))
    
    # 不使用KV Cache
    print("\n1. 不使用KV Cache")
    _, metrics_no_cache = engine_no_cache.generate(
        input_ids, max_new_tokens=max_new_tokens, use_cache=False
    )
    print(f"  延迟: {metrics_no_cache.latency_ms:.2f} ms")
    print(f"  吞吐量: {metrics_no_cache.throughput_tokens_per_sec:.2f} tokens/s")
    
    # 使用KV Cache
    print("\n2. 使用KV Cache")
    _, metrics_with_cache = engine_with_cache.generate(
        input_ids, max_new_tokens=max_new_tokens, use_cache=True
    )
    print(f"  延迟: {metrics_with_cache.latency_ms:.2f} ms")
    print(f"  吞吐量: {metrics_with_cache.throughput_tokens_per_sec:.2f} tokens/s")
    
    # 对比
    print("\n3. 性能提升")
    speedup = metrics_no_cache.latency_ms / metrics_with_cache.latency_ms
    print(f"  延迟降低: {(1 - 1/speedup)*100:.1f}%")
    print(f"  加速比: {speedup:.2f}x")
    print(f"  吞吐量提升: {metrics_with_cache.throughput_tokens_per_sec / metrics_no_cache.throughput_tokens_per_sec:.2f}x")
    
    print("\n" + "="*70)
    print("✅ 性能对比完成!")
    print("="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 大模型推理优化演示")
    print("="*70)
    
    # 1. KV Cache原理
    demonstrate_kv_cache()
    
    # 2. 批量推理
    demonstrate_batched_inference()
    
    # 3. KV Cache性能对比
    demonstrate_cache_vs_no_cache()
    
    print("\n" + "="*70)
    print("✅ 所有推理优化演示完成!")
    print("="*70)
