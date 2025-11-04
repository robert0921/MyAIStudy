"""LLM 架构原理实现 - Attention、位置编码、残差连接"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
import numpy as np

class RotaryPositionalEmbedding(nn.Module):
    """RoPE（旋转位置编码）- LLaMA使用的位置编码"""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 预计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算位置编码
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0
    
    def _compute_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """预计算cos和sin值"""
        if seq_len > self._seq_len_cached or self._cos_cached is None:
            self._seq_len_cached = seq_len
            
            # 计算位置
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)
        
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]
    
    def rotate_half(self, x):
        """旋转张量的一半维度"""
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, q, k, seq_len=None):
        """应用旋转位置编码"""
        if seq_len is None:
            seq_len = q.shape[-2]
        
        cos, sin = self._compute_cos_sin(seq_len, q.device, q.dtype)
        
        # 应用RoPE
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_embed, k_embed

class MultiHeadAttention(nn.Module):
    """多头注意力机制 - LLaMA风格"""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        use_rope: bool = True
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Q, K, V 投影层
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # 位置编码
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] 或 None
            kv_cache: 用于推理时的KV缓存
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算 Q, K, V
        q = self.q_proj(x)  # [batch_size, seq_len, d_model]
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 重塑为多头格式
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # 现在形状为 [batch_size, n_heads, seq_len, head_dim]
        
        # 应用RoPE位置编码
        if self.use_rope:
            q, k = self.rope(q, k, seq_len)
        
        # 处理KV缓存（用于推理）
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # scores: [batch_size, n_heads, seq_len, seq_len]
        
        # 应用掩码
        if mask is not None:
            if mask.dim() == 3:  # [batch_size, seq_len, seq_len]
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 应用softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 计算输出
        out = torch.matmul(attn_weights, v)
        # out: [batch_size, n_heads, seq_len, head_dim]
        
        # 重新组合多头输出
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 最终投影
        out = self.o_proj(out)
        
        # 返回新的KV缓存
        new_kv_cache = (k, v) if kv_cache is not None else None
        
        return out, new_kv_cache

class FeedForward(nn.Module):
    """前馈网络 - LLaMA使用SwiGLU激活"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # SwiGLU: 使用两个线性层和Swish门控
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # gate
        self.w2 = nn.Linear(d_ff, d_model, bias=False)  # down
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # up
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # SwiGLU激活：Swish(xW1) ⊙ (xW3)W2
        gate = F.silu(self.w1(x))  # Swish激活
        up = self.w3(x)
        hidden = gate * up  # 门控机制
        return self.w2(self.dropout(hidden))

class RMSNorm(nn.Module):
    """RMS层归一化 - LLaMA使用的归一化方法"""
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        # RMS归一化：x / sqrt(mean(x^2) + eps) * weight
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight

class TransformerBlock(nn.Module):
    """Transformer块 - 包含残差连接"""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        max_seq_len: int = 2048
    ):
        super().__init__()
        
        # 注意力层
        self.attention = MultiHeadAttention(
            d_model, n_heads, dropout, max_seq_len
        )
        
        # 前馈网络
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # 层归一化（Pre-Norm架构）
        self.attention_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        # 注意力块：Pre-Norm + 残差连接
        norm_x = self.attention_norm(x)
        attn_out, new_kv_cache = self.attention(norm_x, mask, kv_cache)
        x = x + attn_out  # 残差连接
        
        # 前馈块：Pre-Norm + 残差连接
        norm_x = self.ffn_norm(x)
        ffn_out = self.feed_forward(norm_x)
        x = x + ffn_out  # 残差连接
        
        return x, new_kv_cache

class LLaMAModel(nn.Module):
    """简化的LLaMA模型架构"""
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        
        # 词嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, max_seq_len)
            for _ in range(n_layers)
        ])
        
        # 最终层归一化
        self.norm = RMSNorm(d_model)
        
        # 输出投影（语言模型头）
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 权重绑定（可选）
        # self.lm_head.weight = self.token_embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """创建因果掩码（下三角矩阵）"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[list] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            kv_caches: 推理时的KV缓存列表
        """
        batch_size, seq_len = input_ids.shape
        
        # 词嵌入
        x = self.token_embedding(input_ids)  # [batch_size, seq_len, d_model]
        
        # 创建因果掩码
        causal_mask = self._create_causal_mask(seq_len, input_ids.device)
        
        # 合并注意力掩码
        if attention_mask is not None:
            # 扩展attention_mask维度
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_mask = extended_mask.expand(batch_size, 1, seq_len, seq_len)
            causal_mask = causal_mask * extended_mask
        
        # 通过Transformer层
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches else None
            x, new_kv_cache = layer(x, causal_mask, kv_cache)
            new_kv_caches.append(new_kv_cache)
        
        # 最终归一化
        x = self.norm(x)
        
        # 语言模型头
        logits = self.lm_head(x)  # [batch_size, seq_len, vocab_size]
        
        return {
            'logits': logits,
            'kv_caches': new_kv_caches if any(cache is not None for cache in new_kv_caches) else None
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True
    ) -> torch.Tensor:
        """文本生成"""
        self.eval()
        
        with torch.no_grad():
            generated_ids = input_ids.clone()
            kv_caches = None
            
            for _ in range(max_new_tokens):
                # 前向传播
                outputs = self.forward(
                    generated_ids[:, -1:] if kv_caches else generated_ids,
                    kv_caches=kv_caches
                )
                
                logits = outputs['logits'][:, -1, :] / temperature
                kv_caches = outputs['kv_caches']
                
                # 采样策略
                if do_sample:
                    if top_k is not None:
                        # Top-k采样
                        top_k = min(top_k, logits.size(-1))
                        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                        logits[indices_to_remove] = float('-inf')
                    
                    if top_p is not None:
                        # Top-p (nucleus) 采样
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        logits[indices_to_remove] = float('-inf')
                    
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # 贪心采样
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # 检查是否遇到结束符
                if next_token.item() == self.pad_token_id:
                    break
            
            return generated_ids

# 示例使用
def create_llama_example():
    """创建LLaMA模型示例"""
    model = LLaMAModel(
        vocab_size=32000,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        max_seq_len=2048
    )
    
    return model

if __name__ == "__main__":
    # 测试模型
    model = create_llama_example()
    
    # 创建示例输入
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 32000, (batch_size, seq_len))
    
    # 前向传播
    outputs = model(input_ids)
    print(f"Logits shape: {outputs['logits'].shape}")
    
    # 生成文本
    generated = model.generate(input_ids[:1], max_new_tokens=20)
    print(f"Generated shape: {generated.shape}")
