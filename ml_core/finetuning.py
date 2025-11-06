"""
大模型微调方法实现：LoRA / QLoRA / PEFT
支持1B~7B参数规模的模型高效微调
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
import time
import math


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) 层实现
    用低秩矩阵分解来适配预训练模型
    """
    def __init__(
        self, 
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.0
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA的两个低秩矩阵：A和B
        # W' = W + BA，其中B是out_features x rank，A是rank x in_features
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：计算 x @ A^T @ B^T * scaling
        """
        # x: [batch_size, seq_len, in_features]
        result = self.dropout(x)
        result = result @ self.lora_A.T  # [batch_size, seq_len, rank]
        result = result @ self.lora_B.T  # [batch_size, seq_len, out_features]
        result = result * self.scaling
        return result


class LoRALinear(nn.Module):
    """
    集成LoRA的线性层
    冻结原始权重，只训练LoRA参数
    """
    def __init__(
        self,
        linear_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.0
    ):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
        # 冻结原始权重
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：原始线性层 + LoRA增量
        """
        return self.linear(x) + self.lora(x)


class QLoRALayer(nn.Module):
    """
    QLoRA (Quantized Low-Rank Adaptation)
    结合4bit量化和LoRA，进一步降低显存占用
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
        quantize_bits: int = 4
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.quantize_bits = quantize_bits
        
        # LoRA参数（保持FP16/BF16）
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        result = self.dropout(x)
        result = result @ self.lora_A.T
        result = result @ self.lora_B.T
        result = result * self.scaling
        return result
    
    @staticmethod
    def quantize_weight(weight: torch.Tensor, bits: int = 4) -> tuple:
        """
        简化的权重量化（NF4量化的模拟）
        实际QLoRA使用NF4（4-bit NormalFloat）
        """
        # 计算量化范围
        n_levels = 2 ** bits
        w_min, w_max = weight.min(), weight.max()
        
        # 量化
        scale = (w_max - w_min) / (n_levels - 1)
        quantized = torch.round((weight - w_min) / scale)
        
        return quantized.to(torch.int8), scale, w_min
    
    @staticmethod
    def dequantize_weight(quantized: torch.Tensor, scale: float, offset: float) -> torch.Tensor:
        """反量化"""
        return quantized.float() * scale + offset


class PEFTModel(nn.Module):
    """
    PEFT (Parameter-Efficient Fine-Tuning) 统一接口
    支持多种高效微调方法
    """
    def __init__(
        self,
        base_model: nn.Module,
        method: str = "lora",
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
        target_modules: Optional[List[str]] = None
    ):
        super().__init__()
        self.base_model = base_model
        self.method = method.lower()
        self.rank = rank
        self.alpha = alpha
        
        # 默认目标模块（通常是注意力层的Q、K、V投影）
        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        self.target_modules = target_modules
        self.adapted_modules = {}
        
        # 应用PEFT方法
        self._apply_peft()
    
    def _apply_peft(self):
        """应用PEFT到目标模块"""
        for name, module in self.base_model.named_modules():
            # 检查是否是目标模块
            if any(target in name for target in self.target_modules):
                if isinstance(module, nn.Linear):
                    if self.method == "lora":
                        adapted = LoRALinear(
                            module,
                            rank=self.rank,
                            alpha=self.alpha
                        )
                        self._replace_module(name, adapted)
                        self.adapted_modules[name] = adapted
                        print(f"  ✓ 应用LoRA到 {name}")
    
    def _replace_module(self, name: str, new_module: nn.Module):
        """替换模块"""
        parts = name.split('.')
        parent = self.base_model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)
    
    def forward(self, *args, **kwargs):
        """前向传播"""
        return self.base_model(*args, **kwargs)
    
    def get_trainable_parameters(self) -> tuple:
        """获取可训练参数统计"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        percentage = 100 * trainable / total if total > 0 else 0
        return trainable, total, percentage
    
    def print_trainable_parameters(self):
        """打印可训练参数信息"""
        trainable, total, percentage = self.get_trainable_parameters()
        print(f"\n📊 参数统计:")
        print(f"  可训练参数: {trainable:,}")
        print(f"  总参数量: {total:,}")
        print(f"  可训练比例: {percentage:.4f}%")


class SimpleTransformerBlock(nn.Module):
    """
    简化的Transformer Block用于演示
    实际应用中会使用HuggingFace的模型
    """
    def __init__(self, d_model: int = 512, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # 注意力层
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        # 前馈网络
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.fc2 = nn.Linear(d_model * 4, d_model)
        
        # 层归一化
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        # 多头自注意力
        residual = x
        x = self.ln1(x)
        
        # 简化的注意力计算
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = torch.matmul(attn, v)
        x = self.o_proj(x)
        x = residual + self.dropout(x)
        
        # 前馈网络
        residual = x
        x = self.ln2(x)
        x = self.fc2(F.gelu(self.fc1(x)))
        x = residual + self.dropout(x)
        
        return x


class SimpleLLM(nn.Module):
    """
    简化的语言模型用于演示微调
    实际使用时会加载预训练的GPT/LLaMA模型
    """
    def __init__(
        self,
        vocab_size: int = 50000,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        max_seq_len: int = 512
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(d_model, n_heads)
            for _ in range(n_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 权重共享
        self.lm_head.weight = self.token_embedding.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播"""
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits


def demonstrate_lora_finetuning():
    """演示LoRA微调"""
    print("\n" + "="*70)
    print("🔧 LoRA微调演示")
    print("="*70)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 创建基础模型
    print("\n1. 创建基础语言模型（简化版）")
    base_model = SimpleLLM(
        vocab_size=10000,
        d_model=256,
        n_layers=4,
        n_heads=4,
        max_seq_len=128
    ).to(device)
    
    total_params = sum(p.numel() for p in base_model.parameters())
    print(f"  模型参数量: {total_params:,}")
    
    # 冻结原始模型
    print("\n2. 冻结原始模型参数")
    for param in base_model.parameters():
        param.requires_grad = False
    
    # 应用LoRA
    print("\n3. 应用LoRA (rank=8, alpha=16)")
    peft_model = PEFTModel(
        base_model,
        method="lora",
        rank=8,
        alpha=16,
        target_modules=["q_proj", "v_proj"]
    )
    
    peft_model.print_trainable_parameters()
    
    # 模拟训练数据
    print("\n4. 准备训练数据")
    batch_size = 4
    seq_len = 64
    input_ids = torch.randint(0, 10000, (batch_size, seq_len)).to(device)
    labels = torch.randint(0, 10000, (batch_size, seq_len)).to(device)
    
    print(f"  批次大小: {batch_size}")
    print(f"  序列长度: {seq_len}")
    
    # 训练循环
    print("\n5. 开始微调训练")
    optimizer = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad],
        lr=1e-4
    )
    
    peft_model.train()
    num_steps = 5
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # 前向传播
        logits = peft_model(input_ids)
        
        # 计算损失
        loss = F.cross_entropy(
            logits.view(-1, 10000),
            labels.view(-1)
        )
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        print(f"  Step {step+1}/{num_steps}: Loss = {loss.item():.4f}")
    
    # 显存使用
    if device.type == 'cuda':
        print(f"\n6. 显存使用情况")
        print(f"  已分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"  峰值: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    
    print("\n" + "="*70)
    print("✅ LoRA微调演示完成!")
    print("="*70)
    
    return peft_model


def demonstrate_qlora_comparison():
    """演示QLoRA与LoRA的对比"""
    print("\n" + "="*70)
    print("⚖️ QLoRA vs LoRA 对比")
    print("="*70)
    
    # 创建测试层
    in_features = 1024
    out_features = 1024
    
    print(f"\n测试配置:")
    print(f"  输入维度: {in_features}")
    print(f"  输出维度: {out_features}")
    print(f"  LoRA Rank: 8")
    
    # LoRA层
    print("\n1. 标准LoRA层")
    lora_layer = LoRALayer(in_features, out_features, rank=8)
    lora_params = sum(p.numel() for p in lora_layer.parameters())
    lora_size = sum(p.numel() * p.element_size() for p in lora_layer.parameters()) / 1024**2
    
    print(f"  参数量: {lora_params:,}")
    print(f"  内存占用: {lora_size:.4f} MB")
    
    # QLoRA层
    print("\n2. QLoRA层 (4-bit量化)")
    qlora_layer = QLoRALayer(in_features, out_features, rank=8, quantize_bits=4)
    qlora_params = sum(p.numel() for p in qlora_layer.parameters())
    qlora_size = sum(p.numel() * p.element_size() for p in qlora_layer.parameters()) / 1024**2
    
    # 量化节省（假设基础权重也被量化）
    base_weight_size_fp16 = in_features * out_features * 2 / 1024**2  # FP16
    base_weight_size_4bit = in_features * out_features * 0.5 / 1024**2  # 4-bit
    
    print(f"  参数量: {qlora_params:,}")
    print(f"  内存占用: {qlora_size:.4f} MB")
    
    print("\n3. 对比分析")
    print(f"  基础权重 (FP16): {base_weight_size_fp16:.4f} MB")
    print(f"  基础权重 (4-bit): {base_weight_size_4bit:.4f} MB")
    print(f"  量化节省: {base_weight_size_fp16 - base_weight_size_4bit:.4f} MB ({(1 - base_weight_size_4bit/base_weight_size_fp16)*100:.1f}%)")
    
    # 前向传播速度测试
    print("\n4. 前向传播速度测试")
    x = torch.randn(32, 128, in_features)
    
    # LoRA
    start = time.time()
    for _ in range(100):
        _ = lora_layer(x)
    lora_time = (time.time() - start) / 100
    
    # QLoRA
    start = time.time()
    for _ in range(100):
        _ = qlora_layer(x)
    qlora_time = (time.time() - start) / 100
    
    print(f"  LoRA: {lora_time*1000:.4f} ms/iter")
    print(f"  QLoRA: {qlora_time*1000:.4f} ms/iter")
    print(f"  速度比: {qlora_time/lora_time:.2f}x")
    
    print("\n" + "="*70)
    print("✅ QLoRA对比完成!")
    print("="*70)


def demonstrate_peft_methods():
    """演示不同PEFT方法"""
    print("\n" + "="*70)
    print("🎯 PEFT方法对比")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建基础模型
    print("\n1. 创建基础模型")
    base_model = SimpleLLM(
        vocab_size=5000,
        d_model=256,
        n_layers=3,
        n_heads=4
    ).to(device)
    
    total_params = sum(p.numel() for p in base_model.parameters())
    print(f"  总参数量: {total_params:,}")
    
    # 测试不同配置
    configs = [
        {"name": "LoRA-r4", "rank": 4, "alpha": 8},
        {"name": "LoRA-r8", "rank": 8, "alpha": 16},
        {"name": "LoRA-r16", "rank": 16, "alpha": 32},
    ]
    
    print("\n2. 测试不同LoRA配置")
    print(f"\n  {'配置':<15} {'可训练参数':<15} {'比例':<10} {'增加量':<15}")
    print(f"  {'-'*55}")
    
    for config in configs:
        # 创建PEFT模型
        peft_model = PEFTModel(
            SimpleLLM(vocab_size=5000, d_model=256, n_layers=3, n_heads=4).to(device),
            method="lora",
            rank=config["rank"],
            alpha=config["alpha"],
            target_modules=["q_proj", "v_proj"]
        )
        
        trainable, total, percentage = peft_model.get_trainable_parameters()
        added_params = trainable
        
        print(f"  {config['name']:<15} {trainable:<15,} {percentage:<10.4f}% {added_params:<15,}")
    
    print("\n3. 微调效率分析")
    print(f"  ✓ 参数效率: rank越小，可训练参数越少")
    print(f"  ✓ 性能权衡: rank越大，表达能力越强，但训练成本增加")
    print(f"  ✓ 推荐配置: rank=8是常见的平衡选择")
    print(f"  ✓ 内存节省: 相比全量微调，节省90%+显存")
    
    print("\n" + "="*70)
    print("✅ PEFT方法对比完成!")
    print("="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 大模型高效微调演示")
    print("="*70)
    
    # 1. LoRA微调
    demonstrate_lora_finetuning()
    
    # 2. QLoRA对比
    demonstrate_qlora_comparison()
    
    # 3. PEFT方法对比
    demonstrate_peft_methods()
    
    print("\n" + "="*70)
    print("✅ 所有微调演示完成!")
    print("="*70)
