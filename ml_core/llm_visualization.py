"""LLaMA 架构可视化工具"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Arrow
import numpy as np
from typing import List, Tuple, Dict, Any

class LLaMAArchitectureVisualizer:
    """LLaMA架构可视化器"""
    
    def __init__(self, figsize: Tuple[int, int] = (14, 16)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 20)
        self.ax.axis('off')
        
        # 颜色定义
        self.colors = {
            'embedding': '#FFE5B4',      # 桃色
            'attention': '#B4E5FF',      # 淡蓝色
            'ffn': '#B4FFB4',           # 淡绿色
            'norm': '#FFB4B4',          # 淡红色
            'residual': '#E5E5E5',      # 灰色
            'rope': '#DDA0DD',          # 梅花色
            'output': '#FFCCCB'         # 淡粉色
        }
    
    def draw_box(self, x: float, y: float, width: float, height: float, 
                 text: str, color: str, text_size: int = 10) -> FancyBboxPatch:
        """绘制带文本的方框"""
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=1.5
        )
        self.ax.add_patch(box)
        
        # 添加文本
        self.ax.text(
            x + width/2, y + height/2, text,
            ha='center', va='center',
            fontsize=text_size, fontweight='bold'
        )
        
        return box
    
    def draw_arrow(self, start: Tuple[float, float], end: Tuple[float, float], 
                   color: str = 'black', width: float = 2) -> None:
        """绘制箭头"""
        self.ax.annotate(
            '', xy=end, xytext=start,
            arrowprops=dict(
                arrowstyle='->', lw=width, color=color
            )
        )
    
    def draw_residual_connection(self, x_start: float, y_start: float, 
                               x_end: float, y_end: float) -> None:
        """绘制残差连接"""
        # 绘制弯曲的残差连接线
        mid_x = x_start - 0.8
        
        # 创建贝塞尔曲线路径
        from matplotlib.path import Path
        import matplotlib.patches as mpatches
        
        verts = [
            (x_start, y_start),  # 起点
            (mid_x, y_start),    # 控制点1
            (mid_x, y_end),      # 控制点2
            (x_end, y_end),      # 终点
        ]
        
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]
        path = Path(verts, codes)
        
        patch = mpatches.PathPatch(
            path, facecolor='none', edgecolor=self.colors['residual'], 
            linewidth=2, linestyle='--'
        )
        self.ax.add_patch(patch)
        
        # 添加 "+" 符号
        self.ax.text(
            x_end + 0.2, y_end, '+',
            ha='center', va='center',
            fontsize=16, fontweight='bold'
        )
    
    def draw_attention_detail(self, x: float, y: float) -> None:
        """绘制注意力机制的详细结构"""
        # Multi-Head Attention 内部结构
        
        # Q, K, V 投影
        qkv_width = 0.6
        qkv_height = 0.3
        spacing = 0.8
        
        # Q
        self.draw_box(x - 1.5, y + 1.5, qkv_width, qkv_height, 'Q', '#FFE4E1', 8)
        # K  
        self.draw_box(x - 0.6, y + 1.5, qkv_width, qkv_height, 'K', '#FFE4E1', 8)
        # V
        self.draw_box(x + 0.3, y + 1.5, qkv_width, qkv_height, 'V', '#FFE4E1', 8)
        
        # RoPE 位置编码
        self.draw_box(x - 1.5, y + 2.2, 1.2, 0.3, 'RoPE', self.colors['rope'], 8)
        
        # Attention计算
        self.draw_box(x - 0.8, y + 0.8, 1.6, 0.3, 'Scaled Dot-Product', '#E6E6FA', 8)
        
        # 多头连接
        self.draw_box(x - 0.6, y + 0.2, 1.2, 0.3, 'Concat + Linear', '#F0E68C', 8)
        
        # 绘制连接线
        self.draw_arrow((x - 1.2, y + 1.5), (x - 0.4, y + 1.1))
        self.draw_arrow((x - 0.3, y + 1.5), (x - 0.1, y + 1.1))
        self.draw_arrow((x + 0.6, y + 1.5), (x + 0.2, y + 1.1))
        
        self.draw_arrow((x, y + 0.8), (x, y + 0.5))
    
    def draw_swiglu_detail(self, x: float, y: float) -> None:
        """绘制SwiGLU前馈网络的详细结构"""
        # SwiGLU 结构
        
        # Gate 和 Up 投影
        self.draw_box(x - 1, y + 1, 0.8, 0.3, 'Gate (W1)', '#98FB98', 8)
        self.draw_box(x + 0.2, y + 1, 0.8, 0.3, 'Up (W3)', '#98FB98', 8)
        
        # SiLU激活
        self.draw_box(x - 1, y + 0.5, 0.8, 0.3, 'SiLU', '#FFB6C1', 8)
        
        # 元素相乘
        circle = Circle((x, y + 0.05), 0.15, facecolor='#FFFF99', edgecolor='black')
        self.ax.add_patch(circle)
        self.ax.text(x, y + 0.05, '⊙', ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Down 投影
        self.draw_box(x - 0.4, y - 0.5, 0.8, 0.3, 'Down (W2)', '#98FB98', 8)
        
        # 绘制连接线
        self.draw_arrow((x - 0.6, y + 1), (x - 0.6, y + 0.8))
        self.draw_arrow((x + 0.6, y + 1), (x + 0.15, y + 0.2))
        self.draw_arrow((x - 0.6, y + 0.5), (x - 0.15, y + 0.2))
        self.draw_arrow((x, y - 0.1), (x, y - 0.2))
    
    def draw_complete_architecture(self) -> None:
        """绘制完整的LLaMA架构"""
        # 标题
        self.ax.text(5, 19.5, 'LLaMA Architecture', 
                    ha='center', va='center', fontsize=20, fontweight='bold')
        
        # 输入
        self.draw_box(4, 18, 2, 0.5, 'Input Tokens', self.colors['embedding'])
        
        # Token Embedding
        self.draw_box(4, 17, 2, 0.5, 'Token Embedding', self.colors['embedding'])
        self.draw_arrow((5, 18), (5, 17.5))
        
        # Transformer 层数标注
        layer_y = 16
        self.ax.text(8.5, layer_y, 'N × Transformer Layers', 
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
        
        # 第一个Transformer块（详细展示）
        current_y = 15.5
        
        # RMS Norm (Pre-norm)
        self.draw_box(4, current_y, 2, 0.4, 'RMS Norm', self.colors['norm'])
        self.draw_arrow((5, 17), (5, current_y + 0.4))
        
        # Multi-Head Attention
        current_y -= 0.8
        self.draw_box(3.5, current_y, 3, 0.6, 'Multi-Head Attention', self.colors['attention'])
        self.draw_arrow((5, current_y + 1.2), (5, current_y + 0.6))
        
        # 绘制注意力详细结构
        self.draw_attention_detail(7.5, current_y - 0.2)
        
        # 残差连接1
        residual_start_y = current_y + 1.6
        residual_end_y = current_y - 0.2
        self.draw_residual_connection(3.2, residual_start_y, 3.2, residual_end_y)
        
        # RMS Norm (Pre-norm)
        current_y -= 1
        self.draw_box(4, current_y, 2, 0.4, 'RMS Norm', self.colors['norm'])
        self.draw_arrow((5, current_y + 0.8), (5, current_y + 0.4))
        
        # Feed Forward (SwiGLU)
        current_y -= 0.8
        self.draw_box(3.5, current_y, 3, 0.6, 'SwiGLU FFN', self.colors['ffn'])
        self.draw_arrow((5, current_y + 1.2), (5, current_y + 0.6))
        
        # 绘制SwiGLU详细结构
        self.draw_swiglu_detail(7.5, current_y + 0.3)
        
        # 残差连接2
        residual_start_y = current_y + 1.6
        residual_end_y = current_y - 0.2
        self.draw_residual_connection(3.2, residual_start_y, 3.2, residual_end_y)
        
        # 省略号表示更多层
        current_y -= 1.2
        self.ax.text(5, current_y, '⋮', ha='center', va='center', fontsize=24, fontweight='bold')
        self.ax.text(7, current_y, '(更多Transformer层)', ha='center', va='center', fontsize=10)
        
        # 最终 RMS Norm
        current_y -= 1
        self.draw_box(4, current_y, 2, 0.4, 'Final RMS Norm', self.colors['norm'])
        self.draw_arrow((5, current_y + 0.8), (5, current_y + 0.4))
        
        # LM Head
        current_y -= 0.8
        self.draw_box(4, current_y, 2, 0.5, 'LM Head (Linear)', self.colors['output'])
        self.draw_arrow((5, current_y + 1.2), (5, current_y + 0.5))
        
        # 输出
        current_y -= 0.8
        self.draw_box(4, current_y, 2, 0.5, 'Output Logits', self.colors['output'])
        self.draw_arrow((5, current_y + 0.8), (5, current_y + 0.5))
        
        # 添加关键特性说明
        self._add_feature_annotations()
    
    def _add_feature_annotations(self) -> None:
        """添加关键特性注释"""
        # 左侧注释
        annotations = [
            (0.5, 16, "关键特性:", 12, 'bold'),
            (0.5, 15.5, "• RoPE位置编码", 10, 'normal'),
            (0.5, 15.2, "• RMS层归一化", 10, 'normal'),
            (0.5, 14.9, "• SwiGLU激活", 10, 'normal'),
            (0.5, 14.6, "• 残差连接", 10, 'normal'),
            (0.5, 14.3, "• Pre-Norm架构", 10, 'normal'),
            (0.5, 14, "• 因果注意力掩码", 10, 'normal'),
        ]
        
        for x, y, text, size, weight in annotations:
            self.ax.text(x, y, text, ha='left', va='center', 
                        fontsize=size, fontweight=weight)
    
    def draw_attention_visualization(self) -> None:
        """绘制注意力机制可视化"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Attention Scores Heatmap
        ax1 = axes[0, 0]
        seq_len = 8
        attention_scores = np.random.rand(seq_len, seq_len)
        # 创建因果掩码
        mask = np.tril(np.ones((seq_len, seq_len)))
        attention_scores = attention_scores * mask
        
        im1 = ax1.imshow(attention_scores, cmap='Blues', aspect='auto')
        ax1.set_title('Causal Attention Pattern', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Key Position')
        ax1.set_ylabel('Query Position')
        plt.colorbar(im1, ax=ax1)
        
        # 2. RoPE可视化
        ax2 = axes[0, 1]
        positions = np.arange(0, 20)
        freqs = np.outer(positions, 1.0 / (10000 ** (np.arange(0, 8, 2) / 8)))
        
        ax2.plot(positions, np.cos(freqs[:, 0]), label='cos(freq_0)', linewidth=2)
        ax2.plot(positions, np.sin(freqs[:, 0]), label='sin(freq_0)', linewidth=2)
        ax2.plot(positions, np.cos(freqs[:, 1]), label='cos(freq_1)', linewidth=2)
        ax2.plot(positions, np.sin(freqs[:, 1]), label='sin(freq_1)', linewidth=2)
        ax2.set_title('RoPE Positional Encoding', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. SwiGLU激活函数
        ax3 = axes[1, 0]
        x = np.linspace(-3, 3, 100)
        silu = x / (1 + np.exp(-x))  # SiLU/Swish
        relu = np.maximum(0, x)       # ReLU
        gelu = x * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))  # GELU
        
        ax3.plot(x, silu, label='SiLU (Swish)', linewidth=2)
        ax3.plot(x, relu, label='ReLU', linewidth=2, linestyle='--')
        ax3.plot(x, gelu, label='GELU', linewidth=2, linestyle=':')
        ax3.set_title('Activation Functions Comparison', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Input')
        ax3.set_ylabel('Output')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 多头注意力权重
        ax4 = axes[1, 1]
        n_heads = 8
        seq_len = 10
        
        # 模拟不同头的注意力模式
        head_patterns = []
        for head in range(n_heads):
            if head < 2:  # 局部注意力
                pattern = np.eye(seq_len) + np.eye(seq_len, k=1) + np.eye(seq_len, k=-1)
            elif head < 4:  # 长距离注意力
                pattern = np.random.exponential(0.5, (seq_len, seq_len))
            else:  # 随机注意力
                pattern = np.random.rand(seq_len, seq_len)
            
            pattern = np.tril(pattern)  # 应用因果掩码
            pattern = pattern / (pattern.sum(axis=1, keepdims=True) + 1e-8)  # 归一化
            head_patterns.append(pattern)
        
        # 显示平均注意力模式
        avg_pattern = np.mean(head_patterns, axis=0)
        im4 = ax4.imshow(avg_pattern, cmap='Reds', aspect='auto')
        ax4.set_title('Multi-Head Attention (Average)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Key Position')
        ax4.set_ylabel('Query Position')
        plt.colorbar(im4, ax=ax4)
        
        plt.tight_layout()
        return fig
    
    def save_diagrams(self, save_dir: str = './llama_diagrams/') -> None:
        """保存所有图表"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存架构图
        self.fig.savefig(f'{save_dir}llama_architecture.png', 
                        dpi=300, bbox_inches='tight')
        
        # 保存注意力可视化
        attention_fig = self.draw_attention_visualization()
        attention_fig.savefig(f'{save_dir}llama_attention_analysis.png', 
                             dpi=300, bbox_inches='tight')
        
        print(f"图表已保存到 {save_dir}")

def create_llama_visualization():
    """创建LLaMA架构可视化"""
    visualizer = LLaMAArchitectureVisualizer()
    visualizer.draw_complete_architecture()
    
    # 显示图表
    plt.tight_layout()
    plt.show()
    
    # 创建注意力分析图
    attention_fig = visualizer.draw_attention_visualization()
    plt.show()
    
    return visualizer

# 使用示例
if __name__ == "__main__":
    # 创建可视化
    viz = create_llama_visualization()
    
    # 保存图表
    viz.save_diagrams()
    
    print("LLaMA架构可视化完成！")
