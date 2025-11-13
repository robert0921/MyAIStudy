"""卷积神经网络与Transformer基础实现"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import time

class ConvolutionLayer:
    """手写卷积层实现"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, stride: int = 1, padding: int = 0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # He初始化
        self.kernels = np.random.randn(out_channels, in_channels, 
                                     kernel_size, kernel_size) * np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.bias = np.zeros((out_channels, 1))
        
        self.cache = {}
        
        print(f"卷积层初始化:")
        print(f"  输入通道: {in_channels}, 输出通道: {out_channels}")
        print(f"  卷积核大小: {kernel_size}x{kernel_size}")
        print(f"  步长: {stride}, 填充: {padding}")
        print(f"  参数数量: {self.kernels.size + self.bias.size}")
    
    def im2col(self, X: np.ndarray) -> np.ndarray:
        """
        将图像转换为矩阵形式，便于矩阵乘法实现卷积
        X: (N, C, H, W)
        返回: (N * out_H * out_W, C * kernel_size * kernel_size)
        """
        N, C, H, W = X.shape
        
        # 计算输出尺寸
        out_H = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_W = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # 添加填充
        if self.padding > 0:
            X_padded = np.pad(X, ((0, 0), (0, 0), 
                                (self.padding, self.padding), 
                                (self.padding, self.padding)), 
                            mode='constant')
        else:
            X_padded = X
        
        # im2col转换
        col = np.zeros((N, C, self.kernel_size, self.kernel_size, out_H, out_W))
        
        for y in range(self.kernel_size):
            y_lim = y + self.stride * out_H
            for x in range(self.kernel_size):
                x_lim = x + self.stride * out_W
                col[:, :, y, x, :, :] = X_padded[:, :, y:y_lim:self.stride, x:x_lim:self.stride]
        
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_H * out_W, -1)
        return col, out_H, out_W
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        前向传播
        X: (N, C, H, W)
        返回: (N, out_channels, out_H, out_W)
        """
        self.cache['X'] = X
        N, C, H, W = X.shape
        
        # im2col转换
        X_col, out_H, out_W = self.im2col(X)
        self.cache['X_col'] = X_col
        self.cache['out_shape'] = (out_H, out_W)
        
        # 将卷积核转换为矩阵形式
        W_col = self.kernels.reshape(self.out_channels, -1)
        
        # 矩阵乘法实现卷积
        # (out_channels, kernel_size²*in_channels) @ (kernel_size²*in_channels, N*out_H*out_W)
        out = W_col @ X_col.T + self.bias
        
        # 重塑为输出形状
        out = out.reshape(self.out_channels, N, out_H, out_W)
        out = out.transpose(1, 0, 2, 3)  # (N, out_channels, out_H, out_W)
        
        return out
    
    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        反向传播
        dout: (N, out_channels, out_H, out_W)
        """
        X = self.cache['X']
        X_col = self.cache['X_col']
        out_H, out_W = self.cache['out_shape']
        N, C, H, W = X.shape
        
        # 重塑梯度
        dout = dout.transpose(1, 0, 2, 3).reshape(self.out_channels, -1)
        
        # 计算权重梯度
        dW = dout @ X_col
        dW = dW.reshape(self.kernels.shape)
        
        # 计算偏置梯度
        db = np.sum(dout, axis=1, keepdims=True)
        
        # 计算输入梯度
        W_col = self.kernels.reshape(self.out_channels, -1)
        dX_col = W_col.T @ dout
        
        # col2im转换
        dX = self.col2im(dX_col, X.shape)
        
        return dX, dW, db
    
    def col2im(self, col: np.ndarray, X_shape: Tuple[int, ...]) -> np.ndarray:
        """col2im转换"""
        N, C, H, W = X_shape
        out_H, out_W = self.cache['out_shape']
        
        col = col.T.reshape(N, out_H, out_W, C, self.kernel_size, self.kernel_size)
        col = col.transpose(0, 3, 4, 5, 1, 2)
        
        if self.padding > 0:
            X = np.zeros((N, C, H + 2 * self.padding, W + 2 * self.padding))
        else:
            X = np.zeros((N, C, H, W))
        
        for y in range(self.kernel_size):
            y_lim = y + self.stride * out_H
            for x in range(self.kernel_size):
                x_lim = x + self.stride * out_W
                X[:, :, y:y_lim:self.stride, x:x_lim:self.stride] += col[:, :, y, x, :, :]
        
        if self.padding > 0:
            return X[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            return X

class MaxPoolingLayer:
    """最大池化层"""
    
    def __init__(self, pool_size: int = 2, stride: int = 2):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = {}
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        前向传播
        X: (N, C, H, W)
        """
        N, C, H, W = X.shape
        
        out_H = (H - self.pool_size) // self.stride + 1
        out_W = (W - self.pool_size) // self.stride + 1
        
        out = np.zeros((N, C, out_H, out_W))
        
        # 存储最大值位置用于反向传播
        self.cache = {
            'X': X,
            'out_shape': (out_H, out_W)
        }
        
        for i in range(out_H):
            for j in range(out_W):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                pool_region = X[:, :, h_start:h_end, w_start:w_end]
                out[:, :, i, j] = np.max(pool_region, axis=(2, 3))
        
        return out
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """反向传播"""
        X = self.cache['X']
        out_H, out_W = self.cache['out_shape']
        N, C, H, W = X.shape
        
        dX = np.zeros_like(X)
        
        for i in range(out_H):
            for j in range(out_W):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                pool_region = X[:, :, h_start:h_end, w_start:w_end]
                
                # 找到最大值位置
                for n in range(N):
                    for c in range(C):
                        pool_slice = pool_region[n, c, :, :]
                        max_val = np.max(pool_slice)
                        mask = (pool_slice == max_val)
                        dX[n, c, h_start:h_end, w_start:w_end] += mask * dout[n, c, i, j] / np.sum(mask)
        
        return dX

class SelfAttention:
    """手写自注意力机制"""
    
    def __init__(self, d_model: int, n_heads: int = 8):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"
        
        # 权重矩阵初始化
        self.W_q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_k = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_v = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_o = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        
        self.cache = {}
        
        print(f"自注意力机制初始化:")
        print(f"  模型维度: {d_model}")
        print(f"  注意力头数: {n_heads}")
        print(f"  每个头的维度: {self.d_k}")
    
    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, 
                                   mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        缩放点积注意力
        Q, K, V: (batch_size, n_heads, seq_len, d_k)
        """
        d_k = Q.shape[-1]
        
        # 计算注意力分数: QK^T / √d_k
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        
        # 应用掩码（如果有）
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        # Softmax
        attention_weights = self.softmax(scores)
        
        # 应用注意力权重到值
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """稳定的softmax实现"""
        # 沿最后一个维度计算softmax
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, X: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        前向传播
        X: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = X.shape
        
        # 计算Q, K, V
        Q = np.matmul(X, self.W_q)  # (batch_size, seq_len, d_model)
        K = np.matmul(X, self.W_k)
        V = np.matmul(X, self.W_v)
        
        # 重塑为多头形式
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # 计算注意力
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多头
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        
        # 最终线性变换
        output = np.matmul(attention_output, self.W_o)
        
        # 缓存用于反向传播
        self.cache = {
            'X': X,
            'Q': Q,
            'K': K, 
            'V': V,
            'attention_weights': attention_weights,
            'attention_output': attention_output
        }
        
        return output

class SimpleCNN:
    """简单的CNN网络"""
    
    def __init__(self):
        self.conv1 = ConvolutionLayer(1, 32, 3, padding=1)
        self.pool1 = MaxPoolingLayer(2, 2)
        self.conv2 = ConvolutionLayer(32, 64, 3, padding=1)
        self.pool2 = MaxPoolingLayer(2, 2)
        
        # 全连接层权重（假设输入为28x28图像）
        self.fc_input_size = 64 * 7 * 7  # 经过两次池化后的尺寸
        self.W_fc = np.random.randn(10, self.fc_input_size) * np.sqrt(2.0 / self.fc_input_size)
        self.b_fc = np.zeros((10, 1))
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        前向传播
        X: (N, 1, 28, 28)
        """
        # 第一个卷积块
        out = self.conv1.forward(X)
        out = self.relu(out)
        out = self.pool1.forward(out)
        
        # 第二个卷积块
        out = self.conv2.forward(out)
        out = self.relu(out)
        out = self.pool2.forward(out)
        
        # 展平
        N = out.shape[0]
        out = out.reshape(N, -1)  # (N, fc_input_size)
        
        # 全连接层
        out = self.W_fc @ out.T + self.b_fc  # (10, N)
        
        return out.T  # (N, 10)

def demonstrate_matrix_multiplication_meaning():
    """演示矩阵乘法在深度学习中的意义"""
    print("="*60)
    print("矩阵乘法在深度学习中的意义")
    print("="*60)
    
    print("\n1. 线性变换的几何意义")
    
    # 创建2D点
    points = np.array([[1, 0, -1, 0],   # x坐标
                       [0, 1, 0, -1]])   # y坐标
    
    # 不同的变换矩阵
    transformations = {
        '恒等变换': np.array([[1, 0], [0, 1]]),
        '缩放变换': np.array([[2, 0], [0, 0.5]]),
        '旋转变换': np.array([[0, -1], [1, 0]]),  # 逆时针90度
        '反射变换': np.array([[-1, 0], [0, 1]]),  # 关于y轴反射
        '剪切变换': np.array([[1, 0.5], [0, 1]])
    }
    
    plt.figure(figsize=(15, 10))
    
    for i, (name, T) in enumerate(transformations.items()):
        plt.subplot(2, 3, i+1)
        
        # 原始点
        plt.scatter(points[0], points[1], c='blue', s=100, label='原始点', alpha=0.7)
        
        # 变换后的点
        transformed_points = T @ points
        plt.scatter(transformed_points[0], transformed_points[1], 
                   c='red', s=100, label='变换后', alpha=0.7)
        
        # 连线显示变换
        for j in range(points.shape[1]):
            plt.arrow(points[0, j], points[1, j],
                     transformed_points[0, j] - points[0, j],
                     transformed_points[1, j] - points[1, j],
                     head_width=0.1, head_length=0.1, fc='green', ec='green', alpha=0.6)
        
        plt.title(f'{name}\nT = {T}')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.legend()
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
    
    plt.tight_layout()
    plt.savefig('matrix_transformations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n2. 神经网络中的矩阵乘法")
    
    # 模拟神经网络的一层
    batch_size, input_dim, output_dim = 5, 4, 3
    
    X = np.random.randn(batch_size, input_dim)
    W = np.random.randn(output_dim, input_dim)
    b = np.random.randn(output_dim, 1)
    
    print(f"输入 X: {X.shape} (batch_size={batch_size}, input_dim={input_dim})")
    print(f"权重 W: {W.shape} (output_dim={output_dim}, input_dim={input_dim})")
    print(f"偏置 b: {b.shape}")
    
    # 线性变换
    Y = (W @ X.T + b).T  # 或者 Y = X @ W.T + b.T
    print(f"输出 Y: {Y.shape}")
    
    print(f"\n矩阵乘法 W @ X.T 的意义:")
    print(f"• 每一行代表一个输出特征")
    print(f"• 每一列代表一个样本")
    print(f"• W的每一行是一个'特征检测器'")
    print(f"• 输出是输入特征的线性组合")
    
    # 3. 注意力机制中的矩阵乘法
    print(f"\n3. 注意力机制中的矩阵乘法")
    
    seq_len, d_model = 6, 8
    X = np.random.randn(1, seq_len, d_model)  # (batch=1, seq_len, d_model)
    
    # 简化的自注意力
    Q = K = V = X  # 自注意力中Q=K=V
    
    # 注意力分数: Q @ K^T
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_model)  # (1, seq_len, seq_len)
    
    print(f"Query Q: {Q.shape}")
    print(f"Key K: {K.shape}")
    print(f"注意力分数矩阵: {scores.shape}")
    
    # 可视化注意力权重
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_weights[0], cmap='Blues', interpolation='nearest')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title('自注意力权重矩阵\n(每一行表示一个query对所有key的注意力分布)')
    
    # 添加数值标签
    for i in range(seq_len):
        for j in range(seq_len):
            plt.text(j, i, f'{attention_weights[0, i, j]:.2f}', 
                    ha='center', va='center', fontsize=8)
    
    plt.savefig('attention_weights.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n注意力机制中矩阵乘法的意义:")
    print(f"• Q @ K^T: 计算查询与键的相似度")
    print(f"• Softmax(scores) @ V: 加权聚合值向量")
    print(f"• 每个位置都能'看到'其他位置的信息")

def demonstrate_convolution_vs_transformer():
    """对比卷积和Transformer的特点"""
    print("\n" + "="*60)
    print("卷积神经网络 vs Transformer 对比")
    print("="*60)
    
    # 1. 感受野对比
    print("1. 感受野对比:")
    
    # 模拟图像
    image_size = 16
    input_image = np.random.randn(1, 1, image_size, image_size)
    
    # CNN的局部感受野
    conv = ConvolutionLayer(1, 1, 3, padding=1)
    conv_output = conv.forward(input_image)
    
    print(f"   CNN: 3x3卷积核，每个输出像素只'看到'局部3x3区域")
    print(f"   Transformer: 每个位置都能'看到'所有其他位置")
    
    # 2. 计算复杂度对比
    seq_len = 256  # 16x16图像展平
    d_model = 512
    
    # CNN复杂度（近似）
    cnn_flops = seq_len * 9 * d_model  # 9个邻居，线性变换
    
    # Transformer复杂度
    transformer_flops = seq_len * seq_len * d_model  # 自注意力
    
    print(f"\n2. 计算复杂度对比 (序列长度={seq_len}):")
    print(f"   CNN: O(n) ≈ {cnn_flops:,} FLOPs")
    print(f"   Transformer: O(n²) ≈ {transformer_flops:,} FLOPs")
    print(f"   比值: {transformer_flops/cnn_flops:.1f}x")
    
    # 3. 归纳偏置对比
    print(f"\n3. 归纳偏置对比:")
    print(f"   CNN:")
    print(f"   • 平移不变性: 相同的卷积核在所有位置共享")
    print(f"   • 局部性: 只关注局部邻域信息")
    print(f"   • 层次性: 底层检测边缘，高层检测复杂模式")
    
    print(f"   Transformer:")
    print(f"   • 位置无关性: 需要位置编码来感知顺序")
    print(f"   • 全局性: 每个位置都能访问全局信息")
    print(f"   • 灵活性: 学习任意的依赖关系")

def demonstrate_comprehensive_cnn_transformer():
    """综合演示CNN和Transformer"""
    print("🧠 CNN与Transformer基础演示")
    print("="*80)
    
    # 1. 矩阵乘法的意义
    demonstrate_matrix_multiplication_meaning()
    
    # 2. 卷积实现演示
    print("\n" + "="*60)
    print("手写卷积层演示")
    print("="*60)
    
    # 创建简单测试数据
    batch_size, in_channels, height, width = 2, 3, 8, 8
    X = np.random.randn(batch_size, in_channels, height, width)
    
    print(f"输入数据形状: {X.shape}")
    
    # 测试卷积层
    conv_layer = ConvolutionLayer(3, 16, 3, stride=1, padding=1)
    conv_output = conv_layer.forward(X)
    
    print(f"卷积输出形状: {conv_output.shape}")
    
    # 测试池化层
    pool_layer = MaxPoolingLayer(2, 2)
    pool_output = pool_layer.forward(conv_output)
    
    print(f"池化输出形状: {pool_output.shape}")
    
    # 3. 自注意力机制演示
    print("\n" + "="*60)
    print("手写自注意力机制演示")
    print("="*60)
    
    # 创建序列数据
    batch_size, seq_len, d_model = 2, 8, 64
    X_seq = np.random.randn(batch_size, seq_len, d_model)
    
    print(f"序列数据形状: {X_seq.shape}")
    
    # 测试自注意力
    attention = SelfAttention(d_model, n_heads=8)
    attention_output = attention.forward(X_seq)
    
    print(f"注意力输出形状: {attention_output.shape}")
    
    # 可视化注意力权重
    attention_weights = attention.cache['attention_weights']
    print(f"注意力权重形状: {attention_weights.shape}")  # (batch, heads, seq_len, seq_len)
    
    # 绘制第一个样本的注意力权重
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for head in range(8):
        ax = axes[head]
        weights = attention_weights[0, head, :, :]  # 第一个样本的第head个头
        
        im = ax.imshow(weights, cmap='Blues', interpolation='nearest')
        ax.set_title(f'Head {head+1}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('multi_head_attention.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. 性能对比
    print("\n" + "="*60)
    print("性能基准测试")
    print("="*60)
    
    # CNN性能测试
    print("CNN前向传播性能测试...")
    cnn = SimpleCNN()
    
    # 模拟MNIST数据
    mnist_batch = np.random.randn(32, 1, 28, 28)
    
    start_time = time.time()
    for _ in range(10):
        cnn_output = cnn.forward(mnist_batch)
    cnn_time = (time.time() - start_time) / 10
    
    print(f"CNN (32样本): {cnn_time*1000:.2f}ms/batch")
    print(f"CNN输出形状: {cnn_output.shape}")
    
    # Transformer性能测试（小规模）
    print("\nTransformer前向传播性能测试...")
    small_attention = SelfAttention(128, n_heads=8)
    small_seq = np.random.randn(32, 16, 128)  # 较小的序列长度
    
    start_time = time.time()
    for _ in range(10):
        attention_output = small_attention.forward(small_seq)
    transformer_time = (time.time() - start_time) / 10
    
    print(f"Transformer (32样本, 序列长度16): {transformer_time*1000:.2f}ms/batch")
    print(f"Transformer输出形状: {attention_output.shape}")
    
    # 5. 对比总结
    demonstrate_convolution_vs_transformer()
    
    # 6. 实际应用建议
    print(f"\n" + "="*60)
    print("实际应用建议")
    print("="*60)
    
    print(f"🖼️  图像任务:")
    print(f"   • CNN: 适合图像分类、目标检测、图像分割")
    print(f"   • Vision Transformer: 大数据集下性能更好，但需要更多数据")
    
    print(f"📝 序列任务:")
    print(f"   • RNN/LSTM: 适合较短序列，计算效率高")
    print(f"   • Transformer: 长序列建模能力强，并行化友好")
    
    print(f"🚀 混合架构:")
    print(f"   • ConvBERT: 结合CNN的局部性和Transformer的全局性")
    print(f"   • CoAtNet: 在不同阶段使用不同的架构")
    
    return {
        'conv_output': conv_output,
        'attention_output': attention_output,
        'cnn_time': cnn_time,
        'transformer_time': transformer_time
    }

if __name__ == "__main__":
    results = demonstrate_comprehensive_cnn_transformer()
    print("\n🎉 CNN与Transformer演示完成!")
