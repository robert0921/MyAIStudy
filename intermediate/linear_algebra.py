"""线性代数和矩阵运算基础实现"""
import numpy as np
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt

class LinearAlgebraDemo:
    """线性代数演示类"""
    
    @staticmethod
    def matrix_operations():
        """基础矩阵运算演示"""
        print("="*60)
        print("线性代数基础 - 矩阵运算演示")
        print("="*60)
        
        # 矩阵创建
        A = np.array([[1, 2, 3], 
                      [4, 5, 6], 
                      [7, 8, 9]])
        B = np.array([[9, 8, 7], 
                      [6, 5, 4], 
                      [3, 2, 1]])
        
        print("矩阵 A:")
        print(A)
        print("\n矩阵 B:")
        print(B)
        
        # 基础运算
        print("\n1. 矩阵加法 A + B:")
        print(A + B)
        
        print("\n2. 矩阵乘法 A @ B:")
        print(A @ B)
        
        print("\n3. 元素乘法 A * B:")
        print(A * B)
        
        print("\n4. 矩阵转置 A.T:")
        print(A.T)
        
        # 特殊运算
        print("\n5. 矩阵行列式 det(A):")
        print(np.linalg.det(A))
        
        print("\n6. 矩阵的秩 rank(A):")
        print(np.linalg.matrix_rank(A))
        
        # 特征值和特征向量
        symmetric_matrix = A @ A.T  # 创建对称矩阵
        eigenvals, eigenvecs = np.linalg.eig(symmetric_matrix)
        print("\n7. 特征值:")
        print(eigenvals)
        print("\n特征向量:")
        print(eigenvecs)
        
        return A, B
    
    @staticmethod
    def vector_operations():
        """向量运算演示"""
        print("\n" + "="*60)
        print("向量运算演示")
        print("="*60)
        
        # 向量创建
        v1 = np.array([1, 2, 3])
        v2 = np.array([4, 5, 6])
        
        print(f"向量 v1: {v1}")
        print(f"向量 v2: {v2}")
        
        # 点积
        dot_product = np.dot(v1, v2)
        print(f"\n1. 点积 v1·v2: {dot_product}")
        
        # 叉积（3D向量）
        cross_product = np.cross(v1, v2)
        print(f"2. 叉积 v1×v2: {cross_product}")
        
        # 向量范数
        l1_norm = np.linalg.norm(v1, ord=1)
        l2_norm = np.linalg.norm(v1, ord=2)
        print(f"3. L1范数 ||v1||₁: {l1_norm}")
        print(f"4. L2范数 ||v1||₂: {l2_norm}")
        
        # 单位向量
        unit_v1 = v1 / l2_norm
        print(f"5. 单位向量: {unit_v1}")
        
        # 向量夹角
        cos_angle = dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(cos_angle)
        print(f"6. 向量夹角: {np.degrees(angle):.2f}度")
        
        return v1, v2

class AutoDiffDemo:
    """自动微分机制演示"""
    
    @staticmethod
    def jacobian_computation():
        """Jacobian矩阵计算演示"""
        print("\n" + "="*60)
        print("Jacobian 矩阵计算演示")
        print("="*60)
        
        def f(x):
            """示例函数: f(x) = [x₁², x₁*x₂, x₂²]"""
            return np.array([
                x[0]**2,
                x[0] * x[1], 
                x[1]**2
            ])
        
        def jacobian_analytical(x):
            """解析计算的Jacobian矩阵"""
            return np.array([
                [2*x[0], 0],      # ∂f₁/∂x₁, ∂f₁/∂x₂
                [x[1], x[0]],     # ∂f₂/∂x₁, ∂f₂/∂x₂
                [0, 2*x[1]]       # ∂f₃/∂x₁, ∂f₃/∂x₂
            ])
        
        def jacobian_numerical(func, x, h=1e-5):
            """数值方法计算Jacobian矩阵"""
            f_x = func(x)
            n_out, n_in = len(f_x), len(x)
            jacobian = np.zeros((n_out, n_in))
            
            for i in range(n_in):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += h
                x_minus[i] -= h
                
                # 中心差分
                jacobian[:, i] = (func(x_plus) - func(x_minus)) / (2 * h)
            
            return jacobian
        
        # 测试点
        x = np.array([2.0, 3.0])
        print(f"测试点: x = {x}")
        print(f"函数值: f(x) = {f(x)}")
        
        # 解析Jacobian
        J_analytical = jacobian_analytical(x)
        print("\n解析计算的Jacobian矩阵:")
        print(J_analytical)
        
        # 数值Jacobian
        J_numerical = jacobian_numerical(f, x)
        print("\n数值计算的Jacobian矩阵:")
        print(J_numerical)
        
        # 误差分析
        error = np.abs(J_analytical - J_numerical)
        print(f"\n误差矩阵:")
        print(error)
        print(f"最大误差: {np.max(error):.2e}")
        
        return J_analytical, J_numerical
    
    @staticmethod
    def chain_rule_demo():
        """链式法则演示"""
        print("\n" + "="*60)
        print("链式法则演示")
        print("="*60)
        
        # 复合函数: h(x) = g(f(x)) = sin(x²)
        # f(x) = x², g(y) = sin(y)
        # h'(x) = g'(f(x)) * f'(x) = cos(x²) * 2x
        
        def f(x):
            return x**2
        
        def g(y):
            return np.sin(y)
        
        def h(x):
            return g(f(x))  # 复合函数
        
        def f_prime(x):
            return 2*x
        
        def g_prime(y):
            return np.cos(y)
        
        def h_prime_chain_rule(x):
            """使用链式法则计算导数"""
            return g_prime(f(x)) * f_prime(x)
        
        def h_prime_numerical(x, h_val=1e-5):
            """数值方法计算导数"""
            return (h(x + h_val) - h(x - h_val)) / (2 * h_val)
        
        # 测试点
        x = 2.0
        print(f"测试点: x = {x}")
        print(f"f(x) = x² = {f(x)}")
        print(f"g(f(x)) = sin(f(x)) = {h(x)}")
        
        # 链式法则计算
        chain_derivative = h_prime_chain_rule(x)
        print(f"\n链式法则: h'(x) = cos(x²) × 2x = {chain_derivative}")
        
        # 数值验证
        numerical_derivative = h_prime_numerical(x)
        print(f"数值计算: h'(x) ≈ {numerical_derivative}")
        
        # 误差
        error = abs(chain_derivative - numerical_derivative)
        print(f"误差: {error:.2e}")
        
        return chain_derivative, numerical_derivative

class LinearLayerFromScratch:
    """从零实现线性层（含详细推导）"""
    
    def __init__(self, in_features: int, out_features: int):
        # Xavier/Glorot初始化
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.W = np.random.uniform(-limit, limit, (out_features, in_features))
        self.b = np.zeros((out_features, 1))
        
        # 缓存用于反向传播
        self.cache = {}
        
        print(f"线性层初始化: {in_features} -> {out_features}")
        print(f"权重形状: {self.W.shape}, 偏置形状: {self.b.shape}")
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        前向传播: Y = WX + b
        Args:
            X: 输入矩阵 (n_features, n_samples)
        Returns:
            Y: 输出矩阵 (n_outputs, n_samples)
        """
        self.cache['X'] = X
        Y = np.dot(self.W, X) + self.b
        return Y
    
    def backward(self, dY: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        反向传播推导:
        设 Y = WX + b
        
        损失函数对权重的梯度:
        ∂L/∂W = ∂L/∂Y × ∂Y/∂W = dY × X^T
        
        损失函数对偏置的梯度:
        ∂L/∂b = ∂L/∂Y × ∂Y/∂b = dY × 1
        
        损失函数对输入的梯度:
        ∂L/∂X = ∂L/∂Y × ∂Y/∂X = W^T × dY
        """
        X = self.cache['X']
        
        # 计算梯度
        dW = np.dot(dY, X.T)  # (n_outputs, n_samples) × (n_samples, n_features)
        db = np.sum(dY, axis=1, keepdims=True)  # 沿样本维度求和
        dX = np.dot(self.W.T, dY)  # (n_features, n_outputs) × (n_outputs, n_samples)
        
        return dX, dW, db
    
    def manual_gradient_check(self, X: np.ndarray, epsilon: float = 1e-5):
        """手动梯度检查"""
        print("\n执行梯度检查...")
        
        # 前向传播
        Y = self.forward(X)
        
        # 假设损失函数为 L = 0.5 * ||Y||²
        loss = 0.5 * np.sum(Y**2)
        dY = Y  # ∂L/∂Y = Y
        
        # 反向传播计算梯度
        _, dW_analytical, db_analytical = self.backward(dY)
        
        # 数值方法检查权重梯度
        dW_numerical = np.zeros_like(self.W)
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                # W[i,j] + epsilon
                self.W[i, j] += epsilon
                Y_plus = self.forward(X)
                loss_plus = 0.5 * np.sum(Y_plus**2)
                
                # W[i,j] - epsilon
                self.W[i, j] -= 2 * epsilon
                Y_minus = self.forward(X)
                loss_minus = 0.5 * np.sum(Y_minus**2)
                
                # 中心差分
                dW_numerical[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
                
                # 恢复原值
                self.W[i, j] += epsilon
        
        # 比较解析梯度和数值梯度
        diff = np.abs(dW_analytical - dW_numerical)
        max_diff = np.max(diff)
        relative_error = max_diff / (np.max(np.abs(dW_analytical)) + 1e-8)
        
        print(f"权重梯度检查:")
        print(f"最大绝对误差: {max_diff:.2e}")
        print(f"相对误差: {relative_error:.2e}")
        
        if relative_error < 1e-5:
            print("✓ 梯度检查通过!")
        else:
            print("✗ 梯度检查失败!")
        
        return relative_error

class ReLUFromScratch:
    """从零实现ReLU激活函数"""
    
    def __init__(self):
        self.cache = {}
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        前向传播: Y = max(0, X)
        """
        self.cache['X'] = X
        Y = np.maximum(0, X)
        return Y
    
    def backward(self, dY: np.ndarray) -> np.ndarray:
        """
        反向传播推导:
        ReLU导数: ∂Y/∂X = 1 if X > 0 else 0
        链式法则: ∂L/∂X = ∂L/∂Y × ∂Y/∂X
        """
        X = self.cache['X']
        dX = dY * (X > 0)  # 元素级乘法
        return dX
    
    def visualize_function_and_derivative(self):
        """可视化ReLU函数及其导数"""
        x = np.linspace(-5, 5, 1000)
        y = np.maximum(0, x)
        dy = (x > 0).astype(float)
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(x, y, 'b-', linewidth=2, label='ReLU(x)')
        plt.grid(True, alpha=0.3)
        plt.xlabel('x')
        plt.ylabel('ReLU(x)')
        plt.title('ReLU激活函数')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(x, dy, 'r-', linewidth=2, label="ReLU'(x)")
        plt.grid(True, alpha=0.3)
        plt.xlabel('x')
        plt.ylabel("ReLU'(x)")
        plt.title('ReLU导数')
        plt.legend()
        plt.ylim(-0.1, 1.1)
        
        plt.tight_layout()
        plt.savefig('relu_function_and_derivative.png', dpi=300, bbox_inches='tight')
        plt.show()

def demonstrate_comprehensive_linear_algebra():
    """综合演示线性代数功能"""
    print("🧮 综合线性代数与自动微分演示")
    print("="*80)
    
    # 1. 基础矩阵运算
    linear_demo = LinearAlgebraDemo()
    A, B = linear_demo.matrix_operations()
    v1, v2 = linear_demo.vector_operations()
    
    # 2. 自动微分机制
    autodiff_demo = AutoDiffDemo()
    J_analytical, J_numerical = autodiff_demo.jacobian_computation()
    chain_result = autodiff_demo.chain_rule_demo()
    
    # 3. 手写线性层和ReLU
    print("\n" + "="*60)
    print("手写神经网络层演示")
    print("="*60)
    
    # 创建层
    linear_layer = LinearLayerFromScratch(3, 2)
    relu_layer = ReLUFromScratch()
    
    # 测试数据
    X = np.random.randn(3, 5)  # 3个特征，5个样本
    print(f"\n输入数据形状: {X.shape}")
    
    # 前向传播
    Z = linear_layer.forward(X)
    A = relu_layer.forward(Z)
    
    print(f"线性层输出形状: {Z.shape}")
    print(f"ReLU层输出形状: {A.shape}")
    
    # 梯度检查
    error = linear_layer.manual_gradient_check(X)
    
    # 可视化ReLU
    relu_layer.visualize_function_and_derivative()
    
    # 4. 与PyTorch对比
    print("\n" + "="*60)
    print("与PyTorch autograd对比")
    print("="*60)
    
    try:
        import torch
        import torch.nn as nn
        
        # PyTorch实现
        torch_linear = nn.Linear(3, 2, bias=True)
        torch_relu = nn.ReLU()
        
        # 设置相同的权重
        with torch.no_grad():
            torch_linear.weight.copy_(torch.from_numpy(linear_layer.W).float())
            torch_linear.bias.copy_(torch.from_numpy(linear_layer.b.flatten()).float())
        
        # PyTorch前向传播
        X_torch = torch.from_numpy(X.T).float().requires_grad_(True)  # (n_samples, n_features)
        Z_torch = torch_linear(X_torch)
        A_torch = torch_relu(Z_torch)
        
        # 计算损失并反向传播
        loss = 0.5 * torch.sum(A_torch**2)
        loss.backward()
        
        print("手工实现的权重:")
        print(linear_layer.W)
        print("\nPyTorch的权重:")
        print(torch_linear.weight.data.numpy())
        
        print("\n权重差异:")
        print(np.abs(linear_layer.W - torch_linear.weight.data.numpy()))
        
        print("✓ 与PyTorch对比完成!")
        
    except ImportError:
        print("PyTorch未安装，跳过对比")
    
    return {
        'matrix_operations': (A, B),
        'vector_operations': (v1, v2),
        'jacobian': (J_analytical, J_numerical),
        'layers': (linear_layer, relu_layer),
        'gradient_error': error
    }

if __name__ == "__main__":
    results = demonstrate_comprehensive_linear_algebra()
    print("\n🎉 所有演示完成!")
