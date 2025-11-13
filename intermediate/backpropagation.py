"""反向传播详细实现与PyTorch对比"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import time

class TwoLayerNetwork:
    """手工实现的两层神经网络（详细反向传播推导）"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        # Xavier初始化
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((output_size, 1))
        
        # 存储中间计算结果
        self.cache = {}
        
        print(f"网络结构: {input_size} -> {hidden_size} -> {output_size}")
        print(f"参数数量: {self.W1.size + self.b1.size + self.W2.size + self.b2.size}")
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        前向传播
        X: (n_features, n_samples)
        """
        # 第一层: Z1 = W1*X + b1
        Z1 = np.dot(self.W1, X) + self.b1
        
        # ReLU激活: A1 = max(0, Z1)
        A1 = np.maximum(0, Z1)
        
        # 第二层: Z2 = W2*A1 + b2
        Z2 = np.dot(self.W2, A1) + self.b2
        
        # 缓存中间结果用于反向传播
        self.cache = {
            'X': X,
            'Z1': Z1,
            'A1': A1,
            'Z2': Z2
        }
        
        return Z2
    
    def compute_loss(self, Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
        """计算均方误差损失"""
        m = Y_true.shape[1]  # 样本数量
        loss = (1.0 / (2 * m)) * np.sum((Y_pred - Y_true) ** 2)
        return loss
    
    def backward(self, Y_pred: np.ndarray, Y_true: np.ndarray) -> Dict[str, np.ndarray]:
        """
        详细反向传播推导
        
        网络结构: X -> Z1 -> A1 -> Z2 -> Y_pred
        损失函数: L = (1/2m) * ||Y_pred - Y_true||²
        
        反向传播路径: ∂L/∂Y_pred -> ∂L/∂Z2 -> ∂L/∂W2, ∂L/∂b2, ∂L/∂A1 
                    -> ∂L/∂Z1 -> ∂L/∂W1, ∂L/∂b1
        """
        m = Y_true.shape[1]  # 样本数量
        
        # 从缓存中获取前向传播的中间结果
        X = self.cache['X']
        Z1 = self.cache['Z1'] 
        A1 = self.cache['A1']
        Z2 = self.cache['Z2']
        
        # 步骤1: 计算输出层梯度
        # ∂L/∂Z2 = ∂L/∂Y_pred * ∂Y_pred/∂Z2 = (Y_pred - Y_true) / m
        dZ2 = (Y_pred - Y_true) / m
        
        # 步骤2: 计算第二层参数梯度
        # ∂L/∂W2 = ∂L/∂Z2 * ∂Z2/∂W2 = dZ2 * A1^T
        dW2 = np.dot(dZ2, A1.T)
        
        # ∂L/∂b2 = ∂L/∂Z2 * ∂Z2/∂b2 = dZ2 * 1
        db2 = np.sum(dZ2, axis=1, keepdims=True)
        
        # 步骤3: 计算隐藏层梯度
        # ∂L/∂A1 = ∂L/∂Z2 * ∂Z2/∂A1 = W2^T * dZ2
        dA1 = np.dot(self.W2.T, dZ2)
        
        # 步骤4: 通过ReLU反向传播
        # ∂L/∂Z1 = ∂L/∂A1 * ∂A1/∂Z1 = dA1 * (Z1 > 0)
        dZ1 = dA1 * (Z1 > 0)
        
        # 步骤5: 计算第一层参数梯度
        # ∂L/∂W1 = ∂L/∂Z1 * ∂Z1/∂W1 = dZ1 * X^T
        dW1 = np.dot(dZ1, X.T)
        
        # ∂L/∂b1 = ∂L/∂Z1 * ∂Z1/∂b1 = dZ1 * 1
        db1 = np.sum(dZ1, axis=1, keepdims=True)
        
        return {
            'dW1': dW1,
            'db1': db1,
            'dW2': dW2,
            'db2': db2
        }
    
    def update_parameters(self, gradients: Dict[str, np.ndarray], learning_rate: float):
        """更新参数"""
        self.W1 -= learning_rate * gradients['dW1']
        self.b1 -= learning_rate * gradients['db1']
        self.W2 -= learning_rate * gradients['dW2']
        self.b2 -= learning_rate * gradients['db2']
    
    def train_step(self, X: np.ndarray, Y: np.ndarray, learning_rate: float) -> float:
        """执行一步训练"""
        # 前向传播
        Y_pred = self.forward(X)
        
        # 计算损失
        loss = self.compute_loss(Y_pred, Y)
        
        # 反向传播
        gradients = self.backward(Y_pred, Y)
        
        # 更新参数
        self.update_parameters(gradients, learning_rate)
        
        return loss
    
    def numerical_gradient_check(self, X: np.ndarray, Y: np.ndarray, epsilon: float = 1e-5):
        """数值方法验证梯度计算"""
        print("执行数值梯度检查...")
        
        # 计算解析梯度
        Y_pred = self.forward(X)
        analytical_grads = self.backward(Y_pred, Y)
        
        # 数值梯度检查
        def check_gradient(param_name, param, grad):
            print(f"\n检查参数: {param_name}")
            numerical_grad = np.zeros_like(param)
            
            # 展平参数以便迭代
            param_flat = param.flatten()
            grad_flat = grad.flatten()
            numerical_grad_flat = numerical_grad.flatten()
            
            for i in range(min(10, len(param_flat))):  # 只检查前10个参数
                # 保存原值
                original_value = param_flat[i]
                
                # 计算 f(θ + ε)
                param_flat[i] = original_value + epsilon
                param.flat = param_flat
                Y_pred_plus = self.forward(X)
                loss_plus = self.compute_loss(Y_pred_plus, Y)
                
                # 计算 f(θ - ε)
                param_flat[i] = original_value - epsilon
                param.flat = param_flat
                Y_pred_minus = self.forward(X)
                loss_minus = self.compute_loss(Y_pred_minus, Y)
                
                # 中心差分
                numerical_grad_flat[i] = (loss_plus - loss_minus) / (2 * epsilon)
                
                # 恢复原值
                param_flat[i] = original_value
                param.flat = param_flat
            
            numerical_grad.flat = numerical_grad_flat
            
            # 比较梯度
            diff = np.abs(grad - numerical_grad)
            max_diff = np.max(diff)
            relative_error = max_diff / (np.max(np.abs(grad)) + 1e-8)
            
            print(f"  最大绝对误差: {max_diff:.2e}")
            print(f"  相对误差: {relative_error:.2e}")
            
            return relative_error < 1e-5
        
        # 检查所有参数
        results = []
        results.append(check_gradient('W1', self.W1, analytical_grads['dW1']))
        results.append(check_gradient('b1', self.b1, analytical_grads['db1']))
        results.append(check_gradient('W2', self.W2, analytical_grads['dW2']))
        results.append(check_gradient('b2', self.b2, analytical_grads['db2']))
        
        if all(results):
            print("\n✓ 所有梯度检查通过!")
        else:
            print("\n✗ 部分梯度检查失败!")
        
        return all(results)

def compare_with_pytorch():
    """与PyTorch autograd对比"""
    print("\n" + "="*60)
    print("与PyTorch autograd对比")
    print("="*60)
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        # 网络参数
        input_size, hidden_size, output_size = 3, 4, 2
        n_samples = 100
        
        # 生成相同的测试数据
        np.random.seed(42)
        X_np = np.random.randn(input_size, n_samples)
        Y_np = np.random.randn(output_size, n_samples)
        
        # 1. 手工实现
        print("1. 训练手工实现的网络...")
        manual_net = TwoLayerNetwork(input_size, hidden_size, output_size)
        
        manual_losses = []
        learning_rate = 0.01
        n_epochs = 100
        
        start_time = time.time()
        for epoch in range(n_epochs):
            loss = manual_net.train_step(X_np, Y_np, learning_rate)
            manual_losses.append(loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss:.6f}")
        
        manual_time = time.time() - start_time
        
        # 梯度检查
        manual_net.numerical_gradient_check(X_np[:, :5], Y_np[:, :5])
        
        # 2. PyTorch实现
        print("\n2. 训练PyTorch网络...")
        
        class PyTorchNet(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.linear1 = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.linear2 = nn.Linear(hidden_size, output_size)
                
            def forward(self, x):
                x = self.linear1(x)
                x = self.relu(x)
                x = self.linear2(x)
                return x
        
        # 创建PyTorch网络
        torch_net = PyTorchNet(input_size, hidden_size, output_size)
        
        # 设置相同的初始权重
        with torch.no_grad():
            torch_net.linear1.weight.copy_(torch.from_numpy(manual_net.W1).float())
            torch_net.linear1.bias.copy_(torch.from_numpy(manual_net.b1.flatten()).float())
            torch_net.linear2.weight.copy_(torch.from_numpy(manual_net.W2).float())
            torch_net.linear2.bias.copy_(torch.from_numpy(manual_net.b2.flatten()).float())
        
        # 准备PyTorch数据
        X_torch = torch.from_numpy(X_np.T).float()  # (n_samples, n_features)
        Y_torch = torch.from_numpy(Y_np.T).float()  # (n_samples, n_outputs)
        
        criterion = nn.MSELoss()
        optimizer = optim.SGD(torch_net.parameters(), lr=learning_rate)
        
        torch_losses = []
        
        start_time = time.time()
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            Y_pred = torch_net(X_torch)
            loss = criterion(Y_pred, Y_torch)
            
            loss.backward()
            optimizer.step()
            
            torch_losses.append(loss.item())
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
        
        torch_time = time.time() - start_time
        
        # 3. 结果对比
        print("\n" + "="*60)
        print("结果对比")
        print("="*60)
        
        print(f"手工实现训练时间: {manual_time:.4f}秒")
        print(f"PyTorch训练时间: {torch_time:.4f}秒")
        print(f"速度差异: {manual_time/torch_time:.2f}x")
        
        print(f"\n最终损失:")
        print(f"手工实现: {manual_losses[-1]:.6f}")
        print(f"PyTorch: {torch_losses[-1]:.6f}")
        print(f"损失差异: {abs(manual_losses[-1] - torch_losses[-1]):.2e}")
        
        # 绘制训练曲线
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(manual_losses, 'b-', label='手工实现', linewidth=2)
        plt.plot(torch_losses, 'r--', label='PyTorch', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('训练损失对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(manual_losses, 'b-', label='手工实现', linewidth=2)
        plt.plot(torch_losses, 'r--', label='PyTorch', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('训练损失对比 (对数坐标)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 参数对比
        print(f"\n参数对比:")
        print(f"W1差异: {np.max(np.abs(manual_net.W1 - torch_net.linear1.weight.data.numpy())):.2e}")
        print(f"W2差异: {np.max(np.abs(manual_net.W2 - torch_net.linear2.weight.data.numpy())):.2e}")
        
        return {
            'manual_losses': manual_losses,
            'torch_losses': torch_losses,
            'manual_time': manual_time,
            'torch_time': torch_time
        }
        
    except ImportError:
        print("PyTorch未安装，跳过对比")
        return None

def demonstrate_backprop_step_by_step():
    """逐步演示反向传播过程"""
    print("\n" + "="*60)
    print("逐步演示反向传播过程")
    print("="*60)
    
    # 创建简单示例
    net = TwoLayerNetwork(2, 3, 1)
    
    # 单个样本
    X = np.array([[1.0], [2.0]])  # (2, 1)
    Y = np.array([[0.5]])         # (1, 1)
    
    print("输入数据:")
    print(f"X = {X.flatten()}")
    print(f"Y = {Y.flatten()}")
    
    print(f"\n初始权重:")
    print(f"W1 = \n{net.W1}")
    print(f"b1 = {net.b1.flatten()}")
    print(f"W2 = \n{net.W2}")
    print(f"b2 = {net.b2.flatten()}")
    
    # 前向传播详细过程
    print(f"\n前向传播:")
    
    # 第一层
    Z1 = np.dot(net.W1, X) + net.b1
    print(f"Z1 = W1 @ X + b1 = {Z1.flatten()}")
    
    A1 = np.maximum(0, Z1)
    print(f"A1 = ReLU(Z1) = {A1.flatten()}")
    
    # 第二层
    Z2 = np.dot(net.W2, A1) + net.b2
    print(f"Z2 = W2 @ A1 + b2 = {Z2.flatten()}")
    
    Y_pred = Z2
    print(f"Y_pred = {Y_pred.flatten()}")
    
    # 损失
    loss = 0.5 * np.sum((Y_pred - Y) ** 2)
    print(f"Loss = 0.5 * (Y_pred - Y)² = {loss}")
    
    # 反向传播详细过程
    print(f"\n反向传播:")
    
    # 输出层梯度
    dZ2 = Y_pred - Y
    print(f"dZ2 = Y_pred - Y = {dZ2.flatten()}")
    
    # 第二层参数梯度
    dW2 = np.dot(dZ2, A1.T)
    db2 = dZ2
    print(f"dW2 = dZ2 @ A1.T = {dW2}")
    print(f"db2 = dZ2 = {db2.flatten()}")
    
    # 隐藏层梯度
    dA1 = np.dot(net.W2.T, dZ2)
    print(f"dA1 = W2.T @ dZ2 = {dA1.flatten()}")
    
    # ReLU反向传播
    dZ1 = dA1 * (Z1 > 0)
    print(f"dZ1 = dA1 * (Z1 > 0) = {dZ1.flatten()}")
    
    # 第一层参数梯度
    dW1 = np.dot(dZ1, X.T)
    db1 = dZ1
    print(f"dW1 = dZ1 @ X.T = \n{dW1}")
    print(f"db1 = dZ1 = {db1.flatten()}")
    
    # 验证计算
    net.cache = {'X': X, 'Z1': Z1, 'A1': A1, 'Z2': Z2}
    computed_grads = net.backward(Y_pred, Y)
    
    print(f"\n验证梯度计算:")
    print(f"dW1误差: {np.max(np.abs(dW1 - computed_grads['dW1'])):.2e}")
    print(f"db1误差: {np.max(np.abs(db1 - computed_grads['db1'])):.2e}")
    print(f"dW2误差: {np.max(np.abs(dW2 - computed_grads['dW2'])):.2e}")
    print(f"db2误差: {np.max(np.abs(db2 - computed_grads['db2'])):.2e}")

def demonstrate_comprehensive_backprop():
    """综合演示反向传播"""
    print("🔄 综合反向传播演示")
    print("="*80)
    
    # 1. 逐步演示
    demonstrate_backprop_step_by_step()
    
    # 2. 与PyTorch对比
    pytorch_results = compare_with_pytorch()
    
    # 3. 创建综合测试
    print("\n" + "="*60)
    print("综合梯度验证测试")
    print("="*60)
    
    # 不同规模的网络测试
    test_configs = [
        (2, 3, 1),
        (5, 10, 3),
        (10, 20, 5)
    ]
    
    for input_size, hidden_size, output_size in test_configs:
        print(f"\n测试网络: {input_size}-{hidden_size}-{output_size}")
        
        net = TwoLayerNetwork(input_size, hidden_size, output_size)
        
        # 随机数据
        X = np.random.randn(input_size, 10)
        Y = np.random.randn(output_size, 10)
        
        # 梯度检查
        success = net.numerical_gradient_check(X, Y)
        print(f"梯度检查: {'✓ 通过' if success else '✗ 失败'}")
    
    print("\n🎉 反向传播演示完成!")
    return pytorch_results

if __name__ == "__main__":
    results = demonstrate_comprehensive_backprop()
