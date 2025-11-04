"""优化算法详细实现与对比"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time

class OptimizerBase:
    """优化器基类"""
    def __init__(self, learning_rate: float = 0.001):
        self.lr = learning_rate
        self.history = {'loss': [], 'lr': []}
    
    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        """执行一步优化"""
        raise NotImplementedError
    
    def zero_grad(self):
        """清零梯度（某些优化器需要）"""
        pass

class SGDOptimizer(OptimizerBase):
    """随机梯度下降（SGD）"""
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = {}
        
    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        if not self.velocity:  # 初始化速度
            self.velocity = {name: np.zeros_like(param) 
                           for name, param in params.items()}
        
        for name in params:
            # 动量更新
            self.velocity[name] = (self.momentum * self.velocity[name] - 
                                 self.lr * grads[name])
            params[name] += self.velocity[name]

class AdamOptimizer(OptimizerBase):
    """Adam优化器（详细数学推导）"""
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # 一阶矩估计
        self.v = {}  # 二阶矩估计
        self.t = 0   # 时间步
        
        print(f"Adam优化器初始化:")
        print(f"  学习率: {learning_rate}")
        print(f"  β₁: {beta1} (一阶矩衰减率)")
        print(f"  β₂: {beta2} (二阶矩衰减率)")
        print(f"  ε: {epsilon} (数值稳定性)")
    
    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        if not self.m:  # 初始化矩估计
            self.m = {name: np.zeros_like(param) 
                     for name, param in params.items()}
            self.v = {name: np.zeros_like(param) 
                     for name, param in params.items()}
        
        self.t += 1
        
        for name in params:
            g = grads[name]
            
            # 更新有偏一阶矩估计
            # m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g
            
            # 更新有偏二阶矩估计
            # v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (g ** 2)
            
            # 偏差修正
            # m̂_t = m_t / (1 - β₁^t)
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            
            # v̂_t = v_t / (1 - β₂^t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            
            # 参数更新
            # θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
            params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

class RMSPropOptimizer(OptimizerBase):
    """RMSProp优化器"""
    def __init__(self, learning_rate: float = 0.001, decay: float = 0.9, 
                 epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.decay = decay
        self.epsilon = epsilon
        self.cache = {}
        
        print(f"RMSProp优化器初始化:")
        print(f"  学习率: {learning_rate}")
        print(f"  衰减率: {decay}")
        print(f"  ε: {epsilon}")
    
    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        if not self.cache:  # 初始化缓存
            self.cache = {name: np.zeros_like(param) 
                         for name, param in params.items()}
        
        for name in params:
            g = grads[name]
            
            # 更新梯度平方的移动平均
            # cache = decay * cache + (1 - decay) * g²
            self.cache[name] = (self.decay * self.cache[name] + 
                              (1 - self.decay) * (g ** 2))
            
            # 参数更新
            # θ = θ - lr * g / (√cache + ε)
            params[name] -= (self.lr * g / 
                           (np.sqrt(self.cache[name]) + self.epsilon))

class SimpleNetwork:
    """简单的测试网络"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        # He初始化
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((output_size, 1))
        
        self.params = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2
        }
        
        self.cache = {}
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """前向传播"""
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = np.maximum(0, Z1)  # ReLU
        Z2 = np.dot(self.W2, A1) + self.b2
        
        self.cache = {'X': X, 'Z1': Z1, 'A1': A1, 'Z2': Z2}
        return Z2
    
    def backward(self, Y_pred: np.ndarray, Y_true: np.ndarray) -> Dict[str, np.ndarray]:
        """反向传播"""
        m = Y_true.shape[1]
        
        # 获取缓存
        X = self.cache['X']
        Z1 = self.cache['Z1']
        A1 = self.cache['A1']
        
        # 反向传播
        dZ2 = (Y_pred - Y_true) / m
        dW2 = np.dot(dZ2, A1.T)
        db2 = np.sum(dZ2, axis=1, keepdims=True)
        
        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * (Z1 > 0)
        dW1 = np.dot(dZ1, X.T)
        db1 = np.sum(dZ1, axis=1, keepdims=True)
        
        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
    
    def compute_loss(self, Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
        """计算损失"""
        return 0.5 * np.mean((Y_pred - Y_true) ** 2)

def compare_optimizers_on_mnist():
    """在MNIST数据集上比较优化器"""
    print("="*60)
    print("优化器在模拟MNIST数据上的对比")
    print("="*60)
    
    # 模拟MNIST数据（简化版）
    np.random.seed(42)
    n_samples = 1000
    input_size = 784  # 28x28
    hidden_size = 128
    output_size = 10  # 10个类别
    
    # 生成合成数据
    X = np.random.randn(input_size, n_samples) * 0.1
    Y = np.eye(output_size)[:, np.random.randint(0, output_size, n_samples)]
    
    print(f"数据集大小: {n_samples} 样本")
    print(f"网络结构: {input_size} -> {hidden_size} -> {output_size}")
    
    # 创建优化器
    optimizers = {
        'SGD': SGDOptimizer(learning_rate=0.01),
        'SGD+Momentum': SGDOptimizer(learning_rate=0.01, momentum=0.9),
        'Adam': AdamOptimizer(learning_rate=0.001),
        'RMSProp': RMSPropOptimizer(learning_rate=0.001)
    }
    
    # 训练参数
    n_epochs = 200
    batch_size = 32
    n_batches = n_samples // batch_size
    
    # 存储结果
    results = {}
    
    for name, optimizer in optimizers.items():
        print(f"\n训练使用 {name} 优化器...")
        
        # 创建网络（每个优化器使用相同的初始权重）
        np.random.seed(42)  # 确保相同的初始化
        network = SimpleNetwork(input_size, hidden_size, output_size)
        
        losses = []
        times = []
        
        start_time = time.time()
        
        for epoch in range(n_epochs):
            epoch_loss = 0
            
            # 随机打乱数据
            indices = np.random.permutation(n_samples)
            
            for i in range(n_batches):
                # 获取批次数据
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                batch_indices = indices[start_idx:end_idx]
                
                X_batch = X[:, batch_indices]
                Y_batch = Y[:, batch_indices]
                
                # 前向传播
                Y_pred = network.forward(X_batch)
                
                # 计算损失
                batch_loss = network.compute_loss(Y_pred, Y_batch)
                epoch_loss += batch_loss
                
                # 反向传播
                grads = network.backward(Y_pred, Y_batch)
                
                # 更新参数
                optimizer.step(network.params, grads)
            
            # 记录损失和时间
            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)
            times.append(time.time() - start_time)
            
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1:3d}, Loss: {avg_loss:.6f}")
        
        results[name] = {
            'losses': losses,
            'times': times,
            'final_loss': losses[-1],
            'convergence_time': times[-1]
        }
    
    return results

def visualize_optimizer_comparison(results: Dict):
    """可视化优化器对比结果"""
    print("\n生成优化器对比图表...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 损失曲线
    ax1 = axes[0, 0]
    for name, data in results.items():
        ax1.plot(data['losses'], label=name, linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('训练损失对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 对数尺度损失曲线
    ax2 = axes[0, 1]
    for name, data in results.items():
        ax2.plot(data['losses'], label=name, linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('训练损失对比 (对数坐标)')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 收敛速度
    ax3 = axes[1, 0]
    names = list(results.keys())
    final_losses = [results[name]['final_loss'] for name in names]
    bars = ax3.bar(names, final_losses)
    ax3.set_ylabel('Final Loss')
    ax3.set_title('最终损失对比')
    ax3.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, loss in zip(bars, final_losses):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.4f}', ha='center', va='bottom')
    
    # 4. 训练时间
    ax4 = axes[1, 1]
    times = [results[name]['convergence_time'] for name in names]
    bars = ax4.bar(names, times)
    ax4.set_ylabel('Training Time (seconds)')
    ax4.set_title('训练时间对比')
    ax4.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('optimizer_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def demonstrate_optimizer_mathematics():
    """演示优化器的数学原理"""
    print("\n" + "="*60)
    print("优化器数学原理演示")
    print("="*60)
    
    # 创建一个简单的二次函数来演示优化过程
    def f(x, y):
        """目标函数: f(x,y) = x² + 10y²"""
        return x**2 + 10*y**2
    
    def grad_f(x, y):
        """梯度: ∇f = [2x, 20y]"""
        return np.array([2*x, 20*y])
    
    # 优化器参数
    learning_rate = 0.1
    n_steps = 50
    
    # 初始点
    start_point = np.array([5.0, 2.0])
    
    print(f"目标函数: f(x,y) = x² + 10y²")
    print(f"梯度函数: ∇f = [2x, 20y]")
    print(f"初始点: {start_point}")
    print(f"学习率: {learning_rate}")
    
    # 测试不同优化器
    optimizers_2d = {
        'SGD': SGDOptimizer(learning_rate),
        'SGD+Momentum': SGDOptimizer(learning_rate, momentum=0.9),
        'Adam': AdamOptimizer(learning_rate),
        'RMSProp': RMSPropOptimizer(learning_rate)
    }
    
    # 记录轨迹
    trajectories = {}
    
    for name, optimizer in optimizers_2d.items():
        print(f"\n{name} 优化轨迹:")
        
        # 初始化参数
        params = {'position': start_point.copy()}
        trajectory = [params['position'].copy()]
        
        for step in range(n_steps):
            x, y = params['position']
            
            # 计算梯度
            grad = grad_f(x, y)
            grads = {'position': grad}
            
            # 优化步骤
            optimizer.step(params, grads)
            trajectory.append(params['position'].copy())
            
            if step < 5 or (step + 1) % 10 == 0:
                current_loss = f(params['position'][0], params['position'][1])
                print(f"  Step {step+1:2d}: x={params['position'][0]:6.3f}, "
                      f"y={params['position'][1]:6.3f}, f={current_loss:8.3f}")
        
        trajectories[name] = np.array(trajectory)
    
    # 可视化优化轨迹
    plt.figure(figsize=(12, 8))
    
    # 创建等高线图
    x = np.linspace(-6, 6, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    
    plt.contour(X, Y, Z, levels=20, alpha=0.6)
    plt.contourf(X, Y, Z, levels=20, alpha=0.3, cmap='viridis')
    
    # 绘制优化轨迹
    colors = ['red', 'blue', 'green', 'orange']
    for i, (name, trajectory) in enumerate(trajectories.items()):
        plt.plot(trajectory[:, 0], trajectory[:, 1], 
                'o-', color=colors[i], label=name, 
                linewidth=2, markersize=4)
        
        # 标记起点和终点
        plt.plot(trajectory[0, 0], trajectory[0, 1], 
                's', color=colors[i], markersize=8)
        plt.plot(trajectory[-1, 0], trajectory[-1, 1], 
                '*', color=colors[i], markersize=12)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('不同优化器的优化轨迹\n(正方形=起点, 星形=终点)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.savefig('optimizer_trajectories.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return trajectories

def demonstrate_learning_rate_effects():
    """演示学习率对收敛的影响"""
    print("\n" + "="*60)
    print("学习率对收敛的影响")
    print("="*60)
    
    # 简单的一维优化问题: f(x) = x²
    def f(x):
        return x**2
    
    def grad_f(x):
        return 2*x
    
    learning_rates = [0.01, 0.1, 0.5, 1.0, 1.5]
    start_x = 5.0
    n_steps = 20
    
    plt.figure(figsize=(15, 10))
    
    for i, lr in enumerate(learning_rates):
        plt.subplot(2, 3, i+1)
        
        x = start_x
        trajectory = [x]
        losses = [f(x)]
        
        for step in range(n_steps):
            grad = grad_f(x)
            x = x - lr * grad
            trajectory.append(x)
            losses.append(f(x))
        
        # 绘制收敛过程
        steps = range(len(losses))
        plt.plot(steps, losses, 'bo-', linewidth=2, markersize=4)
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title(f'Learning Rate = {lr}')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # 判断收敛状态
        final_loss = losses[-1]
        if final_loss < 1e-10:
            status = "收敛"
        elif final_loss > 100:
            status = "发散"
        else:
            status = "震荡"
        
        plt.text(0.05, 0.95, f'状态: {status}', 
                transform=plt.gca().transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 最后一个子图显示所有学习率的对比
    plt.subplot(2, 3, 6)
    for lr in learning_rates:
        x = start_x
        losses = [f(x)]
        
        for step in range(n_steps):
            grad = grad_f(x)
            x = x - lr * grad
            losses.append(f(x))
        
        plt.plot(range(len(losses)), losses, 'o-', 
                linewidth=2, label=f'lr={lr}')
    
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('学习率对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('learning_rate_effects.png', dpi=300, bbox_inches='tight')
    plt.show()

def demonstrate_comprehensive_optimizers():
    """综合优化器演示"""
    print("⚙️ 综合优化器演示")
    print("="*80)
    
    # 1. 数学原理演示
    trajectories = demonstrate_optimizer_mathematics()
    
    # 2. MNIST对比实验
    mnist_results = compare_optimizers_on_mnist()
    
    # 3. 可视化结果
    visualize_optimizer_comparison(mnist_results)
    
    # 4. 学习率影响
    demonstrate_learning_rate_effects()
    
    # 5. 结果总结
    print("\n" + "="*60)
    print("优化器性能总结")
    print("="*60)
    
    print("📊 MNIST数据集结果:")
    for name, data in mnist_results.items():
        print(f"{name:15s}: 最终损失={data['final_loss']:.6f}, "
              f"训练时间={data['convergence_time']:.2f}s")
    
    # 推荐建议
    print(f"\n💡 优化器选择建议:")
    print(f"• SGD: 简单稳定，适合凸优化问题")
    print(f"• SGD+Momentum: 在SGD基础上加速收敛，减少震荡")
    print(f"• Adam: 自适应学习率，大多数情况下表现良好")
    print(f"• RMSProp: 适合处理稀疏梯度，RNN训练常用")
    
    print(f"\n🎯 最佳性能: {min(mnist_results.items(), key=lambda x: x[1]['final_loss'])[0]}")
    print(f"🚀 最快收敛: {min(mnist_results.items(), key=lambda x: x[1]['convergence_time'])[0]}")
    
    return {
        'trajectories': trajectories,
        'mnist_results': mnist_results
    }

if __name__ == "__main__":
    results = demonstrate_comprehensive_optimizers()
    print("\n🎉 优化器对比演示完成!")
