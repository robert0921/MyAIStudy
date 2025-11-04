"""简单神经网络实现"""
import numpy as np
from typing import List, Tuple, Dict
from .layers import LinearLayer, ReLU, Softmax
from .optimizers import Optimizer

class SimpleNN:
    """简单神经网络实现"""
    def __init__(self, layer_dims: List[int]):
        """
        初始化网络
        Args:
            layer_dims: 每层维度的列表，如 [784, 128, 64, 10]
        """
        self.layers = []
        self.params = {}
        
        # 构建网络架构
        for i in range(len(layer_dims) - 1):
            self.layers.append(LinearLayer(layer_dims[i], layer_dims[i+1]))
            if i < len(layer_dims) - 2:  # 非输出层添加ReLU
                self.layers.append(ReLU())
        
        self.layers.append(Softmax())  # 输出层使用Softmax
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List]:
        """前向传播"""
        activations = []
        current = x
        
        for layer in self.layers:
            current = layer.forward(current)
            activations.append(current)
        
        return current, activations
    
    def backward(self, grad: np.ndarray, 
                activations: List) -> Dict[str, np.ndarray]:
        """反向传播"""
        layer_grads = {}
        current_grad = grad
        
        for i, layer in reversed(list(enumerate(self.layers))):
            if hasattr(layer, 'weight'):  # 是否为LinearLayer
                current_grad, grads = layer.backward(current_grad)
                layer_grads[f'layer_{i}'] = grads
            else:
                current_grad = layer.backward(current_grad)
        
        return layer_grads
    
    def train_step(self, x: np.ndarray, y: np.ndarray, 
                  optimizer: Optimizer) -> float:
        """执行一步训练"""
        # 前向传播
        pred, activations = self.forward(x)
        
        # 计算损失
        loss = -np.sum(y * np.log(pred + 1e-7)) / x.shape[1]
        
        # 计算初始梯度
        grad = pred - y
        
        # 反向传播
        grads = self.backward(grad, activations)
        
        # 更新参数
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weight'):
                optimizer.step(
                    {'weight': layer.weight, 'bias': layer.bias},
                    grads[f'layer_{i}']
                )
        
        return loss
