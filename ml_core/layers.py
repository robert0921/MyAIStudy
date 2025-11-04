"""神经网络层实现模块"""
import numpy as np
from typing import Tuple, Dict

class LinearLayer:
    """线性层实现"""
    def __init__(self, in_features: int, out_features: int):
        # He初始化
        self.weight = np.random.randn(out_features, in_features) * np.sqrt(2.0/in_features)
        self.bias = np.zeros((out_features, 1))
        self.cache = {}
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        self.cache['x'] = x
        return np.dot(self.weight, x) + self.bias
    
    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """反向传播"""
        x = self.cache['x']
        dw = np.dot(grad, x.T)
        db = np.sum(grad, axis=1, keepdims=True)
        dx = np.dot(self.weight.T, grad)
        return dx, {'weight': dw, 'bias': db}

class ReLU:
    """ReLU激活函数"""
    def __init__(self):
        self.cache = {}
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        self.cache['mask'] = (x > 0)
        return x * self.cache['mask']
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """反向传播"""
        return grad * self.cache['mask']

class Softmax:
    """Softmax激活函数"""
    def __init__(self):
        self.cache = {}
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        exp = np.exp(x - np.max(x, axis=0, keepdims=True))
        out = exp / np.sum(exp, axis=0, keepdims=True)
        self.cache['out'] = out
        return out
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """反向传播"""
        out = self.cache['out']
        return out * (grad - np.sum(grad * out, axis=0, keepdims=True))
