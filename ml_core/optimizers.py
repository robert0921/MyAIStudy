"""优化器实现模块"""
from typing import List, Dict
import numpy as np

class Optimizer:
    """优化器基类"""
    def __init__(self, lr: float = 0.001):
        self.lr = lr
    
    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        """执行一步优化"""
        raise NotImplementedError

class SGD(Optimizer):
    """随机梯度下降优化器"""
    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        for name in params:
            params[name] -= self.lr * grads[name]

class Adam(Optimizer):
    """Adam优化器"""
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}  # 一阶矩估计
        self.v = {}  # 二阶矩估计
        self.t = 0   # 时间步
        
    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        if not self.m:  # 首次运行时初始化动量
            self.m = {name: np.zeros_like(param) 
                     for name, param in params.items()}
            self.v = {name: np.zeros_like(param) 
                     for name, param in params.items()}
        
        self.t += 1
        for name in params:
            # 更新动量
            self.m[name] = self.beta1 * self.m[name] + \
                          (1 - self.beta1) * grads[name]
            self.v[name] = self.beta2 * self.v[name] + \
                          (1 - self.beta2) * (grads[name] ** 2)
            
            # 偏差修正
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            
            # 参数更新
            params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
