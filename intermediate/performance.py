"""性能优化模块"""
import numpy as np
import importlib
from typing import Tuple

# 尝试导入numba
try:
    numba = importlib.import_module('numba')
    jit = getattr(numba, 'jit')
    prange = getattr(numba, 'prange')
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        return lambda x: x
    def prange(n):
        return range(n)

@jit(nopython=True, parallel=True)
def fast_matrix_mul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Numba加速的矩阵乘法"""
    M, N = A.shape
    N, P = B.shape
    C = np.zeros((M, P))
    
    for i in prange(M):
        for j in range(P):
            for k in range(N):
                C[i, j] += A[i, k] * B[k, j]
    return C

def benchmark_matmul(size: int = 1000, 
                    methods: Tuple[str] = ('numpy', 'numba')) -> dict:
    """矩阵乘法性能基准测试"""
    import time
    results = {}
    
    # 准备测试数据
    A = np.random.randn(size, size)
    B = np.random.randn(size, size)
    
    if 'numpy' in methods:
        start = time.perf_counter()
        np.dot(A, B)
        results['numpy'] = time.perf_counter() - start
    
    if 'numba' in methods and NUMBA_AVAILABLE:
        # 预热JIT
        fast_matrix_mul(A[:2, :2], B[:2, :2])
        
        start = time.perf_counter()
        fast_matrix_mul(A, B)
        results['numba'] = time.perf_counter() - start
    
    return results
