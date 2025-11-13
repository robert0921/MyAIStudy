"""
第2周：NumPy数组与矩阵操作（重命名自 beginner_ai）
"""
import numpy as np
from typing import Tuple


def demonstrate_numpy_basics():
    """演示NumPy基础操作"""
    print("\n" + "="*60)
    print("🔬 第2周：NumPy数组操作与矩阵运算")
    print("="*60)

    arr1 = np.array([1, 2, 3, 4, 5])
    arr2 = np.arange(0, 10, 2)
    arr3 = np.linspace(0, 1, 5)
    arr4 = np.zeros((3, 3))
    arr5 = np.ones((2, 4))
    arr6 = np.eye(3)

    print(f"  一维数组: {arr1}")
    print(f"  等差数列: {arr2}")
    print(f"  线性空间: {arr3}")

    matrix = np.arange(12).reshape(3, 4)
    print(f"  原始矩阵:\n{matrix}")
    print(f"  第二行: {matrix[1]}")
    print(f"  第三列: {matrix[:, 2]}")
    print(f"  子矩阵:\n{matrix[1:3, 1:3]}")

    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([10, 20, 30])
    result = a + b
    print(f"  广播相加:\n{result}")

    x = np.array([0, np.pi/2, np.pi])
    print(f"  sin(x) = {np.sin(x)}")
    print(f"  exp(x) = {np.exp(x)}")

    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    print(f"  A @ B =\n{A @ B}")
    print(f"  det(A) = {np.linalg.det(A):.2f}")

    try:
        inv_A = np.linalg.inv(A)
        print(f"  inv(A):\n{inv_A}")
    except np.linalg.LinAlgError:
        print("  矩阵不可逆")

    data = np.random.randn(100)
    print(f"  mean: {np.mean(data):.4f}")
    print(f"  std: {np.std(data):.4f}")

    return data


def demonstrate_image_processing():
    """图像处理示例：灰度化"""
    print("\n" + "="*60)
    print("🖼️ 图像处理示例：灰度化算法")
    print("="*60)

    np.random.seed(42)
    image_rgb = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    gray_avg = np.mean(image_rgb, axis=2).astype(np.uint8)
    weights = np.array([0.299, 0.587, 0.114])
    gray_weighted = np.dot(image_rgb, weights).astype(np.uint8)
    diff = np.abs(gray_avg.astype(float) - gray_weighted.astype(float))
    print(f"  avg vs weighted mean diff: {np.mean(diff):.2f}")
    return gray_weighted


def demonstrate_matrix_operations():
    print("\n" + "="*60)
    print("🧮 高级矩阵运算示例")
    print("="*60)

    A = np.array([[4, 2], [1, 3]])
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(f"  eigenvalues: {eigenvalues}")

    M = np.array([[1, 2, 3], [4, 5, 6]])
    U, S, Vt = np.linalg.svd(M)
    Sigma = np.zeros((M.shape[0], M.shape[1]))
    Sigma[:len(S), :len(S)] = np.diag(S)
    M_reconstructed = U @ Sigma @ Vt
    print(f"  reconstruction error: {np.linalg.norm(M - M_reconstructed):.10f}")

    return U


if __name__ == "__main__":
    demonstrate_numpy_basics()
    demonstrate_image_processing()
    demonstrate_matrix_operations()
