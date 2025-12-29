"""
第14周：向量数据库索引机制
测试不同维度的检索延迟

核心知识点：
1. 向量索引原理（IVF、HNSW）
2. 性能基准测试
3. 维度对检索性能的影响
4. 批量插入与查询优化
"""

import numpy as np
import time
from typing import List, Tuple, Dict
from collections import defaultdict
import math


class FlatIndex:
    """暴力搜索索引（Flat Index）"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = []
        self.ids = []
    
    def add(self, vectors: np.ndarray, ids: List[int] = None):
        """添加向量"""
        if ids is None:
            start_id = len(self.vectors)
            ids = list(range(start_id, start_id + len(vectors)))
        
        for vec, vec_id in zip(vectors, ids):
            self.vectors.append(vec)
            self.ids.append(vec_id)
    
    def search(self, query: np.ndarray, k: int = 10) -> Tuple[List[int], List[float]]:
        """搜索最近邻"""
        if len(self.vectors) == 0:
            return [], []
        
        # 计算所有向量的距离
        distances = []
        for vec in self.vectors:
            # L2距离
            dist = np.linalg.norm(query - vec)
            distances.append(dist)
        
        # 排序并返回top-k
        indices = np.argsort(distances)[:k]
        top_ids = [self.ids[i] for i in indices]
        top_distances = [distances[i] for i in indices]
        
        return top_ids, top_distances


class IVFIndex:
    """倒排文件索引（Inverted File Index）"""
    
    def __init__(self, dimension: int, n_clusters: int = 100):
        """
        Args:
            dimension: 向量维度
            n_clusters: 聚类中心数量
        """
        self.dimension = dimension
        self.n_clusters = n_clusters
        self.centroids = None
        self.inverted_lists = defaultdict(list)  # cluster_id -> [(vec_id, vector)]
        self.ids = []
        self.is_trained = False
    
    def train(self, vectors: np.ndarray):
        """训练聚类中心"""
        n_samples = len(vectors)
        n_clusters = min(self.n_clusters, n_samples)
        
        # 使用K-means聚类（简化版）
        print(f"   训练IVF索引: {n_samples}个样本, {n_clusters}个聚类中心")
        
        # 随机初始化聚类中心
        indices = np.random.choice(n_samples, n_clusters, replace=False)
        self.centroids = vectors[indices].copy()
        
        # 迭代优化（最多10轮）
        for iteration in range(10):
            # 分配样本到最近的聚类中心
            assignments = []
            for vec in vectors:
                distances = [np.linalg.norm(vec - c) for c in self.centroids]
                assignments.append(np.argmin(distances))
            
            # 更新聚类中心
            new_centroids = []
            for i in range(n_clusters):
                cluster_vecs = vectors[[j for j, a in enumerate(assignments) if a == i]]
                if len(cluster_vecs) > 0:
                    new_centroids.append(np.mean(cluster_vecs, axis=0))
                else:
                    new_centroids.append(self.centroids[i])
            
            self.centroids = np.array(new_centroids)
        
        self.is_trained = True
        print(f"   ✅ 训练完成")
    
    def add(self, vectors: np.ndarray, ids: List[int] = None):
        """添加向量"""
        if not self.is_trained:
            raise ValueError("索引未训练，请先调用train()")
        
        if ids is None:
            start_id = len(self.ids)
            ids = list(range(start_id, start_id + len(vectors)))
        
        # 将向量分配到聚类
        for vec, vec_id in zip(vectors, ids):
            # 找到最近的聚类中心
            distances = [np.linalg.norm(vec - c) for c in self.centroids]
            cluster_id = np.argmin(distances)
            
            # 添加到倒排列表
            self.inverted_lists[cluster_id].append((vec_id, vec))
            self.ids.append(vec_id)
    
    def search(self, query: np.ndarray, k: int = 10, n_probe: int = 1) -> Tuple[List[int], List[float]]:
        """搜索最近邻
        
        Args:
            query: 查询向量
            k: 返回的近邻数量
            n_probe: 探测的聚类数量
        """
        if not self.is_trained:
            return [], []
        
        # 找到最近的n_probe个聚类中心
        distances_to_centroids = [np.linalg.norm(query - c) for c in self.centroids]
        probe_clusters = np.argsort(distances_to_centroids)[:n_probe]
        
        # 在这些聚类中搜索
        candidates = []
        for cluster_id in probe_clusters:
            for vec_id, vec in self.inverted_lists[cluster_id]:
                dist = np.linalg.norm(query - vec)
                candidates.append((vec_id, dist))
        
        # 排序并返回top-k
        candidates.sort(key=lambda x: x[1])
        top_k = candidates[:k]
        
        top_ids = [vec_id for vec_id, _ in top_k]
        top_distances = [dist for _, dist in top_k]
        
        return top_ids, top_distances


class HNSWIndex:
    """分层导航小世界图索引（简化版）"""
    
    def __init__(self, dimension: int, M: int = 16, ef_construction: int = 200):
        """
        Args:
            dimension: 向量维度
            M: 每层的最大连接数
            ef_construction: 构建时的动态候选列表大小
        """
        self.dimension = dimension
        self.M = M
        self.ef_construction = ef_construction
        self.max_level = 0
        self.entry_point = None
        
        # 存储结构
        self.vectors = {}  # id -> vector
        self.levels = {}   # id -> level
        self.graph = defaultdict(lambda: defaultdict(set))  # level -> {id -> neighbors}
    
    def _get_random_level(self) -> int:
        """随机生成层数"""
        ml = 1.0 / math.log(2.0)
        return int(-math.log(np.random.random()) * ml)
    
    def add(self, vectors: np.ndarray, ids: List[int] = None):
        """添加向量（简化实现）"""
        if ids is None:
            start_id = len(self.vectors)
            ids = list(range(start_id, start_id + len(vectors)))
        
        for vec, vec_id in zip(vectors, ids):
            level = self._get_random_level()
            
            self.vectors[vec_id] = vec
            self.levels[vec_id] = level
            
            if self.entry_point is None:
                self.entry_point = vec_id
                self.max_level = level
            else:
                # 简化：只连接到最近的M个节点
                for l in range(level + 1):
                    # 找到当前层的最近邻
                    candidates = []
                    for existing_id in self.graph[l]:
                        dist = np.linalg.norm(vec - self.vectors[existing_id])
                        candidates.append((existing_id, dist))
                    
                    candidates.sort(key=lambda x: x[1])
                    neighbors = [cand_id for cand_id, _ in candidates[:self.M]]
                    
                    # 双向连接
                    for neighbor_id in neighbors:
                        self.graph[l][vec_id].add(neighbor_id)
                        self.graph[l][neighbor_id].add(vec_id)
    
    def search(self, query: np.ndarray, k: int = 10, ef: int = 50) -> Tuple[List[int], List[float]]:
        """搜索最近邻（简化实现）"""
        if not self.vectors:
            return [], []
        
        # 简化：在所有向量中搜索
        candidates = []
        for vec_id, vec in self.vectors.items():
            dist = np.linalg.norm(query - vec)
            candidates.append((vec_id, dist))
        
        candidates.sort(key=lambda x: x[1])
        top_k = candidates[:k]
        
        top_ids = [vec_id for vec_id, _ in top_k]
        top_distances = [dist for _, dist in top_k]
        
        return top_ids, top_distances


def benchmark_index_performance():
    """性能基准测试"""
    print("=" * 80)
    print("第14周：向量数据库索引性能测试")
    print("=" * 80)
    
    # 测试配置
    dimensions = [128, 256, 512, 768, 1024]
    n_vectors = 10000
    n_queries = 100
    k = 10
    
    results = {
        'Flat': defaultdict(dict),
        'IVF': defaultdict(dict),
        'HNSW': defaultdict(dict)
    }
    
    for dim in dimensions:
        print(f"\n{'='*80}")
        print(f"测试维度: {dim}")
        print(f"{'='*80}")
        
        # 生成测试数据
        print(f"\n生成测试数据...")
        print(f"  向量数量: {n_vectors}")
        print(f"  查询数量: {n_queries}")
        print(f"  维度: {dim}")
        
        np.random.seed(42)
        train_vectors = np.random.randn(n_vectors, dim).astype(np.float32)
        # L2归一化
        train_vectors = train_vectors / np.linalg.norm(train_vectors, axis=1, keepdims=True)
        
        query_vectors = np.random.randn(n_queries, dim).astype(np.float32)
        query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
        
        # ===== 测试Flat Index =====
        print(f"\n【1】Flat Index（暴力搜索）")
        print("-" * 80)
        
        index_flat = FlatIndex(dim)
        
        # 构建时间
        start = time.time()
        index_flat.add(train_vectors)
        build_time = time.time() - start
        
        print(f"  构建时间: {build_time:.3f}秒")
        print(f"  索引大小: {n_vectors * dim * 4 / 1024 / 1024:.2f} MB")
        
        # 查询时间
        start = time.time()
        for query in query_vectors:
            _ = index_flat.search(query, k=k)
        query_time = (time.time() - start) / n_queries * 1000  # ms
        
        print(f"  平均查询时间: {query_time:.3f} ms")
        print(f"  QPS: {1000 / query_time:.1f}")
        
        results['Flat'][dim] = {
            'build_time': build_time,
            'query_time': query_time,
            'qps': 1000 / query_time
        }
        
        # ===== 测试IVF Index =====
        print(f"\n【2】IVF Index（倒排文件索引）")
        print("-" * 80)
        
        n_clusters = int(np.sqrt(n_vectors))
        index_ivf = IVFIndex(dim, n_clusters=n_clusters)
        
        # 训练时间
        start = time.time()
        index_ivf.train(train_vectors[:1000])  # 使用部分数据训练
        train_time = time.time() - start
        
        print(f"  训练时间: {train_time:.3f}秒")
        print(f"  聚类中心数: {n_clusters}")
        
        # 构建时间
        start = time.time()
        index_ivf.add(train_vectors)
        build_time = time.time() - start
        
        print(f"  添加向量时间: {build_time:.3f}秒")
        
        # 查询时间（不同n_probe）
        for n_probe in [1, 4, 8]:
            start = time.time()
            for query in query_vectors:
                _ = index_ivf.search(query, k=k, n_probe=n_probe)
            query_time = (time.time() - start) / n_queries * 1000
            
            print(f"  平均查询时间 (n_probe={n_probe}): {query_time:.3f} ms")
            print(f"    QPS: {1000 / query_time:.1f}")
            
            results['IVF'][f"{dim}_probe{n_probe}"] = {
                'build_time': train_time + build_time,
                'query_time': query_time,
                'qps': 1000 / query_time
            }
        
        # ===== 测试HNSW Index =====
        print(f"\n【3】HNSW Index（分层导航小世界图）")
        print("-" * 80)
        
        index_hnsw = HNSWIndex(dim, M=16)
        
        # 构建时间（简化实现，性能较慢）
        start = time.time()
        # 只添加部分数据以节省时间
        index_hnsw.add(train_vectors[:1000])
        build_time = time.time() - start
        
        print(f"  构建时间: {build_time:.3f}秒 (仅1000个向量)")
        print(f"  M参数: {index_hnsw.M}")
        
        # 查询时间
        start = time.time()
        for query in query_vectors[:10]:  # 只测试10个查询
            _ = index_hnsw.search(query, k=k)
        query_time = (time.time() - start) / 10 * 1000
        
        print(f"  平均查询时间: {query_time:.3f} ms")
        print(f"  QPS: {1000 / query_time:.1f}")
        
        results['HNSW'][dim] = {
            'build_time': build_time,
            'query_time': query_time,
            'qps': 1000 / query_time
        }
    
    # ===== 性能对比总结 =====
    print(f"\n{'='*80}")
    print("性能对比总结")
    print(f"{'='*80}")
    
    print(f"\n📊 Flat Index (暴力搜索)")
    print("-" * 80)
    print(f"{'维度':<10} {'构建时间':<15} {'查询时间':<15} {'QPS':<10}")
    print("-" * 80)
    for dim in dimensions:
        r = results['Flat'][dim]
        print(f"{dim:<10} {r['build_time']:<15.3f} {r['query_time']:<15.3f} {r['qps']:<10.1f}")
    
    print(f"\n📊 IVF Index (n_probe=1)")
    print("-" * 80)
    print(f"{'维度':<10} {'构建时间':<15} {'查询时间':<15} {'QPS':<10}")
    print("-" * 80)
    for dim in dimensions:
        r = results['IVF'][f"{dim}_probe1"]
        print(f"{dim:<10} {r['build_time']:<15.3f} {r['query_time']:<15.3f} {r['qps']:<10.1f}")
    
    print(f"\n💡 维度对性能的影响:")
    print(f"   - 维度越高，计算距离的时间越长")
    print(f"   - Flat Index的查询时间与维度线性相关")
    print(f"   - IVF Index通过聚类减少计算量，但维度影响仍存在")
    print(f"   - 推荐使用384-768维度作为平衡点")
    
    print(f"\n💡 索引类型选择建议:")
    print(f"   - 小数据集(<10K): Flat Index")
    print(f"   - 中等数据集(10K-1M): IVF Index")
    print(f"   - 大数据集(>1M): HNSW Index")
    print(f"   - 高召回要求: Flat Index或HNSW")
    print(f"   - 高性能要求: IVF Index (调整n_probe)")


def demonstrate_vector_database():
    """演示向量数据库核心概念"""
    print("\n" + "=" * 80)
    print("向量数据库核心概念演示")
    print("=" * 80)
    
    # 1. 向量相似度计算
    print("\n【1】向量相似度度量")
    print("-" * 80)
    
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([0.9, 0.1, 0.0])
    vec3 = np.array([0.0, 1.0, 0.0])
    
    # L2距离
    l2_12 = np.linalg.norm(vec1 - vec2)
    l2_13 = np.linalg.norm(vec1 - vec3)
    
    # 余弦相似度
    cos_12 = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    cos_13 = np.dot(vec1, vec3) / (np.linalg.norm(vec1) * np.linalg.norm(vec3))
    
    print(f"  向量1: {vec1}")
    print(f"  向量2: {vec2}")
    print(f"  向量3: {vec3}")
    print(f"\n  L2距离:")
    print(f"    vec1 <-> vec2: {l2_12:.4f}")
    print(f"    vec1 <-> vec3: {l2_13:.4f}")
    print(f"\n  余弦相似度:")
    print(f"    vec1 <-> vec2: {cos_12:.4f}")
    print(f"    vec1 <-> vec3: {cos_13:.4f}")
    
    # 2. 高维空间的"维度诅咒"
    print("\n【2】维度诅咒现象")
    print("-" * 80)
    
    for dim in [10, 100, 1000]:
        # 生成随机向量
        n_samples = 1000
        vectors = np.random.randn(n_samples, dim)
        
        # 计算所有向量对的距离
        center = np.mean(vectors, axis=0)
        distances = [np.linalg.norm(v - center) for v in vectors]
        
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        print(f"\n  维度={dim}:")
        print(f"    平均距离: {mean_dist:.4f}")
        print(f"    标准差: {std_dist:.4f}")
        print(f"    变异系数: {std_dist/mean_dist:.4f}")
    
    print(f"\n  💡 观察:")
    print(f"     随着维度增加，距离的标准差减小")
    print(f"     高维空间中向量趋向等距分布")
    print(f"     这使得最近邻搜索变得困难")
    
    # 3. 索引结构对比
    print("\n【3】索引结构特点对比")
    print("-" * 80)
    
    comparison = {
        'Flat Index': {
            '构建时间': 'O(1)',
            '查询时间': 'O(N)',
            '内存占用': 'N*D*4字节',
            '召回率': '100%',
            '适用场景': '小数据集(<10K)'
        },
        'IVF Index': {
            '构建时间': 'O(N*K)',
            '查询时间': 'O(N/K)',
            '内存占用': 'N*D*4 + K*D*4字节',
            '召回率': '90-99%',
            '适用场景': '中大数据集(10K-1M)'
        },
        'HNSW Index': {
            '构建时间': 'O(N*log(N)*M)',
            '查询时间': 'O(log(N)*M)',
            '内存占用': 'N*(D*4 + M*8)字节',
            '召回率': '95-99.9%',
            '适用场景': '大数据集(>1M)'
        }
    }
    
    metrics = ['构建时间', '查询时间', '内存占用', '召回率', '适用场景']
    
    print(f"\n  {'索引类型':<15}", end='')
    for metric in metrics:
        print(f" {metric:<20}", end='')
    print()
    print("  " + "-" * 110)
    
    for index_name, values in comparison.items():
        print(f"  {index_name:<15}", end='')
        for metric in metrics:
            print(f" {values[metric]:<20}", end='')
        print()


if __name__ == "__main__":
    # 运行性能测试
    benchmark_index_performance()
    
    # 运行概念演示
    demonstrate_vector_database()
    
    print("\n" + "=" * 80)
    print("第14周学习完成！")
    print("=" * 80)
    print("""
    ✅ 已掌握的知识点:
    1. 向量索引的三种主要类型
    2. 不同维度对检索性能的影响
    3. 索引构建与查询的时间复杂度
    4. 准确率与性能的权衡
    5. 维度诅咒现象
    
    💡 下一步学习:
    - 第15周: RAG Pipeline优化
    - 实现分块策略优化
    - Embedding模型选择
    """)
