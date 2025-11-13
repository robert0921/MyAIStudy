# 📖 MyAIStudy 使用指南

**版本：** v2.3  
**更新日期：** 2025年11月10日

---

## 🚀 快速开始

### 方式一：统一入口（推荐新手）

```bash
# 交互式选择入门版或进阶版
python run_example.py

# 直接启动入门版
python run_example.py beginner

# 直接启动进阶版
python run_example.py intermediate
```

### 方式二：直接使用专用入口

```bash
# 入门版（12周基础实战）
python run_beginner_examples.py

# 进阶版（12周深度提升）
python run_intermediate_examples.py
```

---

## 📚 入门版使用指南

### 适合人群
- 有Python编程基础
- 想系统学习数据科学和机器学习
- 希望循序渐进掌握深度学习

### 学习路线（12周）

**第1-4周：Python与数据科学基础**
```bash
python run_beginner_examples.py week1  # Python基础
python run_beginner_examples.py week2  # NumPy数组操作
python run_beginner_examples.py week3  # Pandas数据分析
python run_beginner_examples.py week4  # Matplotlib可视化
```

**第5-8周：机器学习基础**
```bash
python run_beginner_examples.py week5  # Scikit-Learn、分类、回归、聚类
```

**第9-12周：深度学习入门**
```bash
python run_beginner_examples.py week9  # PyTorch、CNN、RNN/LSTM
```

### 学习建议
1. 按周次顺序学习，不要跳过基础内容
2. 每周完成后尝试修改代码参数，观察结果变化
3. 遇到问题查看代码注释和错误提示
4. 完成后可以尝试类似的数据集

---

## 🎓 进阶版使用指南

### 适合人群
- 掌握深度学习基础知识
- 想深入理解算法原理
- 需要工程实践经验
- 对LLM微调和优化感兴趣

### 功能模块（11个）

#### 1. 基础知识演示
```bash
# 在菜单中选择 1
```
内容：线性代数、反向传播、优化器对比、CNN/Transformer

#### 2. LLM架构
```bash
# 在菜单中选择 2
```
内容：注意力机制、RoPE位置编码、RMSNorm、完整LLaMA实现

#### 3. 性能测试
```bash
# 在菜单中选择 3
```
内容：NumPy vs Numba性能对比、矩阵乘法优化

#### 4. 数据仪表盘
```bash
# 在菜单中选择 4
```
内容：交互式数据可视化（会打开浏览器）

#### 5. 深度学习训练
```bash
# 在菜单中选择 5
```
内容：CIFAR-10训练、分布式训练、混合精度、模型量化

#### 6. Prompt Engineering
```bash
# 在菜单中选择 6
```
内容：Few-shot学习、Prompt调试、自动化优化

#### 7. 模型剪枝
```bash
# 在菜单中选择 7
```
内容：幅度/结构化/全局/迭代剪枝、稀疏度分析

#### 8. LLM微调 ⭐
```bash
# 在菜单中选择 8
```
内容：LoRA低秩适配、QLoRA量化微调、PEFT方法对比

#### 9. 推理优化 ⭐
```bash
# 在菜单中选择 9
```
内容：KV Cache加速、批量推理、性能基准测试

#### 10. 快速演示
```bash
# 在菜单中选择 10
```
内容：核心功能精简版（2-5分钟）

#### 11. 完整演示
```bash
# 在菜单中选择 11
```
内容：所有功能完整版（约30分钟）

---

## 🛠️ 环境配置

### Python环境要求
- Python 3.8+
- 建议使用Anaconda或虚拟环境

### 安装依赖

```bash
# 基础依赖（必需）
pip install numpy pandas matplotlib scikit-learn

# PyTorch（根据系统选择）
# CPU版本
pip install torch torchvision

# GPU版本（CUDA 11.8）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 可选依赖（进阶功能）
pip install seaborn plotly dash numba lmdb albumentations
```

### GPU支持
- 深度学习训练建议使用GPU
- 检查GPU是否可用：
```python
import torch
print(torch.cuda.is_available())  # 应该返回True
```

---

## 🔧 常见问题

### Q1: ModuleNotFoundError
**问题：** `ModuleNotFoundError: No module named 'xxx'`  
**解决：** 安装缺失的包 `pip install xxx`

### Q2: CUDA out of memory
**问题：** 显存不足  
**解决：** 
- 减小batch_size
- 使用混合精度训练
- 减少模型层数

### Q3: 数据仪表盘无法打开
**问题：** 浏览器没有自动打开  
**解决：** 手动访问 `http://127.0.0.1:8050`

### Q4: 训练速度慢
**问题：** 训练太慢  
**解决：**
- 确认使用GPU训练
- 启用混合精度（FP16）
- 使用数据缓存（LMDB）

### Q5: 中文显示乱码
**问题：** 图表中文乱码  
**解决：** 
```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
plt.rcParams['axes.unicode_minus'] = False
```

---

## 📊 学习建议

### 入门版学习路径
1. **第1个月**：Week 1-4（Python和数据科学）
2. **第2个月**：Week 5-8（机器学习）
3. **第3个月**：Week 9-12（深度学习）

### 进阶版学习路径
1. **第1个月**：模块1-4（理论基础）
2. **第2个月**：模块5-7（工程实践）
3. **第3个月**：模块8-9（LLM专项）

### 实战项目建议
- 入门版：修改数据集，尝试不同算法
- 进阶版：调整超参数，实现新功能
- 综合项目：结合入门版和进阶版知识

---

## 📚 相关文档

- [README.md](../README.md) - 项目总览
- [README_BEGINNER.md](../README_BEGINNER.md) - 入门版详细文档
- [README_Intermediate.md](../README_Intermediate.md) - 进阶版详细文档
- [OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md) - 优化技术总结
- [COMPLETION_REPORT.md](COMPLETION_REPORT.md) - 项目完成报告

---

## 🆘 获取帮助

- 📖 查看代码注释
- 🔍 搜索错误信息
- 💬 提交Issue到GitHub
- 📧 联系项目维护者

---

**最后更新：** 2025年11月10日  
**文档版本：** v2.3
