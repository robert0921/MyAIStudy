# 🌱 MyAIStudy 入门版 v4.0 - 12周 AI 基础实战

> 对应整合版第1-12周，从编程基础到AI实战，系统掌握Python、数据科学、机器学习和深度学习核心技能

## 📚 项目简介

本项目是**《AI学习12周实战计划表（入门版）》**的完整代码实现，专为有编程基础但不熟悉Python和AI的初学者设计。通过12周循序渐进的学习，您将：

- ✅ 掌握Python语法和面向对象编程
- ✅ 熟练使用NumPy、Pandas、Matplotlib等数据科学工具
- ✅ 理解机器学习基本原理并完成实战项目
- ✅ 学会使用PyTorch构建深度学习模型
- ✅ 完成图像分类、文本分类等实际应用

## 🗂️ 项目结构

```
beginner/
├── __init__.py                      # 模块初始化
├── week1_python_basics.py           # 第1周：Python基础与面向对象
├── week2_numpy_operations.py        # 第2周：NumPy数组操作与矩阵运算
├── week3_pandas_analysis.py         # 第3周：Pandas数据处理与分析
├── week4_visualization.py           # 第4周：Matplotlib/Seaborn可视化
├── week5_8_machine_learning.py      # 第5-8周：机器学习基础
└── week9_12_deep_learning.py        # 第9-12周：深度学习入门

run_beginner_examples.py             # 统一入口程序
README_beginner.md                   # 本文档
```

## 🎯 学习路线

### 阶段一：Python与数据科学基础（第1-4周）

| 周次 | 学习重点 | 核心技能 | 实践项目 |
|------|---------|---------|----------|
| 第1周 | Python基础语法、函数、类、模块 | 面向对象编程、装饰器、异常处理 | 学生成绩管理系统 |
| 第2周 | NumPy数组操作、广播机制、矩阵运算 | 数组操作、线性代数、特征值分解 | 图像灰度化、矩阵运算 |
| 第3周 | Pandas数据处理：读取、清洗、分组、合并 | 数据清洗、分组聚合、透视表 | 电商用户行为分析 |
| 第4周 | Matplotlib/Seaborn可视化 | 多种图表类型、热力图、子图布局 | 销售数据可视化 |

**输出成果：**
- 完整的学生成绩管理系统（OOP实现）
- 电商数据分析报告（含复购率、RFM分析）
- 销售趋势可视化仪表盘

---

### 阶段二：机器学习基础与Scikit-Learn（第5-8周）

| 周次 | 学习重点 | 核心技能 | 实践项目 |
|------|---------|---------|----------|
| 第5周 | 机器学习基础：监督/无监督学习、评估指标 | 分类模型、交叉验证、混淆矩阵 | 鸢尾花分类任务 |
| 第6周 | 线性回归、逻辑回归、决策树 | 回归分析、特征工程、模型对比 | 房价预测项目 |
| 第7周 | 聚类算法：K-Means、DBSCAN | 无监督学习、轮廓系数、肘部法则 | 用户分群分析 |
| 第8周 | 模型调优：交叉验证、网格搜索、特征工程 | 超参数优化、特征重要性、学习曲线 | Kaggle Titanic优化 |

**输出成果：**
- 鸢尾花分类器（准确率>95%）
- 房价预测模型（R²>0.85）
- 用户分群报告（K-Means + DBSCAN）
- 调优实验笔记

---

### 阶段三：深度学习入门与PyTorch（第9-12周）

| 周次 | 学习重点 | 核心技能 | 实践项目 |
|------|---------|---------|----------|
| 第9周 | PyTorch基础：Tensor、自动微分、Dataset | 张量操作、梯度计算、数据加载 | MNIST手写数字识别 |
| 第10周 | CNN原理与实现 | 卷积层、池化层、特征图可视化 | CIFAR-10图像分类 |
| 第11周 | RNN/LSTM基础 | 序列模型、词嵌入、LSTM单元 | 文本情感分类 |
| 第12周 | 综合项目：从数据加载到模型部署 | 完整Pipeline、模型保存、推理优化 | 端到端项目实战 |

**输出成果：**
- MNIST分类器（准确率>98%）
- CIFAR-10 CNN模型
- 文本情感分类器（LSTM）
- 完整的深度学习Pipeline

---

## 🚀 快速开始

### 1. 环境准备

**Python版本要求：** Python 3.8+

**安装依赖：**

```bash
# 基础依赖（阶段1-2）
pip install numpy pandas matplotlib seaborn scikit-learn

# 深度学习依赖（阶段3）
pip install torch torchvision

# 一键安装全部
pip install -r beginner_requirements.txt
```

### 2. 运行示例

**交互式菜单模式：**
```bash
python run_beginner_examples.py
```

**命令行模式：**
```bash
# 运行单周内容
python run_beginner_examples.py week1      # 第1周
python run_beginner_examples.py week2      # 第2周
python run_beginner_examples.py week5-8    # 第5-8周

# 运行整个阶段
python run_beginner_examples.py stage1     # 阶段1（第1-4周）
python run_beginner_examples.py stage2     # 阶段2（第5-8周）
python run_beginner_examples.py stage3     # 阶段3（第9-12周）

# 运行全部12周
python run_beginner_examples.py all

# 查看模块状态
python run_beginner_examples.py status

# 查看帮助
python run_beginner_examples.py help
```

### 3. 单独运行各周模块

```bash
# 第1周：Python基础
python -m beginner.week1_python_basics

# 第2周：NumPy操作
python -m beginner.week2_numpy_operations

# 第3周：Pandas分析
python -m beginner.week3_pandas_analysis

# 第4周：可视化
python -m beginner.week4_visualization

# 第5-8周：机器学习
python -m beginner.week5_8_machine_learning

# 第9-12周：深度学习
python -m beginner.week9_12_deep_learning
```

---

## 📖 详细内容说明

### 第1周：Python基础语法与面向对象

**学习内容：**
- Python基础语法（列表推导、字典推导、Lambda函数）
- 函数式编程（装饰器、闭包）
- 面向对象编程（类、继承、封装）
- 异常处理与文件操作

**实践项目：学生成绩管理系统**
```python
from beginner.week1_python_basics import demonstrate_python_basics

# 运行完整演示
manager = demonstrate_python_basics()

# 输出：
# - 学生信息管理
# - 成绩统计分析
# - 排名计算
```

**核心代码示例：**
```python
class Student:
    def __init__(self, student_id, name, grade):
        self.student_id = student_id
        self.name = name
        self.scores = {}
    
    def add_score(self, subject, score):
        self.scores[subject] = score
    
    def get_average(self):
        return sum(self.scores.values()) / len(self.scores)
```

---

### 第2周：NumPy数组操作与矩阵运算

**学习内容：**
- NumPy数组创建与索引
- 广播机制 (Broadcasting)
- 通用函数 (ufunc)
- 线性代数运算（矩阵乘法、特征值、SVD）

**实践项目：图像灰度化**
```python
from beginner.week2_numpy_operations import demonstrate_image_processing

# 演示灰度化算法
gray_image = demonstrate_image_processing()

# 三种方法对比：
# 1. 平均值法：Gray = (R + G + B) / 3
# 2. 加权法：Gray = 0.299*R + 0.587*G + 0.114*B
# 3. 最大值法：Gray = max(R, G, B)
```

**核心概念：**
- 广播机制让不同形状的数组自动对齐
- 矩阵运算是深度学习的数学基础
- SVD分解用于降维和推荐系统

---

### 第3周：Pandas数据处理与分析

**学习内容：**
- DataFrame创建与操作
- 数据清洗（缺失值、异常值、重复值）
- 分组聚合 (groupby)
- 数据合并与透视表

**实践项目：电商用户行为分析**
```python
from beginner.week3_pandas_analysis import analyze_ecommerce_data

# 运行完整分析
customers, orders, analysis = analyze_ecommerce_data()

# 输出指标：
# - 总订单量、完成率
# - 复购率分析
# - 城市消费排行
# - RFM客户价值分析
```

**分析亮点：**
- 复购率计算：识别高价值客户
- 时间序列分析：发现销售趋势
- 用户分群：年龄段、城市、消费行为

---

### 第4周：Matplotlib/Seaborn可视化

**学习内容：**
- 折线图、散点图、柱状图、直方图
- 子图布局与样式定制
- Seaborn统计图表（箱线图、小提琴图、热力图）
- 相关性分析可视化

**实践项目：销售数据可视化**
```python
from beginner.week4_visualization import demonstrate_sales_visualization

# 生成销售趋势图
df = demonstrate_sales_visualization()

# 图表类型：
# - 日销售额趋势 + 移动平均
# - 月度销售柱状图
# - 产品类别饼图
# - 地区-月度热力图
```

**可视化技巧：**
- 使用移动平均平滑趋势
- 热力图展示多维数据
- 颜色映射增强表达力

---

### 第5-8周：机器学习基础

**第5周：分类任务（鸢尾花）**
```python
from beginner.week5_8_machine_learning import week5_classification

model, X_test, y_test = week5_classification()

# 技术栈：
# - 逻辑回归 (Logistic Regression)
# - 数据标准化 (StandardScaler)
# - 交叉验证 (Cross-Validation)
# - 混淆矩阵 (Confusion Matrix)
```

**第6周：回归任务（房价预测）**
```python
from beginner.week5_8_machine_learning import week6_regression

lr_model, dt_model = week6_regression()

# 模型对比：
# - 线性回归：可解释性强
# - 决策树：处理非线性关系
# - 评估指标：MSE、R²分数
```

**第7周：聚类任务（用户分群）**
```python
from beginner.week5_8_machine_learning import week7_clustering

kmeans, dbscan, data = week7_clustering()

# 聚类算法：
# - K-Means：基于距离的聚类
# - DBSCAN：基于密度的聚类
# - 轮廓系数：评估聚类质量
# - 肘部法则：选择最优K值
```

**第8周：模型调优**
```python
from beginner.week5_8_machine_learning import week8_model_tuning

best_model = week8_model_tuning()

# 调优技术：
# - 网格搜索 (GridSearchCV)
# - 学习曲线分析
# - 特征重要性排序
# - 性能提升对比
```

---

### 第9-12周：深度学习入门

**第9周：MNIST手写数字识别**
```python
from beginner.week9_12_deep_learning import week9_mnist

model = week9_mnist()

# 技术要点：
# - PyTorch基础：Tensor、autograd
# - 多层感知机 (MLP)
# - 训练循环与验证
# - 模型参数统计
```

**第10周：CIFAR-10图像分类（CNN）**
```python
from beginner.week9_12_deep_learning import week10_cifar10

cnn_model = week10_cifar10()

# CNN组件：
# - 卷积层 (Conv2d)
# - 池化层 (MaxPool2d)
# - 全连接层 (Linear)
# - Dropout正则化
```

**第11周：文本情感分类（RNN/LSTM）**
```python
from beginner.week9_12_deep_learning import week11_text_classification

rnn_model = week11_text_classification()

# RNN组件：
# - 词嵌入 (Embedding)
# - LSTM层
# - 序列模型训练
# - 文本分类应用
```

**第12周：综合项目**
```python
from beginner.week9_12_deep_learning import week12_comprehensive_project

final_model = week12_comprehensive_project()

# 完整Pipeline：
# - 数据加载与预处理
# - 模型训练与验证
# - 模型保存与加载
# - 推理与部署建议
```

---

## 🛠️ 技术栈

### 数据科学工具
- **NumPy**: 数值计算与数组操作
- **Pandas**: 数据处理与分析
- **Matplotlib**: 基础可视化
- **Seaborn**: 统计图表可视化

### 机器学习框架
- **Scikit-Learn**: 经典机器学习算法
  - 分类：逻辑回归、决策树、随机森林
  - 回归：线性回归、决策树回归
  - 聚类：K-Means、DBSCAN
  - 工具：网格搜索、交叉验证、特征工程

### 深度学习框架
- **PyTorch**: 深度学习建模
  - 张量操作与自动微分
  - 神经网络模块 (nn.Module)
  - 数据加载器 (DataLoader)
  - 优化器 (Adam, SGD)

---

## 📊 学习成果展示

### 阶段1成果（第1-4周）
- ✅ 学生成绩管理系统（300+行代码）
- ✅ 图像灰度化算法实现（3种方法对比）
- ✅ 电商数据分析报告（10+项指标）
- ✅ 销售可视化仪表盘（15+种图表）

### 阶段2成果（第5-8周）
- ✅ 鸢尾花分类器（准确率>95%）
- ✅ 房价预测模型（R²=0.85+）
- ✅ 用户分群分析（K-Means + DBSCAN）
- ✅ 调优后的随机森林（准确率提升5%+）

### 阶段3成果（第9-12周）
- ✅ MNIST手写数字识别（MLP, 准确率>95%）
- ✅ CIFAR-10图像分类（CNN）
- ✅ 文本情感分类（LSTM）
- ✅ 完整的深度学习Pipeline

---

## 💡 学习建议

### 学习节奏
- 每周投入 **8-10小时**（约 1.5小时/天）
- 完成课程内容 + 动手实践 + 写学习笔记
- 每周末回顾总结，巩固知识点

### 实践建议
1. **动手优先**：每个知识点都要写代码验证
2. **记录笔记**：记录"学到的知识点 + 遇到的坑 + 解决方式"
3. **对比学习**：C++与Python语法对比、不同算法对比
4. **可视化进度**：用Excel或Notion记录学习进度

### 调试技巧
- 使用 `print()` 查看中间结果
- 使用 `shape` 和 `dtype` 检查张量维度
- 使用断点调试 (VSCode Debugger)
- 查看官方文档和Stack Overflow

### 进阶路线
完成12周后，可以继续学习：
- **Transformer架构**（BERT、GPT）
- **计算机视觉**（目标检测、图像分割）
- **自然语言处理**（命名实体识别、机器翻译）
- **强化学习**（Q-Learning、DQN）
- **Kaggle竞赛**（实战项目积累）

---

## 🔧 常见问题

### Q1: 我完全没有Python基础，能学吗？
**A:** 可以！第1周专门讲解Python基础，有C++基础的话学习曲线会很平缓。建议：
- 先快速过一遍Python官方教程
- 重点关注Python与C++的差异（动态类型、GIL、装饰器等）
- 多写代码，熟能生巧

### Q2: 没有GPU，能学深度学习吗？
**A:** 可以！本项目的示例都在CPU上可运行。实际应用中：
- 小模型和demo可以用CPU
- Google Colab 提供免费GPU
- Kaggle Notebooks 提供免费GPU
- 真实项目建议使用GPU加速

### Q3: 每周的时间不够，怎么办？
**A:** 灵活调整节奏：
- 可以延长到16周或24周
- 重点掌握核心概念，跳过部分细节
- 先快速过一遍，再回头深入学习

### Q4: 如何验证学习效果？
**A:** 建议：
- 每周完成配套项目
- 在GitHub创建个人仓库，提交代码
- 写技术博客总结知识点
- 参加Kaggle入门竞赛

### Q5: 学完之后能做什么？
**A:** 可以胜任：
- 数据分析师（初级）
- 机器学习工程师（实习/初级）
- AI算法工程师（实习）
- 继续深造（研究生/在线课程）

---

## 📚 推荐资源

### 在线教程
- [Kaggle Learn](https://www.kaggle.com/learn): 免费的数据科学课程
- [李宏毅机器学习](https://www.bilibili.com/video/BV1Wv411h7kN): 经典ML/DL课程
- [PyTorch官方教程](https://pytorch.org/tutorials/): 官方文档和示例
- [Scikit-Learn文档](https://scikit-learn.org/): 机器学习算法库

### 实战平台
- [Kaggle](https://www.kaggle.com/): 数据竞赛平台
- [天池](https://tianchi.aliyun.com/): 阿里云竞赛平台
- [和鲸社区](https://www.heywhale.com/): 国内数据科学社区

### 书籍推荐
- 《Python编程：从入门到实践》: Python基础
- 《利用Python进行数据分析》: Pandas实战
- 《机器学习实战》: 算法原理与代码
- 《深度学习入门》: PyTorch/TensorFlow

---

## 🤝 贡献与反馈

### 如何贡献
欢迎提交Issue和Pull Request！

- **Bug修复**: 发现代码错误请提Issue
- **功能增强**: 添加新的示例或优化现有代码
- **文档改进**: 完善README或添加注释

### 反馈渠道
- GitHub Issues: 技术问题和bug报告
- Discussions: 学习交流和经验分享

---

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 🎉 致谢

感谢所有开源项目的贡献者：
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-Learn
- PyTorch
- Jupyter/VSCode

---

## 📞 联系方式

- **项目地址**: [GitHub仓库链接]
- **作者**: AI Learning Team
- **版本**: v4.0
- **更新日期**: 2026-05-25

---

**🚀 开始您的AI学习之旅吧！**

```bash
python run_beginner_examples.py
```

---

*"学习AI最好的时间是现在。让我们一起从零开始，系统掌握AI核心技能！"*
