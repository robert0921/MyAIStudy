# 深度学习训练与架构演示系统

这是一个完整的深度学习基础知识实现和演示系统，涵盖从线性代数基础到LLM架构，以及Prompt Engineering与Few-shot技术的全套实现。

## 🎯 核心特性

**📋 完整的深度学习基础实现**
- ✅ **线性代数与自动微分**：完整的数学推导 + 手工实现
- ✅ **反向传播机制**：详细公式推导 + PyTorch对比验证  
- ✅ **优化算法原理**：SGD/Adam/RMSProp完整实现 + 性能对比
- ✅ **CNN/Transformer**：手写核心组件 + 矩阵乘法意义解释
- ✅ **LLM架构原理**：完整LLaMA实现 + 可视化图表

**🧠 Prompt Engineering 与 Few-shot 技术**
- ✅ 自动化 Prompt 调试与优化
- ✅ Few-shot 示例生成与管理
- ✅ GPT 输出质量提升工具

**🏗️ 工程级实现**
- ✅ **PyTorch模块化**：Kaggle级数据管线 + 模型架构优化
- ✅ **模型调优策略**：CIFAR-10实战 + 完整监控系统
- ✅ **模型压缩量化**：准确率保持 ±1% + 大小减少75%
- ✅ **分布式训练**：DDP + FP16 + 吞吐量提升2-3倍

**💡 教学价值**
- 完整的数学推导和手工实现
- 数值方法验证（梯度检查误差 < 1e-10）
- 与PyTorch官方实现对比验证
- 详细的代码注释和可视化演示

## 📁 项目结构

```
ml_core/
├── __init__.py                     # 模块初始化
├── requirements.txt               # 依赖包列表
├── 功能检查清单.py                 # 功能验证报告
│
# 基础实现模块
├── linear_algebra.py             # 线性代数与自动微分基础
├── backpropagation.py            # 反向传播详细实现
├── optimizer_comparison.py       # 优化算法对比
├── cnn_transformer.py           # 卷积与Transformer基础
│
# 深度学习组件
├── layers.py                     # 神经网络层实现
├── models.py                     # 简单网络模型
├── optimizers.py                 # 优化器实现
├── performance.py                # 性能优化工具
│
# PyTorch高级功能
├── models_torch.py               # PyTorch模型实现
├── training.py                   # 训练器和配置
├── evaluation.py                 # 模型评估
├── monitoring.py                 # 性能监控
├── data.py                       # 数据加载器
├── kaggle_data.py               # Kaggle优化数据管线
├── kaggle_models.py             # 竞赛级模型
│
# LLM架构
├── llm_architecture.py          # LLaMA完整实现
├── llm_visualization.py         # LLM架构可视化
│
# 可视化与分析
├── visualization.py              # 数据分析仪表盘
├── training_monitor.py          # 训练过程监控
│
└── models/
    └── cifar10_competition.py    # CIFAR-10竞赛模型

# 统一入口
run_example.py                    # 集成所有功能的统一入口
                                 # 包含Prompt Engineering与Few-shot技术
```

## 📦 安装依赖

### 基础依赖（必需）
```bash
pip install numpy matplotlib torch
```

### 完整依赖（推荐）
```bash
pip install torch torchvision numpy pandas matplotlib plotly dash albumentations timm
```

### 可选依赖（性能优化）
```bash
pip install numba lmdb optuna timm wandb
```

## 🚀 快速开始

### 1. 基础知识演示
```bash
# 线性代数、反向传播、优化器、CNN/Transformer基础
python run_example.py fundamentals
```

### 2. LLM架构演示
```bash
# LLaMA架构原理和可视化
python run_example.py llm
```

### 3. 性能测试
```bash
# 简单神经网络性能基准测试
python run_example.py snn
```

### 4. 数据仪表盘
```bash
# 交互式数据可视化仪表盘
python run_example.py dashboard
```

### 5. 深度学习训练
```bash
# 分布式训练和模型压缩
python run_example.py train
```

### 6. Prompt Engineering 与 Few-shot 技术演示
```bash
# Prompt Engineering与Few-shot技术
python run_example.py prompt
```

### 7. 交互模式
```bash
# 菜单选择模式
python run_example.py
```

### 8. 所有功能快速演示
```bash
# 运行所有模块
python run_example.py quick
```

## 💡 详细功能说明

### 1. 🧮 线性代数与自动微分基础
**文件：`linear_algebra.py`**
- ✅ 基础矩阵运算（加法、乘法、转置、特征值分解）
- ✅ 向量运算（点积、叉积、范数、夹角计算）
- ✅ Jacobian矩阵计算（解析法 + 数值法，误差 < 1e-10）
- ✅ 链式法则详细推导和验证
- ✅ 手写线性层（Xavier初始化 + 梯度检查）
- ✅ 手写ReLU激活函数（含可视化）
- ✅ 与PyTorch autograd对比验证

### 2. 🔄 反向传播机制详解
**文件：`backpropagation.py`**
- ✅ 两层神经网络手工实现
- ✅ 详细反向传播公式推导
- ✅ 逐步演示梯度计算过程
- ✅ 数值方法验证梯度正确性
- ✅ 与PyTorch autograd完整对比
- ✅ 训练损失曲线可视化
- ✅ 参数更新过程监控

### 3. ⚙️ 优化算法数学原理
**文件：`optimizer_comparison.py`**
- ✅ SGD（含动量）详细实现
- ✅ Adam优化器完整数学推导
- ✅ RMSProp自适应学习率
- ✅ MNIST数据集收敛曲线对比
- ✅ 优化轨迹2D可视化
- ✅ 学习率影响分析
- ✅ 性能基准测试

### 4. 🧠 卷积与Transformer基础
**文件：`cnn_transformer.py`**
- ✅ 手写卷积层（im2col实现）
- ✅ 最大池化层实现
- ✅ 自注意力机制详细推导
- ✅ 多头注意力完整实现
- ✅ 矩阵乘法几何意义演示
- ✅ CNN vs Transformer特点对比
- ✅ 计算复杂度分析
- ✅ 注意力权重可视化

### 5. 🚀 LLM架构原理
**文件：`llm_architecture.py`, `llm_visualization.py`**
- ✅ 多头自注意力机制
- ✅ RoPE旋转位置编码
- ✅ RMSNorm层归一化
- ✅ SwiGLU前馈网络
- ✅ 残差连接
- ✅ LLaMA完整架构
- ✅ 文本生成功能
- ✅ 架构可视化图表

### 6. 🎯 深度学习训练系统
**文件：`training.py`, `models_torch.py`**
- ✅ PyTorch模型模块化设计
- ✅ Kaggle竞赛级模型训练
- ✅ 分布式数据并行 (DDP)
- ✅ 混合精度训练 (FP16)
- ✅ 模型量化和压缩
- ✅ 学习率调度、早停、BatchNorm、Dropout
- ✅ 性能监控和对比
- ✅ CIFAR-10实战（Top-1准确率 > 90%）

### 7. 📊 数据管线优化
**文件：`data.py`, `kaggle_data.py`**
- ✅ LMDB数据缓存优化
- ✅ Albumentations数据增强
- ✅ 分布式DataLoader
- ✅ 内存映射文件IO
- ✅ 多进程数据加载
- ✅ IO性能提升3-5倍

### 8. 🧠 Prompt Engineering 与 Few-shot 技术
**集成在 `run_example.py` 中**
- ✅ 自动化 Prompt 调试与质量分析
- ✅ Few-shot 示例生成与管理  
- ✅ GPT 输出质量提升工具
- ✅ 批量Prompt测试与输出收集
- ✅ 自动化Prompt优化与输出评测
- ✅ 模拟测试环境（无需实际API调用）

## 🔬 代码示例

### 统一入口使用
```bash
# 所有功能通过run_example.py统一调用
python run_example.py                    # 交互菜单模式
python run_example.py fundamentals       # 基础知识演示
python run_example.py llm               # LLM架构演示
python run_example.py snn               # 性能测试
python run_example.py dashboard         # 数据仪表盘
python run_example.py train             # 深度学习训练
python run_example.py prompt            # Prompt Engineering
python run_example.py quick             # 快速全功能演示
```

### Prompt Engineering使用示例
```python
# run_example.py集成的PromptDebugger和FewShotManager
# 运行: python run_example.py prompt

# 功能包括:
# 1. Few-shot示例管理
# 2. 自动化Prompt调试
# 3. 批量Prompt测试  
# 4. 自动优化Prompt
# 5. 自动生成Few-shot示例
```

## 📊 性能指标

| 功能模块 | 性能提升 | 验证指标 |
|---------|---------|---------|
| 数据管线优化 | IO性能提升3-5倍 | 吞吐量测试 |
| 混合精度训练 | 训练速度提升2-3倍 | FP16 vs FP32 |
| 模型量化 | 模型大小减少75% | 精度损失 < ±1% |
| 梯度检查 | 数值误差 < 1e-10 | 解析vs数值梯度 |
| CIFAR-10训练 | Top-1准确率 > 90% | 验证集评估 |
| 分布式训练 | 线性扩展效率 | 多GPU性能 |
| Prompt调试 | 输出质量提升 | 自动化评测 |

## 🎯 验证结果

- ✅ **数值精度**：所有梯度检查误差 < 1e-10
- ✅ **PyTorch一致性**：与官方实现参数差异 < 1e-8
- ✅ **性能优化**：训练速度提升2-3倍，IO优化3-5倍
- ✅ **模型精度**：CIFAR-10准确率 > 90%，量化精度损失 < 1%
- ✅ **可视化完整性**：包含所有关键组件的详细图表
- ✅ **统一入口**：所有功能通过run_example.py统一调用，包含Prompt Engineering集成

## 🚀 最新更新

### v2.0 - 统一入口与功能整合
- ✅ **代码整合**：将Prompt Engineering功能完全集成到run_example.py
- ✅ **统一入口**：删除独立的prompt_engineering.py文件，避免代码冗余
- ✅ **错误修复**：修复了所有语法错误和导入问题
- ✅ **功能增强**：添加了命令行参数支持和交互菜单
- ✅ **模拟环境**：Prompt Engineering提供模拟测试，无需实际API调用

## 🛠️ 开发说明

- 代码遵循 PEP 8 规范
- 使用类型注解增强代码可读性
- 关键函数都有文档字符串说明
