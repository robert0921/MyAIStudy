# 快速使用指南

## 🚀 快速开始

### 1. 检查环境
```bash
# 运行帮助命令查看可用功能
python run_example.py help
```

### 2. 选择运行模式

#### 交互模式（推荐新手）
```bash
python run_example.py
```
会显示菜单，输入数字选择功能：
- 1: 基础知识演示
- 2: LLM架构演示
- 3: SNN性能测试
- 4: 数据仪表盘
- 5: 深度学习训练
- 6: Prompt Engineering
- 7: 快速演示（推荐）
- 8: 全部模块

#### 命令行模式（推荐熟练用户）
```bash
# 快速演示（3-5分钟，核心功能）
python run_example.py quick

# 完整演示（需要较长时间）
python run_example.py all

# 单独运行某个模块
python run_example.py llm        # LLM架构
python run_example.py snn        # 性能测试
python run_example.py prompt     # Prompt Engineering
```

## 📋 各模块说明

### 1. 基础知识演示 (fundamentals)
**时间**: 约10-15分钟  
**需要**: NumPy  
**内容**:
- 线性代数基础
- 反向传播机制
- 优化算法对比
- CNN/Transformer原理

### 2. LLM架构演示 (llm)
**时间**: 约5分钟  
**需要**: PyTorch  
**内容**:
- LLaMA模型结构
- 注意力机制
- RoPE位置编码
- 文本生成演示

### 3. SNN性能测试 (snn)
**时间**: 约3分钟  
**需要**: NumPy  
**内容**:
- 矩阵乘法性能
- NumPy vs Numba对比
- 神经网络训练

### 4. 数据仪表盘 (dashboard)
**时间**: 持续运行（Ctrl+C停止）  
**需要**: Plotly, Dash  
**内容**:
- 交互式可视化
- 时间序列分析
- 浏览器自动打开

### 5. 深度学习训练 (train)
**时间**: 取决于模式和硬件  
**需要**: PyTorch, torchvision  
**内容**:
- 4种训练模式可选
- 单GPU/多GPU支持
- FP32/FP16对比

### 6. Prompt Engineering (prompt)
**时间**: 约2分钟  
**需要**: 无特殊依赖  
**内容**:
- Few-shot示例管理
- Prompt调试优化
- 批量测试

### 7. 快速演示 (quick)
**时间**: 约10分钟  
**需要**: PyTorch, NumPy  
**内容**: 运行核心功能（LLM + SNN + Prompt）

### 8. 全部模块 (all)
**时间**: 约30分钟+  
**需要**: 所有依赖  
**内容**: 完整演示所有功能

## 🔧 常见问题

### Q1: 提示缺少模块怎么办？
```bash
# 安装基础依赖
pip install numpy pandas matplotlib

# 安装PyTorch（如果需要深度学习功能）
pip install torch torchvision

# 安装仪表盘依赖（如果需要数据可视化）
pip install plotly dash
```

### Q2: 如何快速测试所有功能？
```bash
# 1. 先运行测试脚本
python test_run_example.py

# 2. 然后运行快速演示
python run_example.py quick
```

### Q3: 程序卡住了怎么办？
- 按 `Ctrl+C` 中断当前操作
- 程序会优雅退出，不会崩溃

### Q4: 如何只运行不需要GPU的功能？
```bash
# 这些功能不需要GPU
python run_example.py snn       # NumPy计算
python run_example.py prompt    # 无需特殊硬件
python run_example.py dashboard # 数据可视化
```

### Q5: 训练需要多长时间？
- 单GPU基础训练: 5-10分钟
- Kaggle模型训练: 30-60分钟
- FP32/FP16对比: 20-40分钟
- 完整流程: 1-2小时

## 💡 推荐使用流程

### 首次使用
1. 运行测试: `python test_run_example.py`
2. 查看帮助: `python run_example.py help`
3. 快速演示: `python run_example.py quick`

### 学习模式
1. 基础知识: `python run_example.py fundamentals`
2. LLM架构: `python run_example.py llm`
3. 实际应用: `python run_example.py train`

### 演示模式
```bash
# 向他人展示（10分钟）
python run_example.py quick

# 深度展示（30分钟）
python run_example.py all
```

## 📊 模块依赖关系

```
run_example.py
├── Prompt Engineering (✓ 无依赖)
├── SNN性能测试 (需要 NumPy)
├── 基础知识演示 (需要 NumPy, Matplotlib)
├── LLM架构演示 (需要 PyTorch)
├── 数据仪表盘 (需要 Plotly, Dash)
└── 深度学习训练 (需要 PyTorch, torchvision)
```

## 🎯 最佳实践

1. **先测试后使用**: 运行 `test_run_example.py` 确保环境正常
2. **逐个尝试**: 从单个模块开始，熟悉后再运行完整演示
3. **查看输出**: 注意控制台输出的模块状态信息
4. **合理安排时间**: 完整演示需要较长时间，建议分批运行
5. **保存结果**: 重要的训练结果会保存在 `checkpoints/` 目录

## 📞 获取帮助

- 查看帮助: `python run_example.py help`
- 查看README: 参考 `README.md`
- 查看优化说明: 参考 `OPTIMIZATION_SUMMARY.md`
- 运行测试: `python test_run_example.py`
