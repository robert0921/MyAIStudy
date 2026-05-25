# 🎓 MyAIStudy - AI工程师全栈培训系统 v4.0

> 从 Python 入门到企业级 AI 应用交付的完整学习路径 | 48 周 整合主线 + 应用开发冲刺

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/numpy-latest-orange.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

---

⭐ **如果这个项目对您有帮助，请给它一个星标！** ⭐

## 📖 项目简介

MyAIStudy 是一套完整的 AI 工程师培训体系，涵盖从编程基础到企业级 AI 应用交付的全部内容。当前仓库提供 **4 份阶段化路线** 和 **1 份去重后的整合总计划**，适合两种使用方式：

- 按阶段逐步推进：入门版 → 进阶版 → 高级版
- 按整合版主线学习：参考 AI学习48周实战计划表（整合版）.md

| 路线 | 周期 | 定位 | 核心内容 | 适合人群 |
|------|------|------|----------|----------|
| 🌱 **[入门版](#-入门版第1-12周)** | 12周 | 基础主线 | Python、NumPy、Pandas、Scikit-Learn、PyTorch基础 | 编程基础，AI零基础 |
| 🎓 **[进阶版](#-进阶版第13-24周)** | 12周 | 原理主线 | 从零实现、LLM架构、优化算法、工程实践 | 想深入理解原理的开发者 |
| 🚀 **[高级版](#-高级版第25-36周)** | 12周 | 系统化主线 | RAG系统、AI Agent、科研化输出、职业化准备 | 想形成系统性工程能力 |
| 🏢 **[应用开发版](#-应用开发版平行实战赛道)** | 12周 | 企业落地冲刺 | RAG工程、Agent架构、LLM微调、高并发部署、AI Coding | 有AI基础、专注企业落地 |
| 🧭 **整合版总计划** | 48周 | 去重主线 | 合并重复内容后的完整学习时间表 | 想按一条主线持续推进的学习者 |

**推荐优先查看：** AI学习48周实战计划表（整合版）.md  
**应用开发快线：** README_application.md + AI学习12周实战计划表（应用开发版）.md  
**统一入口：** run_example.py 支持路线直达与参数透传，例如 `python run_example.py application quick`

---

## 🎯 完整学习路线图

```
┌────────────────────────────────────────────────────────────────────┐
│  阶段一 (第1-4周) - Python 与数据科学基础                           │
├────────────────────────────────────────────────────────────────────┤
│  Python / NumPy / Pandas / Matplotlib                              │
└────────────────────────────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────────┐
│  阶段二 (第5-12周) - 机器学习与深度学习入门                         │
├────────────────────────────────────────────────────────────────────┤
│  Scikit-Learn / PyTorch / CNN / RNN / 综合项目                      │
└────────────────────────────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────────┐
│  阶段三 (第13-24周) - 深度学习原理与 LLM 工程基础                   │
├────────────────────────────────────────────────────────────────────┤
│  反向传播 / 优化器 / Transformer / 微调 / 推理优化 / 训练系统        │
└────────────────────────────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────────┐
│  阶段四 (第25-36周) - 企业级 RAG、Agent 与服务化                    │
├────────────────────────────────────────────────────────────────────┤
│  Prompt / Embedding / RAG / MCP / Agent / LangGraph / FastAPI      │
└────────────────────────────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────────┐
│  阶段五 (第37-48周) - 工程落地、科研化输出与职业化                  │
├────────────────────────────────────────────────────────────────────┤
│  vLLM / SGLang / AI Coding / 论文复现 / 知识管理 / 项目展示         │
└────────────────────────────────────────────────────────────────────┘
```

**整合版时间表：** AI学习48周实战计划表（整合版）.md  
**企业落地冲刺：** README_application.md

---

## 🌱 入门版（第1-12周）

### 学习目标
从零开始，掌握Python编程、数据科学工具、机器学习和深度学习基础

### 核心特色
- ✅ **系统化基础训练** - Python → NumPy → Pandas → Scikit-Learn → PyTorch
- ✅ **实战项目驱动** - 学生管理系统、电商分析、房价预测、图像分类
- ✅ **循序渐进** - 每周一个主题，逐步构建AI知识体系
- ✅ **工具链完整** - 数据处理、可视化、建模、评估全流程

### 学习路线

**第1-4周：Python与数据科学基础**
- 🐍 Week 1: Python基础与面向对象编程
- 🔢 Week 2: NumPy数组操作与矩阵运算
- 📊 Week 3: Pandas数据处理与分析
- 📈 Week 4: Matplotlib/Seaborn可视化

**第5-8周：机器学习基础**
- 🤖 Week 5: 机器学习概念与分类任务（鸢尾花）
- 📉 Week 6: 线性回归与决策树（房价预测）
- 🔍 Week 7: 聚类算法（用户分群）
- ⚙️ Week 8: 模型调优与交叉验证

**第9-12周：深度学习入门**
- 🧠 Week 9: PyTorch基础与MNIST手写数字识别
- 🖼️ Week 10: CNN卷积神经网络（CIFAR-10）
- 📝 Week 11: RNN/LSTM序列模型（文本分类）
- 🎯 Week 12: 综合项目实战

### 快速开始

```bash
# 方式1：交互式菜单
python run_beginner_examples.py

# 方式2：运行某周
python run_beginner_examples.py week1    # Python基础
python run_beginner_examples.py week5-8  # 机器学习
python run_beginner_examples.py week9-12 # 深度学习

# 方式3：运行整个阶段
python run_beginner_examples.py stage1   # 第1-4周
python run_beginner_examples.py all      # 全部12周
```

### 学习成果
- 📊 12个实战项目
- 📈 6个数据分析报告
- 🤖 3个深度学习模型（MLP、CNN、LSTM）
- 📝 1000+行实战代码

**📖 详细文档：** [README_beginner.md](./README_beginner.md)

---

## 🎓 进阶版（第13-24周）

### 学习目标
深入理解深度学习数学原理，掌握大模型训练与优化技术

### 核心特色
- 🧮 **从零实现核心算法** - 手写反向传播、优化器、卷积、注意力机制
- 🔬 **数学严谨验证** - 梯度检查误差 < 1e-10，与PyTorch对比验证
- 🚀 **工程级优化** - 分布式训练、混合精度、模型压缩、推理优化
- 🤖 **LLM全栈技术** - 架构实现、Prompt工程、LoRA微调、KV Cache

### 学习路线

**第13-16周：深度学习数学内核**
- 📐 Week 13: 线性代数与自动微分
- 🔄 Week 14: 反向传播机制详解
- ⚙️ Week 15: 优化算法对比（SGD/Adam/RMSProp）
- 🧠 Week 16: CNN与Transformer基础

**第17-20周：工程实践**
- 🎯 Week 17: PyTorch训练系统（分布式/混合精度）
- 🔪 Week 18: 模型剪枝与压缩
- 🎨 Week 19: 大模型微调（LoRA/QLoRA/PEFT）
- ⚡ Week 20: 推理优化（KV Cache/批量推理）

**第21-24周：LLM专项**
- 🤖 Week 21: LLaMA架构完整实现
- 💬 Week 22: Prompt Engineering与Few-shot学习
- 📊 Week 23: 数据可视化与仪表盘
- 🔧 Week 24: 性能监控与训练管线

### 快速开始

```bash
# 方式1：交互式菜单
python run_intermediate_examples.py

# 方式2：功能模块
python run_intermediate_examples.py fundamentals  # 基础知识演示
python run_intermediate_examples.py llm           # LLM架构
python run_intermediate_examples.py train         # 训练系统
python run_intermediate_examples.py finetuning    # 大模型微调
python run_intermediate_examples.py inference     # 推理优化

# 方式3：演示模式
python run_intermediate_examples.py quick         # 快速演示（5分钟）
python run_intermediate_examples.py all           # 完整演示（30分钟）
```

### 学习成果
- 📐 4个数学核心模块（线性代数、反向传播、优化器、Transformer）
- 🎯 5个工程实践模块（训练系统、剪枝、微调、推理、监控）
- 🧠 2个LLM专项模块（LLaMA架构、Prompt工程）
- 📊 13,000+行完整实现代码

**📖 详细文档：** [README_intermediate.md](./README_intermediate.md)

---

## 🚀 高级版（第25-36周）

### 学习目标
构建生产级RAG系统、AI Agent、知识管理与项目展示能力

### 核心特色
- 📚 **RAG完整实现** - 文档处理、向量检索、Pipeline优化、混合检索
- 🤖 **AI Agent系统** - Memory机制、Tool-Use、Planning、Multi-Agent协作
- 🌐 **服务化部署** - FastAPI、WebSocket、会话管理、性能优化
- 📊 **系统监控** - Prometheus、ELK、异常检测、自动恢复
- 📝 **科研化输出** - 论文管理、实验追踪、知识图谱、技术文档
- 💼 **职业化准备** - 项目展示、白皮书、面试题库

### 学习路线

**第25-30周：RAG系统与智能体**
- 📚 Week 25: LangChain框架与RAG原理
- 🔍 Week 26: 向量数据库索引机制（Flat/IVF/HNSW）
- 🎯 Week 27: RAG Pipeline优化（Chunking/Re-ranking）
- 🤖 Week 28: AI Agent架构设计
- 🌐 Week 29: FastAPI服务化部署
- 📊 Week 30: 系统监控与异常恢复

**第31-36周：科研化与职业化**
- 📄 Week 31-32: 论文管理与实验追踪
- 💰 Week 33: GPU性能优化与成本评估
- 📝 Week 34-35: 知识管理与文档生成
- 💼 Week 36: 项目展示与面试准备

### 快速开始

```bash
# 方式1：交互式菜单
python run_advanced_examples.py

# 方式2：运行某周（阶段四）
python run_advanced_examples.py week13      # LangChain与RAG
python run_advanced_examples.py week14      # 向量数据库
python run_advanced_examples.py week15-18   # RAG优化/Agent/服务化

# 方式3：运行某周（阶段五）
python run_advanced_examples.py week19-20   # 论文管理/实验追踪
python run_advanced_examples.py week21      # GPU优化/成本评估
python run_advanced_examples.py week22-23   # 知识管理/文档生成
python run_advanced_examples.py week24      # 项目展示/面试准备

# 方式4：演示模式
python run_advanced_examples.py quick       # 快速演示（10分钟）
python run_advanced_examples.py all         # 完整演示（40分钟）
```

### 学习成果
- 🔍 6个RAG模块（基础、优化、Agent、服务化、监控）
- 📚 4个科研工具（论文管理、实验追踪、知识管理、项目展示）
- 📊 2,800+行高级功能代码
- 💼 完整的求职作品集

**📖 详细文档：** [README_advanced.md](./README_advanced.md)

---

## 🏢 应用开发版（平行实战赛道）

### 学习目标
面向企业级大模型应用落地，用 12 周完成一条可运行、可演示、可继续扩展的业务原型链路

### 核心特色
- 🗂️ **RAG 工程全链路** - Prompt、Embedding、Chunking、检索、评估一条线跑通
- 🤖 **Agent 架构实战** - Function Calling、MCP、Memory、ReAct 的最小可运行样例
- 🔧 **框架选型意识** - LangChain、LlamaIndex、AutoGen、Coze、Dify 的定位对比
- ⚡ **部署与微调认知** - LoRA / QLoRA、vLLM、SGLang、Ollama 的取舍思路
- 💼 **工程化闭环** - Spec Coding、验收清单、Text-to-SQL 与综合项目演示

### 学习路线

**第1-4周：大模型基础与 RAG 工程**
- 💬 Week 1: Prompt Engineering & Context Engineering
- 🔢 Week 2: Embedding 原理与向量数据库选型
- 📚 Week 3: RAG 核心流程与本地知识库搭建
- 🎯 Week 4: 混合检索 + Reranking + RAG 效果评估

**第5-8周：Agent 架构与框架实战**
- 🔧 Week 5: Function Calling 与 MCP 协议
- 🧠 Week 6: Agent 规划、记忆与 ReAct / LangGraph 实战
- 🛠️ Week 7: LangChain / LlamaIndex / AutoGen 框架精讲
- 🖱️ Week 8: Coze / Dify 低代码平台与企业系统集成

**第9-12周：微调、部署与工程效能**
- ⚙️ Week 9: LoRA / QLoRA 微调与显存优化
- 🚀 Week 10: vLLM / SGLang / Ollama 高并发推理部署
- 💻 Week 11: AI Coding 工程实践与 ChatBI 项目
- 🏆 Week 12: 综合项目 · 企业 RAG + Agent + 部署全栈

### 学习成果
- 📦 企业级知识问答 Agent 原型（轻量代码版）
- 📊 微调策略对比表 + 高并发部署框架对比表
- 📄 一套可继续替换为真实框架的工程骨架

### 快速开始

```bash
# 统一入口
python run_example.py application
python run_example.py application quick

# 应用开发版独立入口
python run_application_examples.py
python run_application_examples.py quick
python run_application_examples.py week1-4
python run_application_examples.py week12
```

**📖 详细文档：** [README_application.md](./README_application.md)  
**📖 详细计划：** [AI学习12周实战计划表（应用开发版）.md](./AI学习12周实战计划表（应用开发版）.md)

---

## 🚀 快速开始

### 统一入口（推荐）

```bash
# 交互式选择版本
python run_example.py

# 菜单选项：
# [1] 入门版 - Python与AI基础
# [2] 进阶版 - 深度学习原理与工程
# [3] 高级版 - RAG与科研化输出
# [4] 应用开发版 - 企业级大模型应用实战
# [5] 查看项目信息
# [6] 退出
```

### 命令行直接启动

```bash
# 直接启动入门版
python run_example.py beginner

# 直接启动进阶版
python run_example.py intermediate

# 直接启动高级版
python run_example.py advanced

# 直接启动应用开发版
python run_example.py application
python run_example.py application quick

# 查看帮助
python run_example.py --help
```

---

## 📁 项目结构

```
MyAIStudy/
├── 📌 统一入口
│   ├── run_example.py              # 主入口（选择版本）
│   ├── run_beginner_examples.py    # 入门版专用入口
│   ├── run_intermediate_examples.py # 进阶版专用入口
│   ├── run_advanced_examples.py    # 高级版专用入口
│   └── run_application_examples.py # 应用开发版专用入口
│
├── 🌱 入门版代码 (beginner/)
│   ├── week1_python_basics.py      # Python基础与OOP
│   ├── week2_numpy_operations.py   # NumPy数组操作
│   ├── week3_pandas_analysis.py    # Pandas数据分析
│   ├── week4_visualization.py      # Matplotlib可视化
│   ├── week5_8_machine_learning.py # 机器学习（4周合并）
│   └── week9_12_deep_learning.py   # 深度学习（4周合并）
│
├── 🎓 进阶版代码 (intermediate/)
│   ├── 📐 数学内核
│   │   ├── linear_algebra.py       # 线性代数与自动微分
│   │   ├── backpropagation.py      # 反向传播机制
│   │   ├── optimizer_comparison.py # 优化算法对比
│   │   └── cnn_transformer.py      # CNN与Transformer
│   │
│   ├── 🎯 训练系统
│   │   ├── models_torch.py         # PyTorch模型
│   │   ├── training.py             # 训练器（含早停/检查点）
│   │   ├── pruning.py              # 模型剪枝
│   │   ├── finetuning.py           # LoRA/QLoRA/PEFT微调
│   │   └── inference_optimization.py # KV Cache/批量推理
│   │
│   └── 🤖 LLM架构
│       ├── llm_architecture.py     # LLaMA完整实现
│       └── llm_visualization.py    # LLM可视化
│
├── 🚀 高级版代码 (advanced/)
│   ├── 📚 RAG系统 (Week 13-18)
│   │   ├── week13_langchain_rag.py
│   │   ├── week14_vector_database.py
│   │   └── week15_18_placeholder.py
│   │
│   └── 🔬 科研工具 (Week 19-24)
│       ├── week19_20_research_tools.py
│       ├── week21_optimization.py
│       ├── week22_23_knowledge_management.py
│       └── week24_presentation.py
│
├── 🏢 应用开发版代码 (application/)
│   ├── week1_4_rag_engineering.py  # Prompt / Embedding / RAG / 评估
│   ├── week5_8_agent_workflows.py  # MCP / Agent / 工作流
│   └── week9_12_delivery.py        # 微调 / 部署 / 工程效能 / 综合项目
│
├── 📚 文档
│   ├── README.md                   # 主文档（本文件）
│   ├── README_beginner.md         # 入门版详细文档
│   ├── README_intermediate.md     # 进阶版详细文档
│   ├── README_advanced.md         # 高级版详细文档
│   ├── README_application.md      # 应用开发版详细文档
│   └── AI学习48周实战计划表（整合版）.md
│
└── 💾 数据与模型
    ├── checkpoints/               # 模型检查点
    └── data/                      # 数据集（自动下载）
```

---

## 🛠️ 环境配置

### Python版本
Python 3.8+

### 依赖安装

**入门版依赖：**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch torchvision
```

**进阶版额外依赖：**
```bash
pip install numba plotly dash
```

**高级版额外依赖（可选）：**
```bash
pip install langchain faiss-cpu chromadb fastapi uvicorn redis prometheus-client
```

**应用开发版当前样例：**

应用开发版的轻量演示默认仅依赖 Python 标准库；如果要替换成真实框架实现，可按 README_application.md 中的建议逐步补齐依赖。

**一键安装全部：**
```bash
pip install -r requirements.txt
```

---

## 💡 适用人群

| 学习阶段 | 前置要求 | 适合人群 | 学习目标 |
|---------|---------|---------|---------|
| 🌱 **入门版** | 基础编程知识 | 编程基础，AI零基础 | 掌握Python与机器学习基础 |
| 🎓 **进阶版** | 完成入门版或具备深度学习基础 | 想深入理解原理的开发者 | 掌握深度学习数学与工程实践 |
| 🚀 **高级版** | 完成进阶版或熟悉深度学习 | 想构建企业级AI应用 | 掌握 RAG、Agent、科研化输出与职业化准备 |
| 🏢 **应用开发版** | 具备基础 Python 与 AI 概念 | 专注企业级大模型应用落地的开发者 | 独立搭建 RAG 知识库、Agent 系统并完成生产部署 |
| 🧭 **整合版主线** | 愿意长期持续推进 | 希望按一条去重后的路线完整学习 | 按 48 周节奏完成从基础到企业交付的主线成长 |
---

## 🏆 学习成果

完成整合版 48 周主线，或完成前三阶段后补齐应用开发版关键模块后，你将：

✅ **扎实的理论基础** - 从数学原理到工程实践，系统掌握AI核心技术  
✅ **丰富的项目经验** - 30+个实战项目，涵盖数据分析、模型训练、系统部署  
✅ **完整的技能栈** - Python、PyTorch、RAG、Agent、API、监控全覆盖  
✅ **科研化思维** - 论文复现、实验管理、技术文档、知识沉淀  
✅ **职业化能力** - 项目展示、技术演讲、面试准备、简历优化

---

## 📊 性能指标

### 进阶版性能指标

| 功能模块 | 性能提升 | 验证指标 |
|---------|---------|---------|
| 数据管线优化 | IO性能提升3-5倍 | 吞吐量测试 |
| 混合精度训练 | 训练速度提升2-3倍 | FP16 vs FP32 |
| 模型量化 | 模型大小减少75% | 精度损失 < ±1% |
| 模型剪枝 | 参数减少30-50% | 精度损失 < ±1% |
| LoRA微调 | 可训练参数减少99% | 只训练0.5-1%参数 |
| KV Cache | 生成速度提升2-10倍 | 延迟降低 |
| 批量推理 | 吞吐量提升3-8倍 | batch_size=4-16 |

### 高级版性能指标

| 功能 | 指标 | 说明 |
|------|------|------|
| 向量检索（IVF） | 10-20x加速 | 相比Flat Index |
| RAG问答 | Top-3准确率 | 相似度检索 |
| 推荐维度 | 384-768 | 性能与精度平衡 |

---

## 📚 推荐学习路径

### 推荐主线
1. 优先参考 AI学习48周实战计划表（整合版）.md。
2. 按阶段一 → 阶段五顺序推进，避免在高级版和应用开发版之间反复切换。
3. 每完成一个阶段，至少沉淀一个项目、一个文档、一个指标表。

### 零基础学习者
1. 从入门版第 1 周开始。
2. 按周完成代码练习与周总结。
3. 第 12 周后进入进阶版，再转向整合版第 25 周之后的应用工程路线。

### 有深度学习基础
1. 从进阶版核心模块或整合版第 13 周开始。
2. 补齐 LoRA、推理优化、训练系统等工程基础。
3. 再进入第 25 周之后的 RAG / Agent / 部署链路。

### 想快速落地企业应用
1. 直接学习应用开发版 12 周冲刺路线。
2. 同步阅读 README_application.md 和 AI学习12周实战计划表（应用开发版）.md。
3. 完成冲刺后，再回补整合版第 41~48 周的科研化与职业化内容。

---

## 📝 学习建议

### 时间安排
- **每周投入**：10-15小时（约1.5-2小时/天）
- **总计周期**：48周主线，或 12周应用开发冲刺
- **建议节奏**：工作日1.5小时，周末5小时

### 学习方法
1. **动手优先** - 每个知识点都要写代码验证
2. **记录笔记** - 记录"学到的知识点 + 遇到的坑 + 解决方式"
3. **对比学习** - 不同算法对比、性能对比
4. **可视化进度** - 用Excel或Notion记录学习进度

### 调试技巧
- 使用 `print()` 查看中间结果
- 使用 `shape` 和 `dtype` 检查张量维度
- 使用断点调试 (VSCode Debugger)
- 查看官方文档和Stack Overflow

---

## 🤝 贡献指南

欢迎提交问题和改进建议！

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

- PyTorch团队提供的优秀深度学习框架
- LangChain社区的RAG技术贡献
- 开源社区的宝贵贡献和支持

---

## 📮 联系方式

- 项目主页：[GitHub Repository](https://github.com/robert0921/MyAIStudy)
- 问题反馈：[Issues](https://github.com/robert0921/MyAIStudy/issues)

---

## 📈 版本历史

### v4.0 (2026-05-25) 🎉
- ✅ **统一入口升级**：`run_example.py` 采用四路线配置驱动，并支持参数透传
- ✅ **应用开发版补齐**：新增 12 周 README、示例代码、独立入口和整合主线说明
- ✅ **48 周主线成型**：新增去重后的整合学习计划，形成推荐学习路径
- ✅ **文档口径统一**：主 README 与各阶段 README 全面对齐 v4.0 和真实脚本名
- ✅ **高级版说明修订**：Week 15-17 从计划占位更新为已实现模块说明

### v3.0 (2025-11-13) 🎉
- ✅ **完成高级版Week 19-24**：科研化输出与职业化准备
- ✅ **新增论文管理工具**：Paper/PaperLibrary/Experiment/ExperimentTracker
- ✅ **新增GPU优化模块**：成本计算、性能分析、模型压缩
- ✅ **新增知识管理系统**：文档生成、知识图谱、笔记管理
- ✅ **新增职业化模块**：项目展示、白皮书、面试题库
- ✅ **文档全面更新**：README整合三个版本，各版本独立详细文档
- ✅ **36周完整学习路线**：入门→进阶→高级全覆盖

### v2.3 (2025-11-06)
- ✅ **进阶版大模型微调**：LoRA/QLoRA/PEFT实现
- ✅ **进阶版推理优化**：KV Cache/批量推理
- ✅ **性能基准测试**：完整的延迟、吞吐量、显存分析

### v2.2 (2025-11-05)
- ✅ **进阶版模型剪枝**：4种剪枝策略（幅度/结构化/全局/迭代）
- ✅ **压缩效果评估**：精度、模型大小、推理速度对比

### v2.1 (2025-11-04)
- ✅ **智能早停**：EarlyStopping类
- ✅ **检查点管理**：CheckpointManager类

### v2.0 (2025-11-04)
- ✅ **统一入口**：run_example.py整合三个版本
- ✅ **模块化依赖**：可选依赖管理

### v1.0 (2025-10)
- ✅ **入门版完成**：12周Python与AI基础
- ✅ **进阶版基础**：深度学习原理实现
- ✅ **高级版Week 13-14**：RAG基础

---

<div align="center">

**Made with ❤️ by [robert0921](https://github.com/robert0921)**

[⬆️ 返回顶部](#-myaistudy---ai工程师全栈培训系统-v40)

</div>
