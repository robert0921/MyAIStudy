# 🏢 MyAIStudy 应用开发版 v4.0 - 企业级大模型应用 12 周实战

> 与主入口 run_example.py v4.0 对齐，用 12 周完成一条可运行、可讲解、可扩展的企业 AI 应用落地链路 | RAG 工程 + 可控 Agent + Harness + 微调部署 + 研发提效

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![Track](https://img.shields.io/badge/track-application--engineering-orange.svg)]()

[中文](./README_application.md) | [日本語](./README_application_ja.md)

## 📋 项目简介

本项目是《AI学习12周实战计划表（应用开发版）》的轻量代码实现，并参考《知乎知学堂「AI大模型应用开发实战训练营」26年新版课纲》做了课程重构。定位不是替代真实生产框架，而是提供一套**低依赖、可运行、可讲解**的训练样例，方便把训练营中的 RAG、Agent、部署与工程化概念快速串起来。

这一版采用双层结构：

- ✅ **第一层：核心可运行主线** - 用纯 Python 示例把企业级 AI 应用的主链路跑通
- ✅ **第二层：进阶专题扩展** - 按新版课纲吸收多模态文档处理、Agent 可控性 / 自反思、Harness 工程、低代码平台、AI 测试与 ChatBI

因此，本文档会明确区分哪些内容仓库已经提供可运行样例，哪些内容更适合在真实框架中做替换实践。

---

## 🎯 学习目标

完成应用开发版学习后，你将能够：

✅ **完成 RAG 工程闭环** - 从 Prompt、Embedding、Chunking 到 Query 改写、混合检索和评估，完成知识库问答主链路  
✅ **理解可控 Agent 执行机制** - 掌握 Function Calling、MCP、JSON 约束、ReAct、Self-Reflection、Human-in-the-Loop 等关键结构  
✅ **建立 Harness 与长期记忆认知** - 能把 Orchestration / Memory / Execution / Feedback 四层闭环讲清楚，并设计任务编排方案  
✅ **掌握落地与提效链路** - 理解 LoRA / QLoRA / 蒸馏、高并发部署、AI 测试、Text-to-SQL 与 ChatBI 的关系  
✅ **输出业务原型与验收材料** - 拼装一个“知识库 + Agent + 服务层”的综合示例，并沉淀架构图、指标表和验收清单  

---

## 🎯 学习路线（第1-12周）

### 阶段一：大模型基础与 RAG 工程（Week 1-4）

| 周次 | 学习重点 | 核心内容 | 实践输出 | 对齐方式 |
|------|----------|----------|----------|----------|
| **第1周** | Prompt / Context Engineering | 提示结构、输出约束、长上下文与可控生成 | Prompt 实验对比 + 输出约束模板 | ✅ 代码主线 |
| **第2周** | Embedding 与向量检索 | 稀疏向量、相似度、Embedding 选型、向量库基础能力 | Top-K 检索演示 + 模型选型记录 | ✅ 代码主线 |
| **第3周** | 文档摄取与 Native RAG | 文档分块、知识摄取、检索生成闭环、PDF / Word / Web 解析思路 | 简化版知识库问答 + 文档摄取流程图 | ✅ 主线 + 🔶 文档解析专题 |
| **第4周** | RAG 调优与评估 | Query 改写、Hybrid Search、Rerank、hit rate / MRR、知识库运维思路 | 检索评估报告 + 调优清单 | ✅ 主线 + 🔶 调优专题 |

### 阶段二：Agent 架构与开发框架（Week 5-8）

| 周次 | 学习重点 | 核心内容 | 实践输出 | 对齐方式 |
|------|----------|----------|----------|----------|
| **第5周** | Function Calling / MCP / A2A | 工具注册、参数传递、协议化调用、A2A 关系理解 | MCP 工具调用演示 | ✅ 代码主线 |
| **第6周** | 可控 Agent 设计 | Planning、Memory、ReAct、Self-Reflection、Human-in-the-Loop | 多步骤 Agent 样例 + 审批节点草图 | ✅ 主线 + 🔶 可控性专题 |
| **第7周** | Harness 与长期记忆 | Orchestration / Memory / Execution / Feedback、长期记忆、任务闭环、多 Agent 调度 | Harness 分层设计图 + 记忆分层草案 | 🔶 专题扩展 |
| **第8周** | 框架与低代码集成 | LangChain / LlamaIndex / AutoGen / Coze / Dify / OpenManus 对比 | 框架选型表 + 工作流方案 | ✅ 主线 + 🔶 集成专题 |

### 阶段三：微调、部署与工程效能（Week 9-12）

| 周次 | 学习重点 | 核心内容 | 实践输出 | 对齐方式 |
|------|----------|----------|----------|----------|
| **第9周** | LoRA / QLoRA / 数据工程 | 微调方式、资源开销、数据准备、显存估算、蒸馏认知 | 微调方案比较表 | ✅ 主线 + 🔶 数据专题 |
| **第10周** | 高并发部署 | vLLM / SGLang / Ollama、PagedAttention、Continuous Batching、监控 | 框架对比基准 | ✅ 代码主线 |
| **第11周** | AI Coding 与工程提效 | Spec Coding、AI Testing、验收清单、Text-to-SQL、ChatBI | 工程检查表 + SQL Copilot / ChatBI 草案 | ✅ 主线 + 🔶 提效专题 |
| **第12周** | 综合项目与多模态扩展 | RAG + Agent + 部署要素集成，并预留多模态 / 质检 / 视频理解扩展点 | 企业 AI 应用原型 + 路演材料 | ✅ 代码主线 |

### 2026 课纲增强点如何落地

- **已实现的代码主线**：`application/week1_4_rag_engineering.py`、`application/week5_8_agent_workflows.py`、`application/week9_12_delivery.py` 覆盖知识库、Agent、部署与工程效能的核心闭环。
- **适合在真实框架中替换实践的专题**：PDF / Word / Web 解析、多模态 RAG、OpenManus / Hermes 风格 Harness、Coze / Dify API、AI Testing 与 ChatBI。
- **建议的学习顺序**：先跑通可执行样例，再把同一业务需求替换到 LangChain / LangGraph / FastAPI / vLLM / Coze / Dify 等真实栈中。

---

## 🚀 快速开始

### 方式一：统一入口

```bash
# 从主入口启动应用开发版
python run_example.py application
python run_example.py application quick
python run_example.py application week9-12

# 或进入主菜单后选择应用开发版
python run_example.py
```

### 方式二：直接运行应用开发版

```bash
# 交互式菜单（推荐）
python run_application_examples.py

# 直接运行某周
python run_application_examples.py week1
python run_application_examples.py week6
python run_application_examples.py week12

# 运行整个阶段
python run_application_examples.py week1-4
python run_application_examples.py week5-8
python run_application_examples.py week9-12

# 快速演示（每阶段一个代表模块）
python run_application_examples.py quick

# 完整演示（全部12周）
python run_application_examples.py all
```

---

## 📚 详细内容

### 第1-4周：RAG 工程主链路

这一阶段不是直接依赖第三方框架，而是先把底层逻辑跑通：Prompt 结构化、向量表示、Chunking、检索与评估。按照新版课纲，建议在这条主线上进一步补入 Query 改写、多模态文档解析与混合检索。代码位于 application/week1_4_rag_engineering.py。

**核心组件：**

```python
from application.week1_4_rag_engineering import (
    PromptWorkbench,
    FixedChunker,
    SimpleVectorStore,
    SimpleRAGPipeline,
)

workbench = PromptWorkbench()
results = workbench.run_experiment("如何搭建企业知识库问答系统？")

chunker = FixedChunker(chunk_size=45, overlap=8)
vector_store = SimpleVectorStore()
pipeline = SimpleRAGPipeline(vector_store=vector_store, chunker=chunker)
```

**你会看到：**

- ✅ 不同 Prompt 结构对结果可控性的影响
- ✅ 用稀疏向量模拟 Embedding 与相似度检索
- ✅ 一个可运行的最小 RAG 流程
- ✅ hit rate@3 和 MRR@3 的基础评估方式
- 🔶 PDF / Word / Web 文档解析应该放在知识摄取链路的哪个位置
- 🔶 Query Rewrite / Hybrid Search / Rerank / RAGAS 在真实项目中的接入点

---

### 第5-8周：Agent 与工作流编排

这一阶段重点不是追求“多智能体越多越好”，而是先理解：工具如何注册、协议如何调用、状态如何保存、任务如何拆解。按照新版课纲，这里还要把 Agent 可控性、自主反思、Harness 分层和低代码集成一起串起来。代码位于 application/week5_8_agent_workflows.py。

**核心组件：**

```python
from application.week5_8_agent_workflows import (
    ToolRegistry,
    SimpleMCPServer,
    ConversationMemory,
    SimpleReActAgent,
    build_default_registry,
)

registry = build_default_registry()
server = SimpleMCPServer(registry)
memory = ConversationMemory()
agent = SimpleReActAgent(registry, memory)
```

**你会看到：**

- ✅ MCP 风格的 tools/list 与 tools/call
- ✅ Agent 根据任务关键词选择工具
- ✅ Memory 如何保存最近上下文
- ✅ LangChain / LlamaIndex / AutoGen 的差异化定位
- ✅ 低代码客服工作流的最小闭环
- 🔶 如何用 JSON 模式、审批节点、反思提示提升 Agent 可控性
- 🔶 Harness / 长期记忆 / 多 Agent 调度如何在 OpenManus / Hermes 风格系统中落地

---

### 第9-12周：微调、部署与综合项目

这一阶段把“能讲概念”推进到“能说明如何落地”。按照新版课纲，这里不只讨论 LoRA 和部署选型，还要把数据工程、蒸馏、高并发调优、AI 测试与 ChatBI 串成一条完整交付链路。代码位于 application/week9_12_delivery.py。

**核心组件：**

```python
from application.week9_12_delivery import (
    FinetuningPlanner,
    DeploymentBenchmarker,
    SpecCodingAssistant,
    EnterpriseAIAssistant,
)

planner = FinetuningPlanner()
benchmarker = DeploymentBenchmarker()
assistant = EnterpriseAIAssistant()
```

**你会看到：**

- ✅ Full Fine-Tuning / LoRA / QLoRA 的资源差异
- ✅ vLLM / SGLang / Ollama 的部署对比思路
- ✅ Spec Coding 如何生成验收清单与测试项
- ✅ 一个整合 RAG 与 Agent 的业务原型响应流程
- 🔶 数据工程、蒸馏、性能监控、Continuous Batching / RadixAttention 的延展方向
- 🔶 AI 测试、ChatBI、质检 / 多模态业务如何接入当前骨架

---

## 📊 项目结构

```
MyAIStudy/
├── application/
│   ├── __init__.py
│   ├── week1_4_rag_engineering.py   # Prompt / Embedding / RAG / 评估
│   ├── week5_8_agent_workflows.py   # Function Calling / MCP / Agent / 工作流
│   └── week9_12_delivery.py         # 微调 / 部署 / 工程效能 / 综合项目
│
├── run_application_examples.py      # 应用开发版统一入口
├── README_application.md            # 本文档
├── AI学习12周实战计划表（应用开发版）.md
└── AI学习48周实战计划表（整合版）.md
```

---

## 🛠️ 环境要求

### 基础依赖

```bash
python >= 3.8
```

这套训练样例当前只依赖 Python 标准库，目的是先把工程结构跑通。后续如果你要升级为真实业务原型，建议按阶段逐步引入：

```bash
# RAG 真实实现
pip install langchain faiss-cpu chromadb sentence-transformers

# API 服务与监控
pip install fastapi uvicorn redis prometheus-client

# 微调与部署
pip install transformers peft trl unsloth
```

多模态解析、长期记忆、低代码平台与 AI 测试等专题建议按实际选型单独补依赖，而不是在教学样例中一次性堆满。

---

## 🔗 与其他阶段的关系

- 🌱 **入门版** 解决 Python、数据处理、机器学习和深度学习入门问题。
- 🎓 **进阶版** 解决数学原理、模型训练、微调和推理优化问题。
- 🚀 **高级版** 更偏系统化输出、科研化能力和职业化准备。
- 🏢 **应用开发版** 则把“企业落地链路”单独压缩成 12 周冲刺路线。

如果你希望走一条**不重复、去重后的推荐主线**，请优先参考 AI学习48周实战计划表（整合版）.md。

---

## 💡 学习建议

### 适合人群

- ✅ 已具备 Python 基础
- ✅ 至少理解过基础深度学习概念
- ✅ 想把 RAG、Agent、部署串成一个完整工程故事
- ✅ 希望做汇报 Demo、面试项目或内部培训样例

### 建议学习方式

1. 每周先跑通对应演示，再反推代码结构。
2. 每个阶段至少产出一个“能讲给别人听”的图或表。
3. 把示例里的静态实现替换成真实框架，例如把稀疏检索替换成 FAISS，把模拟工具替换成真实 API。
4. 每完成一个阶段，就补一份验收清单：输入是什么、输出是什么、风险是什么、如何验证。

---

## 📄 相关文档

- README.md
- README_advanced.md
- AI学习12周实战计划表（应用开发版）.md
- AI学习48周实战计划表（整合版）.md

---

## 📮 反馈与支持

- GitHub Issues: [提交问题](https://github.com/robert0921/MyAIStudy/issues)
- 如果这套应用开发版样例对你有帮助，可以继续把它替换成真实业务接口与真实知识库

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件。

---

<div align="center">

**MyAIStudy 应用开发版**  
*用最小实现理解企业 AI 应用如何落地*

</div>