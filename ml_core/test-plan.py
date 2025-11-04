import pandas as pd

data = [
    ["第1周", "数学基础复盘", "线性代数、概率论、微积分在DL中的应用", "3Blue1Brown 视频；李宏毅 ML 课程（前4讲）", "总结笔记：《AI算法中的数学直觉》"],
    ["第2周", "机器学习经典算法", "逻辑回归、SVM、决策树、聚类等", "《Python机器学习实战》", "复现分类器并分析效果"],
    ["第3周", "PyTorch基础", "tensor操作、autograd、model定义", "《动手学深度学习（PyTorch版）》", "实现CNN分类任务"],
    ["第4周", "深度学习基础", "梯度下降、反向传播、优化器", "FastAI课程 / CS231n", "复现MNIST分类并撰写博客"],
    ["第5周", "Transformer结构", "Self-Attention与位置编码", "Illustrated Transformer / CS224n", "画出Transformer结构图并解释"],
    ["第6周", "预训练与微调机制", "BERT/GPT原理与微调", "Hugging Face官方课程", "微调DistilBERT做分类"],
    ["第7周", "Prompt Engineering", "few-shot、CoT、self-consistency", "Learn Prompting / OpenAI Cookbook", "设计10个高质量Prompt"],
    ["第8周", "LoRA与QLoRA微调", "低资源微调方法", "Hugging Face PEFT文档 / Alpaca-LoRA", "微调LLaMA或Mistral模型"],
    ["第9周", "部署与API调用", "使用FastAPI或Gradio部署", "vLLM / FastAPI Docs", "实现ChatGPT风格Web应用"],
    ["第10周", "向量检索与Embedding", "sentence-transformers与FAISS", "DeepLearning.AI Embeddings课程", "文本相似度搜索系统"],
    ["第11周", "RAG基础", "知识检索+生成架构", "LangChain文档 / OpenAI RAG Cookbook", "RAG问答系统"],
    ["第12周", "LangChain进阶", "ConversationalRetrievalChain", "LangChain教程", "RAG系统记忆功能增强"],
    ["第13周", "工具调用与智能体", "ReAct与Tool-Use机制", "AutoGPT / CrewAI 框架", "Agent具备API调用能力"],
    ["第14周", "多Agent协作", "多个Agent分工协作", "CrewAI / LangGraph", "设计问答+总结双智能体系统"],
    ["第15-16周", "综合实战项目", "企业级RAG智能客服系统", "FastAPI + LangChain + Azure OpenAI", "完整可部署项目Demo"],
    ["第17周", "模型部署", "Docker + FastAPI 部署", "Full Stack Deep Learning Bootcamp", "RAG系统Docker镜像"],
    ["第18周", "加速与优化", "量化、混合精度、缓存机制", "NVIDIA TensorRT / vLLM", "优化推理性能提升20%"],
    ["第19周", "模型监控", "实验追踪与对比", "MLflow / W&B", "训练日志追踪系统"],
    ["第20周", "自动化与CI/CD", "训练/部署流水线", "GitHub Actions / Jenkins", "自动化AI服务"],
    ["第21周", "技术写作与展示", "发布技术博客", "知乎 / Medium / CSDN", "《构建智能RAG系统的经验》"],
    ["第22周", "开源贡献", "贡献PR到开源项目", "GitHub / HF社区", "成功提交PR"],
    ["第23周", "简历与作品集", "项目展示与关键词优化", "Notion / GitHub Pages", "在线Portfolio"],
    ["第24周", "模拟面试与复盘", "AI工程师题 / Kaggle挑战", "LeetCode + HuggingFace Hub", "面试Demo项目准备完毕"]
]

df = pd.DataFrame(data, columns=["周次", "学习主题", "重点任务", "推荐资源", "实践产出"])
df.to_excel("AI学习路线计划表.xlsx", index=False)
print("✅ 已生成文件：AI学习路线计划表.xlsx")
