"""
第15-24周：综合模块导入
所有模块已完整实现 ✅
包含：RAG优化、AI Agent、服务化、监控、论文工具、性能优化、知识管理、项目展示
"""

# Week 15-18: 企业级AI应用
try:
    from advanced.week15_rag_optimization import run_week15_demo as week15_rag_optimization
    print("✅ Week 15模块加载成功 - RAG Pipeline优化")
except ImportError as e:
    print(f"⚠️ Week 15模块导入失败: {e}")
    def week15_rag_optimization():
        print("⚠️ Week 15模块未找到")

try:
    from advanced.week16_ai_agent import run_week16_demo as week16_ai_agent
    print("✅ Week 16模块加载成功 - AI Agent架构")
except ImportError as e:
    print(f"⚠️ Week 16模块导入失败: {e}")
    def week16_ai_agent():
        print("⚠️ Week 16模块未找到")

try:
    from advanced.week17_fastapi_service import run_week17_demo as week17_llm_service
    print("✅ Week 17模块加载成功 - FastAPI服务化")
except ImportError as e:
    print(f"⚠️ Week 17模块导入失败: {e}")
    def week17_llm_service():
        print("⚠️ Week 17模块未找到")

try:
    from advanced.week18_monitoring import run_week18_demo as week18_monitoring
    print("✅ Week 18模块加载成功 - 系统监控")
except ImportError as e:
    print(f"⚠️ Week 18模块导入失败: {e}")
    def week18_monitoring():
        print("⚠️ Week 18模块未找到")

# Week 19-24: 系统化输出与科研化思维
try:
    from advanced.week19_20_research_tools import run_week19_20_demo as week19_20_research_tools
    print("✅ Week 19-20模块加载成功 - 论文复现与实验管理")
except ImportError as e:
    print(f"⚠️ Week 19-20模块导入失败: {e}")
    def week19_20_research_tools():
        print("⚠️ Week 19-20模块未找到")

try:
    from advanced.week21_optimization import run_week21_demo as week21_optimization
    print("✅ Week 21模块加载成功 - GPU性能优化与成本评估")
except ImportError as e:
    print(f"⚠️ Week 21模块导入失败: {e}")
    def week21_optimization():
        print("⚠️ Week 21模块未找到")

try:
    from advanced.week22_23_knowledge_management import run_week22_23_demo as week22_23_knowledge_management
    print("✅ Week 22-23模块加载成功 - 知识管理与文档生成")
except ImportError as e:
    print(f"⚠️ Week 22-23模块导入失败: {e}")
    def week22_23_knowledge_management():
        print("⚠️ Week 22-23模块未找到")

try:
    from advanced.week24_presentation import run_week24_demo as week24_presentation
    print("✅ Week 24模块加载成功 - 项目展示与面试准备")
except ImportError as e:
    print(f"⚠️ Week 24模块导入失败: {e}")
    def week24_presentation():
        print("⚠️ Week 24模块未找到")
