"""
Week 24: 项目展示与面试准备工具
包括：演示文档生成、技术白皮书模板、面试题库系统

本模块提供项目展示和求职准备的实用工具。
"""

import json
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict
import random


class ProjectShowcase:
    """项目展示对象"""
    
    def __init__(self, project_name: str, project_type: str):
        self.project_name = project_name
        self.project_type = project_type  # research, application, tool
        self.summary = ""
        self.highlights = []
        self.tech_stack = []
        self.architecture = ""
        self.results = {}
        self.demos = []
        self.team_size = 1
        self.duration = ""
        self.created_at = datetime.now()
    
    def set_summary(self, summary: str):
        """设置项目概述"""
        self.summary = summary
    
    def add_highlight(self, highlight: str):
        """添加亮点"""
        self.highlights.append(highlight)
    
    def add_tech(self, tech: str):
        """添加技术栈"""
        if tech not in self.tech_stack:
            self.tech_stack.append(tech)
    
    def set_architecture(self, architecture: str):
        """设置架构说明"""
        self.architecture = architecture
    
    def add_result(self, metric: str, value: str):
        """添加成果"""
        self.results[metric] = value
    
    def add_demo(self, title: str, description: str, url: str = ""):
        """添加演示"""
        self.demos.append({
            'title': title,
            'description': description,
            'url': url
        })
    
    def generate_presentation(self) -> str:
        """生成演示文档"""
        doc = f"# {self.project_name}\n\n"
        doc += f"**项目类型**: {self.project_type}\n"
        doc += f"**团队规模**: {self.team_size}人\n"
        doc += f"**项目周期**: {self.duration}\n\n"
        doc += "---\n\n"
        
        # 项目概述
        doc += "## 项目概述\n\n"
        doc += f"{self.summary}\n\n"
        
        # 核心亮点
        if self.highlights:
            doc += "## 核心亮点\n\n"
            for i, highlight in enumerate(self.highlights, 1):
                doc += f"{i}. ✨ {highlight}\n"
            doc += "\n"
        
        # 技术栈
        if self.tech_stack:
            doc += "## 技术栈\n\n"
            doc += " | ".join(f"`{tech}`" for tech in self.tech_stack)
            doc += "\n\n"
        
        # 架构设计
        if self.architecture:
            doc += "## 系统架构\n\n"
            doc += f"{self.architecture}\n\n"
        
        # 项目成果
        if self.results:
            doc += "## 项目成果\n\n"
            doc += "| 指标 | 结果 |\n"
            doc += "|------|------|\n"
            for metric, value in self.results.items():
                doc += f"| {metric} | {value} |\n"
            doc += "\n"
        
        # 演示
        if self.demos:
            doc += "## 项目演示\n\n"
            for demo in self.demos:
                doc += f"### {demo['title']}\n\n"
                doc += f"{demo['description']}\n\n"
                if demo['url']:
                    doc += f"[查看演示]({demo['url']})\n\n"
        
        return doc
    
    def generate_resume_section(self) -> str:
        """生成简历中的项目描述"""
        section = f"### {self.project_name}\n\n"
        section += f"**技术栈**: {', '.join(self.tech_stack[:5])}\n\n"
        section += f"{self.summary}\n\n"
        
        section += "**主要工作**:\n"
        for highlight in self.highlights[:3]:
            section += f"- {highlight}\n"
        
        if self.results:
            section += "\n**项目成果**:\n"
            for metric, value in list(self.results.items())[:3]:
                section += f"- {metric}: {value}\n"
        
        return section


class TechnicalWhitepaper:
    """技术白皮书"""
    
    def __init__(self, title: str, author: str):
        self.title = title
        self.author = author
        self.abstract = ""
        self.sections = []
        self.references = []
        self.version = "1.0"
        self.date = datetime.now().strftime("%Y-%m-%d")
    
    def set_abstract(self, abstract: str):
        """设置摘要"""
        self.abstract = abstract
    
    def add_section(self, title: str, content: str, level: int = 1):
        """添加章节"""
        self.sections.append({
            'title': title,
            'content': content,
            'level': level
        })
    
    def add_reference(self, citation: str):
        """添加参考文献"""
        self.references.append(citation)
    
    def generate_document(self) -> str:
        """生成白皮书文档"""
        doc = f"# {self.title}\n\n"
        doc += f"**作者**: {self.author}\n"
        doc += f"**版本**: {self.version}\n"
        doc += f"**日期**: {self.date}\n\n"
        doc += "---\n\n"
        
        # 摘要
        doc += "## 摘要\n\n"
        doc += f"{self.abstract}\n\n"
        
        # 目录
        if len(self.sections) > 3:
            doc += "## 目录\n\n"
            for i, section in enumerate(self.sections, 1):
                indent = "  " * (section['level'] - 1)
                doc += f"{indent}{i}. {section['title']}\n"
            doc += "\n---\n\n"
        
        # 正文
        for section in self.sections:
            header = "#" * (section['level'] + 1)
            doc += f"{header} {section['title']}\n\n"
            doc += f"{section['content']}\n\n"
        
        # 参考文献
        if self.references:
            doc += "## 参考文献\n\n"
            for i, ref in enumerate(self.references, 1):
                doc += f"[{i}] {ref}\n"
            doc += "\n"
        
        return doc


class WhitepaperGenerator:
    """白皮书生成器"""
    
    @staticmethod
    def create_system_design(
        system_name: str,
        author: str,
        problem: str,
        solution: str,
        architecture: str,
        results: Dict
    ) -> TechnicalWhitepaper:
        """生成系统设计白皮书"""
        
        wp = TechnicalWhitepaper(f"{system_name}系统设计白皮书", author)
        
        wp.set_abstract(
            f"本白皮书介绍了{system_name}的系统设计理念、技术架构和实现细节。"
            f"该系统旨在解决{problem}。"
        )
        
        # 背景与问题
        wp.add_section(
            "背景与问题",
            f"### 问题描述\n\n{problem}\n\n"
            "### 现有方案的局限性\n\n"
            "- 性能瓶颈\n- 可扩展性差\n- 维护成本高",
            level=1
        )
        
        # 解决方案
        wp.add_section(
            "解决方案",
            f"{solution}",
            level=1
        )
        
        # 系统架构
        wp.add_section(
            "系统架构",
            f"{architecture}\n\n"
            "### 核心组件\n\n"
            "系统由以下核心组件组成：\n\n"
            "- **数据层**: 负责数据存储和访问\n"
            "- **业务层**: 实现核心业务逻辑\n"
            "- **接口层**: 提供外部API接口\n"
            "- **监控层**: 系统监控和日志收集",
            level=1
        )
        
        # 技术选型
        wp.add_section(
            "技术选型",
            "### 编程语言\n\n- Python 3.8+\n\n"
            "### 核心框架\n\n- FastAPI (Web框架)\n- SQLAlchemy (ORM)\n- Redis (缓存)\n\n"
            "### 部署方案\n\n- Docker容器化\n- Kubernetes编排\n- Prometheus监控",
            level=1
        )
        
        # 实现细节
        wp.add_section(
            "关键实现",
            "### 性能优化\n\n"
            "1. 异步处理提升并发能力\n"
            "2. 缓存策略减少数据库压力\n"
            "3. 批处理优化吞吐量\n\n"
            "### 可靠性保障\n\n"
            "1. 健康检查机制\n"
            "2. 断路器模式\n"
            "3. 自动重试策略",
            level=1
        )
        
        # 测试与结果
        results_text = "### 性能测试\n\n"
        results_text += "| 指标 | 结果 |\n|------|------|\n"
        for metric, value in results.items():
            results_text += f"| {metric} | {value} |\n"
        
        wp.add_section(
            "测试与评估",
            results_text,
            level=1
        )
        
        # 总结与展望
        wp.add_section(
            "总结与展望",
            f"{system_name}系统成功解决了{problem}，"
            "在性能、可扩展性和可维护性方面都达到了预期目标。\n\n"
            "未来计划：\n"
            "- 支持更多数据源\n"
            "- 优化算法效率\n"
            "- 增强智能化能力",
            level=1
        )
        
        return wp


class InterviewQuestion:
    """面试题"""
    
    def __init__(self, question: str, category: str, difficulty: str):
        self.question = question
        self.category = category  # theory, coding, system_design, ml_theory
        self.difficulty = difficulty  # easy, medium, hard
        self.answer = ""
        self.key_points = []
        self.code_example = ""
        self.follow_ups = []
        self.tags = []
    
    def set_answer(self, answer: str):
        """设置答案"""
        self.answer = answer
    
    def add_key_point(self, point: str):
        """添加要点"""
        self.key_points.append(point)
    
    def set_code_example(self, code: str):
        """设置代码示例"""
        self.code_example = code
    
    def add_follow_up(self, question: str):
        """添加追问"""
        self.follow_ups.append(question)
    
    def add_tag(self, tag: str):
        """添加标签"""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def to_markdown(self) -> str:
        """转换为Markdown"""
        md = f"## {self.question}\n\n"
        md += f"**分类**: {self.category} | **难度**: {self.difficulty}\n\n"
        
        if self.tags:
            md += f"**标签**: {', '.join(self.tags)}\n\n"
        
        md += "### 答案\n\n"
        md += f"{self.answer}\n\n"
        
        if self.key_points:
            md += "### 关键要点\n\n"
            for point in self.key_points:
                md += f"- {point}\n"
            md += "\n"
        
        if self.code_example:
            md += "### 代码示例\n\n"
            md += f"```python\n{self.code_example}\n```\n\n"
        
        if self.follow_ups:
            md += "### 可能的追问\n\n"
            for i, q in enumerate(self.follow_ups, 1):
                md += f"{i}. {q}\n"
            md += "\n"
        
        return md


class InterviewQuestionBank:
    """面试题库"""
    
    def __init__(self, name: str = "AI工程师面试题库"):
        self.name = name
        self.questions: List[InterviewQuestion] = []
        self.category_index: Dict[str, List[InterviewQuestion]] = defaultdict(list)
        self.difficulty_index: Dict[str, List[InterviewQuestion]] = defaultdict(list)
        self.tag_index: Dict[str, List[InterviewQuestion]] = defaultdict(list)
    
    def add_question(self, question: InterviewQuestion):
        """添加题目"""
        self.questions.append(question)
        self.category_index[question.category].append(question)
        self.difficulty_index[question.difficulty].append(question)
        
        for tag in question.tags:
            self.tag_index[tag].append(question)
    
    def get_by_category(self, category: str) -> List[InterviewQuestion]:
        """按分类获取"""
        return self.category_index.get(category, [])
    
    def get_by_difficulty(self, difficulty: str) -> List[InterviewQuestion]:
        """按难度获取"""
        return self.difficulty_index.get(difficulty, [])
    
    def get_by_tag(self, tag: str) -> List[InterviewQuestion]:
        """按标签获取"""
        return self.tag_index.get(tag, [])
    
    def get_random_questions(self, count: int = 5, difficulty: Optional[str] = None) -> List[InterviewQuestion]:
        """随机获取题目"""
        pool = self.questions if not difficulty else self.get_by_difficulty(difficulty)
        
        if not pool:
            return []
        
        count = min(count, len(pool))
        return random.sample(pool, count)
    
    def create_mock_interview(
        self,
        categories: List[str],
        questions_per_category: int = 2
    ) -> List[InterviewQuestion]:
        """创建模拟面试"""
        mock_questions = []
        
        for category in categories:
            questions = self.get_by_category(category)
            if questions:
                selected = random.sample(
                    questions,
                    min(questions_per_category, len(questions))
                )
                mock_questions.extend(selected)
        
        return mock_questions
    
    def generate_study_guide(self) -> str:
        """生成学习指南"""
        guide = f"# {self.name}\n\n"
        guide += f"总题目数: {len(self.questions)}\n\n"
        guide += "---\n\n"
        
        # 按分类整理
        for category in sorted(self.category_index.keys()):
            questions = self.category_index[category]
            guide += f"## {category} ({len(questions)}题)\n\n"
            
            # 按难度分组
            by_diff = defaultdict(list)
            for q in questions:
                by_diff[q.difficulty].append(q)
            
            for difficulty in ['easy', 'medium', 'hard']:
                if difficulty in by_diff:
                    guide += f"### {difficulty.upper()} ({len(by_diff[difficulty])}题)\n\n"
                    for i, q in enumerate(by_diff[difficulty], 1):
                        guide += f"{i}. {q.question}\n"
                    guide += "\n"
        
        return guide
    
    def export_to_json(self, filename: str):
        """导出为JSON"""
        data = {
            'name': self.name,
            'total_questions': len(self.questions),
            'categories': list(self.category_index.keys()),
            'questions': [
                {
                    'question': q.question,
                    'category': q.category,
                    'difficulty': q.difficulty,
                    'answer': q.answer,
                    'key_points': q.key_points,
                    'tags': q.tags
                }
                for q in self.questions
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def demonstrate_project_showcase():
    """演示项目展示"""
    print("\n" + "="*70)
    print("🎬 演示：项目展示文档生成")
    print("="*70)
    
    # 创建项目展示
    project = ProjectShowcase("智能客服RAG系统", "application")
    project.team_size = 3
    project.duration = "3个月"
    
    project.set_summary(
        "基于RAG技术的企业级智能客服系统，支持多轮对话、知识检索和个性化回答。"
    )
    
    print("\n【添加项目信息】")
    
    # 添加亮点
    project.add_highlight("实现了混合检索策略，召回率提升30%")
    project.add_highlight("支持10000+ QPS的高并发访问")
    project.add_highlight("平均响应延迟低于100ms")
    print("  ✅ 添加核心亮点")
    
    # 添加技术栈
    tech_stack = ["Python", "FastAPI", "LangChain", "FAISS", "Redis", "Docker"]
    for tech in tech_stack:
        project.add_tech(tech)
    print(f"  ✅ 添加技术栈: {', '.join(tech_stack)}")
    
    # 添加架构
    project.set_architecture(
        "系统采用微服务架构，包括：\n"
        "- API网关层：负载均衡和请求路由\n"
        "- 业务服务层：对话管理、知识检索\n"
        "- 数据存储层：向量数据库、关系数据库\n"
        "- 监控层：实时性能监控和告警"
    )
    print("  ✅ 添加架构说明")
    
    # 添加成果
    project.add_result("用户满意度", "92%")
    project.add_result("问题解决率", "85%")
    project.add_result("系统可用性", "99.9%")
    print("  ✅ 添加项目成果")
    
    # 生成演示文档
    print("\n【生成演示文档】")
    presentation = project.generate_presentation()
    print(presentation[:600] + "...")
    
    # 生成简历部分
    print("\n【生成简历描述】")
    resume_section = project.generate_resume_section()
    print(resume_section)


def demonstrate_whitepaper():
    """演示技术白皮书"""
    print("\n" + "="*70)
    print("📋 演示：技术白皮书生成")
    print("="*70)
    
    print("\n【生成系统设计白皮书】")
    
    wp = WhitepaperGenerator.create_system_design(
        system_name="RAG增强型对话",
        author="张三",
        problem="传统对话系统无法准确回答领域知识问题",
        solution="采用RAG技术，结合向量检索和大语言模型，实现知识增强的对话生成",
        architecture="系统采用三层架构：检索层、增强层和生成层",
        results={
            "平均延迟": "85ms",
            "准确率": "89%",
            "并发能力": "5000 QPS"
        }
    )
    
    print(f"  ✅ 创建白皮书: {wp.title}")
    print(f"  ✅ 版本: {wp.version}")
    print(f"  ✅ 章节数: {len(wp.sections)}")
    
    # 生成文档
    print("\n【白皮书预览】")
    document = wp.generate_document()
    print(document[:800] + "...")


def demonstrate_interview_bank():
    """演示面试题库"""
    print("\n" + "="*70)
    print("💼 演示：面试题库系统")
    print("="*70)
    
    bank = InterviewQuestionBank()
    
    print("\n【添加面试题目】")
    
    # 添加理论题
    q1 = InterviewQuestion(
        "解释Transformer中的Self-Attention机制",
        category="ml_theory",
        difficulty="medium"
    )
    q1.set_answer(
        "Self-Attention允许模型关注输入序列的不同位置。"
        "通过Query、Key、Value三个矩阵计算注意力权重，"
        "实现序列内部元素之间的关联建模。"
    )
    q1.add_key_point("Q、K、V矩阵的计算")
    q1.add_key_point("Scaled Dot-Product Attention")
    q1.add_key_point("Multi-Head机制")
    q1.add_tag("Transformer")
    q1.add_tag("注意力机制")
    q1.add_follow_up("为什么需要除以sqrt(d_k)？")
    
    bank.add_question(q1)
    print("  ✅ 添加理论题: Transformer")
    
    # 添加编码题
    q2 = InterviewQuestion(
        "实现一个简单的Beam Search",
        category="coding",
        difficulty="hard"
    )
    q2.set_answer("Beam Search是一种贪心搜索算法，保留top-k个候选序列。")
    q2.set_code_example("""
def beam_search(model, start_token, beam_size=3, max_len=20):
    beams = [(start_token, 0)]  # (sequence, score)
    
    for _ in range(max_len):
        candidates = []
        for seq, score in beams:
            if seq[-1] == END_TOKEN:
                candidates.append((seq, score))
                continue
            
            # 获取下一个token的概率
            probs = model.predict(seq)
            top_k = probs.argsort()[-beam_size:]
            
            for token in top_k:
                new_seq = seq + [token]
                new_score = score + np.log(probs[token])
                candidates.append((new_seq, new_score))
        
        # 选择top-k
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
    
    return beams[0][0]
""")
    q2.add_key_point("维护beam_size个候选")
    q2.add_key_point("每步扩展所有候选")
    q2.add_key_point("按分数排序选择top-k")
    q2.add_tag("NLP")
    q2.add_tag("搜索算法")
    
    bank.add_question(q2)
    print("  ✅ 添加编码题: Beam Search")
    
    # 添加系统设计题
    q3 = InterviewQuestion(
        "设计一个支持10000 QPS的LLM推理系统",
        category="system_design",
        difficulty="hard"
    )
    q3.set_answer(
        "需要考虑：\n"
        "1. 负载均衡：使用Nginx进行请求分发\n"
        "2. 模型服务：多GPU并行推理\n"
        "3. 缓存策略：Redis缓存热门query\n"
        "4. 批处理：Dynamic Batching提升吞吐\n"
        "5. 监控告警：Prometheus + Grafana"
    )
    q3.add_key_point("水平扩展")
    q3.add_key_point("异步处理")
    q3.add_key_point("缓存优化")
    q3.add_tag("系统设计")
    q3.add_tag("高并发")
    
    bank.add_question(q3)
    print("  ✅ 添加系统设计题: LLM推理系统")
    
    # 统计
    print("\n【题库统计】")
    print(f"  总题目数: {len(bank.questions)}")
    print(f"  分类: {list(bank.category_index.keys())}")
    print(f"  难度分布: {dict([(k, len(v)) for k, v in bank.difficulty_index.items()])}")
    
    # 模拟面试
    print("\n【生成模拟面试】")
    mock = bank.create_mock_interview(
        categories=["ml_theory", "coding", "system_design"],
        questions_per_category=1
    )
    
    print(f"\n本次模拟面试共{len(mock)}题：")
    for i, q in enumerate(mock, 1):
        print(f"  {i}. [{q.category}] {q.question}")
    
    # 生成学习指南
    print("\n【学习指南】")
    guide = bank.generate_study_guide()
    print(guide)


def run_week24_demo():
    """运行Week 24完整演示"""
    print("\n" + "="*70)
    print("🚀 Week 24: 项目展示与面试准备 - 完整演示")
    print("="*70)
    
    # 1. 项目展示
    demonstrate_project_showcase()
    
    input("\n按Enter继续查看技术白皮书...")
    
    # 2. 技术白皮书
    demonstrate_whitepaper()
    
    input("\n按Enter继续查看面试题库...")
    
    # 3. 面试题库
    demonstrate_interview_bank()
    
    print("\n" + "="*70)
    print("✅ Week 24演示完成！")
    print("="*70)
    print("\n核心收获：")
    print("  1. 掌握了项目展示文档的制作")
    print("  2. 学会了技术白皮书的撰写")
    print("  3. 建立了系统的面试题库")
    print("  4. 完全具备了求职和项目展示能力")
    print("\n🎓 恭喜！完成了全部24周的AI学习旅程！")


if __name__ == "__main__":
    run_week24_demo()
