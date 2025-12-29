"""
Week 22-23: 知识管理与文档生成工具
包括：技术文档生成、知识图谱构建、学习笔记管理

本模块提供知识体系构建和技术输出的实用工具。
"""

import json
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict, Counter
import re


class TechDocument:
    """技术文档对象"""
    
    def __init__(self, title: str, category: str = "General"):
        self.title = title
        self.category = category
        self.sections = []
        self.tags = []
        self.references = []
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def add_section(self, title: str, content: str, level: int = 2):
        """添加章节"""
        self.sections.append({
            'title': title,
            'content': content,
            'level': level
        })
        self.updated_at = datetime.now()
    
    def add_code_block(self, code: str, language: str = "python"):
        """添加代码块"""
        content = f"```{language}\n{code}\n```"
        if self.sections:
            self.sections[-1]['content'] += "\n\n" + content
        self.updated_at = datetime.now()
    
    def add_tag(self, tag: str):
        """添加标签"""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def add_reference(self, title: str, url: str):
        """添加参考资料"""
        self.references.append({'title': title, 'url': url})
    
    def generate_markdown(self) -> str:
        """生成Markdown文档"""
        md = f"# {self.title}\n\n"
        md += f"**分类**: {self.category}\n\n"
        
        if self.tags:
            md += f"**标签**: {', '.join(self.tags)}\n\n"
        
        md += f"**创建时间**: {self.created_at.strftime('%Y-%m-%d')}\n"
        md += f"**更新时间**: {self.updated_at.strftime('%Y-%m-%d')}\n\n"
        md += "---\n\n"
        
        # 目录
        if len(self.sections) > 3:
            md += "## 目录\n\n"
            for section in self.sections:
                indent = "  " * (section['level'] - 2)
                md += f"{indent}- [{section['title']}](#{section['title'].lower().replace(' ', '-')})\n"
            md += "\n"
        
        # 内容
        for section in self.sections:
            header = "#" * section['level']
            md += f"{header} {section['title']}\n\n"
            md += f"{section['content']}\n\n"
        
        # 参考资料
        if self.references:
            md += "## 参考资料\n\n"
            for i, ref in enumerate(self.references, 1):
                md += f"{i}. [{ref['title']}]({ref['url']})\n"
            md += "\n"
        
        return md


class DocumentGenerator:
    """技术文档生成器"""
    
    @staticmethod
    def create_api_documentation(
        api_name: str,
        endpoints: List[Dict]
    ) -> TechDocument:
        """生成API文档"""
        doc = TechDocument(f"{api_name} API文档", "API")
        doc.add_tag("API")
        doc.add_tag("文档")
        
        # 概述
        doc.add_section(
            "概述",
            f"{api_name}是一个RESTful API服务，提供以下功能：\n\n" +
            "\n".join([f"- {ep['description']}" for ep in endpoints])
        )
        
        # 基础信息
        doc.add_section(
            "基础信息",
            "- **Base URL**: `https://api.example.com/v1`\n"
            "- **认证方式**: Bearer Token\n"
            "- **数据格式**: JSON"
        )
        
        # 端点详情
        doc.add_section("API端点", "")
        
        for endpoint in endpoints:
            method = endpoint['method']
            path = endpoint['path']
            desc = endpoint['description']
            
            doc.add_section(
                f"{method} {path}",
                f"**描述**: {desc}\n\n" +
                "**请求参数**:\n\n" +
                "```json\n" + json.dumps(endpoint.get('params', {}), indent=2) + "\n```\n\n" +
                "**响应示例**:\n\n" +
                "```json\n" + json.dumps(endpoint.get('response', {}), indent=2) + "\n```",
                level=3
            )
        
        # 错误码
        doc.add_section(
            "错误码",
            "| 状态码 | 说明 |\n"
            "|--------|------|\n"
            "| 200 | 成功 |\n"
            "| 400 | 请求参数错误 |\n"
            "| 401 | 未授权 |\n"
            "| 429 | 请求过于频繁 |\n"
            "| 500 | 服务器内部错误 |"
        )
        
        return doc
    
    @staticmethod
    def create_tutorial(
        title: str,
        steps: List[Dict]
    ) -> TechDocument:
        """生成教程文档"""
        doc = TechDocument(title, "教程")
        doc.add_tag("教程")
        
        # 简介
        doc.add_section(
            "简介",
            f"本教程将指导你完成{title}的实践。"
        )
        
        # 前置要求
        doc.add_section(
            "前置要求",
            "- Python 3.8+\n"
            "- 基础的机器学习知识\n"
            "- NumPy和PyTorch基础"
        )
        
        # 步骤
        for i, step in enumerate(steps, 1):
            doc.add_section(
                f"步骤 {i}: {step['title']}",
                step['description'],
                level=2
            )
            
            if 'code' in step:
                doc.add_code_block(step['code'], step.get('language', 'python'))
        
        # 总结
        doc.add_section(
            "总结",
            f"通过本教程，你已经学会了{title}的核心概念和实践方法。"
        )
        
        return doc
    
    @staticmethod
    def create_comparison(
        title: str,
        items: List[Dict]
    ) -> TechDocument:
        """生成对比文档"""
        doc = TechDocument(title, "对比分析")
        doc.add_tag("对比")
        
        # 概述
        doc.add_section(
            "概述",
            f"本文对比分析了{len(items)}种不同的方案。"
        )
        
        # 对比表格
        headers = ["特性"] + [item['name'] for item in items]
        table = "| " + " | ".join(headers) + " |\n"
        table += "|" + "|".join(["---"] * len(headers)) + "|\n"
        
        # 收集所有特性
        all_features = set()
        for item in items:
            all_features.update(item.get('features', {}).keys())
        
        for feature in sorted(all_features):
            row = [feature]
            for item in items:
                value = item.get('features', {}).get(feature, "N/A")
                row.append(str(value))
            table += "| " + " | ".join(row) + " |\n"
        
        doc.add_section("特性对比", table)
        
        # 详细分析
        doc.add_section("详细分析", "")
        
        for item in items:
            pros = item.get('pros', [])
            cons = item.get('cons', [])
            
            content = ""
            if pros:
                content += "**优点**:\n" + "\n".join([f"- {p}" for p in pros]) + "\n\n"
            if cons:
                content += "**缺点**:\n" + "\n".join([f"- {c}" for c in cons])
            
            doc.add_section(item['name'], content, level=3)
        
        # 推荐
        if items:
            doc.add_section(
                "推荐方案",
                f"综合考虑各方面因素，推荐使用 **{items[0]['name']}**。"
            )
        
        return doc


class KnowledgeNode:
    """知识节点"""
    
    def __init__(self, id: str, title: str, content: str = "", node_type: str = "concept"):
        self.id = id
        self.title = title
        self.content = content
        self.node_type = node_type  # concept, skill, project, reference
        self.tags = []
        self.links = []  # 相关节点ID
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def add_tag(self, tag: str):
        """添加标签"""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def link_to(self, node_id: str):
        """链接到其他节点"""
        if node_id not in self.links:
            self.links.append(node_id)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'node_type': self.node_type,
            'tags': self.tags,
            'links': self.links,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


class KnowledgeGraph:
    """知识图谱"""
    
    def __init__(self, name: str):
        self.name = name
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.tags_index: Dict[str, Set[str]] = defaultdict(set)
    
    def add_node(self, node: KnowledgeNode):
        """添加节点"""
        self.nodes[node.id] = node
        
        # 更新标签索引
        for tag in node.tags:
            self.tags_index[tag].add(node.id)
    
    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """获取节点"""
        return self.nodes.get(node_id)
    
    def search_by_tag(self, tag: str) -> List[KnowledgeNode]:
        """按标签搜索"""
        node_ids = self.tags_index.get(tag, set())
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]
    
    def search_by_keyword(self, keyword: str) -> List[KnowledgeNode]:
        """按关键词搜索"""
        results = []
        keyword_lower = keyword.lower()
        
        for node in self.nodes.values():
            if (keyword_lower in node.title.lower() or
                keyword_lower in node.content.lower()):
                results.append(node)
        
        return results
    
    def get_related_nodes(self, node_id: str, depth: int = 1) -> List[KnowledgeNode]:
        """获取相关节点"""
        if node_id not in self.nodes:
            return []
        
        related = []
        visited = set()
        queue = [(node_id, 0)]
        
        while queue:
            current_id, current_depth = queue.pop(0)
            
            if current_id in visited or current_depth > depth:
                continue
            
            visited.add(current_id)
            
            if current_id != node_id:
                related.append(self.nodes[current_id])
            
            if current_depth < depth:
                node = self.nodes[current_id]
                for link_id in node.links:
                    if link_id in self.nodes:
                        queue.append((link_id, current_depth + 1))
        
        return related
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        type_counts = Counter(node.node_type for node in self.nodes.values())
        tag_counts = Counter(tag for node in self.nodes.values() for tag in node.tags)
        
        return {
            'total_nodes': len(self.nodes),
            'node_types': dict(type_counts),
            'total_tags': len(self.tags_index),
            'top_tags': tag_counts.most_common(10),
            'avg_links': sum(len(n.links) for n in self.nodes.values()) / len(self.nodes) if self.nodes else 0
        }
    
    def export_to_json(self, filename: str):
        """导出为JSON"""
        data = {
            'name': self.name,
            'nodes': [node.to_dict() for node in self.nodes.values()]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def generate_visualization(self) -> str:
        """生成可视化代码（Mermaid格式）"""
        mermaid = "```mermaid\ngraph TD\n"
        
        # 添加节点
        for node_id, node in self.nodes.items():
            label = node.title.replace('"', "'")
            shape = {
                'concept': f'["{label}"]',
                'skill': f'("{label}")',
                'project': f'{{"{label}"}}',
                'reference': f'[("{label}")]'
            }.get(node.node_type, f'["{label}"]')
            
            mermaid += f"  {node_id}{shape}\n"
        
        # 添加连接
        for node_id, node in self.nodes.items():
            for link_id in node.links:
                if link_id in self.nodes:
                    mermaid += f"  {node_id} --> {link_id}\n"
        
        mermaid += "```\n"
        return mermaid


class LearningNote:
    """学习笔记"""
    
    def __init__(self, title: str, topic: str):
        self.title = title
        self.topic = topic
        self.content = []
        self.tags = []
        self.questions = []
        self.resources = []
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def add_content(self, content: str, content_type: str = "text"):
        """添加内容"""
        self.content.append({
            'type': content_type,
            'data': content,
            'timestamp': datetime.now().isoformat()
        })
        self.updated_at = datetime.now()
    
    def add_question(self, question: str, answer: str = ""):
        """添加问题"""
        self.questions.append({
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        })
    
    def add_resource(self, title: str, url: str, resource_type: str = "link"):
        """添加资源"""
        self.resources.append({
            'title': title,
            'url': url,
            'type': resource_type
        })
    
    def generate_markdown(self) -> str:
        """生成Markdown笔记"""
        md = f"# {self.title}\n\n"
        md += f"**主题**: {self.topic}\n"
        md += f"**日期**: {self.created_at.strftime('%Y-%m-%d')}\n\n"
        
        if self.tags:
            md += f"**标签**: {', '.join(self.tags)}\n\n"
        
        md += "---\n\n"
        
        # 内容
        md += "## 学习内容\n\n"
        for item in self.content:
            if item['type'] == 'text':
                md += f"{item['data']}\n\n"
            elif item['type'] == 'code':
                md += f"```python\n{item['data']}\n```\n\n"
        
        # 问题
        if self.questions:
            md += "## 问题与思考\n\n"
            for i, q in enumerate(self.questions, 1):
                md += f"### Q{i}: {q['question']}\n\n"
                if q['answer']:
                    md += f"**A**: {q['answer']}\n\n"
                else:
                    md += "**A**: _待解答_\n\n"
        
        # 资源
        if self.resources:
            md += "## 参考资源\n\n"
            for res in self.resources:
                md += f"- [{res['title']}]({res['url']}) ({res['type']})\n"
            md += "\n"
        
        return md


class NoteManager:
    """笔记管理器"""
    
    def __init__(self):
        self.notes: List[LearningNote] = []
        self.topics_index: Dict[str, List[LearningNote]] = defaultdict(list)
    
    def add_note(self, note: LearningNote):
        """添加笔记"""
        self.notes.append(note)
        self.topics_index[note.topic].append(note)
    
    def get_notes_by_topic(self, topic: str) -> List[LearningNote]:
        """按主题获取笔记"""
        return self.topics_index.get(topic, [])
    
    def search_notes(self, keyword: str) -> List[LearningNote]:
        """搜索笔记"""
        results = []
        keyword_lower = keyword.lower()
        
        for note in self.notes:
            if (keyword_lower in note.title.lower() or
                keyword_lower in note.topic.lower()):
                results.append(note)
        
        return results
    
    def generate_index(self) -> str:
        """生成笔记索引"""
        index = "# 学习笔记索引\n\n"
        
        # 按主题分组
        for topic in sorted(self.topics_index.keys()):
            index += f"## {topic}\n\n"
            notes = self.topics_index[topic]
            
            for note in notes:
                date = note.created_at.strftime('%Y-%m-%d')
                index += f"- [{note.title}]({note.title.replace(' ', '_')}.md) - {date}\n"
            
            index += "\n"
        
        return index


def demonstrate_document_generation():
    """演示文档生成"""
    print("\n" + "="*70)
    print("📝 演示：技术文档生成")
    print("="*70)
    
    # 生成API文档
    print("\n【生成API文档】")
    
    endpoints = [
        {
            'method': 'POST',
            'path': '/chat/completions',
            'description': '生成聊天回复',
            'params': {
                'messages': [{'role': 'user', 'content': 'Hello'}],
                'model': 'gpt-3.5-turbo',
                'temperature': 0.7
            },
            'response': {
                'id': 'chatcmpl-xxx',
                'choices': [{'message': {'role': 'assistant', 'content': 'Hi!'}}]
            }
        },
        {
            'method': 'GET',
            'path': '/models',
            'description': '获取可用模型列表',
            'params': {},
            'response': {
                'models': ['gpt-3.5-turbo', 'gpt-4']
            }
        }
    ]
    
    api_doc = DocumentGenerator.create_api_documentation("ChatGPT", endpoints)
    print(f"✅ 生成API文档: {api_doc.title}")
    print(f"   章节数: {len(api_doc.sections)}")
    
    # 生成教程
    print("\n【生成教程文档】")
    
    steps = [
        {
            'title': '安装依赖',
            'description': '首先安装必要的Python包：',
            'code': 'pip install transformers torch'
        },
        {
            'title': '加载模型',
            'description': '使用Transformers库加载预训练模型：',
            'code': 'from transformers import AutoModel\nmodel = AutoModel.from_pretrained("bert-base-uncased")'
        },
        {
            'title': '推理测试',
            'description': '运行推理测试：',
            'code': 'output = model(input_ids)\nprint(output.shape)'
        }
    ]
    
    tutorial = DocumentGenerator.create_tutorial("BERT模型使用指南", steps)
    print(f"✅ 生成教程: {tutorial.title}")
    print(f"   步骤数: {len(steps)}")
    
    # 预览文档
    print("\n【文档预览】")
    markdown = api_doc.generate_markdown()
    print(markdown[:400] + "...")


def demonstrate_knowledge_graph():
    """演示知识图谱"""
    print("\n" + "="*70)
    print("🧠 演示：知识图谱构建")
    print("="*70)
    
    # 创建知识图谱
    kg = KnowledgeGraph("AI学习知识图谱")
    
    print("\n【构建知识图谱】")
    
    # 添加概念节点
    transformer = KnowledgeNode(
        "transformer",
        "Transformer架构",
        "基于自注意力机制的神经网络架构",
        "concept"
    )
    transformer.add_tag("深度学习")
    transformer.add_tag("NLP")
    kg.add_node(transformer)
    print("  ✅ 添加节点: Transformer")
    
    attention = KnowledgeNode(
        "attention",
        "注意力机制",
        "允许模型关注输入的不同部分",
        "concept"
    )
    attention.add_tag("深度学习")
    kg.add_node(attention)
    print("  ✅ 添加节点: 注意力机制")
    
    bert = KnowledgeNode(
        "bert",
        "BERT模型",
        "双向Transformer预训练模型",
        "concept"
    )
    bert.add_tag("NLP")
    bert.add_tag("预训练模型")
    kg.add_node(bert)
    print("  ✅ 添加节点: BERT")
    
    # 添加技能节点
    finetune = KnowledgeNode(
        "finetune",
        "模型微调",
        "在预训练模型基础上进行任务特定训练",
        "skill"
    )
    finetune.add_tag("实践技能")
    kg.add_node(finetune)
    print("  ✅ 添加节点: 模型微调")
    
    # 建立连接
    transformer.link_to("attention")
    bert.link_to("transformer")
    finetune.link_to("bert")
    print("\n  ✅ 建立节点连接")
    
    # 统计
    print("\n【知识图谱统计】")
    stats = kg.get_statistics()
    print(f"  总节点数: {stats['total_nodes']}")
    print(f"  节点类型: {stats['node_types']}")
    print(f"  平均连接数: {stats['avg_links']:.2f}")
    
    # 搜索
    print("\n【按标签搜索: NLP】")
    results = kg.search_by_tag("NLP")
    for node in results:
        print(f"  - {node.title}")
    
    # 相关节点
    print("\n【查找相关节点: BERT】")
    related = kg.get_related_nodes("bert", depth=2)
    for node in related:
        print(f"  - {node.title} ({node.node_type})")


def demonstrate_note_management():
    """演示笔记管理"""
    print("\n" + "="*70)
    print("📚 演示:学习笔记管理")
    print("="*70)
    
    manager = NoteManager()
    
    print("\n【创建学习笔记】")
    
    # 创建笔记1
    note1 = LearningNote("Transformer原理解析", "深度学习")
    note1.tags = ["Transformer", "注意力机制"]
    note1.add_content("Transformer是一种基于注意力机制的架构。")
    note1.add_content("核心组件包括：Multi-Head Attention, Position Encoding, Feed-Forward")
    note1.add_question("为什么需要Position Encoding？", "因为Attention本身无法感知位置信息")
    note1.add_resource("原始论文", "https://arxiv.org/abs/1706.03762", "paper")
    
    manager.add_note(note1)
    print(f"  ✅ {note1.title}")
    
    # 创建笔记2
    note2 = LearningNote("BERT微调实战", "NLP应用")
    note2.tags = ["BERT", "微调", "实战"]
    note2.add_content("BERT微调步骤：\n1. 加载预训练模型\n2. 添加任务层\n3. 训练")
    note2.add_content("model = BertForSequenceClassification.from_pretrained('bert-base')", "code")
    note2.add_question("如何选择学习率？", "建议使用2e-5到5e-5之间")
    note2.add_resource("Hugging Face文档", "https://huggingface.co/docs", "tutorial")
    
    manager.add_note(note2)
    print(f"  ✅ {note2.title}")
    
    # 搜索
    print("\n【搜索笔记: BERT】")
    results = manager.search_notes("BERT")
    for note in results:
        print(f"  - {note.title} ({note.topic})")
    
    # 生成索引
    print("\n【生成笔记索引】")
    index = manager.generate_index()
    print(index)
    
    # 生成Markdown
    print("\n【笔记预览】")
    markdown = note1.generate_markdown()
    print(markdown[:500] + "...")


def run_week22_23_demo():
    """运行Week 22-23完整演示"""
    print("\n" + "="*70)
    print("🚀 Week 22-23: 知识管理与文档生成 - 完整演示")
    print("="*70)
    
    # 1. 文档生成
    demonstrate_document_generation()
    
    input("\n按Enter继续查看知识图谱...")
    
    # 2. 知识图谱
    demonstrate_knowledge_graph()
    
    input("\n按Enter继续查看笔记管理...")
    
    # 3. 笔记管理
    demonstrate_note_management()
    
    print("\n" + "="*70)
    print("✅ Week 22-23演示完成！")
    print("="*70)
    print("\n核心收获：")
    print("  1. 掌握了技术文档的自动化生成")
    print("  2. 学会了构建个人知识图谱")
    print("  3. 建立了系统化的笔记管理方法")
    print("  4. 可以高效地整理和输出知识")


if __name__ == "__main__":
    run_week22_23_demo()
