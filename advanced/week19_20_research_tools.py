"""
Week 19-20: 论文复现与实验管理工具
包括：论文阅读助手、实验追踪、性能基准测试

本模块提供学术研究和开源贡献所需的实用工具。
"""

import json
import time
from typing import List, Dict, Optional, Any
from datetime import datetime
from collections import defaultdict
import numpy as np


class Paper:
    """论文对象"""
    
    def __init__(self, title: str, authors: List[str], year: int, 
                 venue: str, url: str = ""):
        self.title = title
        self.authors = authors
        self.year = year
        self.venue = venue
        self.url = url
        self.notes = []
        self.tags = []
        self.key_contributions = []
        self.code_repos = []
    
    def add_note(self, note: str):
        """添加笔记"""
        self.notes.append({
            'content': note,
            'timestamp': datetime.now().isoformat()
        })
    
    def add_tag(self, tag: str):
        """添加标签"""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def add_contribution(self, contribution: str):
        """添加关键贡献"""
        self.key_contributions.append(contribution)
    
    def add_code_repo(self, repo_url: str):
        """添加代码仓库"""
        self.code_repos.append(repo_url)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'title': self.title,
            'authors': self.authors,
            'year': self.year,
            'venue': self.venue,
            'url': self.url,
            'notes': self.notes,
            'tags': self.tags,
            'key_contributions': self.key_contributions,
            'code_repos': self.code_repos
        }
    
    def generate_citation(self, style: str = 'bibtex') -> str:
        """生成引用"""
        if style == 'bibtex':
            author_str = ' and '.join(self.authors)
            return f"""@inproceedings{{{self.title.replace(' ', '_')}_{self.year},
    title = {{{self.title}}},
    author = {{{author_str}}},
    booktitle = {{{self.venue}}},
    year = {{{self.year}}},
    url = {{{self.url}}}
}}"""
        elif style == 'apa':
            authors_str = ', '.join(self.authors)
            return f"{authors_str} ({self.year}). {self.title}. In {self.venue}."
        else:
            return f"{self.authors[0]} et al. ({self.year}). {self.title}"


class PaperLibrary:
    """论文库管理"""
    
    def __init__(self):
        self.papers: List[Paper] = []
        self.tags_index: Dict[str, List[Paper]] = defaultdict(list)
    
    def add_paper(self, paper: Paper):
        """添加论文"""
        self.papers.append(paper)
        for tag in paper.tags:
            self.tags_index[tag].append(paper)
    
    def search_by_tag(self, tag: str) -> List[Paper]:
        """按标签搜索"""
        return self.tags_index.get(tag, [])
    
    def search_by_keyword(self, keyword: str) -> List[Paper]:
        """按关键词搜索"""
        results = []
        keyword_lower = keyword.lower()
        
        for paper in self.papers:
            if (keyword_lower in paper.title.lower() or
                any(keyword_lower in note['content'].lower() for note in paper.notes)):
                results.append(paper)
        
        return results
    
    def get_papers_by_year(self, year: int) -> List[Paper]:
        """按年份获取论文"""
        return [p for p in self.papers if p.year == year]
    
    def get_papers_by_venue(self, venue: str) -> List[Paper]:
        """按会议/期刊获取论文"""
        return [p for p in self.papers if venue.lower() in p.venue.lower()]
    
    def export_bibliography(self, filename: str, style: str = 'bibtex'):
        """导出参考文献"""
        with open(filename, 'w', encoding='utf-8') as f:
            for paper in self.papers:
                f.write(paper.generate_citation(style) + '\n\n')
    
    def generate_reading_list(self) -> str:
        """生成阅读清单"""
        output = "# 论文阅读清单\n\n"
        
        # 按年份分组
        papers_by_year = defaultdict(list)
        for paper in self.papers:
            papers_by_year[paper.year].append(paper)
        
        for year in sorted(papers_by_year.keys(), reverse=True):
            output += f"## {year}年\n\n"
            for paper in papers_by_year[year]:
                output += f"### {paper.title}\n"
                output += f"- **作者**: {', '.join(paper.authors)}\n"
                output += f"- **会议**: {paper.venue}\n"
                if paper.tags:
                    output += f"- **标签**: {', '.join(paper.tags)}\n"
                if paper.key_contributions:
                    output += f"- **关键贡献**:\n"
                    for contrib in paper.key_contributions:
                        output += f"  - {contrib}\n"
                output += "\n"
        
        return output


class Experiment:
    """实验对象"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.hyperparameters = {}
        self.metrics = {}
        self.start_time = None
        self.end_time = None
        self.status = "pending"  # pending, running, completed, failed
        self.logs = []
        self.artifacts = []
    
    def set_hyperparameters(self, params: Dict):
        """设置超参数"""
        self.hyperparameters.update(params)
    
    def start(self):
        """开始实验"""
        self.start_time = datetime.now()
        self.status = "running"
        self.log(f"实验开始: {self.name}")
    
    def complete(self):
        """完成实验"""
        self.end_time = datetime.now()
        self.status = "completed"
        duration = (self.end_time - self.start_time).total_seconds()
        self.log(f"实验完成，耗时: {duration:.2f}秒")
    
    def fail(self, error: str):
        """实验失败"""
        self.end_time = datetime.now()
        self.status = "failed"
        self.log(f"实验失败: {error}")
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """记录指标"""
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append({
            'value': value,
            'step': step,
            'timestamp': datetime.now().isoformat()
        })
    
    def log(self, message: str):
        """记录日志"""
        self.logs.append({
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
    
    def add_artifact(self, name: str, path: str):
        """添加输出文件"""
        self.artifacts.append({
            'name': name,
            'path': path,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_summary(self) -> Dict:
        """获取摘要"""
        summary = {
            'name': self.name,
            'description': self.description,
            'status': self.status,
            'hyperparameters': self.hyperparameters,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
        }
        
        # 添加最终指标
        if self.metrics:
            final_metrics = {}
            for name, values in self.metrics.items():
                if values:
                    final_metrics[name] = values[-1]['value']
            summary['final_metrics'] = final_metrics
        
        return summary


class ExperimentTracker:
    """实验追踪器"""
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.experiments: List[Experiment] = []
        self.current_experiment: Optional[Experiment] = None
    
    def create_experiment(self, name: str, description: str = "") -> Experiment:
        """创建新实验"""
        exp = Experiment(name, description)
        self.experiments.append(exp)
        return exp
    
    def start_experiment(self, experiment: Experiment):
        """开始实验"""
        self.current_experiment = experiment
        experiment.start()
    
    def log_params(self, params: Dict):
        """记录参数"""
        if self.current_experiment:
            self.current_experiment.set_hyperparameters(params)
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """记录指标"""
        if self.current_experiment:
            self.current_experiment.log_metric(name, value, step)
    
    def end_experiment(self, success: bool = True, error: str = ""):
        """结束实验"""
        if self.current_experiment:
            if success:
                self.current_experiment.complete()
            else:
                self.current_experiment.fail(error)
            self.current_experiment = None
    
    def compare_experiments(self, metric_name: str) -> List[Dict]:
        """比较实验结果"""
        results = []
        
        for exp in self.experiments:
            if exp.status == "completed" and metric_name in exp.metrics:
                final_value = exp.metrics[metric_name][-1]['value']
                results.append({
                    'experiment': exp.name,
                    'hyperparameters': exp.hyperparameters,
                    metric_name: final_value
                })
        
        # 按指标值排序
        results.sort(key=lambda x: x[metric_name], reverse=True)
        return results
    
    def generate_report(self) -> str:
        """生成实验报告"""
        report = f"# {self.project_name} 实验报告\n\n"
        report += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # 统计
        total = len(self.experiments)
        completed = sum(1 for exp in self.experiments if exp.status == "completed")
        failed = sum(1 for exp in self.experiments if exp.status == "failed")
        
        report += f"## 统计概览\n\n"
        report += f"- 总实验数: {total}\n"
        report += f"- 完成: {completed}\n"
        report += f"- 失败: {failed}\n"
        report += f"- 成功率: {completed/total*100:.1f}%\n\n"
        
        # 实验详情
        report += f"## 实验详情\n\n"
        
        for exp in self.experiments:
            report += f"### {exp.name}\n\n"
            report += f"- **状态**: {exp.status}\n"
            report += f"- **描述**: {exp.description}\n"
            
            if exp.hyperparameters:
                report += f"- **超参数**:\n"
                for key, value in exp.hyperparameters.items():
                    report += f"  - {key}: {value}\n"
            
            if exp.metrics:
                report += f"- **最终指标**:\n"
                for name, values in exp.metrics.items():
                    if values:
                        final_value = values[-1]['value']
                        report += f"  - {name}: {final_value:.4f}\n"
            
            report += "\n"
        
        return report


class BenchmarkSuite:
    """性能基准测试套件"""
    
    def __init__(self, name: str):
        self.name = name
        self.benchmarks = []
    
    def add_benchmark(self, name: str, func: callable, *args, **kwargs):
        """添加基准测试"""
        self.benchmarks.append({
            'name': name,
            'func': func,
            'args': args,
            'kwargs': kwargs
        })
    
    def run(self, iterations: int = 10) -> Dict:
        """运行基准测试"""
        results = {}
        
        for benchmark in self.benchmarks:
            name = benchmark['name']
            func = benchmark['func']
            args = benchmark['args']
            kwargs = benchmark['kwargs']
            
            print(f"\n运行基准测试: {name}")
            
            times = []
            for i in range(iterations):
                start = time.time()
                try:
                    func(*args, **kwargs)
                    end = time.time()
                    times.append(end - start)
                except Exception as e:
                    print(f"  迭代 {i+1} 失败: {e}")
            
            if times:
                results[name] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'median': np.median(times),
                    'iterations': len(times)
                }
                
                print(f"  平均时间: {results[name]['mean']*1000:.2f}ms")
                print(f"  标准差: {results[name]['std']*1000:.2f}ms")
        
        return results
    
    def generate_report(self, results: Dict) -> str:
        """生成基准测试报告"""
        report = f"# {self.name} 基准测试报告\n\n"
        report += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report += "## 测试结果\n\n"
        report += "| 测试名称 | 平均时间(ms) | 标准差(ms) | 最小(ms) | 最大(ms) | 中位数(ms) |\n"
        report += "|---------|------------|----------|---------|---------|----------|\n"
        
        for name, stats in results.items():
            report += f"| {name} | {stats['mean']*1000:.2f} | {stats['std']*1000:.2f} | "
            report += f"{stats['min']*1000:.2f} | {stats['max']*1000:.2f} | {stats['median']*1000:.2f} |\n"
        
        return report


def demonstrate_paper_management():
    """演示论文管理"""
    print("\n" + "="*70)
    print("📄 演示：论文阅读助手")
    print("="*70)
    
    # 创建论文库
    library = PaperLibrary()
    
    # 添加示例论文
    paper1 = Paper(
        title="Attention Is All You Need",
        authors=["Vaswani", "Shazeer", "Parmar", "et al."],
        year=2017,
        venue="NeurIPS",
        url="https://arxiv.org/abs/1706.03762"
    )
    paper1.add_tag("Transformer")
    paper1.add_tag("Attention")
    paper1.add_contribution("提出了Transformer架构")
    paper1.add_contribution("纯注意力机制，无需RNN")
    paper1.add_note("开创性论文，现代NLP的基石")
    
    paper2 = Paper(
        title="BERT: Pre-training of Deep Bidirectional Transformers",
        authors=["Devlin", "Chang", "Lee", "Toutanova"],
        year=2019,
        venue="NAACL",
        url="https://arxiv.org/abs/1810.04805"
    )
    paper2.add_tag("BERT")
    paper2.add_tag("Pretraining")
    paper2.add_contribution("双向预训练Transformer")
    paper2.add_contribution("MLM和NSP任务")
    
    library.add_paper(paper1)
    library.add_paper(paper2)
    
    print("\n【论文库】")
    print(f"总论文数: {len(library.papers)}")
    
    # 搜索
    print("\n【按标签搜索: Transformer】")
    results = library.search_by_tag("Transformer")
    for paper in results:
        print(f"  - {paper.title} ({paper.year})")
    
    # 生成引用
    print("\n【生成BibTeX引用】")
    print(paper1.generate_citation('bibtex'))
    
    # 生成阅读清单
    print("\n【生成阅读清单】")
    reading_list = library.generate_reading_list()
    print(reading_list[:500] + "...")


def demonstrate_experiment_tracking():
    """演示实验追踪"""
    print("\n" + "="*70)
    print("🔬 演示：实验管理工具")
    print("="*70)
    
    # 创建追踪器
    tracker = ExperimentTracker("BERT Fine-tuning")
    
    # 运行多个实验
    experiments_config = [
        {"lr": 1e-4, "batch_size": 16, "epochs": 3},
        {"lr": 2e-4, "batch_size": 32, "epochs": 3},
        {"lr": 5e-5, "batch_size": 16, "epochs": 5},
    ]
    
    print("\n【运行实验】")
    
    for i, config in enumerate(experiments_config, 1):
        exp = tracker.create_experiment(f"实验{i}", f"学习率={config['lr']}")
        tracker.start_experiment(exp)
        tracker.log_params(config)
        
        # 模拟训练过程
        print(f"\n实验 {i}: lr={config['lr']}, batch_size={config['batch_size']}")
        
        for epoch in range(config['epochs']):
            # 模拟指标
            loss = 2.0 - epoch * 0.3 + np.random.random() * 0.1
            acc = 0.5 + epoch * 0.15 + np.random.random() * 0.05
            
            tracker.log_metric("loss", loss, step=epoch)
            tracker.log_metric("accuracy", acc, step=epoch)
            
            print(f"  Epoch {epoch+1}: loss={loss:.4f}, acc={acc:.4f}")
            
            time.sleep(0.1)
        
        tracker.end_experiment(success=True)
    
    # 比较实验
    print("\n【实验对比 - Accuracy】")
    comparison = tracker.compare_experiments("accuracy")
    
    print(f"\n{'排名':<6} {'实验':<15} {'学习率':<12} {'Batch Size':<12} {'Accuracy':<10}")
    print("-" * 65)
    
    for rank, result in enumerate(comparison, 1):
        lr = result['hyperparameters'].get('lr', 'N/A')
        bs = result['hyperparameters'].get('batch_size', 'N/A')
        acc = result['accuracy']
        print(f"{rank:<6} {result['experiment']:<15} {lr:<12} {bs:<12} {acc:.4f}")
    
    # 生成报告
    print("\n【生成实验报告】")
    report = tracker.generate_report()
    print(report[:600] + "...")


def demonstrate_benchmarking():
    """演示性能基准测试"""
    print("\n" + "="*70)
    print("⚡ 演示：性能基准测试")
    print("="*70)
    
    # 创建基准测试套件
    suite = BenchmarkSuite("向量操作性能测试")
    
    # 定义测试函数
    def test_dot_product():
        a = np.random.randn(1000)
        b = np.random.randn(1000)
        return np.dot(a, b)
    
    def test_matrix_multiply():
        a = np.random.randn(100, 100)
        b = np.random.randn(100, 100)
        return np.matmul(a, b)
    
    def test_vector_norm():
        a = np.random.randn(10000)
        return np.linalg.norm(a)
    
    # 添加基准测试
    suite.add_benchmark("向量点积(1000维)", test_dot_product)
    suite.add_benchmark("矩阵乘法(100x100)", test_matrix_multiply)
    suite.add_benchmark("向量范数(10000维)", test_vector_norm)
    
    # 运行测试
    results = suite.run(iterations=100)
    
    # 生成报告
    print("\n【基准测试报告】")
    report = suite.generate_report(results)
    print(report)


def run_week19_20_demo():
    """运行Week 19-20完整演示"""
    print("\n" + "="*70)
    print("🚀 Week 19-20: 论文复现与实验管理 - 完整演示")
    print("="*70)
    
    # 1. 论文管理
    demonstrate_paper_management()
    
    input("\n按Enter继续查看实验追踪...")
    
    # 2. 实验追踪
    demonstrate_experiment_tracking()
    
    input("\n按Enter继续查看性能基准测试...")
    
    # 3. 基准测试
    demonstrate_benchmarking()
    
    print("\n" + "="*70)
    print("✅ Week 19-20演示完成！")
    print("="*70)
    print("\n核心收获：")
    print("  1. 掌握了论文管理和文献引用")
    print("  2. 学会了实验追踪和超参数管理")
    print("  3. 理解了性能基准测试的重要性")
    print("  4. 可以系统化地进行研究工作")


if __name__ == "__main__":
    run_week19_20_demo()
