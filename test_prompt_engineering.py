"""
Prompt Engineering 与 Few-shot 技术演示
简化版本，独立运行
"""
import sys
from typing import List, Dict, Any

class PromptDebugger:
    """自动化 Prompt 调试与质量分析"""
    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    def test_prompt(self, prompt: str, examples: List[str] = None, temperature: float = 0.7) -> Dict[str, Any]:
        """测试单个Prompt，支持Few-shot示例"""
        print(f"模拟测试 Prompt: {prompt}")
        if examples:
            print(f"使用 {len(examples)} 个Few-shot示例")
        return {
            "prompt": prompt,
            "output": "模拟输出结果",
            "usage": {"tokens": 50}
        }

    def batch_test(self, prompts: List[str], examples: List[str] = None, temperature: float = 0.7) -> List[Dict[str, Any]]:
        """批量测试多个Prompt"""
        results = []
        for prompt in prompts:
            result = self.test_prompt(prompt, examples, temperature)
            results.append(result)
        return results

    def optimize_prompt(self, prompt: str, target: str, examples: List[str] = None, max_iter: int = 5) -> Dict[str, Any]:
        """自动化优化Prompt以提升输出质量"""
        print(f"优化Prompt: {prompt}, 目标: {target}")
        return {"optimized_prompt": f"优化后的Prompt: {prompt}", "score": 0.85}

    def evaluate_output(self, output: str, target: str) -> float:
        """简单的输出质量评估"""
        if target in output:
            return 1.0
        return 0.0

class FewShotManager:
    """Few-shot 示例生成与管理"""
    def __init__(self):
        self.examples = []

    def add_example(self, example: str):
        self.examples.append(example)

    def get_examples(self) -> List[str]:
        return self.examples

    def clear_examples(self):
        self.examples = []

    def auto_generate_examples(self, task_desc: str, n: int = 3) -> List[str]:
        """自动生成Few-shot示例"""
        examples = [f"示例{i+1}: {task_desc}" for i in range(n)]
        self.examples.extend(examples)
        return examples

def demonstrate_prompt_engineering():
    """演示Prompt Engineering与Few-shot技术"""
    print("\n" + "="*60)
    print("🧠 Prompt Engineering 与 Few-shot 技术演示")
    print("="*60)
    
    # Few-shot示例管理
    print("\n1. Few-shot示例管理")
    fewshot = FewShotManager()
    fewshot.add_example("Q: 2+2=?\nA: 4")
    fewshot.add_example("Q: 3+5=?\nA: 8")
    fewshot.add_example("Q: 10-6=?\nA: 4")
    
    print("Few-shot示例:")
    for i, ex in enumerate(fewshot.get_examples(), 1):
        print(f"  示例{i}: {ex}")
    
    # 自动化Prompt调试
    print("\n2. 自动化Prompt调试")
    debugger = PromptDebugger()
    
    result = debugger.test_prompt("Q: 7+6=?", examples=fewshot.get_examples())
    print(f"Prompt测试结果: {result}")
    
    # 批量Prompt测试
    print("\n3. 批量Prompt测试")
    prompts = ["计算5+3", "求解8-2", "计算9×2"]
    batch_results = debugger.batch_test(prompts, examples=fewshot.get_examples())
    for i, result in enumerate(batch_results, 1):
        print(f"  批量测试{i}: {result['prompt']} -> {result['output']}")
    
    # 自动优化Prompt
    print("\n4. 自动优化Prompt")
    opt_result = debugger.optimize_prompt("请计算7+6", target="13", examples=fewshot.get_examples())
    print(f"优化结果: {opt_result}")
    
    # 自动生成Few-shot示例
    print("\n5. 自动生成Few-shot示例")
    auto_examples = fewshot.auto_generate_examples("数学计算", n=2)
    print(f"自动生成的示例: {auto_examples}")
    
    print("\n" + "="*60)
    print("✅ Prompt Engineering 演示完成!")
    print("="*60)

if __name__ == "__main__":
    demonstrate_prompt_engineering()
