"""
Week 16: AI Agent架构设计
包括：Memory机制、Tool-Use工具调用、Planning规划能力、Multi-Agent协作

本模块实现完整的AI Agent系统，包括记忆管理、工具使用和任务规划。
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any
from collections import deque
from datetime import datetime
import json


class Message:
    """消息对象"""
    
    def __init__(self, role: str, content: str, timestamp: Optional[datetime] = None):
        self.role = role  # 'user', 'assistant', 'system'
        self.content = content
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat()
        }
    
    def __repr__(self):
        return f"Message(role='{self.role}', content='{self.content[:30]}...')"


class ShortTermMemory:
    """短期记忆（对话历史）"""
    
    def __init__(self, max_messages: int = 10):
        self.max_messages = max_messages
        self.messages = deque(maxlen=max_messages)
    
    def add_message(self, role: str, content: str):
        """添加消息"""
        message = Message(role, content)
        self.messages.append(message)
    
    def get_recent_messages(self, n: Optional[int] = None) -> List[Message]:
        """获取最近的n条消息"""
        if n is None:
            return list(self.messages)
        return list(self.messages)[-n:]
    
    def clear(self):
        """清空记忆"""
        self.messages.clear()
    
    def __len__(self):
        return len(self.messages)


class LongTermMemory:
    """长期记忆（向量存储）"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.memories = []  # List of (text, vector, metadata)
        self.index = 0
    
    def _simple_embed(self, text: str) -> np.ndarray:
        """简单的文本向量化"""
        words = text.split()
        vector = np.zeros(self.embedding_dim)
        
        for i, word in enumerate(words[:self.embedding_dim]):
            # 使用hash作为简单的embedding
            vector[i] = (hash(word) % 1000) / 1000.0
        
        # 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def store(self, text: str, metadata: Optional[Dict] = None):
        """存储记忆"""
        vector = self._simple_embed(text)
        memory = {
            'id': self.index,
            'text': text,
            'vector': vector,
            'metadata': metadata or {},
            'timestamp': datetime.now()
        }
        self.memories.append(memory)
        self.index += 1
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """检索相关记忆"""
        if not self.memories:
            return []
        
        query_vec = self._simple_embed(query)
        
        # 计算相似度
        similarities = []
        for memory in self.memories:
            similarity = np.dot(query_vec, memory['vector'])
            similarities.append((memory, similarity))
        
        # 排序并返回Top-K
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [mem for mem, _ in similarities[:k]]
    
    def clear(self):
        """清空长期记忆"""
        self.memories.clear()
        self.index = 0
    
    def __len__(self):
        return len(self.memories)


class Tool:
    """工具基类"""
    
    def __init__(self, name: str, description: str, parameters: Dict):
        self.name = name
        self.description = description
        self.parameters = parameters
    
    def execute(self, **kwargs) -> Any:
        """执行工具"""
        raise NotImplementedError
    
    def get_schema(self) -> Dict:
        """获取工具模式（用于LLM理解）"""
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters
        }


class CalculatorTool(Tool):
    """计算器工具"""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="执行数学计算，支持+、-、*、/运算",
            parameters={
                'expression': {
                    'type': 'string',
                    'description': '数学表达式，例如：2+3*4'
                }
            }
        )
    
    def execute(self, expression: str) -> float:
        """执行计算"""
        try:
            # 安全的eval（实际应该用ast.literal_eval）
            result = eval(expression, {"__builtins__": {}}, {})
            return float(result)
        except Exception as e:
            return f"计算错误: {str(e)}"


class SearchTool(Tool):
    """搜索工具（模拟）"""
    
    def __init__(self):
        super().__init__(
            name="search",
            description="搜索相关信息",
            parameters={
                'query': {
                    'type': 'string',
                    'description': '搜索查询'
                }
            }
        )
        # 模拟的知识库
        self.knowledge_base = {
            "Python": "Python是一种高级编程语言，由Guido van Rossum于1991年创建。",
            "AI": "人工智能（AI）是计算机科学的一个分支，致力于创造智能机器。",
            "机器学习": "机器学习是AI的核心技术，让计算机从数据中学习模式。",
            "深度学习": "深度学习是机器学习的子领域，使用多层神经网络。",
        }
    
    def execute(self, query: str) -> str:
        """执行搜索"""
        # 简单的关键词匹配
        for key, value in self.knowledge_base.items():
            if key in query or query in key:
                return f"搜索结果: {value}"
        
        return f"未找到关于'{query}'的信息"


class WeatherTool(Tool):
    """天气查询工具（模拟）"""
    
    def __init__(self):
        super().__init__(
            name="get_weather",
            description="查询指定城市的天气",
            parameters={
                'city': {
                    'type': 'string',
                    'description': '城市名称'
                }
            }
        )
        # 模拟的天气数据
        self.weather_data = {
            "北京": "晴天，气温15-25°C",
            "上海": "多云，气温18-28°C",
            "深圳": "小雨，气温22-30°C",
        }
    
    def execute(self, city: str) -> str:
        """查询天气"""
        return self.weather_data.get(city, f"暂无{city}的天气数据")


class ToolRegistry:
    """工具注册表"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        """注册工具"""
        self.tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """获取工具"""
        return self.tools.get(name)
    
    def list_tools(self) -> List[Dict]:
        """列出所有工具"""
        return [tool.get_schema() for tool in self.tools.values()]
    
    def execute_tool(self, name: str, **kwargs) -> Any:
        """执行工具"""
        tool = self.get_tool(name)
        if tool is None:
            return f"错误: 工具'{name}'不存在"
        
        try:
            return tool.execute(**kwargs)
        except Exception as e:
            return f"工具执行错误: {str(e)}"


class ReActAgent:
    """ReAct (Reasoning + Acting) Agent"""
    
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
        self.short_memory = ShortTermMemory(max_messages=10)
        self.long_memory = LongTermMemory()
        self.max_iterations = 5
    
    def _parse_action(self, text: str) -> Optional[Tuple[str, Dict]]:
        """解析动作（简化版）"""
        # 格式: Action: tool_name(arg1=value1, arg2=value2)
        if "Action:" not in text:
            return None
        
        action_part = text.split("Action:")[1].strip()
        
        # 简单解析工具名和参数
        if "(" in action_part and ")" in action_part:
            tool_name = action_part.split("(")[0].strip()
            params_str = action_part.split("(")[1].split(")")[0]
            
            # 解析参数（简化版）
            params = {}
            if params_str:
                for param in params_str.split(","):
                    if "=" in param:
                        key, value = param.split("=", 1)
                        params[key.strip()] = value.strip().strip("'\"")
            
            return tool_name, params
        
        return None
    
    def run(self, task: str) -> str:
        """运行ReAct循环"""
        print(f"\n{'='*70}")
        print(f"🎯 任务: {task}")
        print(f"{'='*70}")
        
        self.short_memory.add_message("user", task)
        
        for iteration in range(self.max_iterations):
            print(f"\n--- 迭代 {iteration + 1} ---")
            
            # Thought（思考）
            thought = self._generate_thought(task, iteration)
            print(f"💭 Thought: {thought}")
            
            # 检查是否完成
            if "Answer:" in thought or "最终答案" in thought:
                answer = thought.split("Answer:")[-1].strip() if "Answer:" in thought else thought.split("最终答案:")[-1].strip()
                print(f"\n✅ 完成! 答案: {answer}")
                return answer
            
            # Action（行动）
            action_result = self._parse_action(thought)
            
            if action_result:
                tool_name, params = action_result
                print(f"🔧 Action: {tool_name}({params})")
                
                # 执行工具
                observation = self.tool_registry.execute_tool(tool_name, **params)
                print(f"👁️ Observation: {observation}")
                
                # 存储观察结果
                self.short_memory.add_message("assistant", f"Observation: {observation}")
            else:
                print(f"⚠️ 未识别到有效动作")
        
        return "达到最大迭代次数，任务未完成"
    
    def _generate_thought(self, task: str, iteration: int) -> str:
        """生成思考（模拟LLM输出）"""
        # 这里是简化版本，实际应该调用LLM
        
        if iteration == 0:
            # 第一次思考：分析任务
            if "计算" in task or "+" in task or "*" in task:
                return f"Thought: 这是一个数学计算任务。\nAction: calculator(expression='{task.split('计算')[-1].strip()}')"
            elif "天气" in task:
                # 提取城市
                cities = ["北京", "上海", "深圳"]
                for city in cities:
                    if city in task:
                        return f"Thought: 需要查询{city}的天气。\nAction: get_weather(city='{city}')"
            elif "什么是" in task or "介绍" in task:
                query = task.replace("什么是", "").replace("介绍", "").strip()
                return f"Thought: 需要搜索相关信息。\nAction: search(query='{query}')"
        elif iteration == 1:
            # 第二次思考：基于观察结果给出答案
            recent_messages = self.short_memory.get_recent_messages(2)
            if recent_messages:
                last_message = recent_messages[-1]
                if "Observation:" in last_message.content:
                    observation = last_message.content.split("Observation:")[-1].strip()
                    return f"Thought: 根据观察结果，我可以回答了。\nAnswer: {observation}"
        
        return "Thought: 无法处理此任务。\nAnswer: 抱歉，我不知道如何完成这个任务。"


class PlanAndExecuteAgent:
    """Plan-and-Execute Agent"""
    
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
        self.short_memory = ShortTermMemory()
        self.long_memory = LongTermMemory()
    
    def run(self, task: str) -> str:
        """运行计划-执行流程"""
        print(f"\n{'='*70}")
        print(f"🎯 任务: {task}")
        print(f"{'='*70}")
        
        # 1. 制定计划
        plan = self._make_plan(task)
        print(f"\n📋 计划:")
        for i, step in enumerate(plan, 1):
            print(f"   {i}. {step}")
        
        # 2. 执行计划
        results = []
        for i, step in enumerate(plan, 1):
            print(f"\n--- 执行步骤 {i} ---")
            print(f"🔨 {step}")
            
            result = self._execute_step(step)
            print(f"✅ 结果: {result}")
            
            results.append(result)
            self.short_memory.add_message("assistant", f"Step {i}: {result}")
        
        # 3. 汇总结果
        final_answer = self._summarize_results(task, results)
        print(f"\n{'='*70}")
        print(f"✅ 最终答案: {final_answer}")
        print(f"{'='*70}")
        
        return final_answer
    
    def _make_plan(self, task: str) -> List[str]:
        """制定计划（模拟）"""
        # 简化版本：根据任务关键词生成计划
        
        if "天气" in task and "然后" in task:
            # 多步骤任务
            return [
                "查询北京的天气",
                "查询上海的天气",
                "比较两地天气"
            ]
        elif "计算" in task and "搜索" in task:
            return [
                "搜索相关信息",
                "提取数值进行计算",
                "给出最终结果"
            ]
        else:
            # 单步骤任务
            return [task]
    
    def _execute_step(self, step: str) -> str:
        """执行单个步骤"""
        # 判断需要使用的工具
        if "天气" in step:
            cities = ["北京", "上海", "深圳"]
            for city in cities:
                if city in step:
                    return self.tool_registry.execute_tool("get_weather", city=city)
        elif "计算" in step or "+" in step or "*" in step:
            expr = step.split("计算")[-1].strip() if "计算" in step else step
            return str(self.tool_registry.execute_tool("calculator", expression=expr))
        elif "搜索" in step:
            query = step.replace("搜索", "").strip()
            return self.tool_registry.execute_tool("search", query=query)
        else:
            return f"执行: {step}"
    
    def _summarize_results(self, task: str, results: List[str]) -> str:
        """汇总结果"""
        if len(results) == 1:
            return results[0]
        else:
            summary = "综合结果:\n"
            for i, result in enumerate(results, 1):
                summary += f"  {i}. {result}\n"
            return summary.strip()


class MultiAgentSystem:
    """多Agent协作系统"""
    
    def __init__(self):
        self.agents: Dict[str, Any] = {}
        self.message_queue = deque()
    
    def register_agent(self, name: str, agent: Any):
        """注册Agent"""
        self.agents[name] = agent
    
    def send_message(self, from_agent: str, to_agent: str, content: str):
        """发送消息"""
        message = {
            'from': from_agent,
            'to': to_agent,
            'content': content,
            'timestamp': datetime.now()
        }
        self.message_queue.append(message)
    
    def collaborate(self, task: str) -> str:
        """协作完成任务"""
        print(f"\n{'='*70}")
        print(f"👥 多Agent协作任务: {task}")
        print(f"{'='*70}")
        
        # 简化的协作流程
        results = []
        
        # 让每个Agent处理任务的一部分
        for agent_name, agent in self.agents.items():
            print(f"\n🤖 Agent '{agent_name}' 开始工作...")
            
            if hasattr(agent, 'run'):
                result = agent.run(task)
                results.append(f"{agent_name}: {result}")
            else:
                results.append(f"{agent_name}: 无法处理")
        
        # 汇总结果
        final_result = "\n".join(results)
        print(f"\n{'='*70}")
        print(f"✅ 协作完成!")
        print(f"{'='*70}")
        
        return final_result


def demonstrate_memory():
    """演示记忆机制"""
    print("\n" + "="*70)
    print("🧠 演示：Memory机制")
    print("="*70)
    
    # 短期记忆
    print("\n【短期记忆（对话历史）】")
    short_memory = ShortTermMemory(max_messages=5)
    
    short_memory.add_message("user", "你好")
    short_memory.add_message("assistant", "你好！有什么可以帮助你的吗？")
    short_memory.add_message("user", "什么是AI？")
    short_memory.add_message("assistant", "AI是人工智能的缩写...")
    
    print(f"   存储消息数: {len(short_memory)}")
    print(f"   最近3条消息:")
    for msg in short_memory.get_recent_messages(3):
        print(f"      {msg.role}: {msg.content[:40]}...")
    
    # 长期记忆
    print("\n【长期记忆（向量存储）】")
    long_memory = LongTermMemory(embedding_dim=128)
    
    long_memory.store("用户喜欢Python编程", metadata={'type': 'preference'})
    long_memory.store("用户正在学习机器学习", metadata={'type': 'activity'})
    long_memory.store("用户询问过深度学习的问题", metadata={'type': 'history'})
    
    print(f"   存储记忆数: {len(long_memory)}")
    
    # 检索相关记忆
    query = "Python相关的内容"
    results = long_memory.retrieve(query, k=2)
    print(f"\n   查询: '{query}'")
    print(f"   检索到的相关记忆:")
    for memory in results:
        print(f"      - {memory['text']}")


def demonstrate_tool_use():
    """演示工具调用"""
    print("\n" + "="*70)
    print("🔧 演示：Tool-Use工具调用")
    print("="*70)
    
    # 创建工具注册表
    registry = ToolRegistry()
    
    # 注册工具
    registry.register(CalculatorTool())
    registry.register(SearchTool())
    registry.register(WeatherTool())
    
    print(f"\n【可用工具】")
    for tool_schema in registry.list_tools():
        print(f"   - {tool_schema['name']}: {tool_schema['description']}")
    
    # 测试工具
    print(f"\n【工具测试】")
    
    print("\n1. 计算器工具:")
    result = registry.execute_tool("calculator", expression="2+3*4")
    print(f"   计算 2+3*4 = {result}")
    
    print("\n2. 搜索工具:")
    result = registry.execute_tool("search", query="机器学习")
    print(f"   {result}")
    
    print("\n3. 天气工具:")
    result = registry.execute_tool("get_weather", city="北京")
    print(f"   {result}")


def demonstrate_react_agent():
    """演示ReAct Agent"""
    print("\n" + "="*70)
    print("🤖 演示：ReAct Agent")
    print("="*70)
    
    # 创建工具注册表
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(SearchTool())
    registry.register(WeatherTool())
    
    # 创建ReAct Agent
    agent = ReActAgent(registry)
    
    # 测试任务
    tasks = [
        "计算 15+27",
        "北京的天气怎么样",
        "什么是深度学习",
    ]
    
    for i, task in enumerate(tasks, 1):
        print(f"\n{'='*70}")
        print(f"测试任务 {i}")
        agent.run(task)
        
        if i < len(tasks):
            input("\n按Enter继续下一个任务...")


def demonstrate_plan_execute_agent():
    """演示Plan-and-Execute Agent"""
    print("\n" + "="*70)
    print("📋 演示：Plan-and-Execute Agent")
    print("="*70)
    
    # 创建工具注册表
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(SearchTool())
    registry.register(WeatherTool())
    
    # 创建Plan-and-Execute Agent
    agent = PlanAndExecuteAgent(registry)
    
    # 复杂任务
    task = "查询北京的天气，然后告诉我上海的天气"
    agent.run(task)


def run_week16_demo():
    """运行Week 16完整演示"""
    print("\n" + "="*70)
    print("🚀 Week 16: AI Agent架构设计 - 完整演示")
    print("="*70)
    
    # 1. Memory机制
    demonstrate_memory()
    
    input("\n按Enter继续查看Tool-Use...")
    
    # 2. Tool-Use
    demonstrate_tool_use()
    
    input("\n按Enter继续查看ReAct Agent...")
    
    # 3. ReAct Agent
    demonstrate_react_agent()
    
    input("\n按Enter继续查看Plan-and-Execute Agent...")
    
    # 4. Plan-and-Execute Agent
    demonstrate_plan_execute_agent()
    
    print("\n" + "="*70)
    print("✅ Week 16演示完成！")
    print("="*70)
    print("\n核心收获：")
    print("  1. 理解了Agent的记忆机制（短期+长期）")
    print("  2. 掌握了工具调用的设计模式")
    print("  3. 实现了ReAct推理-行动循环")
    print("  4. 学会了Plan-and-Execute任务规划")


if __name__ == "__main__":
    run_week16_demo()
