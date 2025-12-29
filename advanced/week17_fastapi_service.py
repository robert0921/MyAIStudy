"""
Week 17: FastAPI服务化部署
包括：RESTful API设计、会话管理、流式输出、性能优化

本模块演示如何将LLM包装成生产级API服务（代码示例，不实际启动服务器）
"""

from typing import List, Dict, Optional, AsyncIterator
from datetime import datetime, timedelta
import json
import time
import asyncio
from collections import defaultdict


class Session:
    """会话对象"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.last_active = datetime.now()
        self.messages = []
        self.metadata = {}
    
    def add_message(self, role: str, content: str):
        """添加消息"""
        self.messages.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        self.last_active = datetime.now()
    
    def is_expired(self, timeout_minutes: int = 30) -> bool:
        """检查会话是否过期"""
        return datetime.now() - self.last_active > timedelta(minutes=timeout_minutes)


class SessionManager:
    """会话管理器"""
    
    def __init__(self, timeout_minutes: int = 30):
        self.sessions: Dict[str, Session] = {}
        self.timeout_minutes = timeout_minutes
    
    def create_session(self, session_id: str) -> Session:
        """创建会话"""
        session = Session(session_id)
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """获取会话"""
        session = self.sessions.get(session_id)
        if session and not session.is_expired(self.timeout_minutes):
            return session
        elif session:
            # 删除过期会话
            del self.sessions[session_id]
        return None
    
    def cleanup_expired_sessions(self):
        """清理过期会话"""
        expired = [sid for sid, session in self.sessions.items() 
                  if session.is_expired(self.timeout_minutes)]
        for sid in expired:
            del self.sessions[sid]
        return len(expired)


class RateLimiter:
    """速率限制器"""
    
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        """检查是否允许请求"""
        now = time.time()
        window_start = now - self.window_seconds
        
        # 清理过期记录
        self.requests[client_id] = [
            timestamp for timestamp in self.requests[client_id]
            if timestamp > window_start
        ]
        
        # 检查限制
        if len(self.requests[client_id]) < self.max_requests:
            self.requests[client_id].append(now)
            return True
        
        return False
    
    def get_remaining(self, client_id: str) -> int:
        """获取剩余请求数"""
        now = time.time()
        window_start = now - self.window_seconds
        
        recent_requests = [
            timestamp for timestamp in self.requests[client_id]
            if timestamp > window_start
        ]
        
        return max(0, self.max_requests - len(recent_requests))


class LLMService:
    """LLM服务（模拟）"""
    
    def __init__(self):
        self.model_name = "gpt-3.5-turbo-mock"
        self.session_manager = SessionManager(timeout_minutes=30)
        self.rate_limiter = RateLimiter(max_requests=10, window_seconds=60)
    
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """生成文本（同步）"""
        # 模拟LLM生成
        time.sleep(0.1)  # 模拟延迟
        
        # 简单的响应生成
        responses = {
            "你好": "你好！我是AI助手，有什么可以帮助你的吗？",
            "天气": "今天天气不错，适合出行。",
            "编程": "编程是一项有趣的技能，建议从Python开始学习。",
        }
        
        for key, value in responses.items():
            if key in prompt:
                return value
        
        return f"收到您的消息：{prompt[:50]}..."
    
    async def generate_stream(self, prompt: str, max_tokens: int = 100) -> AsyncIterator[str]:
        """生成文本（流式）"""
        response = self.generate(prompt, max_tokens)
        words = response.split()
        
        # 模拟流式输出
        for word in words:
            await asyncio.sleep(0.05)  # 模拟延迟
            yield word + " "
    
    def chat(self, session_id: str, message: str) -> Dict:
        """多轮对话"""
        # 获取或创建会话
        session = self.session_manager.get_session(session_id)
        if session is None:
            session = self.session_manager.create_session(session_id)
        
        # 添加用户消息
        session.add_message("user", message)
        
        # 构建上下文
        context = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in session.messages[-5:]  # 最近5条
        ])
        
        # 生成回复
        response = self.generate(context)
        
        # 添加助手消息
        session.add_message("assistant", response)
        
        return {
            "session_id": session_id,
            "response": response,
            "message_count": len(session.messages)
        }


class APIEndpoints:
    """API端点定义（FastAPI风格的伪代码）"""
    
    def __init__(self):
        self.service = LLMService()
    
    def health_check(self) -> Dict:
        """
        GET /health
        健康检查端点
        """
        return {
            "status": "healthy",
            "model": self.service.model_name,
            "timestamp": datetime.now().isoformat()
        }
    
    def chat_completion(self, request: Dict) -> Dict:
        """
        POST /chat/completions
        单轮对话
        
        请求体:
        {
            "prompt": "用户输入",
            "max_tokens": 100,
            "temperature": 0.7
        }
        """
        prompt = request.get("prompt", "")
        max_tokens = request.get("max_tokens", 100)
        
        # 检查速率限制
        client_id = request.get("client_id", "default")
        if not self.service.rate_limiter.is_allowed(client_id):
            return {
                "error": "Rate limit exceeded",
                "retry_after": 60
            }
        
        response = self.service.generate(prompt, max_tokens)
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.service.model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response.split()),
                "total_tokens": len(prompt.split()) + len(response.split())
            }
        }
    
    async def chat_completion_stream(self, request: Dict) -> AsyncIterator[str]:
        """
        POST /chat/completions/stream
        流式对话
        
        返回Server-Sent Events (SSE)格式
        """
        prompt = request.get("prompt", "")
        max_tokens = request.get("max_tokens", 100)
        
        async for chunk in self.service.generate_stream(prompt, max_tokens):
            data = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": self.service.model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"content": chunk},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(data)}\n\n"
        
        # 结束标记
        yield "data: [DONE]\n\n"
    
    def chat_session(self, request: Dict) -> Dict:
        """
        POST /chat/session
        多轮对话（会话管理）
        
        请求体:
        {
            "session_id": "user-123",
            "message": "用户输入"
        }
        """
        session_id = request.get("session_id", "")
        message = request.get("message", "")
        
        if not session_id:
            return {"error": "session_id is required"}
        
        result = self.service.chat(session_id, message)
        
        return result
    
    def get_session_history(self, session_id: str) -> Dict:
        """
        GET /chat/session/{session_id}/history
        获取会话历史
        """
        session = self.service.session_manager.get_session(session_id)
        
        if session is None:
            return {"error": "Session not found or expired"}
        
        return {
            "session_id": session_id,
            "created_at": session.created_at.isoformat(),
            "last_active": session.last_active.isoformat(),
            "message_count": len(session.messages),
            "messages": session.messages
        }


def demonstrate_rest_api():
    """演示RESTful API设计"""
    print("\n" + "="*70)
    print("🌐 演示：RESTful API设计")
    print("="*70)
    
    api = APIEndpoints()
    
    # 1. 健康检查
    print("\n【1. 健康检查】")
    print("GET /health")
    health = api.health_check()
    print(f"响应: {json.dumps(health, indent=2, ensure_ascii=False)}")
    
    # 2. 单轮对话
    print("\n【2. 单轮对话】")
    print("POST /chat/completions")
    request = {
        "prompt": "你好，请介绍一下自己",
        "max_tokens": 100,
        "client_id": "user-001"
    }
    print(f"请求: {json.dumps(request, indent=2, ensure_ascii=False)}")
    response = api.chat_completion(request)
    print(f"响应: {json.dumps(response, indent=2, ensure_ascii=False)}")
    
    # 3. 多轮对话
    print("\n【3. 多轮对话（会话管理）】")
    print("POST /chat/session")
    
    session_id = "session-001"
    messages = [
        "你好",
        "今天天气怎么样",
        "推荐一门编程语言"
    ]
    
    for msg in messages:
        request = {"session_id": session_id, "message": msg}
        print(f"\n用户: {msg}")
        response = api.chat_session(request)
        print(f"助手: {response['response']}")
        print(f"消息数: {response['message_count']}")
    
    # 4. 获取会话历史
    print("\n【4. 获取会话历史】")
    print(f"GET /chat/session/{session_id}/history")
    history = api.get_session_history(session_id)
    print(f"会话创建: {history['created_at']}")
    print(f"最后活跃: {history['last_active']}")
    print(f"消息总数: {history['message_count']}")


async def demonstrate_streaming():
    """演示流式输出"""
    print("\n" + "="*70)
    print("📡 演示：流式输出")
    print("="*70)
    
    api = APIEndpoints()
    
    print("\n【流式对话】")
    print("POST /chat/completions/stream")
    
    request = {"prompt": "编程是一项有趣的技能", "max_tokens": 50}
    print(f"请求: {json.dumps(request, indent=2, ensure_ascii=False)}")
    print("\n流式响应:")
    print("助手: ", end="", flush=True)
    
    async for chunk in api.chat_completion_stream(request):
        if "[DONE]" not in chunk:
            data = json.loads(chunk.replace("data: ", ""))
            content = data['choices'][0]['delta'].get('content', '')
            print(content, end="", flush=True)
    
    print("\n")


def demonstrate_session_management():
    """演示会话管理"""
    print("\n" + "="*70)
    print("💬 演示：会话管理")
    print("="*70)
    
    manager = SessionManager(timeout_minutes=1)
    
    # 创建会话
    print("\n【创建会话】")
    session1 = manager.create_session("user-001")
    session2 = manager.create_session("user-002")
    print(f"创建会话: {session1.session_id}, {session2.session_id}")
    print(f"当前会话数: {len(manager.sessions)}")
    
    # 添加消息
    print("\n【添加消息】")
    session1.add_message("user", "你好")
    session1.add_message("assistant", "你好！")
    print(f"会话 {session1.session_id} 消息数: {len(session1.messages)}")
    
    # 检索会话
    print("\n【检索会话】")
    retrieved = manager.get_session("user-001")
    if retrieved:
        print(f"✅ 成功检索会话: {retrieved.session_id}")
        print(f"   消息数: {len(retrieved.messages)}")
    
    # 模拟过期
    print("\n【会话过期】")
    print("等待会话过期...")
    time.sleep(2)  # 等待超过timeout
    
    expired_count = manager.cleanup_expired_sessions()
    print(f"清理了 {expired_count} 个过期会话")
    print(f"当前会话数: {len(manager.sessions)}")


def demonstrate_rate_limiting():
    """演示速率限制"""
    print("\n" + "="*70)
    print("⏱️ 演示：速率限制")
    print("="*70)
    
    limiter = RateLimiter(max_requests=5, window_seconds=10)
    client_id = "user-001"
    
    print(f"\n配置: 最多 5 个请求 / 10 秒")
    print(f"客户端: {client_id}")
    
    print("\n【连续请求测试】")
    for i in range(7):
        allowed = limiter.is_allowed(client_id)
        remaining = limiter.get_remaining(client_id)
        
        status = "✅ 允许" if allowed else "❌ 限制"
        print(f"请求 {i+1}: {status}, 剩余: {remaining}")
        
        if not allowed:
            print(f"   → 客户端被限流，请稍后重试")
    
    print("\n【等待窗口重置】")
    print("等待 3 秒...")
    time.sleep(3)
    
    allowed = limiter.is_allowed(client_id)
    remaining = limiter.get_remaining(client_id)
    print(f"新请求: {'✅ 允许' if allowed else '❌ 限制'}, 剩余: {remaining}")


def demonstrate_api_documentation():
    """演示API文档"""
    print("\n" + "="*70)
    print("📚 API文档示例")
    print("="*70)
    
    doc = """
    
API Base URL: https://api.example.com/v1

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

端点1: 健康检查
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GET /health

响应示例:
{
    "status": "healthy",
    "model": "gpt-3.5-turbo",
    "timestamp": "2024-01-15T10:30:00"
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

端点2: 聊天补全
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POST /chat/completions

请求头:
    Content-Type: application/json
    Authorization: Bearer YOUR_API_KEY

请求体:
{
    "prompt": "你好",
    "max_tokens": 100,
    "temperature": 0.7
}

响应示例:
{
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1705305000,
    "model": "gpt-3.5-turbo",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "你好！有什么可以帮助你的吗？"
        },
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 2,
        "completion_tokens": 10,
        "total_tokens": 12
    }
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

端点3: 流式聊天
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POST /chat/completions/stream

响应格式: Server-Sent Events (SSE)

data: {"id":"chatcmpl-123","choices":[{"delta":{"content":"你"}}]}
data: {"id":"chatcmpl-123","choices":[{"delta":{"content":"好"}}]}
data: [DONE]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

端点4: 会话管理
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POST /chat/session

请求体:
{
    "session_id": "user-123",
    "message": "你好"
}

响应示例:
{
    "session_id": "user-123",
    "response": "你好！有什么可以帮助你的吗？",
    "message_count": 2
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

速率限制
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- 免费层: 10 请求/分钟
- 专业版: 100 请求/分钟
- 企业版: 无限制

响应头:
    X-RateLimit-Limit: 10
    X-RateLimit-Remaining: 7
    X-RateLimit-Reset: 1705305060

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    
    print(doc)


def run_week17_demo():
    """运行Week 17完整演示"""
    print("\n" + "="*70)
    print("🚀 Week 17: FastAPI服务化部署 - 完整演示")
    print("="*70)
    
    # 1. RESTful API
    demonstrate_rest_api()
    
    input("\n按Enter继续查看流式输出...")
    
    # 2. 流式输出
    asyncio.run(demonstrate_streaming())
    
    input("\n按Enter继续查看会话管理...")
    
    # 3. 会话管理
    demonstrate_session_management()
    
    input("\n按Enter继续查看速率限制...")
    
    # 4. 速率限制
    demonstrate_rate_limiting()
    
    input("\n按Enter继续查看API文档...")
    
    # 5. API文档
    demonstrate_api_documentation()
    
    print("\n" + "="*70)
    print("✅ Week 17演示完成！")
    print("="*70)
    print("\n核心收获：")
    print("  1. 掌握了RESTful API设计模式")
    print("  2. 实现了流式输出（SSE）")
    print("  3. 学会了会话管理和过期清理")
    print("  4. 理解了速率限制的重要性")
    print("\n注意: 这是演示代码，实际部署需要使用FastAPI框架")


if __name__ == "__main__":
    run_week17_demo()
