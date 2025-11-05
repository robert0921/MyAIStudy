"""
Realtime Voice Assistant 测试脚本
用于测试和验证各项功能
"""
import asyncio
import os
from pathlib import Path
import sys

# 加载.env文件
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️ python-dotenv未安装，跳过.env文件加载")

# 确保可以导入主模块
sys.path.insert(0, str(Path(__file__).parent))

from realtime_voice_assistant import (
    AudioConfig,
    AudioPlayer,
    AudioRecorder,
    ConversationLogger,
    RealtimeClient,
    list_audio_devices
)


def test_audio_config():
    """测试音频配置"""
    print("\n" + "="*70)
    print("测试1: AudioConfig 初始化")
    print("="*70)
    
    config = AudioConfig(
        samplerate=16000,
        channels=1,
        dtype="int16"
    )
    
    assert config.samplerate == 16000
    assert config.channels == 1
    assert config.dtype == "int16"
    print("✓ AudioConfig 测试通过")


def test_conversation_logger():
    """测试会话记录器"""
    print("\n" + "="*70)
    print("测试2: ConversationLogger")
    print("="*70)
    
    # 创建临时目录
    test_dir = Path("./test_logs")
    test_dir.mkdir(exist_ok=True)
    
    try:
        logger = ConversationLogger(str(test_dir))
        
        # 测试保存文本
        text_path = logger.save_text("这是一条测试消息", prefix="test")
        assert text_path is not None
        assert Path(text_path).exists()
        print(f"✓ 文本保存成功: {text_path}")
        
        # 测试保存会话
        conversation = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮助你的？"}
        ]
        conv_path = logger.save_conversation(conversation)
        assert conv_path is not None
        assert Path(conv_path).exists()
        print(f"✓ 会话保存成功: {conv_path}")
        
    finally:
        # 清理
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)
        print("✓ ConversationLogger 测试通过")


def test_audio_devices():
    """测试音频设备列表"""
    print("\n" + "="*70)
    print("测试3: 音频设备检测")
    print("="*70)
    
    try:
        list_audio_devices()
        print("✓ 音频设备检测成功")
    except Exception as e:
        print(f"⚠️ 音频设备检测失败（可能没有音频设备）: {e}")


async def test_realtime_client():
    """测试Realtime客户端初始化"""
    print("\n" + "="*70)
    print("测试4: RealtimeClient 初始化")
    print("="*70)
    
    # 检查环境变量
    required_envs = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_API_MODEL"
    ]
    
    missing = [env for env in required_envs if not os.environ.get(env)]
    
    if missing:
        print(f"⚠️ 缺少环境变量: {', '.join(missing)}")
        print("⚠️ 跳过客户端测试")
        return
    
    try:
        client = RealtimeClient()
        print(f"✓ 客户端初始化成功")
        print(f"  - 端点: {client.endpoint}")
        print(f"  - 模型: {client.model}")
        print(f"  - API版本: {client.api_version}")
    except Exception as e:
        print(f"❌ 客户端初始化失败: {e}")


async def test_simple_text_query():
    """测试简单文本查询"""
    print("\n" + "="*70)
    print("测试5: 简单文本查询（需要API访问）")
    print("="*70)
    
    # 检查环境变量
    if not all(os.environ.get(env) for env in [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_API_MODEL"
    ]):
        print("⚠️ 缺少必要的环境变量，跳过API测试")
        return
    
    try:
        from realtime_voice_assistant import RealtimeSession
        
        audio_config = AudioConfig()
        client = RealtimeClient()
        
        session = RealtimeSession(
            client=client,
            audio_config=audio_config,
            enable_audio=False  # 仅文本模式
        )
        
        print("✓ Session 初始化成功")
        print("⚠️ 实际API调用需要在交互模式中测试")
        
    except Exception as e:
        print(f"❌ Session 测试失败: {e}")


def test_import():
    """测试模块导入"""
    print("\n" + "="*70)
    print("测试0: 模块导入检查")
    print("="*70)
    
    try:
        import sounddevice
        print("✓ sounddevice 已安装")
    except ImportError:
        print("❌ sounddevice 未安装，请运行: pip install sounddevice")
    
    try:
        import numpy
        print("✓ numpy 已安装")
    except ImportError:
        print("❌ numpy 未安装，请运行: pip install numpy")
    
    try:
        from openai import AsyncAzureOpenAI
        print("✓ openai 已安装")
    except ImportError:
        print("❌ openai 未安装，请运行: pip install openai")
    
    try:
        from azure.identity.aio import DefaultAzureCredential
        print("✓ azure-identity 已安装")
    except ImportError:
        print("❌ azure-identity 未安装，请运行: pip install azure-identity")


async def run_all_tests():
    """运行所有测试"""
    print("\n" + "🧪"*35)
    print("Realtime Voice Assistant - 功能测试套件")
    print("🧪"*35)
    
    # 测试导入
    test_import()
    
    # 测试基础功能
    test_audio_config()
    test_conversation_logger()
    test_audio_devices()
    
    # 测试API相关（异步）
    await test_realtime_client()
    await test_simple_text_query()
    
    print("\n" + "="*70)
    print("✅ 所有测试完成！")
    print("="*70)
    
    # 显示使用说明
    print("\n📖 使用说明:")
    print("-" * 70)
    print("1. 交互模式:")
    print("   python realtime_voice_assistant.py")
    print()
    print("2. 单次查询:")
    print("   python realtime_voice_assistant.py --text '你好'")
    print()
    print("3. 列出音频设备:")
    print("   python realtime_voice_assistant.py --list-devices")
    print()
    print("4. 保存会话记录:")
    print("   python realtime_voice_assistant.py --save-dir ./logs")
    print()
    print("5. 仅文本模式:")
    print("   python realtime_voice_assistant.py --no-audio")
    print("-" * 70)


def check_environment():
    """检查环境配置"""
    print("\n" + "="*70)
    print("🔍 环境配置检查")
    print("="*70)
    
    env_vars = [
        ("AZURE_OPENAI_ENDPOINT", "Azure OpenAI 端点"),
        ("AZURE_OPENAI_API_KEY", "API密钥"),
        ("AZURE_OPENAI_API_VERSION", "API版本"),
        ("AZURE_OPENAI_API_MODEL", "模型部署名称")
    ]
    
    all_set = True
    for var_name, description in env_vars:
        value = os.environ.get(var_name)
        if value:
            # 隐藏敏感信息
            if "KEY" in var_name:
                display_value = value[:8] + "..." if len(value) > 8 else "***"
            else:
                display_value = value
            print(f"✓ {description:20s}: {display_value}")
        else:
            print(f"❌ {description:20s}: 未设置")
            all_set = False
    
    print("="*70)
    
    if not all_set:
        print("\n⚠️ 部分环境变量未设置！")
        print("请在 .env 文件中配置或设置系统环境变量")
    else:
        print("\n✓ 所有必需的环境变量已设置")
    
    return all_set


if __name__ == "__main__":
    # 检查环境
    check_environment()
    
    # 运行测试
    asyncio.run(run_all_tests())
