"""
Realtime Voice Assistant 使用示例
演示各种使用场景
"""
import asyncio
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def example_1_list_devices():
    """示例1: 列出音频设备"""
    print("\n" + "="*70)
    print("示例1: 列出音频设备")
    print("="*70)
    print("\n运行命令:")
    print("python realtime_voice_assistant.py --list-devices")
    print("\n说明: 查看系统中可用的音频输入/输出设备")


def example_2_text_only():
    """示例2: 仅文本模式"""
    print("\n" + "="*70)
    print("示例2: 仅文本模式（无需音频设备）")
    print("="*70)
    print("\n运行命令:")
    print("python realtime_voice_assistant.py --no-audio")
    print("\n说明: 适合没有音频设备或只需文本交互的场景")


def example_3_single_query():
    """示例3: 单次文本查询"""
    print("\n" + "="*70)
    print("示例3: 单次文本查询")
    print("="*70)
    print("\n运行命令:")
    print('python realtime_voice_assistant.py --text "你好，请介绍一下自己" --no-audio')
    print("\n说明: 快速获取AI回复，执行后自动退出")


def example_4_with_logging():
    """示例4: 保存会话记录"""
    print("\n" + "="*70)
    print("示例4: 保存会话记录")
    print("="*70)
    print("\n运行命令:")
    print("python realtime_voice_assistant.py --save-dir ./my_conversations")
    print("\n说明: 会话内容会自动保存到指定目录")
    print("  - audio/: 语音文件")
    print("  - text/: 文本记录")
    print("  - conversation_*.json: 完整会话JSON")


def example_5_custom_system():
    """示例5: 自定义系统提示"""
    print("\n" + "="*70)
    print("示例5: 自定义系统提示")
    print("="*70)
    print("\n运行命令:")
    print('python realtime_voice_assistant.py --system "你是一位资深的Python开发专家" --no-audio')
    print("\n说明: 设置AI助手的角色和语气")


def example_6_voice_mode():
    """示例6: 语音交互模式"""
    print("\n" + "="*70)
    print("示例6: 语音交互模式（需要麦克风和扬声器）")
    print("="*70)
    print("\n运行命令:")
    print("python realtime_voice_assistant.py")
    print("\n说明: 启动后选择 [v] 进行语音对话")
    print("  1. 系统会录制3秒音频")
    print("  2. AI会实时播放语音回复")
    print("  3. 支持流式音频输出")


def example_7_specific_devices():
    """示例7: 指定音频设备"""
    print("\n" + "="*70)
    print("示例7: 指定特定音频设备")
    print("="*70)
    print("\n步骤:")
    print("1. 运行: python realtime_voice_assistant.py --list-devices")
    print("2. 查看设备索引")
    print("3. 运行: python realtime_voice_assistant.py --input-device 2 --output-device 3")
    print("\n说明: 使用外部麦克风或音箱时很有用")


async def example_8_api_test():
    """示例8: 实际API测试"""
    print("\n" + "="*70)
    print("示例8: 快速API测试")
    print("="*70)
    
    # 检查环境变量
    required = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY", 
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_API_MODEL"
    ]
    
    if not all(os.environ.get(k) for k in required):
        print("\n⚠️ 环境变量未配置，跳过实际测试")
        print("请配置 .env 文件后再试")
        return
    
    print("\n✓ 环境变量已配置")
    print("\n现在可以运行:")
    print('python realtime_voice_assistant.py --text "你好" --no-audio')


def show_usage_scenarios():
    """展示实际使用场景"""
    print("\n" + "🎯"*35)
    print("实际使用场景")
    print("🎯"*35)
    
    scenarios = [
        {
            "name": "场景1: 语音聊天机器人",
            "command": "python realtime_voice_assistant.py --save-dir ./chat_logs",
            "desc": "与AI进行自然语音对话，自动保存记录"
        },
        {
            "name": "场景2: 技术问答助手",
            "command": 'python realtime_voice_assistant.py --system "你是Python专家" --no-audio',
            "desc": "专注于技术问题的文本问答"
        },
        {
            "name": "场景3: 语言学习伙伴",
            "command": 'python realtime_voice_assistant.py --system "用简单的中文对话" --duration 5',
            "desc": "练习口语，支持更长的录音时间"
        },
        {
            "name": "场景4: 快速查询",
            "command": 'python realtime_voice_assistant.py --text "今天天气" --no-audio',
            "desc": "快速获取信息，无需交互"
        },
        {
            "name": "场景5: 会议记录",
            "command": 'python realtime_voice_assistant.py --system "总结要点" --save-dir ./meetings',
            "desc": "记录和总结会议内容"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   命令: {scenario['command']}")
        print(f"   说明: {scenario['desc']}")


def show_tips():
    """显示使用技巧"""
    print("\n" + "💡"*35)
    print("使用技巧")
    print("💡"*35)
    
    tips = [
        "1. 首次使用前运行测试脚本验证配置",
        "2. 使用 --list-devices 找到最佳音频设备",
        "3. 网络不稳定时使用 --no-audio 纯文本模式",
        "4. 定期清理会话日志目录节省空间",
        "5. 使用系统提示定制AI助手的角色",
        "6. 录音时保持环境安静，距离麦克风适中",
        "7. 长时间会话建议开启 --save-dir 保存记录",
        "8. 生产环境建议使用 --use-aad 而非API Key"
    ]
    
    for tip in tips:
        print(f"  {tip}")


def main():
    """主函数"""
    print("\n" + "🤖"*35)
    print("Realtime Voice Assistant - 使用示例集合")
    print("🤖"*35)
    
    # 显示所有示例
    example_1_list_devices()
    example_2_text_only()
    example_3_single_query()
    example_4_with_logging()
    example_5_custom_system()
    example_6_voice_mode()
    example_7_specific_devices()
    
    # 异步示例
    asyncio.run(example_8_api_test())
    
    # 使用场景
    show_usage_scenarios()
    
    # 使用技巧
    show_tips()
    
    print("\n" + "="*70)
    print("📚 更多信息请参考 REALTIME_README.md")
    print("="*70)


if __name__ == "__main__":
    main()
