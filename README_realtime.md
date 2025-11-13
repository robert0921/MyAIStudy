# Azure OpenAI Realtime Voice Assistant

一个功能完整、优化的Azure OpenAI Realtime API语音助手，支持实时语音对话、文本交互和会话记录。

## ✨ 特性

- 🎙️ **实时语音输入**: 支持麦克风录音并发送到AI模型
- 🔊 **实时语音输出**: 流式播放AI生成的语音回复
- 💬 **文本交互**: 支持纯文本模式对话
- 📝 **会话记录**: 自动保存音频文件、文本记录和完整会话JSON
- 🎚️ **设备管理**: 灵活选择输入/输出音频设备
- 🔐 **双认证支持**: 支持API Key和Azure AD认证
- 🎭 **系统提示**: 自定义助手角色和语气
- ⚡ **非阻塞播放**: 使用流式音频播放，延迟更低

## 📦 安装依赖

```bash
# 基础依赖
pip install openai azure-identity sounddevice numpy

# 可选依赖（用于音频处理）
pip install scipy

# 如果遇到sounddevice问题，可能需要安装PortAudio
# macOS: brew install portaudio
# Ubuntu: sudo apt-get install portaudio19-dev
# Windows: 通常无需额外安装
```

## 🔧 环境配置

在项目根目录创建 `.env` 文件或设置环境变量：

```bash
# Azure OpenAI配置（必需）
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_API_VERSION=2025-04-01-preview
AZURE_OPENAI_API_MODEL=gpt-4o-mini-realtime-preview

# 如果使用Azure AD认证，可以不设置API_KEY
```

## 🚀 快速开始

### 1. 列出音频设备

```bash
python realtime_voice_assistant.py --list-devices
```

输出示例：
```
🎤 可用音频设备:
======================================================================
  0 MacBook Pro Microphone, Core Audio (2 in, 0 out)
> 1 MacBook Pro Speakers, Core Audio (0 in, 2 out)
  2 External Microphone, Core Audio (1 in, 0 out)
======================================================================
```

### 2. 交互模式（默认）

```bash
# 基本使用
python realtime_voice_assistant.py

# 指定音频设备
python realtime_voice_assistant.py --input-device 0 --output-device 1

# 保存会话记录
python realtime_voice_assistant.py --save-dir ./conversation_logs
```

交互界面：
```
======================================================================
📝 输入选项:
  [t] 发送文本消息
  [v] 发送语音消息
  [h] 显示会话历史
  [s] 保存会话
  [q] 退出
======================================================================
请选择操作:
```

### 3. 单次查询模式

```bash
# 文本查询
python realtime_voice_assistant.py --text "介绍一下你自己"

# 带系统提示的查询
python realtime_voice_assistant.py --text "写一首诗" --system "你是一位古典诗人"
```

### 4. 仅文本模式

```bash
# 禁用语音功能，仅使用文本
python realtime_voice_assistant.py --no-audio
```

### 5. Azure AD认证

```bash
# 使用Azure Active Directory认证（无需API Key）
python realtime_voice_assistant.py --use-aad
```

## 📖 详细用法

### 命令行参数

```
设备管理:
  --list-devices           列出可用音频设备
  --input-device N         指定输入设备索引（麦克风）
  --output-device N        指定输出设备索引（扬声器）

音频配置:
  --rate N                 采样率（默认16000 Hz）
  --duration N             录音时长（秒，默认3.0）
  --no-audio               仅文本模式

API配置:
  --api-key KEY            Azure OpenAI API Key
  --use-aad                使用Azure AD认证

会话配置:
  --system TEXT            系统提示词（设置助手角色）
  --text TEXT              单次查询文本

日志配置:
  --save-dir PATH          会话记录保存目录
```

### 系统提示示例

```bash
# 友好的助手
python realtime_voice_assistant.py --system "你是一个友好、耐心的AI助手。"

# 专业顾问
python realtime_voice_assistant.py --system "你是一位资深的技术顾问，擅长解决复杂问题。"

# 创意写作
python realtime_voice_assistant.py --system "你是一位富有创意的作家，善于用生动的语言讲故事。"

# 教育导师
python realtime_voice_assistant.py --system "你是一位有耐心的教师，善于用简单易懂的方式解释复杂概念。"
```

## 🧪 测试

运行测试脚本验证安装和配置：

```bash
python test_realtime_voice_assistant.py
```

测试包括：
- ✅ 模块导入检查
- ✅ 音频配置测试
- ✅ 会话记录器测试
- ✅ 音频设备检测
- ✅ 客户端初始化测试
- ✅ 环境变量检查

## 📁 项目结构

```
realtime_voice_assistant.py          # 主程序
test_realtime_voice_assistant.py     # 测试脚本
REALTIME_README.md                   # 本文档
.env                                 # 环境配置（需要创建）

conversation_logs/                   # 会话记录（可选）
├── audio/                          # 音频文件
│   ├── reply_20250104_143022.wav
│   └── reply_20250104_143156.wav
├── text/                           # 文本记录
│   ├── reply_20250104_143022.txt
│   └── reply_20250104_143156.txt
└── conversation_20250104_143500.json  # 完整会话JSON
```

## ⚙️ 高级配置

### 自定义音频参数

```python
# 在代码中自定义AudioConfig
audio_config = AudioConfig(
    samplerate=24000,      # 更高采样率
    channels=1,            # 单声道
    dtype="int16",         # 16位深度
    input_device=2,        # 外部麦克风
    output_device=3        # 外部音箱
)
```

### 会话记录器扩展

```python
# 自定义日志格式
logger = ConversationLogger("./custom_logs")

# 手动保存
logger.save_audio(audio_bytes, samplerate=16000, prefix="custom")
logger.save_text(text_content, prefix="summary")
logger.save_conversation(conversation_history)
```

## 🐛 常见问题

### Q1: 无法找到音频设备
**A**: 运行 `--list-devices` 查看可用设备，然后使用 `--input-device` 和 `--output-device` 指定。

### Q2: 音频播放延迟高
**A**: 降低采样率（如 `--rate 8000`）或检查系统音频驱动。

### Q3: API认证失败
**A**: 检查 `.env` 文件配置，确保环境变量正确设置。使用 `test_realtime_voice_assistant.py` 验证。

### Q4: 录音时没有声音
**A**: 检查麦克风权限，确保系统允许应用访问麦克风。

### Q5: 语音输出不流畅
**A**: 使用非阻塞播放器（已实现），确保网络连接稳定。

## 🔒 安全建议

1. **不要提交 `.env` 文件到版本控制**
2. **使用Azure AD认证而非API Key（生产环境）**
3. **定期轮换API密钥**
4. **限制会话记录的存储时间**
5. **加密敏感的会话记录**

## 📊 性能优化

- ✅ 使用非阻塞流式音频播放
- ✅ 异步事件处理
- ✅ 内存高效的音频缓冲
- ✅ 按需加载资源
- ✅ 支持自定义采样率降低带宽

## 🔄 版本历史

### v1.0.0 (2025-01-04)
- 🎉 初始版本发布
- ✅ 整合3个原始实现
- ✅ 添加完整测试套件
- ✅ 优化音频播放性能
- ✅ 增强错误处理
- ✅ 完善文档

## 📝 许可证

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📧 联系方式

- 项目主页: [GitHub Repository](https://github.com/robert0921/MyAIStudy)
- 问题反馈: [Issues](https://github.com/robert0921/MyAIStudy/issues)

---

**Made with ❤️ by robert0921**
