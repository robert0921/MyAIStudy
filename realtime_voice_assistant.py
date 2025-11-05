"""
Azure OpenAI Realtime API - 语音助手（集成优化版）
支持文本/语音输入，实时语音输出，会话记录和设备管理
"""
import os
import base64
import asyncio
import argparse
import sounddevice as sd
import numpy as np
import io
import wave
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

from openai import AsyncAzureOpenAI
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider

# 加载.env文件
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv是可选的

# 音频配置常量
AUDIO_FORMAT = "pcm16"
DEFAULT_SAMPLERATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_DTYPE = "int16"


class AudioConfig:
    """音频配置类"""
    def __init__(
        self,
        samplerate: int = DEFAULT_SAMPLERATE,
        channels: int = DEFAULT_CHANNELS,
        dtype: str = DEFAULT_DTYPE,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None
    ):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.input_device = input_device
        self.output_device = output_device


class AudioPlayer:
    """非阻塞流式音频播放器"""
    def __init__(self, config: AudioConfig):
        self.config = config
        self.stream: Optional[sd.OutputStream] = None
        self._is_playing = False
    
    def start(self):
        """启动音频流"""
        if self.stream is None:
            try:
                self.stream = sd.OutputStream(
                    samplerate=self.config.samplerate,
                    channels=self.config.channels,
                    dtype=self.config.dtype,
                    device=self.config.output_device,
                    blocksize=0  # 最小延迟
                )
                self.stream.start()
                self._is_playing = True
                print("[播放器] 已启动")
            except Exception as e:
                print(f"[播放器错误] 无法启动音频流: {e}")
    
    def write_bytes(self, pcm_bytes: bytes):
        """写入PCM音频数据"""
        if not pcm_bytes:
            return
        
        try:
            if not self._is_playing:
                self.start()
            
            if self.stream is not None:
                arr = np.frombuffer(pcm_bytes, dtype=self.config.dtype)
                self.stream.write(arr)
        except Exception as e:
            print(f"[播放器错误] 写入音频失败: {e}")
    
    def close(self):
        """关闭音频流"""
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
                print("[播放器] 已关闭")
            except Exception as e:
                print(f"[播放器错误] 关闭失败: {e}")
            finally:
                self.stream = None
                self._is_playing = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AudioRecorder:
    """音频录制器"""
    def __init__(self, config: AudioConfig):
        self.config = config
    
    def record(self, duration: float = 3.0) -> bytes:
        """录制音频并返回WAV格式字节"""
        print(f"[录音] 请开始说话（{duration}秒）...")
        
        try:
            # 录制音频
            num_frames = int(duration * self.config.samplerate)
            audio_data = sd.rec(
                num_frames,
                samplerate=self.config.samplerate,
                channels=self.config.channels,
                dtype=self.config.dtype,
                device=self.config.input_device
            )
            sd.wait()  # 等待录制完成
            
            # 转换为WAV格式
            wav_bytes = self._to_wav(audio_data)
            print(f"[录音] 完成，大小: {len(wav_bytes)} 字节")
            return wav_bytes
            
        except Exception as e:
            print(f"[录音错误] {e}")
            return b""
    
    def _to_wav(self, audio_array: np.ndarray) -> bytes:
        """将numpy数组转换为WAV格式字节"""
        with io.BytesIO() as buf:
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(self.config.channels)
                wf.setsampwidth(2)  # 16-bit = 2 bytes
                wf.setframerate(self.config.samplerate)
                wf.writeframes(audio_array.tobytes())
            return buf.getvalue()


class ConversationLogger:
    """会话记录器"""
    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.audio_dir = self.save_dir / "audio" if self.save_dir else None
        self.text_dir = self.save_dir / "text" if self.save_dir else None
        
        if self.audio_dir:
            self.audio_dir.mkdir(exist_ok=True)
        if self.text_dir:
            self.text_dir.mkdir(exist_ok=True)
    
    def save_audio(self, pcm_bytes: bytes, samplerate: int, prefix: str = "reply") -> Optional[str]:
        """保存音频文件"""
        if not self.audio_dir or not pcm_bytes:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.wav"
        filepath = self.audio_dir / filename
        
        try:
            with wave.open(str(filepath), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(samplerate)
                wf.writeframes(pcm_bytes)
            
            print(f"[已保存音频] {filepath}")
            return str(filepath)
        except Exception as e:
            print(f"[保存音频错误] {e}")
            return None
    
    def save_text(self, text: str, prefix: str = "reply") -> Optional[str]:
        """保存文本文件"""
        if not self.text_dir or not text:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.txt"
        filepath = self.text_dir / filename
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)
            
            print(f"[已保存文本] {filepath}")
            return str(filepath)
        except Exception as e:
            print(f"[保存文本错误] {e}")
            return None
    
    def save_conversation(self, conversation: List[Dict[str, Any]]) -> Optional[str]:
        """保存完整会话JSON"""
        if not self.save_dir:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.json"
        filepath = self.save_dir / filename
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(conversation, f, ensure_ascii=False, indent=2)
            
            print(f"[已保存会话] {filepath}")
            return str(filepath)
        except Exception as e:
            print(f"[保存会话错误] {e}")
            return None


class RealtimeClient:
    """Azure OpenAI Realtime客户端"""
    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        model: Optional[str] = None,
        use_aad: bool = False
    ):
        self.endpoint = endpoint or self._require_env("AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version or self._require_env("AZURE_OPENAI_API_VERSION")
        self.model = model or self._require_env("AZURE_OPENAI_API_MODEL")
        
        # 创建客户端
        if use_aad or not api_key:
            # 使用Azure Active Directory认证
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default"
            )
            self.client = AsyncAzureOpenAI(
                azure_endpoint=self.endpoint,
                api_version=self.api_version,
                azure_ad_token_provider=token_provider
            )
            print("[认证] 使用Azure Active Directory")
        else:
            # 使用API Key认证
            api_key = api_key or self._require_env("AZURE_OPENAI_API_KEY")
            self.client = AsyncAzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=api_key,
                api_version=self.api_version
            )
            print("[认证] 使用API Key")
    
    @staticmethod
    def _require_env(name: str) -> str:
        """获取必需的环境变量"""
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"缺少环境变量: {name}")
        return value


class RealtimeSession:
    """Realtime会话管理"""
    def __init__(
        self,
        client: RealtimeClient,
        audio_config: AudioConfig,
        logger: Optional[ConversationLogger] = None,
        system_prompt: Optional[str] = None,
        enable_audio: bool = True
    ):
        self.client = client
        self.audio_config = audio_config
        self.logger = logger
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.enable_audio = enable_audio
        
        self.player = AudioPlayer(audio_config) if enable_audio else None
        self.recorder = AudioRecorder(audio_config)
        
        self.conversation_history: List[Dict[str, Any]] = []
    
    async def start(self):
        """启动会话"""
        modalities = ["text", "audio"] if self.enable_audio else ["text"]
        
        print(f"\n{'='*70}")
        print(f"🤖 Azure OpenAI Realtime 语音助手")
        print(f"{'='*70}")
        print(f"模型: {self.client.model}")
        print(f"模式: {'文本+语音' if self.enable_audio else '仅文本'}")
        print(f"采样率: {self.audio_config.samplerate} Hz")
        print(f"系统提示: {self.system_prompt}")
        print(f"{'='*70}\n")
        
        async with self.client.client.beta.realtime.connect(
            model=self.client.model
        ) as connection:
            # 配置会话
            await connection.session.update(
                session={
                    "modalities": modalities,
                    "instructions": self.system_prompt,
                    "voice": "alloy",  # 可选: alloy, echo, fable, onyx, nova, shimmer
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "input_audio_transcription": {"model": "whisper-1"},
                }
            )
            
            # 交互循环
            await self._interactive_loop(connection)
    
    async def _interactive_loop(self, connection):
        """交互式对话循环"""
        try:
            while True:
                print("\n" + "="*70)
                print("📝 输入选项:")
                print("  [t] 发送文本消息")
                if self.enable_audio:
                    print("  [v] 发送语音消息")
                print("  [h] 显示会话历史")
                print("  [s] 保存会话")
                print("  [q] 退出")
                print("="*70)
                
                mode = input("请选择操作: ").strip().lower()
                
                if mode == "q":
                    print("\n👋 再见!")
                    break
                elif mode == "h":
                    self._show_history()
                    continue
                elif mode == "s":
                    self._save_conversation()
                    continue
                elif mode == "t":
                    user_input = input("💬 输入你的问题: ").strip()
                    if not user_input:
                        continue
                    
                    item = {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": user_input}]
                    }
                    
                    # 记录用户输入
                    self.conversation_history.append({
                        "role": "user",
                        "type": "text",
                        "content": user_input,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                elif mode == "v" and self.enable_audio:
                    duration = float(input("录音时长(秒，默认3): ") or "3")
                    wav_bytes = self.recorder.record(duration)
                    
                    if not wav_bytes:
                        continue
                    
                    audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
                    item = {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "audio": audio_b64,
                                "audio_format": "wav"
                            }
                        ]
                    }
                    
                    # 记录用户语音输入
                    self.conversation_history.append({
                        "role": "user",
                        "type": "audio",
                        "content": f"<audio: {len(wav_bytes)} bytes>",
                        "timestamp": datetime.now().isoformat()
                    })
                    
                else:
                    print("❌ 无效选项，请重试")
                    continue
                
                # 处理响应
                await self._handle_response(connection, item)
                
        except KeyboardInterrupt:
            print("\n\n⚠️ 收到中断信号")
        finally:
            if self.player:
                self.player.close()
    
    async def _handle_response(self, connection, item):
        """处理单轮对话响应"""
        print("\n🤖 助手回复:")
        print("-" * 70)
        
        try:
            # 发送消息
            await connection.conversation.item.create(item=item)
            await connection.response.create()
            
            # 收集响应数据
            audio_buffer = bytearray()
            text_buffer = []
            transcript_buffer = []
            
            # 处理事件流
            async for event in connection:
                event_type = event.type
                
                if event_type == "response.text.delta":
                    # 文本增量
                    delta = event.delta or ""
                    print(delta, flush=True, end="")
                    text_buffer.append(delta)
                    
                elif event_type == "response.text.done":
                    print()  # 换行
                    
                elif event_type == "response.audio.delta":
                    # 音频增量
                    pcm_bytes = base64.b64decode(event.delta)
                    if self.player:
                        self.player.write_bytes(pcm_bytes)
                    audio_buffer.extend(pcm_bytes)
                    
                elif event_type == "response.audio_transcript.delta":
                    # 音频转文本增量
                    delta = event.delta or ""
                    transcript_buffer.append(delta)
                    
                elif event_type == "response.audio_transcript.done":
                    transcript = "".join(transcript_buffer)
                    if transcript:
                        print(f"\n[转写] {transcript}")
                    
                elif event_type == "response.error":
                    error_msg = getattr(event, "error", "未知错误")
                    print(f"\n❌ 错误: {error_msg}")
                    
                elif event_type == "response.done":
                    break
            
            # 保存响应
            response_text = "".join(text_buffer)
            response_audio = bytes(audio_buffer)
            
            if response_text or response_audio:
                self.conversation_history.append({
                    "role": "assistant",
                    "type": "text" if response_text else "audio",
                    "content": response_text or f"<audio: {len(response_audio)} bytes>",
                    "timestamp": datetime.now().isoformat()
                })
            
            # 持久化
            if self.logger:
                if response_audio:
                    self.logger.save_audio(response_audio, self.audio_config.samplerate)
                if response_text:
                    self.logger.save_text(response_text)
            
            print("-" * 70)
            
        except Exception as e:
            print(f"\n❌ 处理响应时出错: {e}")
    
    def _show_history(self):
        """显示会话历史"""
        if not self.conversation_history:
            print("\n📭 暂无会话历史")
            return
        
        print("\n📜 会话历史:")
        print("="*70)
        for i, msg in enumerate(self.conversation_history, 1):
            role_emoji = "👤" if msg["role"] == "user" else "🤖"
            print(f"{i}. {role_emoji} [{msg['type']}] {msg['timestamp']}")
            print(f"   {msg['content'][:100]}...")
            print()
    
    def _save_conversation(self):
        """保存会话历史"""
        if self.logger:
            self.logger.save_conversation(self.conversation_history)
        else:
            print("❌ 未配置日志记录器")


def list_audio_devices():
    """列出可用的音频设备"""
    print("\n🎤 可用音频设备:")
    print("="*70)
    devices = sd.query_devices()
    print(devices)
    print("="*70)


async def run_single_query(args):
    """单次查询模式"""
    audio_config = AudioConfig(
        samplerate=args.rate,
        input_device=args.input_device,
        output_device=args.output_device
    )
    
    client = RealtimeClient(
        api_key=args.api_key,
        use_aad=args.use_aad
    )
    
    logger = ConversationLogger(args.save_dir) if args.save_dir else None
    
    session = RealtimeSession(
        client=client,
        audio_config=audio_config,
        logger=logger,
        system_prompt=args.system,
        enable_audio=not args.no_audio
    )
    
    print(f"\n🚀 单次查询模式: {args.text}")
    
    async with client.client.beta.realtime.connect(model=client.model) as connection:
        modalities = ["text"] if args.no_audio else ["text", "audio"]
        await connection.session.update(
            session={
                "modalities": modalities,
                "instructions": args.system or "You are a helpful assistant."
            }
        )
        
        item = {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": args.text}]
        }
        
        await session._handle_response(connection, item)


async def run_interactive(args):
    """交互模式"""
    audio_config = AudioConfig(
        samplerate=args.rate,
        input_device=args.input_device,
        output_device=args.output_device
    )
    
    client = RealtimeClient(
        api_key=args.api_key,
        use_aad=args.use_aad
    )
    
    logger = ConversationLogger(args.save_dir) if args.save_dir else None
    
    session = RealtimeSession(
        client=client,
        audio_config=audio_config,
        logger=logger,
        system_prompt=args.system,
        enable_audio=not args.no_audio
    )
    
    await session.start()


def build_argparser():
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="Azure OpenAI Realtime API - 语音助手（集成优化版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 列出音频设备
  python realtime_voice_assistant.py --list-devices
  
  # 交互模式（默认）
  python realtime_voice_assistant.py
  
  # 仅文本模式
  python realtime_voice_assistant.py --no-audio
  
  # 单次查询
  python realtime_voice_assistant.py --text "你好，介绍一下自己"
  
  # 保存会话记录
  python realtime_voice_assistant.py --save-dir ./logs
  
  # 指定音频设备
  python realtime_voice_assistant.py --input-device 1 --output-device 2
  
  # 使用Azure AD认证
  python realtime_voice_assistant.py --use-aad
        """
    )
    
    # 设备管理
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="列出可用音频设备后退出"
    )
    parser.add_argument(
        "--input-device",
        type=int,
        default=None,
        help="输入设备索引（麦克风）"
    )
    parser.add_argument(
        "--output-device",
        type=int,
        default=None,
        help="输出设备索引（扬声器）"
    )
    
    # 音频配置
    parser.add_argument(
        "--rate",
        type=int,
        default=DEFAULT_SAMPLERATE,
        help=f"采样率，默认 {DEFAULT_SAMPLERATE}"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="录音时长（秒），默认3.0"
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="仅文本模式（不启用语音）"
    )
    
    # API配置
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Azure OpenAI API Key（默认从环境变量读取）"
    )
    parser.add_argument(
        "--use-aad",
        action="store_true",
        help="使用Azure Active Directory认证"
    )
    
    # 会话配置
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="系统提示词（设置助手角色/语气）"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="单次查询文本（执行后退出）"
    )
    
    # 日志配置
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="保存会话记录的目录"
    )
    
    return parser


async def main():
    """主函数"""
    parser = build_argparser()
    args = parser.parse_args()
    
    try:
        # 列出设备
        if args.list_devices:
            list_audio_devices()
            return
        
        # 单次查询模式
        if args.text:
            await run_single_query(args)
            return
        
        # 交互模式
        await run_interactive(args)
        
    except KeyboardInterrupt:
        print("\n\n👋 已退出")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
