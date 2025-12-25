# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple web interactive chat demo based on gradio."""

from argparse import ArgumentParser
from threading import Thread
import openai
import os
import gradio as gr
import torch
from queue import Queue
from dotenv import load_dotenv, find_dotenv

# 加载 .env 文件中定义的环境变量
_ = load_dotenv(find_dotenv())

openai.api_type = "azure"
openai.api_version = os.getenv("AZURE_SERVICE_VERSION")
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.azure_endpoint = os.getenv("AZURE_SERVICE_ENDPOINT")
azure_oai_model = os.getenv("AZURE_SERVICE_MODEL")

def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--share", action="store_true", default=True,
                        help="Create a publicly shareable link for the interface.")
    parser.add_argument("--inbrowser", action="store_true", default=False,
                        help="Automatically launch the interface in a new tab on the default browser.")
    parser.add_argument("--server-port", type=int, default=8000,
                        help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="127.0.0.1",
                        help="Demo server name.")  

    args = parser.parse_args()
    return args

def _chat_stream(query, history):
    conversation = [
        {'role': 'system', 'content': '你是一个全能型小助手, 请回答下面的问题：'},
    ]
    # history is a list of [user, assistant] pairs
    for query_h, response_h in history:
        conversation.append({'role': 'user', 'content': query_h})
        conversation.append({'role': 'assistant', 'content': response_h})
    conversation.append({'role': 'user', 'content': query})

    generation_kwargs = dict(
        model=azure_oai_model,
        messages=conversation,
        temperature=0.7,
    )

    # 创建一个队列来存储返回值
    result = Queue()

    def worker(result, **generation_kwargs):
        response = openai.chat.completions.create(**generation_kwargs)
        result.put(response.choices[0].message.content)

    thread = Thread(target=worker, args=(result,), kwargs=generation_kwargs)
    thread.start()

    yield result.get()

def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _launch_demo(args):

    def predict(_query, _chatbot, _task_history):
        print(f"User: {_query}")
        # Use dict messages so Gradio Chatbot accepts them
        _chatbot.append({'role': 'user', 'content': _query})
        _chatbot.append({'role': 'assistant', 'content': ''})
        full_response = ""
        response = ""
        for new_text in _chat_stream(_query, history=_task_history):
            response += new_text
            # update last assistant message content
            if isinstance(_chatbot[-1], dict):
                _chatbot[-1]['content'] = response

            yield _chatbot
            full_response = response

        print(f"History: {_task_history}")
        # store history as list of dict pairs for consistency
        _task_history.append([{'role': 'user', 'content': _query}, {'role': 'assistant', 'content': full_response}])
        print(f"GPT5.2-Chat: {full_response}")

    def regenerate(_chatbot, _task_history):
        if not _task_history:
            yield _chatbot
            return
        item = _task_history.pop(-1)
        # remove last two displayed entries (user and assistant)
        if _chatbot:
            # pop assistant
            _chatbot.pop(-1)
        if _chatbot:
            # pop user
            _chatbot.pop(-1)
        # item is [user_dict, assistant_dict]
        user_message = item[0]['content'] if isinstance(item[0], dict) else item[0]
        yield from predict(user_message, _chatbot, _task_history)

    def reset_user_input():
        return gr.update(value="")

    def reset_state(_chatbot, _task_history):
        _task_history.clear()
        _chatbot.clear()
        _gc()
        return _chatbot

    with gr.Blocks() as demo:
        gr.Markdown("""<center><font size=8>GPT5.2-Chat For DDC</center>""")

        chatbot = gr.Chatbot(label='GPT5.2-Chat For DDC', height=320, elem_classes="control-height")
        query = gr.Textbox(lines=2, label='Input')
        task_history = gr.State([])

        with gr.Row():
            empty_btn = gr.Button("🧹 Clear History (清除历史)")
            submit_btn = gr.Button("🚀 Submit (发送)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")

        submit_btn.click(predict, [query, chatbot, task_history], [chatbot], show_progress=True)
        submit_btn.click(reset_user_input, [], [query])
        empty_btn.click(reset_state, [chatbot, task_history], outputs=[chatbot], show_progress=True)
        regen_btn.click(regenerate, [chatbot, task_history], [chatbot], show_progress=True)

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()
    _launch_demo(args)


if __name__ == '__main__':
    main()