import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import gradio as gr
from chatbot.chatbot import ChatBot
import pandas as pd
from gradio import ChatMessage,Markdown,HTML,Image
from random import choice
import re
import requests
import json
import argparse
from typing import List

class GrBot:
    def __init__(self,bot) -> None:
        self.bot = bot
    
    def chat(self,query:str,history):
        history = self.bot.chat(query,history)
        return history,history

# class TestBot:
#     def __init__(self) -> None:
#         pass
#     def chat(self,query,history):
#         history.append({"role":"user","content":query})
#         history.append({"role":"assistant","content":f"回复-{query}"})
#         print(history)
#         return history,history


def main():
    parser = argparse.ArgumentParser("运行基于Milvus数据库的Chatbot")
    parser.add_argument("--chatbot_config",type=str,help="Chatbot的Configuration")
    parser.add_argument("--max_length_of_chat_history",type=int,default=4096,help="对话历史上下文长度限制")
    args = parser.parse_args()

    # 选择使用什么模型获取标题和摘要信息
    # 初始化chatbot
    print("初始化ChatBot!")
    cooker = ChatBot(args.chatbot_config)
    gr_bot = GrBot(cooker)
    print("初始化ChatBot完成!")

    # gr_bot = TestBot()

    # 设置gradio
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(type="messages")
        query = gr.Textbox(placeholder="Type your question here")
        # clear = gr.Button("清空对话历史")
        history = gr.State(value=[])
        gr.Interface(
            fn= gr_bot.chat,
            inputs=[query,history],
            outputs=[chatbot,history],
            title="神厨小福贵",
        )

    demo.launch(server_name="0.0.0.0",server_port=10086,share=True)
if __name__=="__main__":
    main()