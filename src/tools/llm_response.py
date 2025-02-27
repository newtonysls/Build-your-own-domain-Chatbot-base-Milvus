import json
from openai import OpenAI
from openai import AzureOpenAI
import os
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig
import argparse
import math
from tqdm import trange
import json
# from vllm import LLM, SamplingParams
import re
from typing import List

# 访问Openai的GPT LLM需要调用代理，可以在py程序内设置，推荐在系统变量里设置
os.environ["http_proxy"] = "http://127.0.0.1:7897"
os.environ["https_proxy"] = "http://127.0.0.1:7897"

class AzureChatGPT:
    """
    微软Azure平台Openai GPT的调用方法
    """
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version= os.getenv("AZURE_OPENAI_API_VERSION") if os.getenv("AZURE_OPENAI_API_VERSION") is not None else "2024-02-01"
        )
        self.MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    def get_llm_response(self,messages):
        """
        messages:组转的messages，符合对话格式
        """
        completion = self.client.chat.completions.create(
            model=self.MODEL,
            messages=messages
        )
        res = json.loads(completion.model_dump_json())
        return res["choices"][0]["message"]["content"]

class OpenaiChatGPT:
    
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.MODEL = os.getenv("OPENAI_MODEL")

    def get_llm_response(self,messages):
        """
        messages:组转的messages，符合对话格式
        """
        completion = self.client.chat.completions.create(
            model=self.MODEL,
            messages=messages
        )
        res = json.loads(completion.model_dump_json())
        return res["choices"][0]["message"]["content"]

class Qwen:
    def __init__(self):
        self.MODEL = os.getenv("QWEN_OPENAI_MODEL")
        self.client = OpenAI(
            api_key=os.getenv("QWEN_OPENAI_API_KEY"),
            base_url=os.getenv("QWEN_OPENAI_ENDPOINT") if os.getenv("QWEN_OPENAI_ENDPOINT") is not None else "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    
    def get_llm_response(self,messages):
        completion = self.client.chat.completions.create(
            model=self.MODEL,
            messages=messages
        )
        res = json.loads(completion.model_dump_json())

        return res["choices"][0]["message"]["content"]


def loadModelAndTokenizer(model_path:str,gpu_memory_utilization:float=0.9,max_model_len:int=4096):
    """
    基于VLLM框架加载本地LLM
    input：
        model_path：本地模型路径
        gpu_memory_utilization：GPU预先加载占用率，VLLM会自动按照这个比例占据显卡的显存
        max_model_len：max_model_len
    
    output：
        model：VLLM model
        tokenizer：tokenizer
    """
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    # Input the model name or path. Can be GPTQ or AWQ models.
    # model = LLM(model=model_path,quantization="gptq",gpu_memory_utilization=gpu_memory_utilization,max_model_len=max_model_len)
    model = LLM(model=model_path,gpu_memory_utilization=gpu_memory_utilization,max_model_len=max_model_len)
    return model, tokenizer

class LocalLLM:
    def __init__(self,
        model_path:str="Qwen2.5-14B-Instruct-GPTQ-Int4",
        gpu_memory_utilization:float=0.8,
        max_model_len:int=4096
    ) -> None:
        self.model_path = model_path
        self.model, self.tokenizer = loadModelAndTokenizer(model_path,gpu_memory_utilization,max_model_len)
    
    def get_llm_response(self,
            messages:List[dict],
            temperature:float=0.8,
            top_p:float=0.8,
            repetition_penalty:float=1.1,
            max_tokens:int=2048
        ):
        
        assert type(messages)==list
        chat_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        sampling_params = SamplingParams(
            temperature=temperature, 
            top_p=top_p, 
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            )
        gen_results = self.model.generate(
            [chat_prompt],
            sampling_params
        )
        generated_text = gen_results[0].outputs[0].text.strip()
        first_token_time = gen_results[0].metrics.first_token_time
        return generated_text