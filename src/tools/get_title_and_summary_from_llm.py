from typing import List,Union,Any,Optional
from .llm_response import AzureChatGPT,OpenaiChatGPT,Qwen
import json

def get_title_and_summary(
        LLM:Union[AzureChatGPT,OpenaiChatGPT,Qwen,Any],
        document:str,
    )->tuple:
    system = "你是一名AI助手，你需要按照用户要求完成指定的文本任务"
    prompt_template = (
        "## 任务：请你仔细阅读下面文本，为其核心主题内容生成标题和概要。概要和标题是为了达到更好的检索效果。\n"
        "## 要求：\n1.为你认为该段文本的核心内容生成标题，要求简短且有代表性\n2.概要需要简短，不能冗余，主要描述文本的核心内容且字数小于100。\n"
        "## 输出格式\n"
        "使用json格式输出，概要请不要以“本文介绍了、该文档介绍了”开始，请直接描述核心内容。以下是返回示例\n"
        "{\"title\":XXXX,\"summary\":XXXXX}\n"
        "\n"
        "文本：%s"
    )
    p = prompt_template%(document)
    messages = [
        {"role":"system","content":system},
        {"role":"user","content":p},
    ] 
    llm_response = LLM.get_llm_response(messages)
    try:
        json_r = json.loads(llm_response)
    except:
        try:
            r = r.strip("```")
            r = r.strip("json")
            json_r = json.loads(r)
        except:
            json_r = {"title":"","summary":""}
    
    return json_r["title"], json_r["summary"]