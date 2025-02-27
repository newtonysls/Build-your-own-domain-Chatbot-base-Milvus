from typing import List,Union,Any,Optional
import re
from .llm_response import AzureChatGPT,OpenaiChatGPT,Qwen

def refine_query(
        LLM:Union[AzureChatGPT,OpenaiChatGPT,Qwen,Any],
        query:str,
        history:List[dict]=[],
        logger=None
    )->List[str]:

    query_refine_prompt = "任务：输入对话历史上文history和当前提问query，你需要按照要求对query进行补全和拆解，输出改写后的query。\n\n问题改写类型：\n1.拆解：当query含义复杂，范围较广，可以拆分为多个意图更清晰的sub-query。\n2.补全：query简单，需要结合上下文进行理解，需要对query进行信息补全（时间、地点、主体信息等等补全）。\n3.无需改写：若query为打招呼、结束语、有害问题或者已完整清晰的问题则无需改写。\n\n要求：\n1.拆解query不能改变问题意图；\n2.补全query不要增加冗余的信息，要保证补全信息合理；\n\n输出格式：\n-首先输出开始符<start>\n-输出拆解或者补全的问题。序号.问题？[改写类型(拆解、补全和无需改写)：理由说明]\n-结尾输出结束符<end>\n\n示例：\n输入：\n对话历史history：\nQ：介绍一下XX公司\nA：XX公司是一家成立于2018年的AI虚拟人公司...\n\n当前提问query：招人吗\n\n输出：\n<start>\n1.XX公司还在招聘吗？[补全：补全公司名称让意图更为清晰]\n<end>\n\n输入：\n对话历史history：\nQ：北京有什么好玩的？\nA：北京是中国的首都，也是明清古朝的首都...\n\n当前提问query：哪些景点是必去的？另外有什么美食吗\n\n输出：\n<start>\n1.北京有哪些必去景点呢？[拆解：询问北京有哪些景点]\n2.北京有哪些美食？[拆解：询问北京有哪些美食]\n<end>\n\n输入：\n对话历史history：无\n\n当前提问query：你好呀\n\n输出：\n<start>\n1.你好呀[无需改写：打招呼用语，无需改写]\n<end>\n\n输入：对话历史history：{}\n\n当前提问query：{}\n\n输出："
    if history==[]:
        # 对首个query不进行改写
        logger.debug("对首个query不进行改写")
        return [query]
    qa_history = []
    for item in history:
        if item["role"] =="system":
            continue
        else:
            if item["role"]=="user":
                qa_history.append(f'Q：{item["content"]}')
            else:
                assistant = item["content"]
                assistant = re.sub(r"[\s\S]*</think>","",assistant)
                assistant = assistant.strip("\n").strip(" ")
                qa_history.append(f"A：{assistant}")
    
    prompt = query_refine_prompt.format("\n".join(qa_history),query)
    messages = [
        {"role":"system","content":"你是一名AI助手，负责解决用户问题。"},
        {"role":"user","content":prompt}
    ]
    refine_query_llm_res = LLM.get_llm_response(
        messages,
    )
    
    logger.debug(f"Tools-Query改写-【LLM返回内容】：{refine_query_llm_res}")
    
    
    # 模型llm是否返回"无需改写字样"
    if "无需改写" in refine_query_llm_res:
        return [query]
    
    # 这基于正则表达式提取改写（补全+拆解）的query结果
    try:
        refine_query_search = re.findall(r"<start>([\s\S]*)<end>",refine_query_llm_res)
        refine_query_str = refine_query_search[0]
        refine_query_str = refine_query_str.strip("\n").strip(" ")
        refine_queries = refine_query_str.split("\n")
        refine_queries = [re.sub(r"[1-9]?\.","",q) for q in refine_queries]
        refine_queries = [re.sub(r"\[[^\]]*\]","",q) for q in refine_queries]
        return refine_queries
    except:
        logger.debug(f"Tools-Query改写-【改写出错】")
        return [query]