import os
import json
from typing import List,Union,Optional
import re

markdown_title_types = ["#","##","###","####"]
markdown_title_types = [t +" " for t in markdown_title_types]

def md_split_by_level(texts:List[str],level:Union[int,str]=1)->List[dict]:
    """
    Function：按照level对texts进行拆分，保存level级别的标题以方便补全
    input:
        texts：List[str],逐行读取的md文件
        level：Union[int,str]，1代表一级标题，等价于“#”
    
    output:
        splited_texts：List[dict]
        {
            "title":"level级别对应的title",
            "level":"level",
            "texts":"level级别下对应的文本内容",
        }
    """

    # pre-processing,去掉texts中头尾没有意义的内容
    while texts != [] and re.sub(r"[\n ]*","",texts[0]) =="":
        texts.pop(0)

    while texts != [] and re.sub(r"[\n ]*","",texts[-1]) =="":
        texts.pop()
    

    spliter = markdown_title_types
    level_spliter = spliter[level-1]
    splited_texts = []
    cur_text = []

    index_count = 0

    for line in texts:
        if line.startswith(level_spliter):
            if cur_text!=[]:
                if cur_text[0].startswith(level_spliter):
                    title = cur_text[0]
                    cur_text.pop(0)
                else:
                    title = None
                splited_item = {}
                splited_item["level"] = level
                splited_item["texts"] = cur_text
                if title is not None:
                    title = title.lstrip(level_spliter)
                    title = f"{level_spliter}{index_count+1}.{title}"
                    index_count +=1
                splited_item["title"] = title
                splited_texts.append(splited_item)
                cur_text = []
        cur_text.append(line)
    

    if cur_text !=[]:
        if cur_text[0].startswith(level_spliter):
            title = cur_text[0]
            cur_text.pop(0)
        else:
            title = None
        splited_item = {}
        splited_item["level"] = level
        splited_item["texts"] = cur_text
        # if title is not None:
        #     title = title.lstrip(level_spliter)
        #     title = f"{level_spliter}{index_count+1}.{title}"
            
        splited_item["title"] = title
        splited_texts.append(splited_item)
    return splited_texts

def md_split(texts:List[str],min_split_level:int=2)->List[str]:
    """
    function：从一级标题逐级切分文档，并补全对应的标题
    input:
        texts:List[str],逐行读取的md文件
        min_split_level:int，最小切分标题粒度，1等价于“#”，2等价于“##”
    
    output:
        results:List[str],每个item为按照min_split_level为最小切分粒度的知识块
    """
    ready_to_split = []
    ready_to_split.append((0,[],texts))
    results = []

    while ready_to_split!=[]:
        item = ready_to_split.pop()
        pre_level = item[0]
        pre_title = item[1]
        ready_texts = item[2]
        if pre_level < min_split_level:
            level_md_split_results = md_split_by_level(ready_texts,pre_level+1)
            for split_item in level_md_split_results:
                ready_to_split.append((pre_level+1,pre_title+[split_item["title"]],split_item["texts"]))
        else:
            titles = [t for t in pre_title if t is not None]
            all_chunk_texts = titles+ready_texts
            all_chunk_texts = [c.strip("\n") for c in all_chunk_texts]
            results.append("\n".join(all_chunk_texts))
    
    return results