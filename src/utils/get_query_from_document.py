import json
from random import sample,choice,randint
from tqdm import tqdm
import re
from collections import Counter
import argparse
from ..tools.llm_response import AzureChatGPT,OpenaiChatGPT,Qwen


PROMPT="""## 任务：输入文本document，输出满足下面要求的问题。
在和用户进行对话的任务中，用户问题可以从问题范围和问题类型两个方面进行分类：
一、范围
-全局：全局性、范围性的提问，问题对应的范围比较广。
-局部：问题聚焦于小范围，提问比较具体。
二、类型
-查询：只需要某个片段的文档就可以回答的问题，不需要推理和思考
-操作：问题意图为询问某个操作流程是怎么样，怎么做？怎么办？
-推理：问题意图为需要结合文本进行思考和推理的，不能够直接回答。
查询和操作问题可以是可以根据对应的文本进行直接回答的，推理问题要求一定的思考和推理流程。

## 要求
-以用户的提问方式和风格，例如口语化、正式等风格
-输出所有满足上述要求的问题
-不要输出傻瓜式问题

## 输出格式
-严格按照下面格式输出问题
   -首先输出<start>
   -输出问题：序号.问题？[范围(全局/局部)-类型(查询/操作/推理)：属于该范围/类型的理由]
   -最后输出<end>
输出示例：
<start>
1.什么是整数？[局部-查询：该问题是询问整数定义，不需要推理，属于局部-查询问题]
2.整数和小数怎么相加？[全局-操作：该问题询问的意图是整数和小数的加法流程，范围涉及到整数和小数加法，属于全局-操作问题]
3.7+0.12的结果是多少？[局部-推理：该问题是计算整数和小数相加的结果，涉及到推理计算，属于局部-推理问题]
<end>

## 示例
文本Document：
# 3D人物
## 二、3D人物的定制
#### 套装设置
**套装编辑选项**
- 上衣：更换适合的上衣。
- 下装：更换适合的下装。
- 鞋子：更换适合的鞋子。

输出:
<start>
1.3D人物可以定制吗？[局部-查询：可直接根据文档内容回答，不需要思考，因此属于局部-查询]
2.可以定制3D人物套装吗？[局部-查询：可直接根据文档回答，不需要思考，因此属于局部-查询]
3.3D人物可以定制哪些服装？[全局-查询：可以根据文档内容直接回答，但是范围较广，因此属于全局-查询]
4.可以定制3D人物的上衣吗[局部-查询：可直接根据文档内容回答，不需要思考，因此属于局部-查询]
5.可以定制3D人物的下装吗[局部-查询：可直接根据文档内容回答，不需要思考，因此属于局部-查询]
6.可以定制3D人物的鞋子吗[局部-查询：可直接根据文档内容回答，不需要思考，因此属于局部-查询]
7.有没有什么方法能够让我的3D人物看起来个性化？[全局-推理：需要理解通过定制套装（衣服和鞋子等）得到更加个性化的形象，需要理解和推理，因此属于全局-推理]
<end>

## 输入
文本Document：
{}

输出："""


        

def main():
    """
    说明：本py代码的输入参数为已经拆分好的单个知识点。因为在抽取问题的prompt定义中，包含了全局问题的提取，
    理想情况是将多个知识点（相关的知识点）作为文本Document加入到PROMPT中，生成需要结合多个知识点理解的问题检索数据。
    但是LLM生成的全局问题，可能是只是来源于其中的2、3几个知识点，不一定是全部，因此query-documents并不是严格对应的关系。
    **注意：为了避免训练检索模型带来的干扰，本代码只将单个知识点加入到PROMPT中**
    """
    parser = argparse.ArgumentParser("通过文档知识点抽取高质量query，并用于检索模型的微调。")
    parser.add_argument("--splited_documents_file_path",type=str,default="..\data\documents\splited_documents.json",help="切分为知识点之后的文件路径")
    parser.add_argument("--queries_from_document_res_file_path",type=str,default="..\data\documents\query_from_splited_documents.json",help="抽取问题的保存路径")
    args = parser.parse_args()

    with open(args.splited_documents_file_path,"r",encoding="utf-8") as f:
        data = json.load(f)

    res = []
    # 这里的LLM可以自行选择
    llm = OpenaiChatGPT()
    llm.MODEL = "gpt-4o"
    for idx,doc in tqdm(enumerate(data),desc="问题抽取中..."):
        p = PROMPT.format(doc)
        messages = [{"role": "user", "content": p}]
        attempt = 0
        max_num_attempt = 3
        while attempt<max_num_attempt:
            # 防止因为网络问题，访问api出错
            attempt+=1
            try:
                llm_response = llm.get_llm_response(messages)
            except:
                llm_response = ""
                print(f"进行第{attempt+1}次尝试！")
            if llm_response!="":
                break
        
        if llm_response!="":
            #  基于正则表达式提取query
            try:
                re_search_res=re.findall(r"<start>([\s\S]*)<end>",llm_response)
                assert len(re_search_res)==1,ValueError(f"提取出错，LLM：{llm_response}")
                re_queries_str = re_search_res[0]
                queries = re_queries_str.strip(" ").strip("\n").strip(" ").split("\n")
                queries = [re.sub(r"\[[^\]]*\]","",q) for q in queries]
                queries = [re.sub(r"^[1-9]?[0-9]\.","",q) for q in queries]
                queries_of_single_doc = queries
            except:
                print(f"解析错误，LLM：{llm_response}")
        else:
            queries_of_single_doc = []

        print(f"文档：{doc}")
        print(f"生成问题：{llm_response}")

        res.append({"doc":doc,"query":queries_of_single_doc})

        if len(res)>0 and len(res)%10==0:
            with open(args.queries_from_document_res_file_path.replace(".json","-cache.json"),"w",encoding="utf-8") as f:
                f.write(json.dumps(res,ensure_ascii=False,indent=2))

    with open(args.queries_from_document_res_file_path,"w",encoding="utf-8") as f:
        f.write(json.dumps(res,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()