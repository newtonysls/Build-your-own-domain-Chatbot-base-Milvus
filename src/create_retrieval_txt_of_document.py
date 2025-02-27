import json
import os
import argparse
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tools.get_title_and_summary_from_llm import get_title_and_summary
from tools.llm_response import AzureChatGPT,OpenaiChatGPT,Qwen
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser("构建检索文本到知识块的映射关系")
    parser.add_argument("--chunk_size", type=int, nargs='*',default=256,help="将知识块切分为检索文本块的分块大小，可用空格分开输入多个chunk size")
    parser.add_argument("--chunk_overlap", type=int, nargs='*',default=16,help="将知识块切分为检索文本块的分块之间的重合长度，可用空格分开输入多个chunk overleap")
    parser.add_argument("--documents_file_path",type=str,default="..\..\data\documents\splited_documents.json",help="文档经过切分好的知识块文件路径")
    parser.add_argument("--add_title_and_summary_to_retrieval",action="store_true",help="是否自动生成标题和摘要信息辅助检索")
    parser.add_argument("--retrieval_texts_save_path",type=str,default=r"..\..\data\documents\retrival_texts_to_documents.json",help="文档经过切分好的知识块文件路径")
    parser.add_argument("--retrieval_texts_cache_save_path",type=str,default=r"..\..\data\documents\retrival_texts_to_documents_cahce.json",help="文档经过切分好的知识块文件路径")

    args = parser.parse_args()
    
    # 选择使用什么模型获取标题和摘要信息
    llm = OpenaiChatGPT()

    if not os.path.exists(args.documents_file_path):
        FileExistsError(f"文件：{args.documents_file_path}不存在，请输入正确的知识块路径。")

    splited_documents = []
    with open(args.documents_file_path,"r",encoding="utf-8") as f:
        splited_documents = json.load(f)
    
    separators = ["\n\n", "\n",'。','！','?',"!", "?"]
    retrieval_text_to_document = []

    for document_id,document in tqdm(enumerate(splited_documents),desc="知识块subchunk..."):
        print(f"Processing:{document_id+1}/{len(splited_documents)}")
        unique_chunked_texts = set()

        # 对document知识块，按照chunk size和chunk overlap进行拆分，方便后续转化为向量
        for chunk_size,chunk_overlap in zip(args.chunk_size,args.chunk_overlap):
            child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators,keep_separator=True)
            document_lc = Document(page_content=document,metadata={"doc_id":document_id})
            subchunk_document_lc = child_text_splitter.split_documents([document_lc])
            for _doc in subchunk_document_lc:
                if _doc.page_content not in unique_chunked_texts:
                    unique_chunked_texts.add(_doc.page_content)
        
        # 获取标题、摘要辅助检索信息
        if args.add_title_and_summary_to_retrieval:
            title,summary = get_title_and_summary(llm,document)
            if title !="":
                unique_chunked_texts.add(title)
            if summary !="":
                unique_chunked_texts.add(summary)
        
        unique_chunked_texts = list(unique_chunked_texts)
        for chunked_text in unique_chunked_texts:
            retrieval_text_to_document.append(
                dict(
                    retrieval_text=chunked_text,
                    document=document,
                )
            )
        
        if (document_id+1)%10==0:
            # 保存cache，防止访问GPT出错或者过久无结果
            with open(args.retrieval_texts_cache_save_path,"w",encoding="utf-8") as f:
                f.write(json.dumps(retrieval_text_to_document,ensure_ascii=False,indent=2))

    # 保存检索文本到知识点的映射
    retrieval_texts_save_folder = os.path.split(args.retrieval_texts_save_path)[0]
    if not os.path.exists(retrieval_texts_save_folder):
        os.makedirs(retrieval_texts_save_folder,exist_ok=True)

    with open(args.retrieval_texts_save_path,"w",encoding="utf-8") as f:
        f.write(json.dumps(retrieval_text_to_document,ensure_ascii=False,indent=2))

    print("--对知识块进行chunk已完成.")



if __name__=="__main__":
    main()