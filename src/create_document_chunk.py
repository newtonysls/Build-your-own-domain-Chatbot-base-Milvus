from doc_segmentation import md_split
import json
import os
import argparse

def main():

    parser = argparse.ArgumentParser("对markdown文件根据标题层级进行拆分")
    parser.add_argument("--md_folder_path",type=str,default="..\..\data\HowToCook-master\dishes")
    parser.add_argument("--min_split_level",type=int,default=2,help="划分最小层级标题，1表示一级标题，依次类推")
    parser.add_argument("--document_save_path",type=str,default="..\..\data\documents\splited_documents.json",help="拆分好的知识块存储路径")

    args = parser.parse_args()

    total_splited_documents = []
    for root, dirs, files in os.walk(args.md_folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith(".md"):
                print(f"文件路径：{file_path}")
                document_md_texts = []
                with open(file_path,"r",encoding="utf-8") as f:
                    for line in f.readlines():
                        document_md_texts.append(line)
                splited_document = md_split(document_md_texts,args.min_split_level)
                total_splited_documents.extend(splited_document)

    
    # 保存拆分好的知识块
    document_save_folder = os.path.split(args.document_save_path)[0]
    if not os.path.exists(document_save_folder):
        os.makedirs(document_save_folder,exist_ok=True)

    with open(args.document_save_path,"w",encoding="utf-8") as f:
        f.write(json.dumps(total_splited_documents,ensure_ascii=False,indent=2))
    
    print("--拆分知识块完成.")

if __name__=="__main__":
    main()