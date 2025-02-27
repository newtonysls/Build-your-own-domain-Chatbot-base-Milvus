from init_embedding_model import (
    DenseEmbedding,
    BM25Preprocessor,
    BaseTokenizer,
    BM25ReviseForMultiTurn,
    Analyzer
)
import os
import json
import argparse
import time
from tqdm import tqdm,trange
import numpy as np
import math

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    MilvusClient
)

def main():
    parser = argparse.ArgumentParser("基于milvus构建hybrid search检索知识库")
    parser.add_argument("--retrieval_texts_to_documents_file_path",type=str,default="..\..\data\documents\retrival_texts_to_documents.json",help="检索文本和知识块的映射文件")
    parser.add_argument("--stop_words_file",type=str,default="..\..\data\bm25\stopwords.txt",help="BM25所使用的停用词")
    parser.add_argument("--bm25_k1",type=float,default=1.5,help="bm25参数k1")
    parser.add_argument("--bm25_b",type=float,default=0.75,help="bm25参数b")
    parser.add_argument("--embedding_model_name_or_path",type=str,default="..\..\models\bge-base-zh-v1.5",help="向量化or检索模型路径")
    parser.add_argument("--bm25_embedding_save_path",type=str,default="..\..\data\bm25\bm25_model.json",help="bm25算法模型保存")
    parser.add_argument("--gpu_id",type=str,default="0",help="将检索文本转化为向量时，需要使用gpu")
    parser.add_argument("--batch_size_for_data_insert",type=int,default=32,help="将检索向量批量插入milvus数据的batch size")
    parser.add_argument("--collection_name",type=str,default="hybrid_search",help="milvus数据库中的collection name")
    parser.add_argument("--milvus_database_url",type=str,help="milvus数据库链接")
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # 读取划分好的检索文本数据到知识块的映射
    retrieval_text_to_document = []
    with open(args.retrieval_texts_to_documents_file_path,"r",encoding="utf-8") as f:
        retrieval_text_to_document = json.load(f)
    
    total_documents = set([item["document"] for item in retrieval_text_to_document])
    total_documents = list(total_documents)
    
    # 稠密向量化模型设置
    print("初始化稠密向量化模型。")
    dense_embedding = DenseEmbedding(args.embedding_model_name_or_path)
    dense_dim = dense_embedding.model.model.config.hidden_size

    # 稀疏向量化模型设置
    # bm25 embeddings
    print("初始化稀疏向量化模型。")
    tokenizer = BaseTokenizer(args.stop_words_file)
    processor = BM25Preprocessor()
    analyzer = Analyzer("bm25",tokenizer,[processor],[])
    bm25_embedding = BM25ReviseForMultiTurn(analyzer,total_documents)
    bm25_embedding.save(args.bm25_embedding_save_path)
    
    # 连结Milvus 数据库，安装好Milvus容易docker后的服务器链接和端口
    print("连结milvus数据库。")
    try:
        connections.connect(uri=args.milvus_database_url)
        print("连结milvus数据库成功！")
    except:
        print(f"error：连结milvus数据库失败！请检查milvus_database_url：{args.milvus_database_url}有效性！")
    # Specify the data schema for the new Collection
    fields = [
        # Use auto generated id as primary key
        FieldSchema(
            name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100
        ),
        # Store the original text to retrieve based on semantically distance
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8000), # 这里milvus有bug，明明text不超过max_length但依然报长度错误，百度为编码原因，长度设置更大一点就好
        # Milvus now supports both sparse and dense vectors,
        # we can store each in a separate field to conduct hybrid search on both vectors
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
    ]
    schema = CollectionSchema(fields)
    print("构建milvus数据库fields。")

    # Create collection (drop the old one if exists)
    col_name = args.collection_name
    if utility.has_collection(col_name):
        print(f"collection name：{col_name} 已存在milvus数据库-{args.milvus_database_url}，删除该collection name。")
        Collection(col_name).drop()
    col = Collection(col_name, schema, consistency_level="Strong")
    
    # To make vector search efficient, we need to create indices for the vector fields
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    col.create_index("sparse_vector", sparse_index)
    dense_index = {"index_type": "AUTOINDEX", "metric_type": "COSINE"}
    col.create_index("dense_vector", dense_index)
    col.load()
    
    
    st = time.time()
    
    num_inser_steps = math.ceil(len(retrieval_text_to_document)/args.batch_size_for_data_insert)
    # Insert Data into Milvus Collection
    for step in trange(num_inser_steps,desc="将检索文本向量化插入milvus数据库..."):
        batch_data = retrieval_text_to_document[step*args.batch_size_for_data_insert:(step+1)*args.batch_size_for_data_insert]
        batch_documents = [item["document"] for item in batch_data]
        retrieval_texts = [item["retrieval_text"] for item in batch_data]

        # bm25将完整的知识块作为检索内容
        bm25_text_embeds = bm25_embedding.encode_documents(batch_documents)

        # 向量化将所有subchunk和辅助检索信息转化为向量
        dense_text_embeds = dense_embedding.embedding_document(retrieval_texts)
        try:
            batched_entities = [
                batch_documents,
                bm25_text_embeds,
                np.array(dense_text_embeds,dtype=np.float32),
            ]
            col.insert(batched_entities)
        except:
            print([len(s) for s in batch_documents])
    col.flush()
    print("Number of entities inserted:", col.num_entities)
    et = time.time()
    t_s= et-st
    print(f"构造数据库总共花时：{t_s:.4f}")

if __name__=="__main__":
    main()
