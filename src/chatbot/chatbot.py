from .config import *
from tools.llm_response import AzureChatGPT,OpenaiChatGPT,Qwen
import os
import time
from typing import List,Union,Optional
from tools.set_logger import set_logger
from tools.rerank import rerank
from tools.query_refine import refine_query
from pymilvus import (
    MilvusClient
)
from build_database.init_embedding_model import (
    DenseEmbedding,
    BM25Preprocessor,
    BaseTokenizer,
    BM25ReviseForMultiTurn,
    Analyzer
)
from transformers import  AutoTokenizer,AutoModelForSequenceClassification
import numpy as np
import torch

class ChatBot:
    def __init__(self,config_or_file_path):
        """
        function：根据config初始化RAG各项功能（system prompt、向量化模型、连结milvus数据库、加载LLM等）
        """
        
        if type(config_or_file_path)==str:
            config = RagConfig(config_or_file_path)
        elif isinstance(config_or_file_path,RagConfig):
            config = config_or_file_path
        else:
            raise TypeError("必须提供config实例或者config json文件路径。")
        
        self.config = config
        
        # set log
        if not os.path.exists(self.config.log_folder):
            os.makedirs(self.config.log_folder,exist_ok=True)
        self.logger = set_logger(self.config.log_folder)
        
        # LLM
        llm_type_map = dict(
            openai=OpenaiChatGPT,
            azure=AzureChatGPT,
            qwen=Qwen
        )
        if self.config.llm_type=="local_llm":
            self.LLM = LocalLLM(
                self.config.local_llm.llm_name_or_path,
                self.config.local_llm.gpu_memory_utilization,
                self.config.local_llm.max_model_len
            )
        else:
            self.LLM = llm_type_map[self.config.llm_type]()
        self.logger.info("成功加载LLM.")
        

        # database, dense embedding model and bm25 model

        # database link
        self.milvus_client = MilvusClient(uri=self.config.retrieval.milvus_client)
        if self.milvus_client.has_collection(self.config.retrieval.collection_name):
            self.logger.debug("连结数据库成功！")
            col_status=self.milvus_client.get_collection_stats(collection_name=self.config.retrieval.collection_name)
            self.logger.debug(col_status)
        else:
            ValueError("请检查milvus数据连结或者collection name。")

        # bm25 sparse embedding
        if self.config.bm25.enable_bm25:
            tokenizer = BaseTokenizer(self.config.bm25.stop_words_file)
            processor = BM25Preprocessor()
            analyzer = Analyzer("bm25",tokenizer,[processor],[])
            self.bm25_embedding = BM25ReviseForMultiTurn(analyzer)
            self.bm25_embedding.load(self.config.bm25.bm25_model_path)
        
        # encoder
        self.dense_embedding = DenseEmbedding(self.config.retrieval.embedding_model_or_path)

        # rerank
        if self.config.rerank.enable_rerank:
            # rerank model
            self.logger.info("准备加载rerank模型.")
            self.rerank_model_name_or_path =self.config.rerank.rerank_model_name_or_path
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(self.rerank_model_name_or_path)
            self.reranker_model = AutoModelForSequenceClassification.from_pretrained(self.rerank_model_name_or_path).cuda()
            self.reranker_model.eval()
            self.logger.info("加载rerank模型结束.")

        # chat info
        ## system prompt
        self.SYSTEM_PROMPT = ""
        with open(self.config.prompt_file_path,"r",encoding="utf-8") as f:
            for line in f.readlines():
                self.SYSTEM_PROMPT += line

    def _dense_retrieval(self,
                  query:List[str],
                  retrieval_args:Retrieval,
        )->set:
        """
        function:进行稠密检索
        input:
            query:List[str],对一个或者多个query进行检索，问题拆解的结果可能是多个问题
            retrieval_args:config.Retrieval，检索参数
        output：
            return：set，按照余弦相似度降序排序召回不重复文档
        """
        unique_dense_retrieval_documents = set()
        st_of_dense_search = time.time()

        # 获取queries的embedding
        queries_embeddings_tensor:torch.tensor = self.dense_embedding.embedding_query(query)
        batch_size = len(query)
        data = [np.array(queries_embeddings_tensor[i,:],dtype=np.float32) for i in range(batch_size)]

        dense_search_res = self.milvus_client.search(
            collection_name=self.config.retrieval.collection_name,
            anns_field="dense_vector",
            data=data,  # Use the `emb_text` function to convert the question to an embedding vector
            limit=retrieval_args.retrieval_top_k,  # Return top k results
            search_params={"metric_type": "COSINE", "params": {}},  # COSINE distance
            output_fields=["text"],  # Return the text field
        )

        for idx,dense_res_item in enumerate(dense_search_res):
            for res_item in dense_res_item:
                document = res_item["entity"].get("text")
                score = res_item["distance"]
                # if (score >=retrieval_args.retrieval_similarity_threshold) and (document not in unique_dense_retrieval_documents):
                if document not in unique_dense_retrieval_documents:
                    unique_dense_retrieval_documents.add(document)
                    self.logger.debug(f"向量检索：检索问题-{query[idx]}-检索分数-{score:.4f}-检索文档：{document}")

        et_of_dense_search = time.time()
        dense_search_t_s = et_of_dense_search - st_of_dense_search
        self.logger.info(f"向量检索耗时:{dense_search_t_s:.4f}")

        return unique_dense_retrieval_documents
    
    def _sparse_retrieval(self,
                       query:str,
                       bm25_args:BM25,
                       history:List[dict]=[])->set:
        """
        function:进行稀疏bm25检索（多轮，bm25编码输入包含历史query）
        input:
            query:str,当前提问，只能为字符串，为原问题或者改写后的问题（若改写后为多个问题则需要进行拼接）
            bm25_args:config.BM25
        output：
            return：set，按照bm25 score降序排序召回不重复文档
        """
        
        unique_sparse_retrieval_documents = set()
        st_of_sparse_search = time.time()
        previous_queries = [item["content"] for item in history if item["role"]=="user"]
        if previous_queries != [] and previous_queries[-1]==query:
            previous_queries.pop()
        previous_queries.append(query)

        sparse_search_res = self.milvus_client.search(
            collection_name=self.config.retrieval.collection_name,
            anns_field="sparse_vector",
            data=[
                self.bm25_embedding.encode_queries([previous_queries]) # 输入多轮对话的所有query，按照权重衰减计算bm25分数
            ],  # Use the `emb_text` function to convert the question to an embedding vector
            limit=bm25_args.retrieval_top_k,  # Return top 3 results
            search_params={"metric_type": "IP", "params": {}},  # Inner product distance
            output_fields=["text"],  # Return the text field
        )

        et_of_sparse_search = time.time()
        for res_item in sparse_search_res[0]:
            document = res_item["entity"].get("text")
            score = res_item["distance"]
            if document not in unique_sparse_retrieval_documents:
                unique_sparse_retrieval_documents.add(document)
                self.logger.debug(f"稀疏BM25检索：检索问题-{query}-检索分数-{score:.4f}-检索文档：{document}")
        
        sparse_search_t_s = et_of_sparse_search - st_of_sparse_search
        self.logger.info(f"稀疏BM25检索耗时:{sparse_search_t_s:.4f}")

        return unique_sparse_retrieval_documents

    def _rerank_retrieval_docs(self,
        query:str,
        retrieval_documents:List[str],
        rerank_args:Rerank)->List[str]:
        """
        function:对召回的文档进行重排
        """
        pairs = [[query,d] for d in retrieval_documents]
        rerank_scores,rerank_index = rerank(self.reranker_model,self.reranker_tokenizer,pairs)
        reranked_retrieval_docs = [retrieval_documents[idx] for idx in rerank_index]
        reranked_retrieval_docs = reranked_retrieval_docs[0:rerank_args.rerank_top_k]
        return reranked_retrieval_docs
        
    def concat_docs(self,docs):
        """
        function:拼接召回或者重排之后的文档
        docs:List[str]
        """
        docs = [f"{d}" for i,d in enumerate(docs)]
        return "\n\n".join(docs)
    
    def retrieval_documents(self,
        query:str,
        history:List[dict])->List[str]:
        """
        function:执行召回文档功能
        """
        # 启动RAG工作流

        ## query理解/改写(补全和拆解)
        if self.config.query_refine.enable_query_refine:
            refined_queries = refine_query(self.LLM,query,history,self.logger)
        else:
            refined_queries = [query]

        ## 文档召回
        ### 向量召回
        dense_retrieval_docs_set = self._dense_retrieval(refined_queries,self.config.retrieval)
        ### 稀疏bm25召回
        if self.config.bm25.enable_bm25:
            bm25_retrieval_query = "".join(refined_queries)
            sparse_retrieval_docs_set = self._sparse_retrieval(bm25_retrieval_query,self.config.bm25,history)
            dense_retrieval_docs_set.update(sparse_retrieval_docs_set)
        
        return list(dense_retrieval_docs_set)

    def chat(self,query:str,
             history,
        ):

        st = time.time()

        # 检索文档
        retrieval_docs= self.retrieval_documents(
            query,
            history
        )
        retrieval_docs_str = self.concat_docs(retrieval_docs)
        self.logger.debug(f"召回文档：{retrieval_docs_str}")
        # 更新system prompt里面的文档内容
        system_prompt = self.SYSTEM_PROMPT.format(retrieval_docs_str)
            
        #
        if history==[] or history[0]["role"]!="system":
            history = [{"role":"system","content":system_prompt}] + history
        else:
            assert history[0]["role"] == "system"
            history = system_prompt
            
        history.append(
            {"role":"user","content":query}
        )
        
        messages = history
        response = self.LLM.get_llm_response(messages)
        response = response.strip("\n")
        history.append(
            {"role":"assistant","content":response}
        )
        if history!=[] and history[0]["role"]=="system":
            history.pop(0)
        return history
        
        
        
        