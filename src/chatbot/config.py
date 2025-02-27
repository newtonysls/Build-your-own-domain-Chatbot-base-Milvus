import json

class Retrieval:
    milvus_client = None
    collection_name = None
    embedding_model_or_path="bge-base-zh-v1.5"
    search_k=30
    retrieval_top_k=5
    retrieval_similarity_threshold=0.6
    
class BM25:
    enable_bm25=False
    stop_words_file=None
    bm25_model_path=None
    retrieval_top_k=5

    
class Rerank:
    enable_rerank=False
    rerank_model_name_or_path="bge-reranker-large"
    rerank_top_k=10
    
class QueryRefine:
    enable_query_refine=False

class LocalLLMArgs:
    enable_local_llm = False
    llm_name_or_path = None
    gpu_memory_utilization=0.8
    max_model_len=4096

    
class RagConfig:
    def __init__(self,config_file_path:str):
        self.config_file_path = config_file_path
        with open(config_file_path,"r",encoding="utf-8") as f:
            config = json.load(f)
        self.config = config
        self.log_folder = config.get("log_folder",None)
        self.prompt_file_path = config["prompt_file_path"]
        self.top_p = config.get("top_p",0.8)
        self.temperature = config.get("temperature",0.6)
        self.max_tokens = config.get("max_tokens",1024)
        self.stream = config.get("stream",False)
        self.only_retrieval = config.get("only_retrieval",False)
        self.max_token_of_chat_history = config.get("max_token_of_chat_history",2048)

        self.llm_type = config.get("llm_type",None)
        assert self.llm_type is not None,ValueError("llm type应为下面值其中之一：[\"openai\",\"azure\",\"qwen\",\"local_llm\"]")

        if self.llm_type =="local_llm":
            self.local_llm = LocalLLMArgs()
            self.local_llm.enable_local_llm = config.get("enable_local_llm",True)
            self.local_llm.llm_name_or_path = config.get("llm_name_or_path",None)
            assert self.local_llm.llm_name_or_path is not None,ValueError("选择本地LLM，你需要指定LLM路径")
            self.local_llm.gpu_memory_utilization = config.get("gpu_memory_utilization",0.8)
            self.local_llm.max_model_len = config.get("max_model_len",4096)
        
        self.retrieval = Retrieval()
        assert "retrieval" in config,"必须在config里面设置检索参数."
        assert "embedding_model_or_path" in config["retrieval"],"config里必须指定embedding_model_or_path."
        assert "milvus_client" in config["retrieval"],"config里必须指定已经建立后milvus检索数据库链接."
        assert "collection_name" in config["retrieval"],"config里必须指定数据表collection名称."
        
        assert config["retrieval"]["embedding_model_or_path"] !="","The embedding_model_or_path can not be empty."
        
        self.retrieval.embedding_model_or_path = config["retrieval"]["embedding_model_or_path"]
        self.retrieval.milvus_client = config["retrieval"]["milvus_client"]
        self.retrieval.collection_name = config["retrieval"]["collection_name"]

        self.retrieval.search_k = config["retrieval"].get("search_k",30)
        self.retrieval.retrieval_top_k = config["retrieval"].get("retrieval_top_k",10)
        self.retrieval.retrieval_similarity_threshold = config["retrieval"].get("retrieval_similarity_threshold",0.6)
        
        self.bm25 = BM25()
        if "bm25" in config:
            self.bm25.enable_bm25 = config["bm25"].get("enable_bm25",True)
            self.bm25.stop_words_file = config["bm25"].get("stop_words_file",None)
            self.bm25.bm25_model_path = config["bm25"].get("bm25_model_path",None)
            self.bm25.retrieval_top_k = config["bm25"].get("retrieval_top_k",5)
        
        self.rerank = Rerank()
        if "rerank" in config:
            self.rerank.enable_rerank=config["rerank"].get("enable_rerank",False)
            self.rerank.rerank_model_name_or_path=config["rerank"].get("rerank_model_name_or_path","bge-reranker-large")
            self.rerank.rerank_top_k=config["rerank"].get("rerank_top_k",10)
            
        self.query_refine = QueryRefine()
        if "query_refine" in config:
            self.query_refine.enable_query_refine=config["query_refine"].get("enable_query_refine",False)