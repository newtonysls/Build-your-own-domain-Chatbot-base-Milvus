from transformers import AutoModel,AutoTokenizer
from FlagEmbedding import FlagModel
from typing import List
from milvus_model.sparse.bm25.tokenizers import Analyzer,build_default_analyzer
import jieba
import math
import re
from pymilvus.model.sparse import BM25EmbeddingFunction
from milvus_model.base import BaseEmbeddingFunction
import jieba
import math
import re
import json
import logging
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional

import requests
from scipy.sparse import csr_array, vstack
import numpy as np


class DenseEmbedding:
    def __init__(self,embedding_model_name_or_path:str) -> None:
        self.embedding_model_name_or_path = embedding_model_name_or_path
        self.model = FlagModel(embedding_model_name_or_path, 
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True)
    
    def embedding_query(self,queries:List[str]):
        return self.model.encode_queries(queries)
    
    def embedding_document(self,documents:List[str]):
        return self.model.encode(documents)

class Tokenizer:
    
    def __init__(self,stop_words_file:Optional[str]) -> None:
        self.stop_words_file = stop_words_file
        if self.stop_words_file is not None:
            stop_words = []
            with open(stop_words_file,"r",encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip("\n")
                    if len(line) > 0:
                        stop_words.append(line)
            self.stop_words_pattern = "[,./?!;:@#$%^&*()，。、？《》【】“”！：；]|\[|\]|\"|'|\\\\|\n|\t|[-—]" + "|" + "|".join(stop_words)
        else:
            self.stop_words_pattern = "[,./?!;:@#$%^&*()，。、？《》【】“”！：；]|\[|\]|\"|'|\\\\|\n|\t|[-—]"
    
    def tokenize(self, text: str):
        error_message = "Each tokenizer must implement its 'tokenize' method."
        raise NotImplementedError(error_message)
    
    def stopword_filter(self,text:str):
        return re.sub(self.stop_words_pattern,"",text)

class JiebaTokenizer(Tokenizer):
    def __init__(self,stop_words_file:Optional[str]):
        super().__init__(stop_words_file)

    def tokenize(self, text: str):
        text = self.stopword_filter(text)
        return jieba.lcut(text)
    
class BaseTokenizer(Tokenizer):
    def __init__(self,stop_words_file:Optional[str]):
        super().__init__(stop_words_file)

    def tokenize(self, text: str):
        text = self.stopword_filter(text)
        return list(text)

class Preprocessor:
    def apply(self, text: str):
        error_message = "Each preprocessor must implement its 'apply' method."
        raise NotImplementedError(error_message)

class BM25Preprocessor(Preprocessor):
    def apply(self, text: str):
        """
        将文本均转化为小写，避免英文出现大小写不匹配
        """
        return text.lower()
    
class TextFilter:
    def apply(self, tokens: List[str]):
        error_message = "Each filter must implement the 'apply' method."
        raise NotImplementedError(error_message)

class Analyzer:
    def __init__(
        self,
        name: str,
        tokenizer: Tokenizer,
        preprocessors: Optional[List[Preprocessor]] = None,
        filters: Optional[List[TextFilter]] = None,
    ):
        self.name = name
        self.tokenizer = tokenizer
        self.preprocessors = preprocessors
        self.filters = filters

    def __call__(self, text: str):
        for preprocessor in self.preprocessors:
            text = preprocessor.apply(text)
        tokens = self.tokenizer.tokenize(text)
        for _filter in self.filters:
            tokens = _filter.apply(tokens)
        return tokens

BM25MilvusClass = BM25EmbeddingFunction(build_default_analyzer(language="zh")).__class__

class BM25ReviseForMultiTurn(BM25MilvusClass):
    
    def __init__(self,
            analyzer: Analyzer = None,
            corpus: Optional[List] = None,
            k1: float = 1.5,
            b: float = 0.75,
            epsilon: float = 0.25,
            num_workers: int = 1,
        ):
        super().__init__(analyzer,corpus,k1,b,epsilon,num_workers)
        
    
    def _encode_query_revise(self, query_list: List[str]) -> csr_array:
        """
        query:List[str]，当前对话session的所有query，[query1,query2,query3,...]
        """
        query_weights = []
        terms = []
        for index,query in enumerate(query_list):
            term_single_query = self.analyzer(query)
            terms.extend(term_single_query)
            query_weights.extend([index**2+1 for _ in range(len(term_single_query))])
        
        query_weights = [weight/max(query_weights) for weight in query_weights]
        
        values, rows, cols = [], [], []
        for index,term in enumerate(terms):
            if term in self.idf:
                values.append(self.idf[term][0] * query_weights[index])
                rows.append(0)
                cols.append(self.idf[term][1])
        return csr_array((values, (rows, cols)), shape=(1, len(self.idf))).astype(np.float32)
    
    def encode_queries(self, queries_of_session: List[List[str]]) -> csr_array:
        sparse_embs = [self._encode_query_revise(queries) for queries in queries_of_session]
        return vstack(sparse_embs).tocsr()