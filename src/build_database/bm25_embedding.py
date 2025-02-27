from init_embedding_model import (
    DenseEmbedding,
    JiebaTokenizer,
    BaseTokenizer,
    BM25ReviseForMultiTurn,
    Analyzer
)
from typing_extensions import List

# 停用词表，去除corpus中的无意义词
STOP_WORDS_FILE = "/data/megastore/Projects/PanHe/code/agent/rag_hp/dialogue/rag_projects/data/stop_words/hit_stopwords.txt"

# 知识库
CORPUS:List[str] = [
    "罗辑静立了一会儿，也走了。当地面的震动消失后，褐蚁从孤峰的另一边向下爬去，它要赶回蚁穴报告那只死甲虫的位置。天空中的星星密了起来，在孤峰的脚下，褐蚁又与蜘蛛交错而过，它们再次感觉到了对方的存在，但仍然没有交流。",
    "章北海轻轻摇摇头：“不好，也就是维持吧。”“你请个假吧。”“他刚住院时我已经请过一次了，现在这形势，到时候再说吧。”然后两人就义沉默了，他们之间每一次关于个人生活的交流都是这样，关于工作的谈话肯定会多一些，但也总是隔着一层东西。",
    "“罗老师，请把衣服换了吧。”刚进门的年轻人说，蹲下来拉开他带进来的提包，尽管他显得彬彬有礼，罗辑心里还是像吃了苍蝇似的不舒服。但当年轻人把包中的衣服拿出来时，罗辑才知道那不是给嫌犯穿的东西，而是一件看上去很普通的棕色夹克，他接过衣服翻着看了看，夹克的料子很厚实，接着发现史强和年轻人也穿着这种夹克，只是颜色不同。",
]

# 存储bm25计算好的IDF信息
BM25_MODEL_SAVE = "/data/megastore/Projects/PanHe/code/agent/rag_hp/dialogue/data/bm25_model.json"

# init bm25 embeddings
tokenizer = BaseTokenizer(STOP_WORDS_FILE)
analyzer = Analyzer("bm25",tokenizer,[],[])

"""
tokenizer:对文本进行分词，进阶可以选择基于jieba的分词，详见init_embedding_model
analyzer：对文本处理的包装，进阶可支持预处理、分词和过滤
"""

bm25_embedding = BM25ReviseForMultiTurn(analyzer,CORPUS) # BM25ReviseForMultiTurn会自动计算IDF等数据


"""
BM25ReviseForMultiTurn的初始化需要两个必要参数：analyzer和CORPUS
1.如果analyzer为None，则会自动初始化analyzer，会下载NLTK词表数据，需要外网，服务器无法访问
2.如果CORPUS为None，则BM25ReviseForMultiTurn不会自动计算IDF
"""

# bm25_embedding.fit(CORPUS) # 如果初始化CORPUS为None，则需要调用fit进行IDF的计算

bm25_embedding.save(BM25_MODEL_SAVE) # 保存bm25基于corpus计算的IDF

"""
#######获取document和query的向量#######
注：bm25算法对query和document的embedding是不一样的，需要调用不一样的函数

一、 编码query（基于优化后的多轮bm25检索）
bm25_embedding.encode_queries(queries_of_session:List[List[str]]) #对多个session的queries进行编码
bm25_embedding._encode_query_revise(queries:List[str]) #对单个session的queries进行编码

二、 编码document
bm25_embedding.encode_documents(documents:List[str]) #对批量documents进行编码
"""
embeds_of_batch_sessions_queries = bm25_embedding.encode_queries(
    [
        [
            "罗辑是谁？",
            "为什么他成为了执剑人？"
        ],
        [
            "章北海是谁？",
            "他被烙下思想钢印了吗"
        ],
    ]
)
embeds_of_single_session_queries = bm25_embedding._encode_query_revise(
    [
        "罗辑是谁？",
        "为什么他成为了执剑人？"
    ]
)
print(embeds_of_single_session_queries)
print(embeds_of_single_session_queries.toarray())
print(embeds_of_single_session_queries.toarray().shape)
print(f"BM25算法向量维度:{len(bm25_embedding.idf)}")
embeds_of_batch_documents = bm25_embedding.encode_documents(
    CORPUS
)
