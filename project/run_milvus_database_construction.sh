D:/conda/envs/pytorch2/python src/build_database/build_database_with_milvus.py\
    --retrieval_texts_to_documents_file_path "data\documents\retrival_texts_to_documents.json"\
    --stop_words_file "data\bm25\stopwords.txt"\
    --embedding_model_name_or_path "models\bge-base-zh-v1.5"\
    --bm25_embedding_save_path "data\bm25\bm25_model.json"\
    --collection_name "hybrid_search_for_cooker"\
    --milvus_database_url "http://127.0.0.1:19530"