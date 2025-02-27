from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def seperate_docs(docs,doc_ids,chunk_size=256, chunk_overlap=8,separators = ["\n\n", "\n",'。','！','?',"!", "?"]):
        """
        对每篇文档进行分块处理
        """
        child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators,keep_separator=True)
        sub_docs = []
        for i, doc in enumerate(docs):
            _id = doc_ids[i]
            doc = Document(page_content=doc)
            _sub_docs = child_text_splitter.split_documents([doc])
            for _doc in _sub_docs:
                _doc.metadata["id"] = _id
            sub_docs.extend(_sub_docs)
        print(f'文档分块,chunk_size:{chunk_size},chunk_overlap:{chunk_overlap},共{len(sub_docs)}个分块')
        return sub_docs