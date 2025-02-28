# 基于Milvus构造菜谱小助手-神厨小福贵

## 知识库
文本知识库采用Github上Anduin2017这位程序员高级厨师开源项目：[程序员做饭指南](https://github.com/Anduin2017/HowToCook.git)。
这个菜谱知识块存储于*data\HowToCook-master\dishes*路径下。

选择这个知识库作为示例主要一下几个原因：
1. 这里面的菜谱都是以Markdown格式，可以直接读取，不需要额外的解析
2. 这里面的菜谱文档格式非常规范，标题层级合理，可以非常容易地基于规则进行知识块的拆分

### 1. 知识块的拆分
> 将文档拆分为含义独立的知识块。将文档拆分为独立、合格的知识块是一个非常重要的过程。这些知识块会被送入到大模型作为知识参考进而回答用的问题。因此结构清晰、内容独立完整的知识块是更加容易方便LLM去理解的。特别是在约束模型回答要求时。因此需要将知识块拆分完整，同时需要对知识块内容进行补全。

基于规则拆分markdown文档代码位于:*src\create_document_chunk.py*
```shell
# project\run_document_chunk.sh
python src\create_document_chunk.py\
    --md_folder_path "data\HowToCook-master\dishes"\
    --min_split_level 2\
    --document_save_path "data\documents\splited_documents.json"
```
运行上面代码，就会自动对菜谱文档进行切分和层级标题的补全。该代码的拆分思路是：根据markdown的层级标题，从大标题递归地拆分到小标题，并同时记录层级标题，然后对每个拆分好的知识块进行补全。
*min_split_level*参数的含义是最小拆分到哪个级别的标题，这里对菜谱文档进行分析最小拆到二级标题比较好，再小就容易让知识不完整。

### 2. subchunk和辅助检索信息的构建
