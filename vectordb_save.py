from __future__ import annotations
import os
from dotenv import load_dotenv, find_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from zhipuai import ZhipuAI
from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, root_validator

from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader
from ZhipuAIEmbeddings import ZhipuAIEmbeddings
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)
# 读取本地/项目的环境变量。
# find_dotenv()寻找并定位.env文件的路径
# load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中  
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

# 如果你需要通过代理端口访问，你需要如下配置
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
# os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890'

# 获取folder_path下所有文件路径，储存在file_paths里
file_paths = []
folder_path = 'data_base/knowledge_db'
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        file_paths.append(file_path)
# print(file_paths[:3])


# 遍历文件路径并把实例化的loader存放在loaders里
loaders = []
cnt=0
for file_path in file_paths:

    file_type = file_path.split('.')[-1]
    if file_type == 'pdf':
        loaders.append(PyMuPDFLoader(file_path))
        cnt=cnt+1
    elif file_type == 'md':
        loaders.append(UnstructuredMarkdownLoader(file_path))
        cnt=cnt+1
    if cnt is 2:break

# 下载文件并存储到text
texts = []

for loader in loaders: texts.extend(loader.load())


text = texts[1]
# print(f"每一个元素的类型：{type(text)}.", 
#     f"该文档的描述性数据：{text.metadata}", 
#     f"查看该文档的内容:\n{text.page_content[0:]}", 
#     sep="\n------\n")




# 切分文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=40, chunk_overlap=20)

split_docs = text_splitter.split_documents(texts)


# 创建一个 ZhipuEmbeddings 实例
zhipu_embedding = ZhipuAIEmbeddings()
embedding = zhipu_embedding



# def zhipu_embedding(text: str):
#     response = client.embeddings.create(
#         model="embedding-2",
#         input=text,
#     )
#     return response

# text = '要生成 embedding 的输入文本，字符串形式。'
# response = zhipu_embedding(text=text)
# 使用 OpenAI Embedding
# from langchain.embeddings.openai import OpenAIEmbeddings
# 使用百度千帆 Embedding
# from langchain.embeddings.baidu_qianfan_endpoint import QianfanEmbeddingsEndpoint
# 使用我们自己封装的智谱 Embedding，需要将封装代码下载到本地使用

# 定义 Embeddings
# embedding = OpenAIEmbeddings() 
# embedding = QianfanEmbeddingsEndpoint()

# 定义持久化路径
persist_directory = 'data_base/vector_db/chroma'

print(split_docs[:5])
vectordb = Chroma.from_documents(
    documents=split_docs[:5], # 为了速度，只选择前 20 个切分的 doc 进行生成；使用千帆时因QPS限制，建议选择前 5 个doc
    embedding=embedding,
    persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
)

vectordb.persist()

print(f"向量库中存储的数量：{vectordb._collection.count()}")
