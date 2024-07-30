import os 
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatZhipuAI
from dotenv import find_dotenv,load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain.memory import ConversationBufferMemory
_ =load_dotenv(find_dotenv())
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")

embedding = ZhipuAIEmbeddings(api_key=ZHIPUAI_API_KEY)
# 初始化并加载 Chroma 向量数据库
vectordb = Chroma(
    persist_directory='data_base/vector_db/chroma',
    embedding_function=embedding
)
llm = ChatZhipuAI(model="glm-4-flash", temperature = 0,api_key=ZHIPUAI_API_KEY)

llm.invoke("请你自我介绍一下自己！")


template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
{context}
问题: {question}
"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                 template=template)

qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
question_1 = "什么是南瓜书？"
question_2 = "王阳明是谁？"
result = qa_chain.invoke({"query": question_1})
print("大模型+知识库后回答 question_1 的结果：")
print(result["result"])
result = qa_chain.invoke({"query": question_2})
print("大模型+知识库后回答 question_2 的结果：")
print(result["result"])

prompt_template = """请回答下列问题:
                            {}""".format(question_1)

### 基于大模型的问答
res=llm.invoke(prompt_template)
print(res.content)
prompt_template = """请回答下列问题:
                            {}""".format(question_2)

### 基于大模型的问答
print(llm.invoke(prompt_template).content)


memory = ConversationBufferMemory(
    memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
    return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
)