import streamlit as st
import os
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv, find_dotenv
from langchain_core.output_parsers import StrOutputParser

_ = load_dotenv(find_dotenv())    # read local .env file
ZHIPU_api_key=os.environ['ZHIPUAI_API_KEY']

def generate_response(input_text):
    llm = ChatZhipuAI(temperature=0.7,ZHIPU_api_key=ZHIPU_api_key)
    output = llm.invoke(input_text)
    output_parser = StrOutputParser()
    output = output_parser.invoke(output)
    # st.info(output)
    return output

# with st.form('my_form'):
#     text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
#     submitted = st.form_submit_button('Submit')
#     # if not openai_api_key.startswith('sk-'):
#     #     st.warning('Please enter your OpenAI API key!', icon='⚠')
#     # if submitted and openai_api_key.startswith('sk-'):
#     generate_response(text)


def get_vectordb():
    # 定义 Embeddings
    embedding = ZhipuAIEmbeddings()
    # 向量数据库持久化路径
    persist_directory = 'data_base/vector_db/chroma'
    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embedding
    )
    return vectordb

#带有历史记录的问答链
def get_chat_qa_chain(question:str,ZHIPU_api_key:str):
    vectordb = get_vectordb()
    llm = ChatZhipuAI(model_name = "glm-4-flash", temperature = 0.4,ZHIPU_api_key = ZHIPU_api_key)
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
        return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
    )
    retriever=vectordb.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )
    result = qa({"question": question})
    return result['answer']

#不带历史记录的问答链
def get_qa_chain(question:str,ZHIPU_api_key:str):
    vectordb = get_vectordb()
    llm = ChatZhipuAI(model_name = "glm-4-flash", temperature = 0.4,ZHIPU_api_key = ZHIPU_api_key)
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
    result = qa_chain({"query": question})
    return result["result"]




# Streamlit 应用程序界面
def main():
    st.title('🦜🔗 动手学大模型应用开发')
    ZHIPU_api_key = st.sidebar.text_input('ZhiPu API Key', type='password')

  
    # 用于跟踪对话历史
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    messages = st.container(height=300)
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append({"role": "user", "text": prompt})

        # 调用 respond 函数获取回答
        answer = generate_response(prompt)
        # 检查回答是否为 None
        if answer is not None:
            # 将LLM的回答添加到对话历史中
            st.session_state.messages.append({"role": "assistant", "text": answer})

        # 显示整个对话历史
        for message in st.session_state.messages:
            if message["role"] == "user":
                messages.chat_message("user").write(message["text"])
            elif message["role"] == "assistant":
                messages.chat_message("assistant").write(message["text"])   

        selected_method = st.radio(
        "你想选择哪种模式进行对话？",
        ["None", "qa_chain", "chat_qa_chain"],
        captions = ["不使用检索问答的普通模式", "不带历史记录的检索问答模式", "带历史记录的检索问答模式"])



if __name__ == "__main__":
    main()